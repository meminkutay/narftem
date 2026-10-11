"""nartfem.axi.multitire as the nartfem-apd response engine.

For every climate case (month x temperature quintile) and every axle type the moduli of the
stress-dependent layers are made self-consistent with the superposed stresses of all tires of
the axle group (iterated superposition, as ``nartfem.axi.multitire.run_multitire``), and the
strains are evaluated exactly inside the elements at nartfem-apd's analysis points (no element-center
interpolation).

Reuse across the 12 months x 5 quintiles x 4 axle types:
  * one mesh, one set of B matrices / integration constants, one solver pattern, one force
    vector per axle type (the tire is identical for all cases: nartfem-apd's 18-kip reference tire);
  * when only the AC moduli change, the element stiffness matrices of the AC elements are
    rescaled (K_e is proportional to E at fixed nu) instead of being recomputed;
  * the unbound modulus field of the previous case (same axle type) is the warm start of the
    nonlinear iteration (``warm_start`` option) - typically 2-4 iterations instead of 3-6.

Engineering choices (see also the report):
  * AC sublayers of nartfem-apd (0.5, 0.5, 1, 1 ... in) become separate elastic layers of the nartfem
    model (``ac_sublayer_interfaces``) so that each sublayer carries its own |E*|(T, f) exactly
    like MatLEA; the mesh therefore has node rows at the sublayer boundaries.
  * Points on an interior interface are evaluated in the element BELOW, which is what
    MatLEA_Sub01_Find_LayNo does for the layered-elastic solution (z < cumulative thickness).
  * Shear: nartfem's gamma_rz/2 is the tensor shear used in the rotation/superposition
    (``shear = "tensor"``); ``shear = "legacy"`` reproduces the nartfem-apd convention
    (ff_superpose_strains puts the engineering shear in the tensor), which only affects the
    maximum principal strain (top-down cracking).
  * ``load_scaling = "power"``: the case is solved at several pressures (``power_levels`` x
    the reference pressure) and the strains are log-log interpolated over the axle load in the
    damage model instead of nartfem-apd's linear scaling.
"""
from __future__ import annotations
import math
import time
import warnings
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import norm

from nartfem.axi.model import Model, Layer, Geogrid, model_from_dict
from nartfem.axi import mesh as meshmod
from nartfem.axi.elements import ElementData, overburden, d_matrix
from nartfem.axi.assembly import force_vector, fixed_dofs, Solver, assemble
from nartfem.axi.responses import State
from nartfem.axi.multitire import _superposed_tensor, _row_profiles, PSI_TO_KPA
from nartfem.solid3d.material import resilient_modulus as mr_3d
from nartfem.solid3d import geogrid as gg3d

from . import schema
from .engine_base import ResponseEngine, AXLE_KEYS, tire_positions

AC_SEED_PSI = 500000.0        # placeholder stiffness of AC elements before the first |E*| assignment
NARTFEM_LAYER_KEYS = ("name", "type", "thickness_in", "E_psi", "nu", "unit_weight_pcf", "k", "K0",
                      "residual_psi", "Mr_min_psi", "E_table", "Er_over_Ez", "nu_hor", "geogrids")


# ----------------------------------------------------------------------------- model building
def build_model(structure: dict, pressure_psi: float, radius_in: float, TH_sub=None, LayNo_sub=None,
                ac_sublayer_interfaces=True, mesh_scale=1.3, name="", subgrade_split_in=0.0) -> Model:
    """nartfem Model from the structure block.  With ``ac_sublayer_interfaces`` the AC layers are
    split into nartfem-apd's sublayers (TH_sub / LayNo_sub from ff_sublayer_MEPDGmethod)."""
    layers = structure["layers"]; roles = schema.layer_roles(layers)
    out = []
    for i, (L, r) in enumerate(zip(layers, roles)):
        ld = {k: L[k] for k in NARTFEM_LAYER_KEYS if k in L}
        ld.setdefault("type", "elastic")
        if r in ("AC", "EAC"):
            # viscoelastic AC: the modulus is |E*|(T, f) of the climate case, set per element in
            # solve_case before every solve; the value here is only a non-zero seed for the stiffness
            ld["type"] = "elastic"; ld["E_psi"] = AC_SEED_PSI; ld.pop("E_table", None)
        if r in ("AC", "EAC") and ac_sublayer_interfaces and TH_sub is not None:
            idx = np.flatnonzero(np.asarray(LayNo_sub) == i)
            for j, k in enumerate(idx):
                sub = dict(ld); sub["name"] = "%s[%d]" % (L.get("name", "AC"), j + 1)
                sub["thickness_in"] = float(TH_sub[k]); sub["geogrids"] = []
                out.append(sub)
        elif i == len(layers) - 1 and subgrade_split_in:
            # split the subgrade top into a layer of its own (same properties): the mesh then has node
            # rows / refinement around depth h + subgrade_split_in, which keeps the strain at the second
            # nartfem-apd subgrade point (h + 6 in) mesh-converged - the subgrade rut model uses the ratio of
            # the strains at h and h + 6 in and is very sensitive to it.
            top = dict(ld); top["name"] = ld.get("name", "Subgrade") + "[top]"; top["thickness_in"] = float(subgrade_split_in)
            top["geogrids"] = []
            out.append(top); out.append(ld)
        else:
            out.append(ld)
    d = {"name": name, "load": {"pressure_psi": pressure_psi, "radius_in": radius_in}, "layers": out,
         "domain": {"width_in": structure.get("domain", {}).get("width_in"),
                    "depth_in": structure.get("domain", {}).get("depth_in")},
         "mesh": {"scale": mesh_scale, "import_file": None},
         "solver": dict(structure.get("solver", {}))}
    m = model_from_dict(d)
    m.verbose = False
    return m


# ----------------------------------------------------------------------------- in-element evaluation
class StructuredEvaluator:
    """Exact in-element strains of the single-tire field at arbitrary (r, depth) for the
    structured (tensor-product) mesh of nartfem.axi: vectorized over points."""

    def __init__(self, mesh, ed):
        self.mesh, self.ed = mesh, ed
        self.rs = np.unique(np.round(mesh.nodes[:, 0], 9))
        self.zs = np.unique(np.round(-mesh.nodes[:, 1], 9))              # positive depths
        nr, nz = self.rs.size, self.zs.size
        if (nr - 1) * (nz - 1) != mesh.n_elements:
            raise ValueError("StructuredEvaluator needs the generated tensor-product mesh")
        # element e = i*(nz-1) + j  (i radial column, j vertical row) - verify
        i = np.arange(nr - 1)[:, None]; j = np.arange(nz - 1)[None, :]
        e = (i * (nz - 1) + j).ravel()
        rc = (0.5 * (self.rs[:-1, None] + self.rs[1:, None]) + 0 * j).ravel()
        zc = (0 * i - 0.5 * (self.zs[None, :-1] + self.zs[None, 1:])).ravel()
        if not (np.allclose(mesh.center[e, 0], rc) and np.allclose(mesh.center[e, 1], zc)):
            raise ValueError("unexpected element numbering of the nartfem mesh")
        self.coords = mesh.nodes[mesh.conn]
        self.rtol = 1e-9 * max(mesh.width, 1.0)

    def locate(self, r, depth, side="below"):
        r = np.asarray(r, float); depth = np.asarray(depth, float)
        rs, zs = self.rs, self.zs
        i = np.searchsorted(rs, r, side="right") - 1
        i = np.clip(i, 0, rs.size - 2)
        if side == "below":
            j = np.searchsorted(zs, depth, side="right") - 1
        else:
            j = np.searchsorted(zs, depth, side="left") - 1
        j = np.clip(j, 0, zs.size - 2)
        e = i * (zs.size - 1) + j
        xi = 2.0 * (r - rs[i]) / (rs[i + 1] - rs[i]) - 1.0
        eta = 2.0 * (zs[j + 1] - depth) / (zs[j + 1] - zs[j]) - 1.0
        return e, np.clip(xi, -1, 1), np.clip(eta, -1, 1)

    def strains(self, u, r, depth, side="below"):
        """(npts, 4): eps_r, eps_z, eps_t, gamma_rz (engineering) at the points."""
        e, xi, eta = self.locate(r, depth, side)
        c = self.coords[e]                                              # (n,4,2)
        N = 0.25 * np.stack([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)], axis=1)         # (n,4)
        dN = 0.25 * np.stack([np.stack([-(1 - eta), -(1 - xi)], 1), np.stack([(1 - eta), -(1 + xi)], 1),
                              np.stack([(1 + eta), (1 + xi)], 1), np.stack([-(1 + eta), (1 - xi)], 1)], axis=1)  # (n,4,2)
        J = np.einsum("nki,nkj->nji", c, dN)
        detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        invJ = np.empty_like(J)
        invJ[:, 0, 0] = J[:, 1, 1]; invJ[:, 0, 1] = -J[:, 0, 1]; invJ[:, 1, 0] = -J[:, 1, 0]; invJ[:, 1, 1] = J[:, 0, 0]
        invJ /= detJ[:, None, None]
        dNdrz = np.einsum("nij,nkj->nki", invJ, dN)                     # (n,4,2)
        rr = np.einsum("nk,nk->n", N, c[:, :, 0])
        on_axis = rr <= self.rtol
        rsafe = np.where(on_axis, 1.0, rr)
        d = u[self.ed.dof[e]]                                           # (n,8)
        ur, uz = d[:, 0::2], d[:, 1::2]
        eps_r = np.einsum("nk,nk->n", dNdrz[:, :, 0], ur)
        eps_z = np.einsum("nk,nk->n", dNdrz[:, :, 1], uz)
        eps_t = np.where(on_axis, eps_r, np.einsum("nk,nk->n", N, ur) / rsafe)
        gam = np.einsum("nk,nk->n", dNdrz[:, :, 1], ur) + np.einsum("nk,nk->n", dNdrz[:, :, 0], uz)
        return np.column_stack([eps_r, eps_z, eps_t, gam])


def superpose_points(evalr, u, tires, xyz, shear="tensor", side="below"):
    """Superposed Cartesian strains at the points xyz (n,3) [x, y, depth] for identical tires at
    ``tires`` [(x, y), ...]; returns dict eps_xx, eps_yy, eps_zz, eps_princ (nartfem-apd sign: tension +)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    n = x.size
    T = np.zeros((n, 3, 3))
    fac = 1.0 if shear == "legacy" else 0.5
    for (tx, ty) in tires:
        dx, dy = x - tx, y - ty
        r = np.hypot(dx, dy); th = np.arctan2(dy, dx)
        s4 = evalr.strains(u, r, z, side)
        er, ez, et, g = s4[:, 0], s4[:, 1], s4[:, 2], s4[:, 3] * fac
        c, s = np.cos(th), np.sin(th)
        T[:, 0, 0] += c * c * er + s * s * et
        T[:, 1, 1] += s * s * er + c * c * et
        T[:, 2, 2] += ez
        exy = c * s * (er - et); exz = c * g; eyz = s * g
        T[:, 0, 1] += exy; T[:, 1, 0] += exy
        T[:, 0, 2] += exz; T[:, 2, 0] += exz
        T[:, 1, 2] += eyz; T[:, 2, 1] += eyz
    princ = np.linalg.eigvalsh(T)[:, -1]
    return {"eps_xx": T[:, 0, 0], "eps_yy": T[:, 1, 1], "eps_zz": T[:, 2, 2], "eps_princ": princ}


# ----------------------------------------------------------------------------- the engine
class NartfemEngine(ResponseEngine):
    name = "nartfem"

    def __init__(self, ctx: dict, structure: dict, options: dict = None, log=None):
        """``ctx``: the Step-1 quantities (see engine_base.context_from_step1)."""
        super().__init__(ctx)
        self.opt = dict(schema.DEFAULT_ENGINE_OPTIONS); self.opt.update(options or {})
        self.log = log or (print if self.opt.get("verbose") else (lambda *a, **k: None))
        self.structure = structure
        self.roles = schema.layer_roles(structure["layers"])
        self.model = build_model(structure, ctx["tirePressure"], ctx["CR_in"], ctx["TH_sub"], ctx["LayNo_sub"],
                                 self.opt["ac_sublayer_interfaces"], self.opt["mesh_scale"], "nartfem-apd case",
                                 self.opt.get("subgrade_split_in", 0.0))
        m = self.model
        self.mesh = meshmod.generate(m)
        self.res = m.resolved()
        self.ed = ElementData(self.mesh, m.gauss_points)
        self.ed.add_interface_points(np.concatenate([[0.0], m.thickness_cum, [self.mesh.depth]]))
        self.evalr = StructuredEvaluator(self.mesh, self.ed)
        self.solver = Solver(2 * self.mesh.n_nodes, fixed_dofs(self.mesh))
        self.F = force_vector(self.mesh, m.pressure_psi, m.radius_in)
        self.gz = overburden(self.mesh, self.res, m.thickness)
        # map nartfem-apd sublayers (E_sub entries) -> nartfem elements
        self.sub_of_elem = self._sublayer_of_elements(ctx["TH_sub"])
        self.ac_elems = np.flatnonzero(np.isin(self.sub_of_elem, ctx["IACsub"]))
        self.nonlin_layers = [i for i, L in enumerate(m.layers) if L.is_nonlinear]
        self.has_nonlin = bool(self.nonlin_layers)
        self.rcols, _, self.nrow, self.ncol = _row_profiles(self.mesh, np.zeros((self.mesh.n_elements, 1)))
        self.row_of = np.arange(self.mesh.n_elements) % self.nrow
        self.th_cum = np.concatenate([[0.0], m.thickness_cum, [self.mesh.depth]])
        self.states = {}          # per axle type: State (warm start)
        self.stats = {"solves": 0, "iterations": 0, "time_s": 0.0, "not_converged": 0}
        self.log("nartfem engine: mesh %d elements, %d nodes; nonlinear layers: %s" %
                 (self.mesh.n_elements, self.mesh.n_nodes, [m.layers[i].name for i in self.nonlin_layers]))

    # ---- helpers
    def _sublayer_of_elements(self, TH_sub):
        cum = np.concatenate([[0.0], np.cumsum(TH_sub)])
        depth = -self.mesh.center[:, 1]
        k = np.searchsorted(cum, depth, side="right") - 1
        return np.clip(k, 0, len(TH_sub))                 # == len(TH_sub) -> subgrade (last E_sub entry)

    def _state(self, key):
        st = self.states.get(key)
        if st is None:
            st = State(self.model, self.mesh, self.ed, self.res)
            st.gz = self.gz
            self.states[key] = st
        return st

    def _set_moduli(self, st, Mr_new, idx):
        ratio = Mr_new / st.Mr[idx]
        st.D[idx] *= ratio[:, None, None]; st.Ke[idx] *= ratio[:, None, None]
        st.Mr[idx] = Mr_new

    def _mr_superposed(self, st, tires, n_phi):
        """Mr_calc for all elements from the superposed stress state (copy of the closure in
        nartfem.axi.multitire.run_multitire - kept here so that multitire.py stays untouched)."""
        model, mesh, ed, res = self.model, self.mesh, self.ed, self.res
        d = st.u[ed.dof]
        st.strain = np.einsum("nij,nj->ni", ed.Bc, d)
        st.stress = np.einsum("nij,nj->ni", st.D, st.strain)
        Mr_new = st.Mr.copy(); st.delsig[:] = 0.0
        Pa = model.atm_pressure_psi; row_of = self.row_of; th_cum = self.th_cum
        for i in self.nonlin_layers:
            L = model.layers[i]
            idx = np.flatnonzero(mesh.layer == i)
            K0, ResH = res["K0"][i], res["residual"][i]
            S = _superposed_tensor(mesh, st.stress, st.gz, res, i, tires, n_phi, mesh.center[idx, 0], row_of[idx])
            g = st.gz[idx][:, None]
            S[:, :, 0] += g * K0 + ResH; S[:, :, 1] += g * K0 + ResH; S[:, :, 2] += g
            delsig = np.zeros(idx.size)
            if L.geogrids:
                z_bottom = -th_cum[i + 1]
                rc = np.concatenate([np.linspace(0, 10, 30), 3.0 ** np.arange(math.log(10.5) / math.log(3), math.log(mesh.width) / math.log(3) + 1e-12, 0.1)])
                if rc[-1] != mesh.width:
                    rc = np.append(rc, mesh.width)
                for gsp in L.geogrids:
                    zg = z_bottom + gsp.l_in
                    rows_lay = np.unique(row_of[idx]); zrow = np.array([mesh.center[np.flatnonzero(row_of == j)[0], 1] for j in rows_lay])
                    jrow = rows_lay[np.argmin(np.abs(zrow - zg))]
                    Sg = _superposed_tensor(mesh, st.stress, st.gz, res, i, tires, n_phi, rc, np.full(rc.size, jrow))
                    gzg = np.interp(-zg, -mesh.center[np.flatnonzero(row_of == jrow), 1], st.gz[np.flatnonzero(row_of == jrow)])
                    Sg[:, :, 0] += gzg * K0 + ResH; Sg[:, :, 1] += gzg * K0 + ResH; Sg[:, :, 2] += gzg
                    Sg = Sg.reshape(-1, 6); Sg[:, :3] = np.maximum(Sg[:, :3], 0.0)
                    Mr0 = mr_3d(Sg, res["mat_type"][i], res["kmat"][:, i], Pa, res["Mr_min"][i])
                    ds = gg3d.delta_sigma_max(Sg * PSI_TO_KPA, Mr0 * PSI_TO_KPA, res["nu"][i], gsp) / PSI_TO_KPA
                    ds = ds.reshape(rc.size, n_phi).mean(axis=1)
                    ak = Akima1DInterpolator(rc, ds, method="makima")
                    dz = np.abs(mesh.center[idx, 1] - zg); half = 0.5 * gsp.delta_in
                    fac = np.where(dz < half, 1 - dz / half, 0.0)
                    delsig += fac * np.maximum(0.0, ak(np.clip(mesh.center[idx, 0], rc[0], rc[-1])))
            S[:, :, 0] += delsig[:, None]; S[:, :, 1] += delsig[:, None]
            S[:, :, :3] = np.maximum(S[:, :, :3], 0.0)
            Mr_phi = mr_3d(S.reshape(-1, 6), res["mat_type"][i], res["kmat"][:, i], Pa, res["Mr_min"][i]).reshape(idx.size, n_phi)
            Mr_new[idx] = np.exp(np.log(Mr_phi).mean(axis=1))
            st.delsig[idx] = delsig
        return Mr_new

    def solve_case(self, key, E_sub, tires, load_factor=1.0):
        """Set the AC sublayer moduli (E_sub: legacy Ez row without the 2 index columns), solve
        the nonlinear multitire problem for the tires; returns the State (u, Mr)."""
        t0 = time.time()
        model = self.model; st = self._state(key)
        E_elem = np.asarray(E_sub, float)[self.sub_of_elem[self.ac_elems]]
        self._set_moduli(st, E_elem, self.ac_elems)
        if not self.opt["warm_start"] and self.has_nonlin:
            nl = np.flatnonzero(st.nonlin); self._set_moduli(st, st.Mr_seed[nl], nl)
        F = self.F * load_factor
        st.K = assemble(st.Ke, self.ed.dof, 2 * self.mesh.n_nodes)
        st.u = self.solver.solve(st.K, F); self.stats["solves"] += 1
        st.converged, st.iterations, st.conv_error = True, 0, 0.0
        if self.has_nonlin:
            n_phi = int(self.opt["n_phi"])
            nl = np.flatnonzero(st.nonlin)
            a = model.radius_in
            active = (st.nonlin & ((self.mesh.center[:, 0] - a) <= -2 * self.mesh.center[:, 1]))[nl]
            zval = norm.ppf(model.reliability_pct / 100.0); tol = model.max_error_pct / 100.0
            Mr_calc = self._mr_superposed(st, tires, n_phi)
            x = np.log(st.Mr[nl]); gx = np.log(Mr_calc[nl]) - x
            X, G = [], []; st.converged = False; st.iterations = model.max_iter
            for it in range(1, model.max_iter + 1):
                X.append(x.copy()); G.append(gx.copy())
                if len(X) > model.anderson_memory:
                    X.pop(0); G.pop(0)
                if len(X) >= 2:
                    dG = np.array([G[i + 1] - G[i] for i in range(len(G) - 1)]).T
                    dX = np.array([X[i + 1] - X[i] for i in range(len(X) - 1)]).T
                    gamma = np.linalg.lstsq(dG, gx, rcond=None)[0]
                    x = x + model.anderson_beta * gx - (dX + model.anderson_beta * dG) @ gamma
                else:
                    x = x + model.anderson_beta * gx
                self._set_moduli(st, np.exp(x), nl)
                st.K = assemble(st.Ke, self.ed.dof, 2 * self.mesh.n_nodes)
                st.u = self.solver.solve(st.K, F); self.stats["solves"] += 1
                Mr_calc = self._mr_superposed(st, tires, n_phi)
                gx = np.log(Mr_calc[nl]) - x
                err = np.abs(1 - np.exp(-gx))
                e = err[active].mean() + zval * err[active].std(ddof=1); st.conv_error = e
                if e < tol:
                    st.converged, st.iterations = True, it
                    break
            self.stats["iterations"] += st.iterations
            if not st.converged:
                self.stats["not_converged"] += 1
                warnings.warn("nartfem engine: case %s did not converge (error %.1f%%)" % (str(key), 100 * st.conv_error))
        self.stats["time_s"] += time.time() - t0
        return st

    def responses(self, E_sub, levels=(1.0,)):
        """{axle: [resp_level0, resp_level1, ...]} with resp = dict(eps_xx, eps_yy, eps_zz, eps_princ)
        at ctx['xyz_<axle>'] for each load factor in ``levels`` (pressure x level)."""
        ctx = self.ctx; out = {}
        shear = self.opt["shear"]
        for ax in AXLE_KEYS:
            tires = tire_positions(ctx, ax)
            xyz = ctx["xyz_" + ax]
            lst = []
            for lev in levels:
                st = self.solve_case((ax, lev), E_sub, tires, lev)
                lst.append(superpose_points(self.evalr, st.u, tires, xyz, shear=shear, side="below"))
            out[ax] = lst
        return out

    # ---- diagnostics
    def layer_moduli(self, key=None):
        """Mean modulus per nartfem layer of the last solved state (psi)."""
        st = self.states.get(key) if key else next(iter(self.states.values()), None)
        if st is None:
            return None
        return [float(np.exp(np.log(st.Mr[self.mesh.layer == i]).mean())) for i in range(self.model.n_layers)]
