"""Element responses, modulus update and the nonlinear iteration for the 3D model.
Linear layers are assembled once (K_lin); nonlinear elements keep their Ke and are rescaled."""
from __future__ import annotations
import time
import warnings
import numpy as np
from scipy.stats import norm
from .elements import stress_from_strain
from .material import field_stress, resilient_modulus
from .assembly import assemble
from . import geogrid as gg


class State3D:
    def __init__(self, model, mesh, ed, res, log=None):
        self.model, self.mesh, self.ed, self.res = model, mesh, ed, res
        ne = mesh.n_elements
        self.layer = mesh.layer
        self.nonlin = np.array([model.layers[i].is_nonlinear for i in self.layer])
        self.nl = np.flatnonzero(self.nonlin)
        self.gg_layers = [i for i, L in enumerate(model.layers) if L.geogrids]
        self.Mr = np.zeros(ne)
        for i, L in enumerate(model.layers):
            m = self.layer == i
            self.Mr[m] = L.modulus_function()(-mesh.center[m, 2]) if L.type == "elastic_depth" else res["E0"][i]
        self.Mr_seed = self.Mr.copy()
        self.nu = res["nu"][self.layer]
        ndof = 3 * mesh.n_nodes
        t0 = time.time()
        lin = np.flatnonzero(~self.nonlin)
        self.K_lin = assemble(ed.stiffness(lin, self.Mr[lin], self.nu[lin]), ed.dof[lin], ndof) if lin.size else None
        self.Ke_nl = ed.stiffness(self.nl, self.Mr[self.nl], self.nu[self.nl]) if self.nl.size else np.zeros((0, 24, 24))
        self._assemble()
        if log:
            log("  - element stiffness: %d linear + %d nonlinear elements (%.1f s)" % (lin.size, self.nl.size, time.time() - t0))
        self.u = None
        self.strain = np.zeros((ne, 6)); self.stress = np.zeros((ne, 6))
        self.delsig = np.zeros(ne); self.MrError = np.zeros(ne); self.gg_planes = {}
        self.gz = None

    def _assemble(self):
        ndof = 3 * self.mesh.n_nodes
        K = assemble(self.Ke_nl, self.ed.dof[self.nl], ndof) if self.nl.size else None
        if self.K_lin is not None and K is not None:
            self.K = (self.K_lin + K).tocsr()
        else:
            self.K = (self.K_lin if K is None else K).tocsr()

    def set_Mr_nl(self, Mr_nl):
        ratio = Mr_nl / self.Mr[self.nl]
        self.Ke_nl *= ratio[:, None, None]
        self.Mr[self.nl] = Mr_nl
        self._assemble()

    def responses(self):
        ed, mesh, model, res = self.ed, self.mesh, self.model, self.res
        d = self.u[ed.dof]
        self.strain = np.einsum("nij,nj->ni", ed.Bc, d)
        self.stress = stress_from_strain(self.strain, self.Mr, self.nu)
        Mr_new = self.Mr.copy(); self.delsig[:] = 0.0
        for i, L in enumerate(model.layers):
            if not L.is_nonlinear:
                continue
            if L.geogrids:
                eid, Mr_i, ds_i, planes = gg.update_layer(model, res, mesh, i, self.stress, self.gz)
                Mr_new[eid] = Mr_i; self.delsig[eid] = ds_i; self.gg_planes[i] = planes
            else:
                idx = np.flatnonzero(self.layer == i)
                s = field_stress(self.stress[idx], self.gz[idx], res["K0"][i], res["residual"][i])
                Mr_new[idx] = resilient_modulus(s, res["mat_type"][i], res["kmat"][:, i], model.atm_pressure_psi, res["Mr_min"][i])
        self.MrError = np.where(self.nonlin, np.abs(1 - self.Mr / Mr_new), 0.0)
        self.Mr_calc = Mr_new


def iterate(state, solver, F, log=print):
    """Anderson mixing on ln(Mr) of the nonlinear elements (same scheme as nartfem.axi)."""
    model, mesh = state.model, state.mesh
    nl = state.nl
    if nl.size == 0:
        state.converged, state.iterations, state.conv_error = True, 0, 0.0
        return
    # active zone: nonlinear elements within a 1:2 cone under any tire
    a = model.radius_ref
    c = mesh.center
    dist = np.min([np.hypot(c[:, 0] - t.x_in, c[:, 1] - t.y_in) for t in model.tires], axis=0)
    active = (state.nonlin & ((dist - a) <= -2 * c[:, 2]))[nl]
    if not np.any(active):
        active[:] = True
    zval = norm.ppf(model.reliability_pct / 100.0); tol = model.max_error_pct / 100.0
    tol_vol = getattr(model, "max_error_vol_pct", 1.5) / 100.0
    vol = state.ed.vol[nl][active]; wv = vol / vol.sum()
    t0 = time.time()
    state.converged = False

    def measures(err):
        """(MatFEA criterion: mean + z*std ; volume-weighted mean) over the active zone."""
        return err[active].mean() + zval * err[active].std(ddof=1), float(np.dot(wv, err[active]))

    err = state.MrError[nl]
    e, ev = measures(err); state.conv_error = e; state.conv_error_vol = ev
    if e < tol:
        state.converged, state.iterations = True, 0; return
    x = np.log(state.Mr[nl]); gx = np.log(state.Mr_calc[nl]) - x
    X, G = [], []; hist = []
    mhist, beta = model.anderson_memory, model.anderson_beta
    for it in range(1, model.max_iter + 1):
        if model.scheme == "picard":
            x_new = x + gx
        else:
            X.append(x.copy()); G.append(gx.copy())
            if len(X) > mhist:
                X.pop(0); G.pop(0)
            if len(X) >= 2:
                dG = np.array([G[i + 1] - G[i] for i in range(len(G) - 1)]).T
                dX = np.array([X[i + 1] - X[i] for i in range(len(X) - 1)]).T
                gamma = np.linalg.lstsq(dG, gx, rcond=None)[0]
                x_new = x + beta * gx - (dX + beta * dG) @ gamma
            else:
                x_new = x + beta * gx
        x = x_new
        state.set_Mr_nl(np.exp(x))
        state.u = solver.solve(state.K, F)
        state.responses()
        gx = np.log(state.Mr_calc[nl]) - x
        err = np.abs(1 - np.exp(-gx))
        e, ev = measures(err); state.conv_error = e; state.conv_error_vol = ev
        log("    - iter %2d  error %.2f%% (mean+1.645 std)  %.2f%% (volume mean)  max %.1f%%  solve %.1f s  (%.0f s)"
            % (it, 100 * e, 100 * ev, 100 * err[active].max(), solver.last_time, time.time() - t0))
        # converged: MatFEA criterion (mean + 1.645 std < max_error_pct), or the volume-weighted mean
        # change below max_error_vol_pct.  A few discontinuous elements in the no-tension zone keep the
        # std-based measure at 3-6 % long after the critical responses have settled (<0.5 %).
        hist.append(e)
        if e < tol or (it >= 3 and ev < tol_vol):
            state.converged, state.iterations = True, it
            log("    - converged after %d iterations (%.2f%% mean+std, %.2f%% volume mean)" % (it, 100 * e, 100 * ev)); break
        if it >= 12 and min(hist[-8:]) > min(hist[:-8]) - 0.001:
            state.converged, state.iterations = True, it
            log("    - stalled: no improvement over 8 iterations (%.2f%% mean+std, %.2f%% volume mean); accepted" % (100 * e, 100 * ev)); break
    else:
        state.iterations = model.max_iter
    full = np.zeros(mesh.n_elements); full[nl] = err; state.MrError = full
    if not state.converged:
        warnings.warn("3D nonlinear iterations did not converge in %d iterations (error %.2f%%)." % (model.max_iter, 100 * state.conv_error))
