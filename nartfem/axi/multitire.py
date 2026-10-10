"""Iterated superposition for a group of identical tires (dual tires, axle groups).

Plain superposition (MEAPA) uses the single-tire nonlinear modulus field everywhere, so the
base under two overlapping tire bulbs is too soft.  Here the moduli are made self-consistent
with the superposed stresses:

    repeat
        1. solve the axisymmetric single tire with the current moduli Mr(r, z)
        2. at every nonlinear element (r, z) - taken around tire 1 - sample n_phi azimuths, place
           the point in the x-y plane, superpose the Cartesian stress tensors of all tires
           (interpolated from the single-tire field along the element row), add overburden,
           K0 and residual stress, remove tension on the normal components
        3. geogrid layers: Delta-sigma-3 from the superposed stress state on the geogrid plane
           (3D residual with the mean lateral extension), azimuth-averaged, triangular decay
        4. Mr from the superposed state at each azimuth, geometric mean over the azimuths
    until the moduli stop changing (Anderson mixing, same criteria as nartfem.axi)

The modulus field stays axisymmetric (an azimuthal average), so the asymmetry close to the
surface between the tires is not represented; at base/subgrade depth, where the overlap is
largest, the superposed field is nearly axisymmetric anyway.

    res, info = run_multitire(model, tires)        # tires: list of (x, y) positions, identical tires
    strains = superposed_strain(res, tires, x, y, depth, side)
"""
from __future__ import annotations
import math
import time
import warnings
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import norm
from . import mesh as meshmod
from .elements import ElementData, overburden
from .assembly import force_vector, fixed_dofs, Solver, assemble
from .responses import State
from .results import Results
from .material import resilient_modulus as mr_axi
from ..solid3d.material import resilient_modulus as mr_3d
from ..solid3d import geogrid as gg3d

PSI_TO_KPA = 6.89


def _row_profiles(mesh, values):
    """Element-row structure of the generated axi mesh: values (ne, m) -> (nz_rows, nr_cols, m),
    plus the column center radii."""
    rcols = np.unique(np.round(mesh.center[:, 0], 9))
    ncol = rcols.size; nrow = mesh.n_elements // ncol
    # generated axi mesh: element e = i*(nrow) + j  (i radial column, j vertical row)
    prof = np.asarray(values).reshape(ncol, nrow, -1).transpose(1, 0, 2)      # (nrow, ncol, m)
    return rcols, prof, nrow, ncol


def _superposed_tensor(mesh, stress, gz, res_layer, layer, tires, n_phi, r_pts, z_rows_idx):
    """Superposed Cartesian stress (compression positive, overburden + K0 + residual added, no
    tension clip yet) at points (r_pts, row z) around tire 1 for n_phi azimuths.
    Returns (npts, n_phi, 6) Voigt [xx, yy, zz, xy, yz, xz] and the per-point layer index."""
    rcols, prof, nrow, ncol = _row_profiles(mesh, stress)        # prof: (nrow, ncol, 4) r, z, t, rz
    phis = np.linspace(0.0, math.pi, n_phi, endpoint=False) + math.pi / (2 * n_phi)
    out = np.zeros((r_pts.size, n_phi, 6))
    x0, y0 = tires[0]
    for p, phi in enumerate(phis):
        px = x0 + r_pts * math.cos(phi); py = y0 + r_pts * math.sin(phi)
        for (tx, ty) in tires:
            dx, dy = px - tx, py - ty
            R = np.hypot(dx, dy); th = np.arctan2(dy, dx)
            c, s = np.cos(th), np.sin(th)
            # interpolate the single-tire stress components along the element row of each point
            comp = np.empty((r_pts.size, 4))
            for j in np.unique(z_rows_idx):
                m = z_rows_idx == j
                for q in range(4):
                    comp[m, q] = np.interp(R[m], rcols, prof[j, :, q])
            sr, sz, st, trz = -comp[:, 0], -comp[:, 1], -comp[:, 2], -comp[:, 3]     # compression positive
            out[:, p, 0] += c ** 2 * sr + s ** 2 * st
            out[:, p, 1] += s ** 2 * sr + c ** 2 * st
            out[:, p, 2] += sz
            out[:, p, 3] += c * s * (sr - st)
            out[:, p, 4] += s * trz
            out[:, p, 5] += c * trz
    return out


def run_multitire(model, tires, n_phi=8, log=None):
    """Iterated-superposition solution. ``tires``: list of (x, y) of identical tires (pressure and
    radius of ``model``).  Returns (Results of the final single-tire field, info dict)."""
    log = log or (print if model.verbose else (lambda *a, **k: None))
    tires = [tuple(map(float, t)) for t in tires]
    t0 = time.time()
    log("nartfem.axi.multitire: %s  (%d tires, iterated superposition)" % (model.name, len(tires)))
    mesh = meshmod.generate(model)
    res = model.resolved()
    ed = ElementData(mesh, model.gauss_points)
    ed.add_interface_points(np.concatenate([[0.0], model.thickness_cum, [mesh.depth]]))
    state = State(model, mesh, ed, res)
    state.gz = overburden(mesh, res, model.thickness)
    F = force_vector(mesh, model.pressure_psi, model.radius_in)
    solver = Solver(2 * mesh.n_nodes, fixed_dofs(mesh))
    nl = np.flatnonzero(state.nonlin)
    rcols, _, nrow, ncol = _row_profiles(mesh, np.zeros((mesh.n_elements, 1)))
    row_of = np.arange(mesh.n_elements) % nrow
    th_cum = np.concatenate([[0.0], model.thickness_cum, [mesh.depth]])
    Pa = model.atm_pressure_psi

    def mr_superposed():
        """Mr_calc for all elements from the superposed stress state (azimuth geometric mean)."""
        d = state.u[ed.dof]
        state.strain = np.einsum("nij,nj->ni", ed.Bc, d)
        state.stress = np.einsum("nij,nj->ni", state.D, state.strain)
        Mr_new = state.Mr.copy(); state.delsig[:] = 0.0
        for i, L in enumerate(model.layers):
            if not L.is_nonlinear:
                continue
            idx = np.flatnonzero(mesh.layer == i)
            K0, ResH = res["K0"][i], res["residual"][i]
            S = _superposed_tensor(mesh, state.stress, state.gz, res, i, tires, n_phi, mesh.center[idx, 0], row_of[idx])
            g = state.gz[idx][:, None]
            S[:, :, 0] += g * K0 + ResH; S[:, :, 1] += g * K0 + ResH; S[:, :, 2] += g
            delsig = np.zeros(idx.size)
            if L.geogrids:
                z_bottom = -th_cum[i + 1]
                rc = np.concatenate([np.linspace(0, 10, 30), 3.0 ** np.arange(math.log(10.5) / math.log(3), math.log(mesh.width) / math.log(3) + 1e-12, 0.1)])
                if rc[-1] != mesh.width:
                    rc = np.append(rc, mesh.width)
                for gsp in L.geogrids:
                    zg = z_bottom + gsp.l_in
                    # row whose center is nearest the geogrid plane (within the layer)
                    rows_lay = np.unique(row_of[idx]); zrow = np.array([mesh.center[np.flatnonzero(row_of == j)[0], 1] for j in rows_lay])
                    jrow = rows_lay[np.argmin(np.abs(zrow - zg))]
                    Sg = _superposed_tensor(mesh, state.stress, state.gz, res, i, tires, n_phi, rc, np.full(rc.size, jrow))
                    gzg = np.interp(-zg, -mesh.center[np.flatnonzero(row_of == jrow), 1], state.gz[np.flatnonzero(row_of == jrow)])
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
            state.delsig[idx] = delsig
        return Mr_new

    # ---- iteration (Anderson on ln Mr, criteria as nartfem.axi)
    state.u = solver.solve(state.K, F)
    Mr_calc = mr_superposed()
    a = model.radius_in
    active = (state.nonlin & ((mesh.center[:, 0] - a) <= -2 * mesh.center[:, 1]))[nl]
    zval = norm.ppf(model.reliability_pct / 100.0); tol = model.max_error_pct / 100.0
    x = np.log(state.Mr[nl]); gx = np.log(Mr_calc[nl]) - x
    X, G = [], []; state.converged = False; state.iterations = model.max_iter
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
        Mr_used = np.exp(x); ratio = Mr_used / state.Mr[nl]
        state.D[nl] *= ratio[:, None, None]; state.Ke[nl] *= ratio[:, None, None]
        state.K = assemble(state.Ke, ed.dof, 2 * mesh.n_nodes); state.Mr[nl] = Mr_used
        state.u = solver.solve(state.K, F)
        Mr_calc = mr_superposed()
        gx = np.log(Mr_calc[nl]) - x
        err = np.abs(1 - np.exp(-gx))
        e = err[active].mean() + zval * err[active].std(ddof=1); state.conv_error = e
        log("    - iter %2d  error %.2f%%  (max %.1f%%)   (%.1f s)" % (it, 100 * e, 100 * err[active].max(), time.time() - t0))
        if e < tol:
            state.converged, state.iterations = True, it
            log("    - converged after %d iterations (error %.2f%%)" % (it, 100 * e)); break
    full = np.zeros(mesh.n_elements); full[nl] = err; state.MrError = full
    if not state.converged:
        warnings.warn("multitire iterations did not converge in %d iterations (error %.2f%%)." % (model.max_iter, 100 * state.conv_error))
    state.gg_planes = {}
    r = Results(model, mesh, state, time.time() - t0)
    log("  - done in %.1f s" % r.elapsed_s)
    return r, dict(tires=tires, n_phi=n_phi)


def superposed_strain(res, tires, x, y, depth, side="above"):
    """Strains (exx, eyy, ezz, exy[tensor]) at (x, y, depth) by superposing the single-tire field
    of ``res`` (exact in-element values) for the tire positions."""
    x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
    exx = np.zeros(x.size); eyy = np.zeros(x.size); ezz = np.zeros(x.size); exy = np.zeros(x.size)
    for (tx, ty) in tires:
        dx, dy = x - tx, y - ty
        r = np.hypot(dx, dy); th = np.arctan2(dy, dx)
        er = np.array([res.fe_at("strain_r", ri, depth, side=side) for ri in r])
        et = np.array([res.fe_at("strain_t", ri, depth, side=side) for ri in r])
        ez = np.array([res.fe_at("strain_z", ri, depth, side=side) for ri in r])
        c, s = np.cos(th), np.sin(th)
        exx += c ** 2 * er + s ** 2 * et; eyy += s ** 2 * er + c ** 2 * et; exy += c * s * (er - et); ezz += ez
    return exx, eyy, ezz, exy
