"""Element responses and modulus update (DefineResponses.m) and the nonlinear iteration
(DefineNonlinearity.m)."""
from __future__ import annotations
import time
import warnings
import numpy as np
from scipy.stats import norm
from .elements import d_matrix
from .material import field_stress, resilient_modulus
from .assembly import assemble
from . import geogrid as gg


class State:
    """Everything that changes between iterations."""

    def __init__(self, model, mesh, ed, res):
        self.model, self.mesh, self.ed, self.res = model, mesh, ed, res
        ne = mesh.n_elements
        self.layer = mesh.layer
        # initial moduli
        self.Mr = np.zeros(ne)
        for i, L in enumerate(model.layers):
            m = self.layer == i
            if L.type == "elastic_depth":
                self.Mr[m] = L.modulus_function()(-mesh.center[m, 1])
            else:
                self.Mr[m] = res["E0"][i]
        self.nu = res["nu"][self.layer]
        self.D = d_matrix(self.Mr, self.nu)
        self.Ke = ed.stiffness(self.D)
        self.K = assemble(self.Ke, ed.dof, 2 * mesh.n_nodes)
        self.u = None
        self.strain = np.zeros((ne, 4)); self.stress = np.zeros((ne, 4))
        self.delsig = np.zeros(ne)
        self.MrError = np.zeros(ne)
        self.gg_planes = {}
        self.gg_cache = {}       # per layer: precomputed interpolation weights (point set is fixed)
        self.nonlin = np.array([model.layers[i].is_nonlinear for i in self.layer])
        self.gg_layers = [i for i, L in enumerate(model.layers) if L.geogrids]
        self.gz = None   # set by run
        self.Mr_seed = self.Mr.copy()

    # ------------------------------------------------------------------ responses
    def responses(self):
        """Strains/stresses at the element centers from the current displacements, then the
        modulus update of the nonlinear layers (regular and geogrid branches)."""
        ed, mesh, model, res = self.ed, self.mesh, self.model, self.res
        d = self.u[ed.dof]                                     # (ne,8)
        self.strain = np.einsum("nij,nj->ni", ed.Bc, d)
        self.stress = np.einsum("nij,nj->ni", self.D, self.strain)
        Mr_new = self.Mr.copy()
        self.delsig[:] = 0.0

        for i, L in enumerate(model.layers):
            if not L.is_nonlinear:
                continue
            idx = np.flatnonzero(self.layer == i)
            if L.geogrids and i in self.gg_layers:
                pts = self._layer_points(i, idx)
                Mr_i, ds_i, planes = gg.update_layer(model, res, i, pts, idx, mesh.center[idx],
                                                     self.stress[idx], self.gz[idx], mesh.width,
                                                     cache=self.gg_cache.setdefault(i, {}))
                Mr_new[idx] = Mr_i
                self.delsig[idx] = ds_i
                self.gg_planes[i] = planes
            else:
                s = field_stress(self.stress[idx], self.gz[idx], res["K0"][i], res["residual"][i])
                Mr_new[idx] = resilient_modulus(s, res["mat_type"][i], res["kmat"][:, i],
                                                model.atm_pressure_psi, res["Mr_min"][i])

        ratio = Mr_new / self.Mr
        self.MrError = np.where(self.nonlin, np.abs(1 - 1 / ratio), 0.0)
        changed = np.flatnonzero(self.nonlin & (ratio != 1.0))
        if changed.size:
            # isotropic element with fixed nu: D and Ke are proportional to Mr -> just rescale
            self.D[changed] *= ratio[changed, None, None]
            self.Ke[changed] *= ratio[changed, None, None]
            self.K = assemble(self.Ke, ed.dof, 2 * mesh.n_nodes)
        self.Mr = Mr_new

    def _layer_points(self, i, idx):
        """Element centers plus edge points of a geogrid layer, with their FE stresses and
        overburden (the GeogridLyrData block of DefineResponses)."""
        mesh, ed, model = self.mesh, self.ed, self.model
        gam = self.res["gamma_pci"][i]
        th = np.concatenate([[0.0], model.thickness_cum, [model.depth]])
        r = [mesh.center[idx, 0]]; z = [mesh.center[idx, 1]]
        s = [self.stress[idx]]; gz = [self.gz[idx]]
        for B, xy, el in zip(ed.extra_B, ed.extra_xy, ed.extra_elem):
            m = np.isin(el, idx)
            if not np.any(m):
                continue
            e = el[m]
            strain = np.einsum("nij,nj->ni", B[m], self.u[ed.dof[e]])
            stress = np.einsum("nij,nj->ni", self.D[e], strain)
            zz = xy[m, 1].copy()
            on_if = np.any(np.abs(zz[:, None] + th[None, :]) < 1e-9, axis=1)
            below = mesh.center[e, 1] < zz
            zz[on_if & below] -= 1e-4
            zz[on_if & ~below] += 1e-4
            r.append(xy[m, 0]); z.append(zz); s.append(stress)
            gz.append(self.gz[e] + gam * (mesh.center[e, 1] - zz))
        return dict(r=np.concatenate(r), z=np.concatenate(z),
                    stress_fe=np.vstack(s), gamma_z=np.concatenate(gz))


# ----------------------------------------------------------------------- iteration
def iterate(state, solver, F, log=print):
    """Nonlinear iterations on the modulus field of the nonlinear elements.

    scheme = "anderson" (default): Anderson mixing on x = ln(Mr).  The plain Picard update of
             MatFEA is unstable for thin AC on soft subgrade (elements at the load edge flip
             between the floor and ~10 ksi); Anderson mixing converges to a true fixed point.
    scheme = "picard": the MatFEA update (Mr := Mr_calc), kept for one-to-one comparisons.
    Convergence (both): mean + z(reliability) * std of |1 - Mr/Mr_calc| over the active zone
    ((r - a) <= 2|z|, nonlinear elements) below max_error_pct."""
    model, mesh, ed = state.model, state.mesh, state.ed
    if not np.any(state.nonlin):
        state.converged, state.iterations, state.conv_error = True, 0, 0.0
        return
    a = model.radius_in
    nl = np.flatnonzero(state.nonlin)
    active = (state.nonlin & ((mesh.center[:, 0] - a) <= -2 * mesh.center[:, 1]))[nl]
    if not np.any(active):
        active[:] = True
    zval = norm.ppf(model.reliability_pct / 100.0)
    tol = model.max_error_pct / 100.0
    scheme = getattr(model, "scheme", "anderson")
    mhist = getattr(model, "anderson_memory", 10)
    beta = getattr(model, "anderson_beta", 0.8)
    ndof = 2 * mesh.n_nodes

    def set_Mr(Mr_nl):
        ratio = Mr_nl / state.Mr[nl]
        ch = nl[ratio != 1.0]
        if ch.size:
            r = ratio[ratio != 1.0][:, None, None]
            state.D[ch] *= r
            state.Ke[ch] *= r
            state.K = assemble(state.Ke, ed.dof, ndof)
        state.Mr[nl] = Mr_nl

    t0 = time.time()
    state.converged = False
    # state.responses() has already been called once after the linear solve: state.Mr = Mr_calc
    # of the seed moduli, state.MrError = first residual.
    err = state.MrError[nl]
    e = err[active].mean() + zval * err[active].std(ddof=1)
    state.conv_error = e
    if e < tol:
        state.converged, state.iterations = True, 0
        log("    - converged with the seed moduli (error %.2f%%)" % (100 * e))
        return
    if scheme == "picard":
        for it in range(1, model.max_iter + 1):
            state.u = solver.solve(state.K, F)
            state.responses()
            err = state.MrError[nl]
            e = err[active].mean() + zval * err[active].std(ddof=1)
            state.conv_error = e
            log("    - iter %2d  error %.2f%%   (%.1f s)" % (it, 100 * e, time.time() - t0))
            if e < tol:
                state.converged, state.iterations = True, it
                return
        state.iterations = model.max_iter
    else:
        # Anderson mixing.  x_k = ln Mr used in the solve; g(x) = ln Mr_calc(x) - x.
        # The first evaluation is the one already done in run(): x0 = ln(seed), g0 = ln Mr_calc - x0.
        Mr_calc = state.Mr[nl].copy()
        x = np.log(state.Mr_seed[nl]) if hasattr(state, "Mr_seed") else np.log(Mr_calc) - 0.0
        gx = np.log(Mr_calc) - x
        X, G = [], []
        for it in range(1, model.max_iter + 1):
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
            set_Mr(np.exp(x))
            state.u = solver.solve(state.K, F)
            state.responses()                       # state.Mr <- Mr_calc, MrError <- |1 - Mr_used/Mr_calc|
            gx = np.log(state.Mr[nl]) - x
            err = np.abs(1 - np.exp(-gx))
            e = err[active].mean() + zval * err[active].std(ddof=1)
            state.conv_error = e
            log("    - iter %2d  error %.2f%%  (max %.1f%%)   (%.1f s)" % (it, 100 * e, 100 * err[active].max(), time.time() - t0))
            if e < tol:
                state.converged, state.iterations = True, it
                log("    - converged after %d iterations (error %.2f%%)" % (it, 100 * e))
                break
        else:
            state.iterations = model.max_iter
        # leave the state consistent with the moduli actually used in the last solve:
        # Mr = exp(x) (the iterate), responses/stresses belong to that modulus field.
        Mr_used = np.exp(x)
        full_err = np.zeros(mesh.n_elements); full_err[nl] = err
        state.MrError = full_err
        ratio = Mr_used / state.Mr[nl]
        state.D[nl] *= ratio[:, None, None]
        state.Ke[nl] *= ratio[:, None, None]
        state.K = assemble(state.Ke, ed.dof, ndof)
        state.Mr[nl] = Mr_used
    if not state.converged:
        warnings.warn("Nonlinear iterations did not converge in %d iterations (error %.2f%% > %.2f%%)."
                      % (model.max_iter, 100 * state.conv_error, model.max_error_pct))
