"""NCHRP 1-50 geogrid confinement (DefineGeogrid.m, with the September 2026 corrections).

For every layer that contains geogrids:
  1. interpolants of the four stress components over the layer (overburden, K0 and the
     residual stress added; no overburden on the shear stress), from the element centers and
     the edge points collected in responses.py;
  2. ~60 radial points on each geogrid plane (30 from 0 to 10 in, then log-spaced to the model
     width); stresses there, tension removed;
  3. Mr without geogrid at those points (then kPa);
  4. the confinement Delta-sigma-3-max from the implicit NCHRP 1-50 equation (bracketed root,
     >= 0; 0 when the grid is not mobilized);
  5. a modified-Akima interpolant of Delta-sigma-3-max(r), clamped at 0;
  6. for each element of the layer: triangular decay over +/- delta/2, sum over geogrids, add to
     sigma_r and sigma_theta, clip tension, new Mr.
"""
from __future__ import annotations
import math
import warnings
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Akima1DInterpolator
from scipy.spatial import Delaunay, cKDTree
import scipy.sparse as sp
from scipy.optimize import brentq
from .material import resilient_modulus

PSI_TO_KPA = 6.89
IN_TO_M = 0.0254


class _Interp2:
    """Linear interpolation inside the hull, nearest outside (MATLAB scatteredInterpolant
    'linear','linear' extrapolates linearly; the hull here covers the geogrid plane, so the
    fallback is only a safety net)."""

    def __init__(self, r, z, v):
        pts = np.column_stack([r, z])
        self.lin = LinearNDInterpolator(pts, v)
        self.near = NearestNDInterpolator(pts, v)

    def __call__(self, r, z):
        out = self.lin(r, z)
        bad = ~np.isfinite(out)
        if np.any(bad):
            out[bad] = self.near(np.asarray(r)[bad], np.asarray(z)[bad])
        return out


class _PlaneWeights:
    """Linear (barycentric) interpolation weights from a fixed scattered point set to fixed target
    points, nearest point outside the hull.  Built once; each evaluation is a sparse mat-vec."""

    def __init__(self, r, z, rt, zt):
        pts = np.column_stack([r, z]); tgt = np.column_stack([rt, zt])
        tri = Delaunay(pts)
        simp = tri.find_simplex(tgt)
        n, m = tgt.shape[0], pts.shape[0]
        rows, cols, vals = [], [], []
        inside = simp >= 0
        if np.any(inside):
            T = tri.transform[simp[inside]]                      # (k,3,2)
            d = tgt[inside] - T[:, 2]
            b = np.einsum("kij,kj->ki", T[:, :2], d)
            w = np.column_stack([b, 1 - b.sum(axis=1)])         # (k,3)
            v = tri.simplices[simp[inside]]                     # (k,3)
            ii = np.flatnonzero(inside)
            rows.append(np.repeat(ii, 3)); cols.append(v.ravel()); vals.append(w.ravel())
        if np.any(~inside):
            _, near = cKDTree(pts).query(tgt[~inside])
            ii = np.flatnonzero(~inside)
            rows.append(ii); cols.append(near); vals.append(np.ones(ii.size))
        self.W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, m))

    def __call__(self, values):
        return self.W @ values


def _gg_params(g, Pa):
    """Derived geogrid constants (kPa / m units of DefineGeogrid)."""
    phi = math.radians(g.phi_deg)
    alpha = 2 * math.sin(phi) / (math.sqrt(3) * (3 - math.sin(phi)))
    K_kPa = g.c_psi * PSI_TO_KPA * 6 * math.cos(phi) / (math.sqrt(3) * (3 - math.sin(phi)))
    kGG = 2 * g.M_kNm / ((1 - g.vg) * g.delta_in * IN_TO_M * alpha)     # kPa
    return alpha, K_kPa, kGG


def _residual(ds, s, Mrh, Mrv, PRh, PRv, g, alpha, K_kPa, kGG):
    """Residual of the confinement equation (kPa): ds - kGG*(elastic lateral extension + 0.85*eps_p)."""
    s1, s2, s3, s4 = s[0] + ds, s[1], s[2] + ds, s[3]
    I1 = s1 + s2 + s3
    J2 = ((s1 - s2) ** 2 + (s1 - s3) ** 2 + (s2 - s3) ** 2 + 6 * s4 ** 2) / 6.0
    base = alpha * I1 + K_kPa
    if base <= 0 or s2 <= max(s1, s3):
        ep_v = 0.0
    else:
        ep_v = g.eps0 * math.exp(-((g.rho / g.N) ** g.beta)) * (math.sqrt(J2) ** g.m) * (base ** g.n)
        if not math.isfinite(ep_v):
            ep_v = 0.0
    lat_ext = PRv * s2 / Mrv + PRh * s3 / Mrh - s1 / Mrh
    return ds - kGG * (lat_ext + 0.85 * ep_v)


def delta_sigma_max(s_kPa, Mrh, Mrv, PRh, PRv, g, alpha, K_kPa, kGG):
    """Maximum additional confinement at the geogrid plane, kPa (delSigMaxCalculator)."""
    f = lambda ds: _residual(ds, s_kPa, Mrh, Mrv, PRh, PRv, g, alpha, K_kPa, kGG)
    f0 = f(0.0)
    if not math.isfinite(f0) or f0 >= 0:
        return 0.0
    upper = max(-f0, 1e-3)
    fU = f(upper)
    for _ in range(60):
        if math.isfinite(fU) and fU >= 0:
            break
        upper *= 2
        fU = f(upper)
    if not (math.isfinite(fU) and fU >= 0):
        warnings.warn("Geogrid confinement: no root found; delSigMax set to 0.")
        return 0.0
    return max(brentq(f, 0.0, upper, xtol=1e-6 * max(upper, 1.0)), 0.0)


# --------------------------------------------------------------------------- vectorized
def _residual_vec(ds, s, Mrh, Mrv, PRh, PRv, g, alpha, K_kPa, kGG):
    """Residual of the confinement equation for all radial points at once (arrays, kPa)."""
    s1 = s[:, 0] + ds; s2 = s[:, 1]; s3 = s[:, 2] + ds; s4 = s[:, 3]
    I1 = s1 + s2 + s3
    J2 = ((s1 - s2) ** 2 + (s1 - s3) ** 2 + (s2 - s3) ** 2 + 6 * s4 ** 2) / 6.0
    base = alpha * I1 + K_kPa
    ok = (base > 0) & (s2 > np.maximum(s1, s3))
    with np.errstate(all="ignore"):
        ep_v = np.where(ok, g.eps0 * math.exp(-((g.rho / g.N) ** g.beta)) *
                        np.sqrt(np.maximum(J2, 0.0)) ** g.m * np.where(ok, base, 1.0) ** g.n, 0.0)
    ep_v = np.where(np.isfinite(ep_v), ep_v, 0.0)
    lat_ext = PRv * s2 / Mrv + PRh * s3 / Mrh - s1 / Mrh
    return ds - kGG * (lat_ext + 0.85 * ep_v)


def delta_sigma_max_vec(s_kPa, Mrh, Mrv, PRh, PRv, g, alpha, K_kPa, kGG, tol=1e-6, max_iter=60):
    """Vectorized counterpart of delta_sigma_max: all radial points solved simultaneously with
    a bracketing step (doubling) followed by Illinois (regula falsi) iterations.  Same equation,
    same root; points where the grid is not mobilized (f(0) >= 0) return 0."""
    f = lambda ds: _residual_vec(ds, s_kPa, Mrh, Mrv, PRh, PRv, g, alpha, K_kPa, kGG)
    n = s_kPa.shape[0]
    lo = np.zeros(n)
    flo = f(lo)
    active = np.isfinite(flo) & (flo < 0)
    hi = np.where(active, np.maximum(-flo, 1e-3), 0.0)
    fhi = f(hi)
    for _ in range(60):                                   # bracket: double until f(hi) >= 0
        need = active & ~(np.isfinite(fhi) & (fhi >= 0))
        if not np.any(need):
            break
        hi = np.where(need, 2 * hi, hi)
        fhi = np.where(need, f(hi), fhi)
    bad = active & ~(np.isfinite(fhi) & (fhi >= 0))
    if np.any(bad):
        warnings.warn("Geogrid confinement: no root found at %d points; set to 0." % bad.sum())
        active &= ~bad
    # Illinois method on the bracket [lo, hi]
    side = np.zeros(n)
    x = np.where(active, hi, 0.0)
    for _ in range(max_iter):
        if not np.any(active):
            break
        with np.errstate(all="ignore"):
            xn = hi - fhi * (hi - lo) / (fhi - flo)
        xn = np.where(active, np.clip(xn, lo, hi), x)
        fx = f(xn)
        conv = active & ((np.abs(fx) <= tol * np.maximum(np.abs(xn), 1.0)) |
                         (hi - lo <= tol * np.maximum(hi, 1.0)))
        pos = active & (fx > 0)
        neg = active & (fx <= 0)
        # move the bracket; halve the stale side (Illinois)
        flo = np.where(pos & (side == 1), flo / 2, flo)
        fhi = np.where(neg & (side == -1), fhi / 2, fhi)
        hi = np.where(pos, xn, hi); fhi = np.where(pos, fx, fhi)
        lo = np.where(neg, xn, lo); flo = np.where(neg, fx, flo)
        side = np.where(pos, 1, np.where(neg, -1, side))
        x = np.where(active, xn, x)
        active &= ~conv
    return np.maximum(x, 0.0)


def update_layer(model, res, layer_idx, pts, elem_idx, elem_center, elem_stress_fe, elem_gz, width, cache=None):
    """Confinement and new moduli for the elements of one geogrid layer.

    pts            dict with r, z, stress_fe (n,4), gamma_z (n,) of all data points of the layer
    elem_*         arrays for the elements of the layer (subset)
    Returns (Mr_new (nel,), delSig (nel,), diag dict)."""
    L = model.layers[layer_idx]
    Pa = model.atm_pressure_psi
    K0 = res["K0"][layer_idx]; ResH = res["residual"][layer_idx]
    kmat = res["kmat"][:, layer_idx]; MrMin = res["Mr_min"][layer_idx]
    mat = res["mat_type"][layer_idx]
    PRv = res["nu"][layer_idx]; PRh = res["nu_hor"][layer_idx]; ratio = res["Er_over_Ez"][layer_idx]
    th_cum = np.concatenate([[0.0], model.thickness_cum, [model.depth]])
    z_bottom = -th_cum[layer_idx + 1]

    r, z, sfe, gz = pts["r"], pts["z"], pts["stress_fe"], pts["gamma_z"]
    vals = np.column_stack([-sfe[:, 0] + gz * K0 + ResH, -sfe[:, 1] + gz, -sfe[:, 2] + gz * K0 + ResH, -sfe[:, 3]])

    rc = np.concatenate([np.linspace(0, 10, 30),
                         3.0 ** np.arange(math.log(10.5) / math.log(3), math.log(width) / math.log(3) + 1e-12, 0.1)])
    if rc[-1] != width:
        rc = np.append(rc, width)
    use_cache = cache is not None and getattr(model, "geogrid_solver", "vector") != "scalar"
    if cache is not None and cache.get("n_pts") != r.size:
        cache.clear(); cache["n_pts"] = r.size

    grids = []
    for gi, g in enumerate(L.geogrids):
        alpha, K_kPa, kGG = _gg_params(g, Pa)
        zg = z_bottom + g.l_in
        zz = np.full(rc.shape, zg)
        if use_cache:
            if gi not in cache:
                cache[gi] = _PlaneWeights(r, z, rc, zz)
            s = cache[gi](vals)
        else:
            S11 = _Interp2(r, z, vals[:, 0]); S22 = _Interp2(r, z, vals[:, 1])
            S33 = _Interp2(r, z, vals[:, 2]); S12 = _Interp2(r, z, vals[:, 3])
            s = np.column_stack([S11(rc, zz), S22(rc, zz), S33(rc, zz), S12(rc, zz)])
        s[:, :3] = np.maximum(s[:, :3], 0.0)
        Mrv = resilient_modulus(s, mat, kmat, Pa, MrMin) * PSI_TO_KPA
        Mrh = Mrv * ratio
        if getattr(model, "geogrid_solver", "vector") == "scalar":
            ds = np.array([delta_sigma_max(s[j] * PSI_TO_KPA, Mrh[j], Mrv[j], PRh, PRv, g, alpha, K_kPa, kGG)
                           for j in range(rc.size)]) / PSI_TO_KPA
        else:
            ds = delta_sigma_max_vec(s * PSI_TO_KPA, Mrh, Mrv, PRh, PRv, g, alpha, K_kPa, kGG) / PSI_TO_KPA
        ak = Akima1DInterpolator(rc, ds, method="makima")
        grids.append(dict(z=zg, half=0.5 * g.delta_in,
                          interp=lambda x, ak=ak: np.maximum(0.0, ak(np.clip(x, rc[0], rc[-1]))),
                          r=rc, delsig=ds))

    # elements of the layer
    delsig = np.zeros(elem_idx.size)
    for G in grids:
        dz = np.abs(elem_center[:, 1] - G["z"])
        m = dz < G["half"]
        if np.any(m):
            delsig[m] += (1 - dz[m] / G["half"]) * G["interp"](elem_center[m, 0])
    s = -elem_stress_fe.copy()
    s[:, 0] += elem_gz * K0 + ResH + delsig
    s[:, 1] += elem_gz
    s[:, 2] += elem_gz * K0 + ResH + delsig
    s[:, :3] = np.maximum(s[:, :3], 0.0)
    Mr_new = resilient_modulus(s, mat, kmat, Pa, MrMin)
    return Mr_new, delsig, grids
