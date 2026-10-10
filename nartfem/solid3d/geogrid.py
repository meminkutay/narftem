"""NCHRP 1-50 geogrid confinement on a geogrid plane of the 3D tensor mesh.

Per (x, y) element column of the geogrid layer:
  1. stresses at the geogrid depth by linear interpolation in z between the bracketing element
     centers of that column (overburden, K0 and residual stress added, tension removed);
  2. Mr without geogrid at those points;
  3. Delta-sigma-3-max from the confinement equation, vectorized over all columns (Illinois
     iterations).  The elastic lateral extension is the mean of the x and y directions:
         eps_lat = nu*sz/E - (1-nu)*(sx+sy)/(2E)   with the confinement added to sx and sy;
  4. triangular decay over +/- delta/2 along the column, summed over geogrids;
  5. new Mr of the layer's elements with the confinement added to sx and sy.
"""
from __future__ import annotations
import math
import warnings
import numpy as np
from .material import resilient_modulus

PSI_TO_KPA = 6.89
IN_TO_M = 0.0254


def _params(g):
    phi = math.radians(g.phi_deg)
    alpha = 2 * math.sin(phi) / (math.sqrt(3) * (3 - math.sin(phi)))
    K_kPa = g.c_psi * PSI_TO_KPA * 6 * math.cos(phi) / (math.sqrt(3) * (3 - math.sin(phi)))
    kGG = 2 * g.M_kNm / ((1 - g.vg) * g.delta_in * IN_TO_M * alpha)
    return alpha, K_kPa, kGG


def _residual(ds, s, Mr, nu, g, alpha, K_kPa, kGG):
    """s: (n,6) kPa compression positive [xx, yy, zz, xy, yz, xz]."""
    sx = s[:, 0] + ds; sy = s[:, 1] + ds; sz = s[:, 2]
    I1 = sx + sy + sz
    J2 = ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) / 6.0 + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
    base = alpha * I1 + K_kPa
    ok = (base > 0) & (sz > np.maximum(sx, sy))
    with np.errstate(all="ignore"):
        ep_v = np.where(ok, g.eps0 * math.exp(-((g.rho / g.N) ** g.beta)) *
                        np.sqrt(np.maximum(J2, 0)) ** g.m * np.where(ok, base, 1.0) ** g.n, 0.0)
    ep_v = np.where(np.isfinite(ep_v), ep_v, 0.0)
    lat = nu * sz / Mr - (1 - nu) * (sx + sy) / (2 * Mr)
    return ds - kGG * (lat + 0.85 * ep_v)


def delta_sigma_max(s_kPa, Mr_kPa, nu, g, tol=1e-6, max_iter=60):
    alpha, K_kPa, kGG = _params(g)
    f = lambda ds: _residual(ds, s_kPa, Mr_kPa, nu, g, alpha, K_kPa, kGG)
    n = s_kPa.shape[0]
    lo = np.zeros(n); flo = f(lo)
    active = np.isfinite(flo) & (flo < 0)
    hi = np.where(active, np.maximum(-flo, 1e-3), 0.0); fhi = f(hi)
    for _ in range(60):
        need = active & ~(np.isfinite(fhi) & (fhi >= 0))
        if not np.any(need):
            break
        hi = np.where(need, 2 * hi, hi); fhi = np.where(need, f(hi), fhi)
    bad = active & ~(np.isfinite(fhi) & (fhi >= 0))
    if np.any(bad):
        warnings.warn("Geogrid confinement: no root at %d columns; set to 0." % bad.sum()); active &= ~bad
    side = np.zeros(n); x = np.where(active, hi, 0.0)
    for _ in range(max_iter):
        if not np.any(active):
            break
        with np.errstate(all="ignore"):
            xn = hi - fhi * (hi - lo) / (fhi - flo)
        xn = np.where(active, np.clip(xn, lo, hi), x)
        fx = f(xn)
        conv = active & ((np.abs(fx) <= tol * np.maximum(np.abs(xn), 1.0)) | (hi - lo <= tol * np.maximum(hi, 1.0)))
        pos = active & (fx > 0); neg = active & (fx <= 0)
        flo = np.where(pos & (side == 1), flo / 2, flo); fhi = np.where(neg & (side == -1), fhi / 2, fhi)
        hi = np.where(pos, xn, hi); fhi = np.where(pos, fx, fhi)
        lo = np.where(neg, xn, lo); flo = np.where(neg, fx, flo)
        side = np.where(pos, 1, np.where(neg, -1, side))
        x = np.where(active, xn, x); active &= ~conv
    return np.maximum(x, 0.0)


def update_layer(model, res, mesh, layer_idx, stress_fe, gz):
    """Returns (Mr_new, delsig) for ALL elements of the layer (arrays over the layer's elements,
    in mesh element order) plus a dict of the plane data."""
    L = model.layers[layer_idx]
    Pa = model.atm_pressure_psi
    K0 = res["K0"][layer_idx]; ResH = res["residual"][layer_idx]
    kmat = res["kmat"][:, layer_idx]; MrMin = res["Mr_min"][layer_idx]; mat = res["mat_type"][layer_idx]
    nu = res["nu"][layer_idx]
    nx, ny, nz = mesh.shape
    th_cum = np.concatenate([[0.0], model.thickness_cum, [model.depth]])
    z_bottom = -th_cum[layer_idx + 1]
    # k-rows belonging to the layer
    zc_rows = -0.5 * (mesh.zs[:-1] + mesh.zs[1:])           # depth of element rows
    k_rows = np.flatnonzero((zc_rows > th_cum[layer_idx]) & (zc_rows < th_cum[layer_idx + 1]))
    zrows = mesh.zs[:-1][k_rows] * 0.5 + mesh.zs[1:][k_rows] * 0.5     # z (negative) of row centers, descending
    # element ids of the layer as (nx, ny, nk) block
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), k_rows, indexing="ij")
    eid = (I + nx * (J + ny * K))                              # (nx,ny,nk)
    s_all = -stress_fe[eid.ravel()].reshape(nx, ny, k_rows.size, 6)   # compression positive
    g_all = gz[eid.ravel()].reshape(nx, ny, k_rows.size)
    s_all[..., 0] += g_all * K0 + ResH; s_all[..., 1] += g_all * K0 + ResH; s_all[..., 2] += g_all
    delsig = np.zeros(eid.shape)
    planes = []
    for g in L.geogrids:
        zg = z_bottom + g.l_in
        # interpolate in z (rows are descending in z): find bracketing rows
        order = np.argsort(zrows); zr = zrows[order]
        w = np.interp(zg, zr, np.arange(zr.size))
        k0 = int(np.clip(np.floor(w), 0, zr.size - 1)); k1 = int(np.clip(k0 + 1, 0, zr.size - 1)); f = w - k0
        sg = (1 - f) * s_all[:, :, order[k0], :] + f * s_all[:, :, order[k1], :]
        sg = sg.reshape(-1, 6); sg[:, :3] = np.maximum(sg[:, :3], 0.0)
        Mr0 = resilient_modulus(sg, mat, kmat, Pa, MrMin)
        ds = delta_sigma_max(sg * PSI_TO_KPA, Mr0 * PSI_TO_KPA, nu, g) / PSI_TO_KPA     # psi, (nx*ny,)
        ds = ds.reshape(nx, ny)
        dz = np.abs(zrows - zg)                                 # (nk,)
        fac = np.maximum(0.0, 1 - dz / (0.5 * g.delta_in))
        delsig += ds[:, :, None] * fac[None, None, :]
        planes.append(dict(z_in=-zg, x=mesh.xs[:-1] * 0.5 + mesh.xs[1:] * 0.5, y=mesh.ys[:-1] * 0.5 + mesh.ys[1:] * 0.5,
                           delsig_psi=ds))
    s = s_all.reshape(-1, 6).copy()
    d = delsig.ravel()
    s[:, 0] += d; s[:, 1] += d
    s[:, :3] = np.maximum(s[:, :3], 0.0)
    Mr_new = resilient_modulus(s, mat, kmat, Pa, MrMin)
    return eid.ravel(), Mr_new, d, planes
