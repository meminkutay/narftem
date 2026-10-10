"""Superposition of the single-tire axisymmetric strain field for a group of tires
(port of ff_superpose_strains.m with the September 2026 corrections: tensor shear = gamma/2,
principal strains of the summed tensor).

    eps = superpose_strains(results, RM, theta, ZC)

RM, theta : (npoints, ntires) distance (in) and angle (deg) from each tire to each evaluation
            point in the x-y plane (as MEAPA's RM_si / theta_si); ZC: depths (in, positive down).
Returns a dict of arrays of length npoints*len(ZC), ordered point-major like MATLAB's combvec
(for each tire: all points for ZC[0], then all points for ZC[1], ...).
"""
from __future__ import annotations
import numpy as np


def superpose_strains(results, RM, theta, ZC):
    RM = np.atleast_2d(np.asarray(RM, float)); theta = np.atleast_2d(np.asarray(theta, float))
    ZC = np.asarray(ZC, float).ravel()
    npts, ntires = RM.shape
    n = npts * ZC.size
    exx = np.zeros(n); eyy = np.zeros(n); ezz = np.zeros(n)
    exy = np.zeros(n); exz = np.zeros(n); eyz = np.zeros(n)
    Zg = np.repeat(ZC, npts)                        # combvec order: points vary fastest
    for t in range(ntires):
        Rg = np.tile(RM[:, t], ZC.size)
        Tg = np.tile(theta[:, t], ZC.size)
        er = results.interp("strain_r", Rg, Zg)
        et = results.interp("strain_t", Rg, Zg)
        ez = results.interp("strain_z", Rg, Zg)
        erz = results.interp("strain_rz", Rg, Zg) / 2.0          # engineering -> tensor
        c = np.cos(np.radians(Tg)); s = np.sin(np.radians(Tg))
        exx += c ** 2 * er + s ** 2 * et
        eyy += s ** 2 * er + c ** 2 * et
        exy += c * s * (er - et)
        exz += c * erz
        eyz += s * erz
        ezz += ez
    T = np.zeros((n, 3, 3))
    T[:, 0, 0] = exx; T[:, 1, 1] = eyy; T[:, 2, 2] = ezz
    T[:, 0, 1] = T[:, 1, 0] = exy; T[:, 0, 2] = T[:, 2, 0] = exz; T[:, 1, 2] = T[:, 2, 1] = eyz
    ok = np.all(np.isfinite(T.reshape(n, -1)), axis=1)
    pr = np.full((n, 3), np.nan)
    pr[ok] = np.linalg.eigvalsh(T[ok])                           # ascending
    return dict(eps_xx=exx, eps_yy=eyy, eps_zz=ezz, eps_xy=exy, eps_xz=exz, eps_yz=eyz,
                eps_11=pr[:, 0], eps_22=pr[:, 1], eps_33=pr[:, 2])
