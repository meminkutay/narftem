"""Stress-dependent modulus in 3D (same models as nartfem.axi.material).
Stresses (n,6) Voigt [xx, yy, zz, xy, yz, xz]; FE tension positive -> compression positive."""
from __future__ import annotations
import numpy as np


def field_stress(stress_fe, gamma_z, K0, residual, delsig=0.0):
    """Compression-positive stresses with overburden (zz), K0*overburden + residual (+ geogrid
    confinement) on xx and yy, tension removed on the normal components."""
    s = -np.asarray(stress_fe, float).copy()
    s[:, 0] += gamma_z * K0 + residual + delsig
    s[:, 1] += gamma_z * K0 + residual + delsig
    s[:, 2] += gamma_z
    s[:, :3] = np.maximum(s[:, :3], 0.0)
    return s


def resilient_modulus(s, mat_type, k, Pa, Mr_min):
    bulk = s[:, 0] + s[:, 1] + s[:, 2]
    if "KTheta" in mat_type:
        Mr = k[0] * bulk ** k[1]
    else:
        J2 = ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) / 6.0 \
             + s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
        toct = np.sqrt(2.0 / 3.0 * J2)
        Mr = k[0] * Pa * (bulk / Pa) ** k[1] * (toct / Pa + 1.0) ** k[2]
    Mr = np.where(np.isfinite(Mr), Mr, Mr_min)
    return np.maximum(Mr, Mr_min)


def overburden(mesh, res, thickness):
    depth = -mesh.center[:, 2]
    cum = np.concatenate([[0.0], np.cumsum(thickness)])
    gam = res["gamma_pci"]; nl = len(gam)
    gz = np.zeros(depth.size)
    for i in range(nl):
        top = cum[i]; bot = cum[i + 1] if i < nl - 1 else np.inf
        m = (depth > top) & (depth <= bot) if i > 0 else (depth <= bot)
        gz[m] = sum((cum[k + 1] - cum[k]) * gam[k] for k in range(i)) + (depth[m] - top) * gam[i]
    for i, mt in enumerate(res["mat_type"]):
        if mt == "Elastic":
            gz[mesh.layer == i] = 0.0
    return gz
