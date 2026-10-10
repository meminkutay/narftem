"""Stress-dependent resilient modulus models (NewMrCalculation in DefineResponses.m and
NewMrCalculator in DefineGeogrid.m).  All stresses compression-positive, psi."""
from __future__ import annotations
import numpy as np


def field_stress(stress_fe, gamma_z, K0, residual):
    """FE stresses (tension positive, (n,4): r, z, theta, rz) -> compression-positive stresses
    with overburden and horizontal residual stress added, tension removed (shear untouched)."""
    s = -np.asarray(stress_fe, float).copy()
    s[:, 0] += gamma_z * K0 + residual
    s[:, 1] += gamma_z
    s[:, 2] += gamma_z * K0 + residual
    s[:, :3] = np.maximum(s[:, :3], 0.0)
    return s


def resilient_modulus(s, mat_type, k, Pa, Mr_min):
    """Mr (psi) for stresses s (n,4) compression-positive (already clipped).

    mat_type: 'Nonlin:KTheta[-GG]' or 'Nonlin:Universal[-GG]'."""
    bulk = s[:, 0] + s[:, 1] + s[:, 2]
    if "KTheta" in mat_type:
        Mr = k[0] * bulk ** k[1]
    else:
        J2 = ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 0] - s[:, 2]) ** 2 +
              (s[:, 1] - s[:, 2]) ** 2 + 6 * s[:, 3] ** 2) / 6.0
        toct = np.sqrt(2.0 / 3.0 * J2)
        Mr = k[0] * Pa * (bulk / Pa) ** k[1] * (toct / Pa + 1.0) ** k[2]
    Mr = np.where(np.isfinite(Mr), Mr, Mr_min)
    return np.maximum(Mr, Mr_min)
