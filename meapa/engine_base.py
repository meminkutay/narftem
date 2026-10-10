"""Common interface of the response engines and the Step-1 context they need."""
from __future__ import annotations
import numpy as np

AXLE_KEYS = ("si", "ta", "tr", "qd")
AXLE_NAMES = {"si": "single", "ta": "tandem", "tr": "tridem", "qd": "quad"}
N_AXLES = {"si": 1, "ta": 2, "tr": 3, "qd": 4}
REF_LOAD_LB = {"si": 18000.0, "ta": 36000.0, "tr": 54000.0, "qd": 72000.0}


def tire_positions(ctx, ax):
    """(x, y) of the tires of an axle group, exactly as f_analysis_locations: dual tires at
    x = +/- Sx/2, axles at y = 0, Sy, 2Sy ...  (x across, y along the traffic direction)."""
    Sx = ctx["dualTireSpec"]
    Sy = {"si": 0.0, "ta": ctx["taAxleSpec"], "tr": ctx["trAxleSpec"], "qd": ctx["qdAxleSpec"]}[ax]
    return [(sx * Sx / 2, k * Sy) for k in range(N_AXLES[ax]) for sx in (-1.0, 1.0)]


def context_from_step1(Step1Output, inps, LayNo_sub):
    """Everything an engine needs, as plain arrays."""
    al = Step1Output["analysisLoc"]
    TH_sub = np.asarray(Step1Output["TH_sub"], float)
    nAC = len(Step1Output["TH_subAC"])
    ctx = {
        "ZC_in_subM": Step1Output["ZC_in_subM"], "ZC_in": np.asarray(Step1Output["ZC_in_subM"][0], float),
        "RC": np.asarray(Step1Output["RC"], float), "tirePressure": float(Step1Output["tirePressure"]),
        "CR_in": float(Step1Output["CR_in"]), "PR_sub": np.asarray(Step1Output["PR_sub"], float),
        "TH_sub": TH_sub, "LayNo_sub": np.asarray(LayNo_sub, int), "IACsub": np.arange(nAC),
        "dualTireSpec": float(inps["dualTireSpec"]), "taAxleSpec": float(inps["taAxleSpec"]),
        "trAxleSpec": float(inps["trAxleSpec"]), "qdAxleSpec": float(inps["qdAxleSpec"]),
        "analysisLoc": al,
    }
    for ax in AXLE_KEYS:
        ctx["xyz_" + ax] = np.asarray(al["xyz_" + ax], float)
        ctx["RM_" + ax] = np.asarray(al["RM_" + ax], float)
        ctx["theta_" + ax] = np.asarray(al["theta_" + ax], float)
    return ctx


class ResponseEngine:
    """responses(E_sub, levels) -> {axle: [dict(eps_xx, eps_yy, eps_zz, eps_princ) per level]}
    where E_sub are the sublayer moduli of one climate case (psi, incl. the subgrade) and the
    arrays are ordered like analysisLoc['xyz_<axle>'] (depth outer, point inner)."""
    name = "base"

    def __init__(self, ctx):
        self.ctx = ctx
        self.stats = {}

    def responses(self, E_sub, levels=(1.0,)):
        raise NotImplementedError
