"""Traffic summaries of an nartfem-apd-1 (or legacy) input: ESALs and axle counts.

ESALs are the AASHTO-93 equivalents computed by the legacy preprocessor
(f_rawTraf_to_AXL2.processTrafficData: load equivalency with the 18/33.7/48/62.9-kip standard
axles and the 3.9-type exponents, summed over the design life).  The legacy code prints the
total to the status file but does not keep it; this module recomputes it from the same
function so that it can be reported in the output JSON (``run.py``) or on its own
(``scripts/apd_calc_esal.py``).
"""
from __future__ import annotations
import contextlib
import io
import numpy as np

from .legacy_bridge import to_legacy_inps, legacy_modules


def esal_from_inps(inps: dict) -> dict:
    """ESAL totals from a legacy inps dict (no climate / structure needed)."""
    legacy_modules()
    from f_rawTraf_to_AXL2 import processTrafficData
    tdesign = int(inps["tDesign"])
    grType = inps.get("grType", inps.get("crType"))
    month = inps.get("monthStart", inps.get("monthText"))
    alsData = {"alsSingle": inps["alsSingle"], "alsTandem": inps["alsTandem"], "alsTridem": inps["alsTridem"],
               "alsQuad": inps["alsQuad"], "siLoadsLb": inps["siLoadsLb"], "taLoadsLb": inps["taLoadsLb"],
               "trLoadsLb": inps["trLoadsLb"], "qdLoadsLb": inps["qdLoadsLb"], "DataType": 2}
    with contextlib.redirect_stdout(io.StringIO()):
        tn = processTrafficData(inps["aadtt"], inps["aptQd"], inps["aptSi"], inps["aptTa"], inps["aptTr"], grType,
                                inps["ddf"], inps["ldf"], inps["mdf"], inps["truckClass"], inps["classPercent"],
                                inps["growthPercent"], month, tdesign, alsData)
    # per axle group, per month (same formula as the legacy ESAL block: axles/day x 30.4166 days x (P/Pstd)^n)
    std = {"si": (18464.6, 3.962), "ta": (33737.0, 3.907), "tr": (47951.2, 3.924), "qd": (62869.7, 3.9155)}
    keys = {"si": ("NAij_SitwDay", "siLoadsLb"), "ta": ("NAij_TatwDay", "taLoadsLb"),
            "tr": ("NAij_TrtwDay", "trLoadsLb"), "qd": ("NAij_QdtwDay", "qdLoadsLb")}
    per_month = {}
    for ax, (nk, lk) in keys.items():
        N = np.asarray(tn[nk], float); P = np.asarray(inps[lk], float)
        per_month[ax] = (N * 30.4166 * (P / std[ax][0]) ** std[ax][1]).sum(axis=1)
    total_month = sum(per_month.values())
    nyr = tdesign
    per_year = total_month.reshape(nyr, 12).sum(axis=1)
    axles_month = {ax: np.asarray(tn[keys[ax][0]], float).sum(axis=1) * 30.4166 for ax in keys}
    return {"ESAL_total": float(tn["ESALtot"]), "ESAL_design_life_yr": nyr,
            "ESAL_first_year": float(per_year[0]), "ESAL_per_year": per_year.tolist(),
            "ESAL_by_axle_group": {ax: float(v.sum()) for ax, v in per_month.items()},
            "axles_total_by_group": {ax: float(v.sum()) for ax, v in axles_month.items()},
            "aadtt": float(inps["aadtt"])}


def esal(d: dict) -> dict:
    """ESAL summary of an nartfem-apd-1 input dict."""
    return esal_from_inps(to_legacy_inps(d))
