"""meapa-2 input -> the flat ``inps`` dict that the unchanged legacy preprocessor
(``_UPDAPSAC_MinersLaw/UPDAPSAC_STEP1.py``) and reliability post-processor expect, plus the
climate-file resolution (the .hcm files on disk are MATLAB MAT-files; the legacy reader wants a
gzipped CSV, so a cached .csv.gz is written next to the output folder).

The legacy package is imported by path (its modules use flat imports), see ``legacy_modules``.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import os
import sys

from . import schema

LEGACY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_UPDAPSAC_MinersLaw")


def legacy_modules():
    """Make the legacy flat modules importable; returns the folder."""
    if LEGACY_DIR not in sys.path:
        sys.path.insert(0, LEGACY_DIR)
    return LEGACY_DIR


# --------------------------------------------------------------------------- climate file
def _is_matfile(path):
    with open(path, "rb") as f:
        return f.read(6) == b"MATLAB"


def resolve_climate_file(climate: dict, cache_dir: str, base_dir: str = None) -> str:
    """Return a .csv.gz path readable by MCLIM_01ReadHCD.ReadClimateData (converting a MATLAB
    .hcm once, cached by content hash in ``cache_dir``)."""
    folder = climate.get("folder") or ""
    if base_dir and folder and not os.path.isabs(folder):
        folder = os.path.join(base_dir, folder)
    src = os.path.join(folder, climate["file"]) if folder else climate["file"]
    if not os.path.exists(src):
        raise FileNotFoundError("climate file not found: %s" % src)
    if src.endswith(".gz") or not _is_matfile(src):
        return src
    h = hashlib.sha256()
    with open(src, "rb") as f:
        h.update(f.read())
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, os.path.splitext(os.path.basename(src))[0] + "-" + h.hexdigest()[:10] + ".csv.gz")
    if not os.path.exists(dst):
        import scipy.io as sio
        import numpy as np
        m = sio.loadmat(src)
        arrs = [v for k, v in m.items() if not k.startswith("__")]
        A = np.asarray(arrs[0], float)
        tmp = dst + ".%d.tmp" % os.getpid()          # write-then-rename: safe for parallel runs
        with gzip.open(tmp, "wt") as f:
            for row in A:
                f.write("%d," % int(round(row[0])) + ",".join("%.6g" % x for x in row[1:]) + "\n")
        os.replace(tmp, dst)
    return dst


# --------------------------------------------------------------------------- inps
def to_legacy_inps(d: dict, climate_csv_gz: str = None) -> dict:
    """Build the legacy input dict.  ``climate_csv_gz`` overrides fldAirTemp/fnameAirTemp."""
    st, tr, cl, an = d["structure"], d["traffic"], d["climate"], d["analysis"]
    layers = st["layers"]; roles = schema.layer_roles(layers)
    n = len(layers)
    inps = {}
    inps["projectname"] = d.get("name", "")
    inps["userName"] = d.get("legacy", {}).get("userName", "")
    # ---- layers (legacy: flat lists; AC-specific lists are per AC layer in order)
    inps["layerType"] = list(roles)                     # nonlinear unbound layers are 'BASE' for STEP1
    inps["layerTH"] = [float(L["thickness_in"]) for L in layers[:-1]]
    inps["layerPR"] = [float(L.get("nu", 0.35)) for L in layers]
    inps["layerMR"] = [0.0 if r in ("AC", "EAC") else float(L.get("E_psi", 0.0)) for L, r in zip(layers, roles)]
    inps["layerGD"] = [float(L["unit_weight_pcf"]) for L in layers]
    inps["layerK"] = [float(L["thermal"]["conductivity"]) for L in layers]
    inps["layerC"] = [float(L["thermal"]["heat_capacity"]) for L in layers]
    inps["layerName"] = [L.get("name", "") for L in layers]
    ac_layers = [(i, L) for i, (L, r) in enumerate(zip(layers, roles)) if r in ("AC", "EAC")]
    inps["layerVa"] = [float(L["ac"]["Va_pct"]) for _, L in ac_layers]
    inps["layerVbeff"] = [float(L["ac"]["Vbeff_pct"]) for _, L in ac_layers]
    inps["layerTref"] = [float(L["ac"]["Tref_F"]) for _, L in ac_layers]
    inps["layerRTFO"] = [float(ac_layers[0][1]["ac"].get("RTFO", 0))]
    inps["layerIDT"] = [float(L["ac"]["IDT_psi"]) for _, L in ac_layers]
    inps["layerAbsorp"] = [float(cl.get("surface_absorptivity", 0.85))]
    inps["layerEstarDamage"] = [float(L["ac"].get("Estar_damage", 0.0)) for _, L in ac_layers] or [0.0]
    # |E*| master curve: either precomputed coefficients or raw data to be fitted by STEP1
    if all("mastercurve" in L["ac"] for _, L in ac_layers):
        inps["AClayers"] = {}
        for i, L in ac_layers:
            mc = L["ac"]["mastercurve"]
            inps["AClayers"]["Layer%d" % i] = {"EstarCoeff": list(mc["c"]), "aTF": list(mc["aT"]),
                                               "Abinder": mc.get("A", 0), "VTSbinder": mc.get("VTS", 0)}
    else:
        est, gst = [], []
        for i, L in enumerate(layers):
            if roles[i] in ("AC", "EAC"):
                e = L["ac"]["estar"]; g = L["ac"].get("gstar")
                est.append(json.dumps({"estar1M": [{"freq": r["freq_Hz"], "temp": r["temp_F"],
                                                    "estar": r["estar_psi"], "phaseAngle": r["phase_deg"]} for r in e],
                                       "uniquefreqs": sorted({r["freq_Hz"] for r in e}, reverse=True),
                                       "uniquetemps": sorted({r["temp_F"] for r in e})}))
                gst.append(json.dumps({"gstar1M": [{"temp": r["temp_F"], "gstar": r["gstar_psi"],
                                                    "phaseAngle": r["phase_deg"]} for r in g],
                                       "uniquetemps": sorted({r["temp_F"] for r in g})}) if g else "N/A")
            else:
                est.append("N/A"); gst.append("N/A")
        inps["estarJSON"] = est; inps["gstarJSON"] = gst
    # CSM
    csm = [L.get("csm") for L in layers if L.get("csm")]
    inps["csmEmax"] = [c["Emax_psi"] for c in csm] or [0.0]
    inps["csmEmin"] = [c["Emin_psi"] for c in csm] or [0.0]
    inps["csmMoR"] = [c["MoR_psi"] for c in csm] or [0.0]
    inps["csmBetaC1"] = [c["beta_c1"] for c in csm] or [0.0]
    inps["csmBetaC2"] = [c["beta_c2"] for c in csm] or [0.0]
    # subgrade / site
    sub = layers[-1]
    inps["p4Subg"] = float(sub["p4_pct"]); inps["p200Subg"] = float(sub["p200_pct"])
    inps["gwt"] = float(st["gwt_ft"])
    # ---- traffic
    inps["aadtt"] = float(tr["aadtt"]); inps["ddf"] = float(tr["ddf"]); inps["ldf"] = float(tr["ldf"])
    inps["monthText"] = tr["open_month"]
    inps["vehicleSpeed"] = float(tr["speed_mph"]); inps["tirePressure"] = float(tr["tire_pressure_psi"])
    inps["dualTireSpec"] = float(tr["dual_spacing_in"])
    inps["taAxleSpec"] = float(tr["axle_spacing_in"]["tandem"])
    inps["trAxleSpec"] = float(tr["axle_spacing_in"]["tridem"])
    inps["qdAxleSpec"] = float(tr["axle_spacing_in"]["quad"])
    inps["wheelWander"] = float(tr["wheel_wander_in"]); inps["avg_Axle_Width"] = float(tr.get("avg_axle_width_in", 8.5))
    inps["truckClass"] = list(tr["classes"]); inps["classPercent"] = list(tr["class_pct"])
    inps["growthPercent"] = [g["pct"] for g in tr["growth"]]
    inps["crType"] = [g["type"] for g in tr["growth"]]
    apt = tr["axles_per_truck"]
    inps["aptSi"], inps["aptTa"], inps["aptTr"], inps["aptQd"] = apt["single"], apt["tandem"], apt["tridem"], apt["quad"]
    inps["mdf"] = tr["mdf"]; inps["mdfMonth"] = list(range(1, 13))
    sp = tr["spectra"]
    inps["alsSingle"], inps["siLoadsLb"], inps["monthClassSi"] = sp["single"]["pct"], sp["single"]["loads_lb"], sp["single"]["month_class"]
    inps["alsTandem"], inps["taLoadsLb"], inps["monthClassTa"] = sp["tandem"]["pct"], sp["tandem"]["loads_lb"], sp["tandem"]["month_class"]
    inps["alsTridem"], inps["trLoadsLb"], inps["monthClassTr"] = sp["tridem"]["pct"], sp["tridem"]["loads_lb"], sp["tridem"]["month_class"]
    inps["alsQuad"], inps["qdLoadsLb"], inps["monthClassQd"] = sp["quad"]["pct"], sp["quad"]["loads_lb"], sp["quad"]["month_class"]
    # ---- climate
    if climate_csv_gz:
        inps["fldAirTemp"], inps["fnameAirTemp"] = os.path.split(climate_csv_gz)
    else:
        inps["fldAirTemp"], inps["fnameAirTemp"] = cl.get("folder", ""), cl["file"]
    inps["climateType"] = cl["type"]; inps["climateModel"] = cl["model"]
    inps["latitude"] = float(cl["latitude"]); inps["longitude"] = float(cl["longitude"])
    inps["utc"] = float(cl["utc"]); inps["daylightSavingSelected"] = float(cl.get("daylight_saving", 1))
    # ---- analysis
    inps["tDesign"] = float(an["design_life_yr"])
    inps.update(an["calibration"])
    inps["stdevData"] = dict(an["stdev_models"])
    inps["reliabilityThresholds"] = [{"distressName": t["distress"], "distressThreshold": t["threshold"],
                                      "distressReliability": t["reliability_pct"]} for t in an["thresholds"]]
    inps["distressChoice"] = an.get("distress_choice", "Cracking+Rutting")
    inps["savePeriod"] = an.get("save_period") or [2.0, 120.0]
    inps["savePeriodDistress"] = an.get("save_period_distress") or 1.0
    return inps
