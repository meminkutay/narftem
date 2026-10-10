"""Legacy MEAPA input JSON (flat lists per layer, JSON strings inside strings) -> meapa-2 format.

    python -m meapa.convert_legacy                       (uses the CONFIG block)
    python -m meapa.convert_legacy in.json [out.json]

Mapping notes (engineering choices):
  * layerType  AC/EAC/CSM/BASE/SUB -> role;  UNBOUNDNONLIN -> role BASE with type "universal"
    (k from nonlinGGJSON.kunb, k1 dimensionless as in nartfem).  layerMR of such a layer is kept
    as E_psi: it is the linear modulus MatLEA uses and the seed modulus of the nonlinear solve.
  * nonlinGGJSON.unbGG becomes a nartfem geogrid ONLY when unbGG.a != 0 (the legacy files carry
    the same unbGG block with a = 0 in the "no geosynthetic" case, so `a` is read as the on/off
    flag - see GEOGRID_FLAG below).  m_lbin (lb/in) -> M_kNm; vg = 0 -> nartfem default 0.35;
    ncyc = 0 -> N = 1e6.
  * The legacy code has one RTFO flag and one absorptivity for all AC layers; they are copied to
    every AC layer.
  * The reliability thresholds keep MEAPA's fixed order (IRI, TD, BU, TC, total rut, AC rut, RC).
"""
from __future__ import annotations
import json
import os
import sys

# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    IN_FILES=[
        "_UPDAPSAC_MinersLaw/exampleinputs/inputMEAPA_MATFEA_FL131487_AC3_B5_HSR5050eq.json",
        "_UPDAPSAC_MinersLaw/exampleinputs/inputMEAPA_MATFEA_FL131487_AC3_B5_noGS.json",
    ],
    OUT_DIR="examples/meapa",          # output folder (files keep their base name + .meapa2.json)
    GEOGRID_FLAG="a",                  # unbGG key used as on/off flag (None -> always include)
    CLIMATE_FOLDER=None,               # override climate folder (None -> legacy fldAirTemp)
)
# -------------------------------------------------------------------------------------

LBIN_TO_KNM = 1.0 / 224.81 * 39.37           # as nartfem.axi.model
DISTRESS_ORDER = ["IRI (in/mile)", "AC Top-Down Fatigue Cracking (ft/mile)",
                  "AC Bottom-up Fatigue Cracking (%)", "AC Thermal Cracking (ft/mile)",
                  "Total Rutting (in)", "AC Rutting (in)", "AC Reflective Cracking (%)"]
CALIB_KEYS = ["biBu", "kiBu", "cTfBu", "biTd", "kiTd", "cTfTd", "briAC", "kriAC", "briB", "briS",
              "cIRI", "iriO", "tCoefK", "kri_base", "kri_subgrade", "bri_base", "bri_subgrade"]


def _jsonstr(s):
    if s is None or s == "N/A":
        return None
    return json.loads(s) if isinstance(s, str) else s


def _geogrid_from_unbgg(g: dict) -> dict:
    return {"l_in": float(g["l_in"]), "delta_in": float(g["delta_in"]),
            "M_kNm": float(g["m_lbin"]) * LBIN_TO_KNM,
            "vg": float(g["vg"]) if g.get("vg", 0) > 0 else 0.35,
            "phi_deg": float(g["fhi"]), "c_psi": float(g["cpsi"]),
            "eps0": float(g["eo"]), "rho": float(g["rrho"]), "beta": float(g["bbeta"]),
            "m": float(g["mm"]), "n": float(g["nn"]),
            "N": float(g["ncyc"]) if g.get("ncyc", 0) > 0 else 1.0e6}


def convert(d: dict, geogrid_flag="a", climate_folder=None) -> dict:
    ltype = list(d["layerType"]); n = len(ltype)
    th = list(d["layerTH"]) + [0.0]
    names = d.get("layerName") or ["Layer %d" % (i + 1) for i in range(n)]
    nonlin = d.get("nonlinGGJSON") or ["N/A"] * n
    estar = d.get("estarJSON") or ["N/A"] * n
    gstar = d.get("gstarJSON") or ["N/A"] * n
    ac_i = 0
    layers = []
    for i in range(n):
        role = {"UNBOUNDNONLIN": "BASE"}.get(ltype[i], ltype[i])
        L = {"name": names[i], "type": "elastic", "thickness_in": float(th[i]),
             "E_psi": float(d["layerMR"][i]), "nu": float(d["layerPR"][i]),
             "unit_weight_pcf": float(d["layerGD"][i]), "role": role,
             "thermal": {"conductivity": float(d["layerK"][i]), "heat_capacity": float(d["layerC"][i])}}
        if i == n - 1:
            L.pop("thickness_in")
            L["p4_pct"] = float(d["p4Subg"]); L["p200_pct"] = float(d["p200Subg"])
        if role in ("AC", "EAC"):
            if L["E_psi"] <= 0:
                L["E_psi"] = 400000.0          # placeholder: MEAPA replaces it by |E*|(T, f) per climate case
            e = _jsonstr(estar[i]); g = _jsonstr(gstar[i])
            ac = {"Va_pct": float(d["layerVa"][ac_i]), "Vbeff_pct": float(d["layerVbeff"][ac_i]),
                  "Tref_F": float(d["layerTref"][ac_i]), "RTFO": float(d["layerRTFO"][0]),
                  "IDT_psi": float(d["layerIDT"][ac_i])}
            if e is not None:
                ac["estar"] = [{"temp_F": r["temp"], "freq_Hz": r["freq"], "estar_psi": r["estar"],
                                "phase_deg": r["phaseAngle"]} for r in e["estar1M"]]
            if g is not None:
                ac["gstar"] = [{"temp_F": r["temp"], "gstar_psi": r["gstar"], "phase_deg": r["phaseAngle"]}
                               for r in g["gstar1M"]]
            if "AClayers" in d:                       # precomputed master curve
                lay = d["AClayers"]["Layer%d" % i]
                ac["mastercurve"] = {"c": list(lay["EstarCoeff"]), "aT": list(lay["aTF"]),
                                     "A": lay.get("Abinder", 0), "VTS": lay.get("VTSbinder", 0)}
            if role == "EAC" and d.get("layerEstarDamage"):
                ac["Estar_damage"] = d["layerEstarDamage"][0]
            L["ac"] = ac
            ac_i += 1
        if role == "CSM":
            L["csm"] = {"Emax_psi": d["csmEmax"][0], "Emin_psi": d["csmEmin"][0], "MoR_psi": d["csmMoR"][0],
                        "beta_c1": d["csmBetaC1"][0], "beta_c2": d["csmBetaC2"][0]}
        nl = _jsonstr(nonlin[i])
        if ltype[i] == "UNBOUNDNONLIN" and nl:
            L["type"] = "universal"
            L["k"] = [float(x) for x in nl["kunb"]]
            L["K0"] = 0.5; L["residual_psi"] = 3.0; L["Mr_min_psi"] = 1000
            gg = nl.get("unbGG")
            if gg and (geogrid_flag is None or float(gg.get(geogrid_flag, 0)) != 0):
                L["geogrids"] = [_geogrid_from_unbgg(gg)]
            L["legacy_nonlin"] = {k: v for k, v in nl.items() if k not in ("kunb", "unbGG", "layerType")}
        layers.append(L)

    structure = {"layers": layers, "gwt_ft": float(d["gwt"]), "bedrock_depth_in": 1000.0}

    def spectrum(als, loads, mc):
        return {"loads_lb": [float(x) for x in loads], "month_class": [list(p) for p in mc],
                "pct": [list(map(float, r)) for r in als]}

    growth_type = d.get("crType") or d.get("grType")
    traffic = {
        "aadtt": float(d["aadtt"]), "ddf": float(d["ddf"]), "ldf": float(d["ldf"]),
        "open_month": d.get("monthText") or d.get("monthStart"),
        "speed_mph": float(d["vehicleSpeed"]), "tire_pressure_psi": float(d["tirePressure"]),
        "dual_spacing_in": float(d["dualTireSpec"]),
        "axle_spacing_in": {"tandem": float(d["taAxleSpec"]), "tridem": float(d["trAxleSpec"]),
                            "quad": float(d["qdAxleSpec"])},
        "wheel_wander_in": float(d["wheelWander"]), "avg_axle_width_in": float(d.get("avg_Axle_Width", 8.5)),
        "classes": list(d["truckClass"]), "class_pct": [float(x) for x in d["classPercent"]],
        "growth": [{"type": t, "pct": float(p)} for t, p in zip(growth_type, d["growthPercent"])],
        "axles_per_truck": {"single": list(d["aptSi"]), "tandem": list(d["aptTa"]),
                            "tridem": list(d["aptTr"]), "quad": list(d["aptQd"])},
        "mdf": [list(map(float, r)) for r in d["mdf"]],
        "spectra": {"single": spectrum(d["alsSingle"], d["siLoadsLb"], d["monthClassSi"]),
                    "tandem": spectrum(d["alsTandem"], d["taLoadsLb"], d["monthClassTa"]),
                    "tridem": spectrum(d["alsTridem"], d["trLoadsLb"], d["monthClassTr"]),
                    "quad": spectrum(d["alsQuad"], d["qdLoadsLb"], d["monthClassQd"])},
    }
    climate = {"file": d["fnameAirTemp"], "folder": climate_folder or d["fldAirTemp"],
               "type": d["climateType"], "model": d["climateModel"],
               "latitude": float(d["latitude"]), "longitude": float(d["longitude"]),
               "utc": float(d["utc"]), "daylight_saving": float(d["daylightSavingSelected"]),
               "surface_absorptivity": float(d["layerAbsorp"][0])}
    thr = {t["distressName"]: t for t in d["reliabilityThresholds"]}
    thresholds = [{"distress": nm, "threshold": thr[nm]["distressThreshold"],
                   "reliability_pct": thr[nm]["distressReliability"]} for nm in DISTRESS_ORDER if nm in thr]
    analysis = {
        "design_life_yr": int(d["tDesign"]),
        "response_engine": {"engine": "auto", "load_scaling": "linear"},
        "thresholds": thresholds,
        "stdev_models": {k: v for k, v in d["stdevData"].items() if k != "projectId"},
        "calibration": {k: d[k] for k in CALIB_KEYS if k in d},
        "distress_choice": d.get("distressChoice", "Cracking+Rutting"),
        "save_period": d.get("savePeriod"), "save_period_distress": d.get("savePeriodDistress"),
    }
    return {"format": "meapa-2", "name": d.get("projectname", ""),
            "legacy": {k: d.get(k) for k in ("userName", "yrstart", "projectFolder") if k in d},
            "structure": structure, "traffic": traffic, "climate": climate, "analysis": analysis}


def convert_file(src, dst=None, **kw):
    with open(src) as f:
        d = json.load(f)
    out = convert(d, **kw)
    if dst is None:
        dst = os.path.splitext(src)[0] + ".meapa2.json"
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    return dst


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    kw = dict(geogrid_flag=CONFIG["GEOGRID_FLAG"], climate_folder=CONFIG["CLIMATE_FOLDER"])
    if argv:
        src = argv[0]; dst = argv[1] if len(argv) > 1 else None
        print("->", convert_file(src, dst, **kw))
    else:
        for src in CONFIG["IN_FILES"]:
            base = os.path.splitext(os.path.basename(src))[0] + ".meapa2.json"
            print("->", convert_file(src, os.path.join(CONFIG["OUT_DIR"], base), **kw))


if __name__ == "__main__":
    main()
