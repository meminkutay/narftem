"""MEAPA input format, version 2 (nartfem style).

One JSON file with four blocks.  Units: in, psi, lb, pcf, °F, mph, kN/m, degrees (the units
of MEAPA/nartfem; nothing is converted on input except where stated).

    {
      "format": "meapa-2",
      "name": "FL-131487 | AC 3in / base 5in / SG",
      "structure": {
        "layers": [                       <- the nartfem.axi layer schema AS-IS (model.py), plus:
          {"name": "AC", "type": "ac", "thickness_in": 3.0, "nu": 0.25, "unit_weight_pcf": 150,
                                          <- "ac": viscoelastic; the modulus is |E*|(T, f) from the
                                             "ac" block per month/quintile (no E_psi / E_table)
           "role": "AC",                  <- MEAPA role: AC | EAC | CSM | BASE | SUB (optional,
                                             derived from position/"ac" block when absent)
           "ac": {                        <- only for AC/EAC layers (MEAPA |E*| / aging / thermal)
             "Va_pct": 7.0, "Vbeff_pct": 10.0, "Tref_F": 70, "RTFO": 0, "IDT_psi": 461.7,
             "estar": [{"temp_F":13.8,"freq_Hz":25.1,"estar_psi":5.0e6,"phase_deg":4.8}, ...],
             "gstar": [{"temp_F":40,"gstar_psi":1.4e7,"phase_deg":56.1}, ...],
             "mastercurve": {"c": [c1,c2,c3,c4], "aT": [a1,a2], "A": .., "VTS": ..}   <- optional,
                                             used instead of fitting "estar"/"gstar" when present
           },
           "thermal": {"conductivity": 0.67, "heat_capacity": 0.23}   <- BTU/hr-ft-F, BTU/lb-F
          },
          {"name": "Base", "type": "universal", "thickness_in": 5, "k": [k1,k2,k3], "E_psi": 13412,
           "nu": 0.3, "unit_weight_pcf": 124, "thermal": {...},
           "geogrids": [{"l_in": 0.5, "delta_in": 6, "M_kNm": 427.8, ...}]},  <- nartfem geogrid
          {"name": "Subgrade", "type": "elastic", "E_psi": 4000, "nu": 0.35, "unit_weight_pcf": 98,
           "thermal": {...}, "p4_pct": 95, "p200_pct": 79}
        ],
        "gwt_ft": 5.0,
        "bedrock_depth_in": 1000
      },
      "traffic": {
        "aadtt": 7500, "ddf": 0.5, "ldf": 0.95, "open_month": "JANUARY", "speed_mph": 60,
        "tire_pressure_psi": 120, "dual_spacing_in": 12,
        "axle_spacing_in": {"tandem": 51.6, "tridem": 49.2, "quad": 49.2},
        "wheel_wander_in": 10, "avg_axle_width_in": 8.5,
        "classes": ["Class 4", ...], "class_pct": [...],
        "growth": [{"type": "compound", "pct": 3.0}, ...]   (one per class),
        "axles_per_truck": {"single": [...], "tandem": [...], "tridem": [...], "quad": [...]},
        "mdf": [[12 x nclass]],
        "spectra": {"single": {"loads_lb": [...], "month_class": [[m, c], ...], "pct": [[...]]},
                    "tandem": {...}, "tridem": {...}, "quad": {...}}
      },
      "climate": {
        "file": "CFM-131487.hcm", "folder": "/Users/Kutay/HCM",   <- .hcm (MATLAB) or .csv.gz
        "type": "MERRA2", "model": "Original",
        "latitude": 24.0, "longitude": -80.6, "utc": -5, "daylight_saving": 1,
        "surface_absorptivity": 0.85
      },
      "analysis": {
        "design_life_yr": 20,
        "response_engine": {"engine": "auto",        <- auto | matlea | nartfem
                            "load_scaling": "linear",<- linear (MEAPA) | power (nartfem runs at
                                                        several pressures, log-log interpolation)
                            "power_levels": [0.5, 1.0, 2.0],
                            "shear": "legacy",       <- legacy (MEAPA: gamma_rz as tensor shear) | tensor
                            "n_phi": 8, "workers": 1},
        "thresholds": [{"distress": "IRI (in/mile)", "threshold": 172, "reliability_pct": 90}, ...],
        "stdev_models": {"actdfcstdev": "...", ...},     <- MEAPA reliability stdev equations
        "calibration": {"biBu": [...], "kiBu": [...], "cTfBu": [...], "biTd": [...], "kiTd": [...],
                        "cTfTd": [...], "briAC": [...], "kriAC": [...], "briB": .., "briS": ..,
                        "cIRI": [...], "iriO": .., "tCoefK": ..},
        "distress_choice": "Cracking+Rutting", "save_period": [2, 120]
      }
    }

``engine = "auto"`` uses MatLEA when every layer is "elastic"/"elastic_depth" without geogrids
and nartfem.axi.multitire otherwise.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List

FORMAT_TAG = "meapa-2"
ROLES = ("AC", "EAC", "CSM", "BASE", "SUB")
AXLES = ("single", "tandem", "tridem", "quad")


def load(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    if not is_v2(d):
        raise ValueError("%s is not a meapa-2 input (use meapa.convert_legacy for legacy MEAPA files)" % path)
    validate(d)
    d.setdefault("_source_path", os.path.abspath(path))
    return d


def is_v2(d: dict) -> bool:
    return d.get("format") == FORMAT_TAG or ("structure" in d and "traffic" in d)


def is_legacy(d: dict) -> bool:
    return "layerTH" in d and "layerType" in d


def save(d: dict, path: str):
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def layer_roles(layers: List[dict]) -> List[str]:
    """MEAPA role of each layer: explicit "role", else AC when an "ac" block exists, SUB for the
    last layer, BASE otherwise."""
    roles = []
    for i, L in enumerate(layers):
        r = L.get("role")
        if r is None:
            r = "SUB" if i == len(layers) - 1 else ("AC" if ("ac" in L or L.get("type") == "ac") else "BASE")
        if r not in ROLES:
            raise ValueError("layer %d: unknown role %r" % (i, r))
        roles.append(r)
    return roles


def is_all_linear(structure: dict) -> bool:
    """True when MatLEA can represent the structure exactly (no stress-dependent layer, no geogrid)."""
    for L in structure["layers"]:
        if L.get("type", "elastic") not in ("ac", "elastic", "elastic_depth") or L.get("geogrids"):
            return False
    return True


def validate(d: dict):
    for blk in ("structure", "traffic", "climate", "analysis"):
        if blk not in d:
            raise ValueError("missing block %r" % blk)
    layers = d["structure"]["layers"]
    if len(layers) < 2:
        raise ValueError("at least two layers (one above the subgrade) are required")
    roles = layer_roles(layers)
    if roles[-1] != "SUB":
        raise ValueError("the last layer must be the subgrade (role SUB)")
    if roles[0] not in ("AC",):
        raise ValueError("the first layer must be an AC layer with an 'ac' block")
    for i, L in enumerate(layers[:-1]):
        if not L.get("thickness_in", 0) > 0:
            raise ValueError("layer %d (%s): thickness_in must be > 0" % (i, L.get("name", "")))
    for i, (L, r) in enumerate(zip(layers, roles)):
        if r in ("AC", "EAC") and "ac" not in L:
            raise ValueError("layer %d is AC/EAC but has no 'ac' block" % i)
        if r in ("AC", "EAC") and L.get("type", "ac") not in ("ac", "elastic"):
            raise ValueError("layer %d: an AC layer has type 'ac' (modulus = |E*| from the 'ac' block)" % i)
        if r in ("AC", "EAC") and ("E_psi" in L or "E_table" in L):
            raise ValueError("layer %d: E_psi / E_table are not used for AC layers (|E*| from the 'ac' block); remove them" % i)
        if "thermal" not in L:
            raise ValueError("layer %d: 'thermal' block (conductivity, heat_capacity) is required" % i)
        if L.get("type", "elastic") in ("universal", "ktheta") and not L.get("k"):
            raise ValueError("layer %d: nonlinear layer needs 'k'" % i)
    t = d["traffic"]
    for key in ("aadtt", "classes", "class_pct", "growth", "axles_per_truck", "mdf", "spectra",
                "tire_pressure_psi", "dual_spacing_in", "axle_spacing_in", "speed_mph", "open_month"):
        if key not in t:
            raise ValueError("traffic.%s is missing" % key)
    for ax in AXLES:
        if ax not in t["spectra"] or ax not in t["axles_per_truck"]:
            raise ValueError("traffic.spectra/axles_per_truck need %r" % ax)
    a = d["analysis"]
    for key in ("design_life_yr", "calibration", "thresholds", "stdev_models"):
        if key not in a:
            raise ValueError("analysis.%s is missing" % key)


def engine_choice(d: dict) -> str:
    eng = d["analysis"].get("response_engine", {}).get("engine", "auto")
    if eng == "auto":
        return "matlea" if is_all_linear(d["structure"]) else "nartfem"
    if eng not in ("matlea", "nartfem"):
        raise ValueError("analysis.response_engine.engine must be auto|matlea|nartfem")
    return eng


DEFAULT_ENGINE_OPTIONS: Dict = {
    "engine": "auto", "load_scaling": "linear", "power_levels": [0.5, 1.0, 2.0],
    "shear": "legacy", "n_phi": 8, "workers": 1, "mesh_scale": 1.3,
    "ac_sublayer_interfaces": True, "warm_start": True, "verbose": False,
    "subgrade_split_in": 6.0,      # node row at h + 6 in (MEAPA 2nd subgrade point; 0 = off), see engine_nartfem.build_model
}


def engine_options(d: dict) -> dict:
    o = dict(DEFAULT_ENGINE_OPTIONS)
    o.update(d["analysis"].get("response_engine", {}))
    return o
