"""Build meapa-2 inputs for the nartfem example structures (examples/*.json) with a
representative traffic / climate / analysis block taken from a legacy MEAPA input.

    python scripts/meapa_make_examples.py           (CONFIG block)

For each nartfem structure the AC layer(s) get the 'ac' block (|E*| data, Va, Vbeff, Tref,
IDT) and every layer the 'thermal' block of the corresponding legacy layer (AC -> legacy AC,
unbound -> legacy base, last -> legacy subgrade).  The East Lansing case gets the nearest MERRA2
station of the HCM folder (CFM-147034: 42.5 N, -84.375 W) and its coordinates.
"""
from __future__ import annotations
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from meapa.convert_legacy import convert as convert_legacy     # noqa: E402

# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    LEGACY="_UPDAPSAC_MinersLaw/exampleinputs/inputMEAPA_MATFEA_FL131487_AC3_B5_noGS.json",
    CASES=[  # (nartfem structure, output name, climate override or None)
        ("examples/FL_AC3_B5_SG4k_GG.json", "examples/meapa/FL_AC3_B5_SG4k_GG.meapa2.json", None),
        ("examples/FL_AC3_B5_SG4k_noGG.json", "examples/meapa/FL_AC3_B5_SG4k_noGG.meapa2.json", None),
        ("examples/EL_AC4_B8_SG15k_GGmid.json", "examples/meapa/EL_AC4_B8_SG15k_GGmid.meapa2.json",
         {"file": "CFM-147034.hcm", "latitude": 42.5, "longitude": -84.375, "utc": -5.0}),
    ],
)
# -------------------------------------------------------------------------------------


def make(legacy_path, structure_path, out_path, climate_override=None):
    with open(os.path.join(ROOT, legacy_path)) as f:
        leg = convert_legacy(json.load(f))
    with open(os.path.join(ROOT, structure_path)) as f:
        nf = json.load(f)
    L_ac, L_base, L_sub = leg["structure"]["layers"][0], leg["structure"]["layers"][1], leg["structure"]["layers"][-1]
    layers = []
    n = len(nf["layers"])
    for i, L in enumerate(nf["layers"]):
        L = copy.deepcopy(L)
        if i == 0:
            L["role"] = "AC"; L["ac"] = copy.deepcopy(L_ac["ac"]); L["thermal"] = dict(L_ac["thermal"])
            L["unit_weight_pcf"] = L.get("unit_weight_pcf", L_ac["unit_weight_pcf"])
        elif i == n - 1:
            L["role"] = "SUB"; L["thermal"] = dict(L_sub["thermal"])
            L["p4_pct"] = L_sub["p4_pct"]; L["p200_pct"] = L_sub["p200_pct"]
        else:
            L["role"] = "BASE"; L["thermal"] = dict(L_base["thermal"])
            if L.get("type") in ("universal", "ktheta") and not L.get("E_psi"):
                # linear modulus for MatLEA / seed modulus: the legacy representative Mr of the base
                L["E_psi"] = L_base["E_psi"]
        layers.append(L)
    d = {"format": "meapa-2", "name": nf.get("name", ""),
         "structure": {"layers": layers, "gwt_ft": leg["structure"]["gwt_ft"],
                       "bedrock_depth_in": leg["structure"]["bedrock_depth_in"],
                       "domain": nf.get("domain", {}), "solver": nf.get("solver", {})},
         "traffic": copy.deepcopy(leg["traffic"]), "climate": copy.deepcopy(leg["climate"]),
         "analysis": copy.deepcopy(leg["analysis"])}
    if climate_override:
        d["climate"].update(climate_override)
    d["analysis"]["response_engine"] = {"engine": "auto", "load_scaling": "linear", "shear": "legacy", "workers": 1}
    os.makedirs(os.path.dirname(os.path.join(ROOT, out_path)), exist_ok=True)
    with open(os.path.join(ROOT, out_path), "w") as f:
        json.dump(d, f, indent=1)
    print("->", out_path)


if __name__ == "__main__":
    for s, o, c in CONFIG["CASES"]:
        make(CONFIG["LEGACY"], s, o, c)
