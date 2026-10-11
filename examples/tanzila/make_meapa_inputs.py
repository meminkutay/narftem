"""
make_meapa_inputs.py
--------------------
Step 4: write MEAPA (nartfem-apd) input files for a before/after comparison.

Takes an existing nartfem-apd design as the template and changes only the base layer
(k's and the seed modulus E_psi) and the base-rutting calibration factor briB:

  *_base-control.apd.json     control base:  k = control fit,              briB = BRIB_CONTROL
  *_base-HSR-equiv.apd.json   reinforced:    k = equivalent reinforced fit, briB = briB_R (recommended)
  *_base-HSR-rutonly.apd.json bracketing:    k = control fit,              briB = BRIB_CONTROL x R_lab
                              (benefit only through rutting; no stiffness gain - lower bound on
                               the fatigue benefit, upper bound on the rutting-only credit)

No geogrid block is written: the NCHRP 1-50 confinement model cannot reproduce the lab gain
for this stress-hardening base (see fit_mr.py), and adding it on top of the equivalent k's
would double count. The equivalent material is applied to the whole base thickness.

Reads OUT_DIR/mr_fit_results.json and OUT_DIR/rutting_results.json.
"""
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = Path(__file__).resolve().parent
OUT_DIR = DATA_ROOT / "out"
MEAPA_DIR = OUT_DIR / "apd_jsons"
TEMPLATE = DATA_ROOT.parent / "apd" / "FL_AC3_B5_SG4k_noGG.apd.json"   # any nartfem-apd design
BASE_LAYER_NAME = "Base"
WRITE_RUT_ONLY_CASE = True
# =============================================================================

import copy
import json
import math

PCI_PER_PCF = 1 / 1728.0


def mr_universal(k, sx, sz, Pa):
    theta = 2 * sx + sz
    toct = math.sqrt(2) / 3 * abs(sz - sx)
    return k[0] * Pa * (theta / Pa) ** k[1] * (toct / Pa + 1) ** k[2]


def at_rest_seed(layers, i, k, Pa):
    """Seed modulus E_psi (also the legacy 'layerMR'): Mr at mid-depth of the layer under
    overburden, K0 and the residual stress only. It is a starting value; the nonlinear solve
    replaces it. (For the template's own k's this gives ~12.6 ksi vs its 11.75 ksi.)"""
    z, sv = 0.0, 0.0
    for L in layers[:i]:
        sv += L["thickness_in"] * L.get("unit_weight_pcf", 0) * PCI_PER_PCF
    B = layers[i]
    sv += 0.5 * B["thickness_in"] * B.get("unit_weight_pcf", 0) * PCI_PER_PCF
    sh = B.get("K0", 0.5) * sv + B.get("residual_psi", 3.0)
    return mr_universal(k, sh, sv, Pa)


def main():
    MEAPA_DIR.mkdir(parents=True, exist_ok=True)
    tpl = json.load(open(TEMPLATE))
    mr = json.load(open(OUT_DIR / "mr_fit_results.json"))
    rut = json.load(open(OUT_DIR / "rutting_results.json"))
    Pa = tpl["structure"].get("solver", {}).get("atm_pressure_psi", 14.706)
    layers = tpl["structure"]["layers"]
    ib = next(i for i, L in enumerate(layers) if L.get("name") == BASE_LAYER_NAME)
    assert layers[ib]["type"] == "universal", "template base must be a universal-model layer"

    k_tpl = layers[ib]["k"]
    print(f"Template: {TEMPLATE.name}; base k = {k_tpl}, E_psi = {layers[ib].get('E_psi')} "
          f"(at-rest check: {at_rest_seed(layers, ib, k_tpl, Pa):.0f} psi)")

    kC = [round(v, 4) for v in mr["control"]["k"]]
    kR = [round(v, 4) for v in mr["reinforced_equivalent"]["k"]]
    bC = rut["briB_control"]
    cases = [("base-control", kC, bC, "control base (no geosynthetic), lab-calibrated k"),
             ("base-HSR-equiv", kR, round(rut["briB_reinforced"], 3),
              "HSR100-100 reinforced base as equivalent material (k_R,eq, briB_R)")]
    if WRITE_RUT_ONLY_CASE:
        cases.append(("base-HSR-rutonly", kC, round(rut["briB_rut_only"], 3),
                      "HSR100-100 benefit in rutting only (control k, briB x R_lab) - bracketing case"))
    stem = TEMPLATE.name.replace("_noGG", "").replace(".apd.json", "")
    for tag, k, briB, desc in cases:
        d = copy.deepcopy(tpl)
        L = d["structure"]["layers"][ib]
        L["k"] = k
        L.pop("geogrids", None)
        L["E_psi"] = round(at_rest_seed(d["structure"]["layers"], ib, k, Pa), 0)
        d["analysis"]["calibration"]["briB"] = briB
        d["name"] = f"{tpl['name'].split(' - ')[0]} - {desc}; k={k}, briB={briB}"
        path = MEAPA_DIR / f"{stem}_{tag}.apd.json"
        json.dump(d, open(path, "w"), indent=1)
        print(f"  {path.name}: k = {k}, E_psi = {L['E_psi']:.0f}, briB = {briB}")

    # the base-layer block on its own (to paste into other designs)
    blocks = {tag: dict(name=BASE_LAYER_NAME, type="universal", k=k, calibration=dict(briB=briB))
              for tag, k, briB, _ in cases}
    json.dump(blocks, open(MEAPA_DIR / "base_layer_blocks.json", "w"), indent=1)
    print("Wrote", MEAPA_DIR)


if __name__ == "__main__":
    main()
