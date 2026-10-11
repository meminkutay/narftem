"""Generate meapa-2 inputs from a template for every combination of AC and base thickness.

    python scripts/prep_meapa_inputs_v1.py          (CONFIG block)

The template's first layer is the AC (role AC) and the second the base; all other content
(traffic, climate, analysis, materials, geogrids) is copied unchanged.  Geogrid positions are
given from the layer bottom (nartfem `l_in`), so a geogrid at the base bottom stays at the
bottom when the base gets thicker; a geogrid that would end up above the new base top is
clipped to the top (and reported).  Output: <OUT_DIR>/<prefix>_AC<h>_B<h>.meapa2.json
(thicknesses in inches, '.' -> 'p', e.g. AC2p5_B6) - ready for scripts/meapa_batch.py.
"""
from __future__ import annotations
import copy
import itertools
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    TEMPLATE="examples/meapa/FL_AC3_B5_SG4k_noGG.meapa2.json",   # repo-relative or absolute
    # TEMPLATE="examples/meapa/FL_AC3_B5_SG4k_GG.meapa2.json",   # repo-relative or absolute
    AC_THICKNESS_IN=[4.0, 5.0, 6.0, 7.0],
    BASE_THICKNESS_IN=[6.0, 8.0, 10.0, 12.0],
    OUT_DIR="/Users/Kutay/MEAPARuns/meapa_nartfem/no_gg",
    PREFIX="no_gg",                      # None -> template file name without .meapa2.json
    ENGINE="nartfem",                      # None -> keep the template's; or "nartfem" | "matlea" | "auto"
    DESIGN_LIFE_YR=20,              # None -> keep the template's
    # constant overrides applied to every generated file: dotted path -> value
    # (list items by index, e.g. "structure.layers.2.E_psi"; a missing path is created)
    SET_FIELDS={
        "traffic.aadtt": 4000,
        # "climate.folder": "/mnt/lustre/arrow/home/ac.mkutay/HCM/",
    },
    # extra sweep dimensions (dotted path -> list of values), combined with the thicknesses;
    # the file name gets _<lastkey><value> per dimension, e.g. {"traffic.aadtt": [1000, 5000]}
    SWEEP_FIELDS={},
)
# -------------------------------------------------------------------------------------


def _fmt(h):
    return ("%g" % h).replace(".", "p")


def _set(d, path, value):
    """Set a dotted path in nested dicts/lists ("traffic.aadtt", "structure.layers.2.E_psi")."""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur.setdefault(k, {})
    last = keys[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def _rename(name, h_ac, h_b):
    """Replace the thicknesses in the template's name ("AC 3in", "base 5in", "AC3_B5" ...); when
    the name has no such pattern the new thicknesses are appended."""
    n, hits = name, 0
    for pat, rep in ((r"(?i)(AC\s*)\d+(\.\d+)?(\s*in)", r"\g<1>%g\g<3>" % h_ac),
                     (r"(?i)(base\s*)\d+(\.\d+)?(\s*in)", r"\g<1>%g\g<3>" % h_b),
                     (r"(?i)(_AC)\d+(p\d+)?(_B)\d+(p\d+)?", r"\g<1>%s\g<3>%s" % (_fmt(h_ac), _fmt(h_b)))):
        n, k = re.subn(pat, rep, n); hits += k
    return n if hits else "%s | AC %g in / base %g in" % (name, h_ac, h_b)


def make_inputs(template, ac_list, base_list, out_dir, prefix=None, engine=None, years=None,
                set_fields=None, sweep_fields=None):
    set_fields = set_fields or {}; sweep_fields = sweep_fields or {}
    template = template if os.path.isabs(template) else os.path.join(ROOT, template)
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    with open(template) as f:
        tpl = json.load(f)
    layers = tpl["structure"]["layers"]
    if len(layers) < 3 or layers[0].get("role", "AC") != "AC":
        raise ValueError("template needs AC / base / ... / subgrade layers with the AC first")
    prefix = prefix or os.path.basename(template).replace(".meapa2.json", "").replace(".json", "")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    sweep_keys = list(sweep_fields)
    for combo in itertools.product(ac_list, base_list, *[sweep_fields[k] for k in sweep_keys]):
        h_ac, h_b, extra = combo[0], combo[1], combo[2:]
        d = copy.deepcopy(tpl)
        for path, val in set_fields.items():              # constant overrides
            _set(d, path, val)
        for path, val in zip(sweep_keys, extra):           # swept fields
            _set(d, path, val)
        L = d["structure"]["layers"]
        L[0]["thickness_in"] = float(h_ac)
        L[1]["thickness_in"] = float(h_b)
        for g in L[1].get("geogrids", []):
            if g["l_in"] >= h_b:                        # keep the geogrid inside the (thinner) base
                g["l_in"] = round(h_b - 0.5, 3)
                print("  note: geogrid l_in clipped to %.3f in for base %.3g in" % (g["l_in"], h_b))
        d["name"] = _rename(tpl.get("name", prefix), h_ac, h_b)
        d.setdefault("analysis", {})
        if engine:
            d["analysis"].setdefault("response_engine", {})["engine"] = engine
        if years:
            d["analysis"]["design_life_yr"] = int(years)
        d["sweep"] = {"template": os.path.basename(template), "ac_in": h_ac, "base_in": h_b,
                      "set_fields": set_fields, **{k: v for k, v in zip(sweep_keys, extra)}}
        suffix = "".join("_%s%s" % (k.split(".")[-1], _fmt(v) if isinstance(v, (int, float)) else str(v))
                         for k, v in zip(sweep_keys, extra))
        name = "%s_AC%s_B%s%s.meapa2.json" % (prefix, _fmt(h_ac), _fmt(h_b), suffix)
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
        written.append(path)
        print("->", os.path.relpath(path, ROOT))
    print("%d inputs written to %s" % (len(written), out_dir))
    return written


if __name__ == "__main__":
    c = CONFIG
    make_inputs(c["TEMPLATE"], c["AC_THICKNESS_IN"], c["BASE_THICKNESS_IN"], c["OUT_DIR"], c["PREFIX"],
                c["ENGINE"], c["DESIGN_LIFE_YR"], c.get("SET_FIELDS"), c.get("SWEEP_FIELDS"))
