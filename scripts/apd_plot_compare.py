"""Compare several nartfem-apd output JSONs: one figure per distress (all runs as curves), saved as
<OUT_DIR>/<distress>.png, plus a CSV of the end-of-life values.

    python scripts/apd_plot_compare.py                      (CONFIG block)
    python scripts/apd_plot_compare.py run1-output.json run2-output.json ... [--out DIR]

Legend labels: LABELS entry if given, else the file name without '-output.json'.  A glob is
accepted in FILES (e.g. "runs/sweep/*-nartfem-output.json").
"""
from __future__ import annotations
import csv
import glob
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_FLD = "/Users/Kutay/MEAPARuns/meapa_nartfem/no_gg_2/out"
# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    FILES=[f"{BASE_FLD}/*-nartfem-output.json"],   # paths or globs
    LABELS={},            # optional: {"FL_AC3_B5_SG4k_noGG-nartfem": "no geogrid", ...}
    OUT_DIR=f"{BASE_FLD}/plots",
    AT_RELIABILITY=True, # plot the at-reliability curves instead of the mean ones
    DPI=130,
    THRESHOLDS_FROM_INPUT=True,   # read analysis.thresholds of the first run's input JSON; else THRESHOLD_DEFAULTS
)
# -------------------------------------------------------------------------------------

DISTRESSES = [("FCbu_mo", "fcbuAtRel", "bottom_up_cracking", "Bottom-up fatigue cracking (%)"),
              ("FCtd_mo", "fctdtAtRel", "top_down_cracking", "Top-down fatigue cracking (ft/mile)"),
              ("TC_mo", "tc_ACAtRel", "thermal_cracking", "Thermal cracking (ft/mile)"),
              ("rut_tot_AC_mo", "acrutAtRel", "ac_rutting", "AC rutting (in)"),
              ("rut_tot_BASE_mo", None, "base_rutting", "Base rutting (in)"),
              ("rut_tot_SUBG_mo", None, "subgrade_rutting", "Subgrade rutting (in)"),
              ("rutTOT_mo", "totrutAtRel", "total_rutting", "Total rutting (in)"),
              ("IRI_mo", "iriAtRel", "iri", "IRI (in/mile)")]


def expand(files):
    out = []
    for f in files:
        hits = sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f]
        out.extend(hits)
    return out


def compare(files, out_dir, labels=None, at_rel=False, dpi=130):
    labels = labels or {}
    files = expand(files)
    if not files:
        raise SystemExit("no output files found")
    os.makedirs(out_dir, exist_ok=True)
    runs = []
    for f in files:
        with open(f) as fh:
            o = json.load(fh)
        key = os.path.basename(f).replace("-output.json", "")
        runs.append((labels.get(key, key), o))
    thr = thresholds_for(runs[0][1], CONFIG.get("THRESHOLDS_FROM_INPUT", True))
    rows = []
    for mkey, rkey, fname, lab in DISTRESSES:
        use = rkey if (at_rel and rkey) else mkey
        fig, ax = plt.subplots(figsize=(8, 5))
        n = 0
        for name, o in runs:
            if use not in o:
                continue
            y = np.asarray(o[use], float); t = np.arange(1, y.size + 1) / 12.0
            ax.plot(t, y, lw=1.6, label=name); n += 1
        if n == 0:
            plt.close(fig); continue
        if mkey in thr:
            ax.axhline(thr[mkey], color="r", ls="--", lw=1.2, label="threshold %g" % thr[mkey])
        ax.set_xlabel("years"); ax.set_ylabel(lab + (" - at reliability" if (at_rel and rkey) else ""))
        ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=1 if n <= 12 else 2)
        fig.tight_layout()
        p = os.path.join(out_dir, fname + ("_atrel" if (at_rel and rkey) else "") + ".png")
        fig.savefig(p, dpi=dpi); plt.close(fig)
        print("->", p)
    for name, o in runs:
        row = {"run": name, "engine": o.get("engine"), "years": o.get("design_life_yr")}
        for mkey, rkey, fname, lab in DISTRESSES:
            if mkey in o:
                row[fname] = float(o[mkey][-1])
            if rkey and rkey in o:
                row[fname + "_atrel"] = float(o[rkey][-1])
        rows.append(row)
    # ---- end-of-life bar charts: mean and at-reliability per run, threshold as a red line
    names = [r["run"] for r in rows]; x = np.arange(len(names)); w = 0.38
    for mkey, rkey, fname, lab in DISTRESSES:
        mean = [r.get(fname, np.nan) for r in rows]
        rel = [r.get(fname + "_atrel", np.nan) for r in rows] if rkey else None
        if all(np.isnan(mean)):
            continue
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(names) + 3), 5))
        if rel is not None and not all(np.isnan(rel)):
            ax.bar(x - w / 2, mean, w, label="end of life (mean)")
            ax.bar(x + w / 2, rel, w, label="end of life at reliability")
        else:
            ax.bar(x, mean, w * 1.6, label="end of life (mean)")
        if mkey in thr:
            ax.axhline(thr[mkey], color="r", ls="--", lw=1.2, label="threshold %g" % thr[mkey])
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(out_dir, fname + "_bars.png")
        fig.savefig(p, dpi=dpi); plt.close(fig)
        print("->", p)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("run", "engine", "years"), k))
    with open(os.path.join(out_dir, "end_of_life.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("-> %s (end-of-life values)" % os.path.join(out_dir, "end_of_life.csv"))


# ---- distress thresholds: from the run's input JSON (analysis.thresholds) when it can be found,
# ---- otherwise THRESHOLD_DEFAULTS (edit to taste)
THRESHOLD_DEFAULTS = {"FCbu_mo": 20.0, "FCtd_mo": 2000.0, "TC_mo": 1000.0, "rut_tot_AC_mo": 0.25,
                      "rutTOT_mo": 0.5, "IRI_mo": 172.0}
_THR_NAMES = {"IRI (in/mile)": "IRI_mo", "AC Top-Down Fatigue Cracking (ft/mile)": "FCtd_mo",
              "AC Bottom-up Fatigue Cracking (%)": "FCbu_mo", "AC Thermal Cracking (ft/mile)": "TC_mo",
              "Total Rutting (in)": "rutTOT_mo", "AC Rutting (in)": "rut_tot_AC_mo"}


def thresholds_for(o, use_input=True):
    thr = dict(THRESHOLD_DEFAULTS)
    inp = o.get("input")
    if use_input and inp and os.path.exists(inp):
        try:
            with open(inp) as fh:
                d = json.load(fh)
            for t in d.get("analysis", {}).get("thresholds", []):
                k = _THR_NAMES.get(t.get("distress"))
                if k:
                    thr[k] = float(t["threshold"])
        except Exception:
            pass
    return thr


if __name__ == "__main__":
    a = sys.argv[1:]
    out = CONFIG["OUT_DIR"]
    if "--out" in a:
        i = a.index("--out"); out = a[i + 1]; a = a[:i] + a[i + 2:]
    compare(a or CONFIG["FILES"], out, CONFIG["LABELS"], CONFIG["AT_RELIABILITY"], CONFIG["DPI"])
