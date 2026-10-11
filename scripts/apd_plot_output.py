"""Plot one nartfem-apd output JSON (<name>-<engine>-output.json): the monthly distresses, with the
at-reliability curves and thresholds, as a 2x3 figure saved next to the output (or OUT_PNG).

    python scripts/apd_plot_output.py                       (CONFIG block)
    python scripts/apd_plot_output.py path/to/x-nartfem-output.json [out.png]
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    OUTPUT_JSON="/Users/Kutay/MEAPARuns/meapa_nartfem/gg_anl/retry/out/gg_AC6_B8-nartfem-output.json",
    OUT_PNG=None,          # None -> same name with .png
    SHOW=False,            # also open a window (PyCharm)
    THRESHOLDS_FROM_INPUT=True,   # read analysis.thresholds of the run's input JSON; else THRESHOLD_DEFAULTS
)
# -------------------------------------------------------------------------------------

# (monthly key, at-reliability key, label, threshold name in the input)
PANELS = [("FCbu_mo", "fcbuAtRel", "Bottom-up fatigue cracking (%)"),
          ("FCtd_mo", "fctdtAtRel", "Top-down fatigue cracking (ft/mile)"),
          ("TC_mo", "tc_ACAtRel", "Thermal cracking (ft/mile)"),
          ("rut_tot_AC_mo", "acrutAtRel", "AC rutting (in)"),
          ("rutTOT_mo", "totrutAtRel", "Total rutting (in)"),
          ("IRI_mo", "iriAtRel", "IRI (in/mile)")]


def plot_output(path, out_png=None, show=False):
    with open(path) as f:
        o = json.load(f)
    thr = thresholds_for(o, CONFIG.get("THRESHOLDS_FROM_INPUT", True))
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (key, rel, lab) in zip(axs.ravel(), PANELS):
        y = np.asarray(o.get(key, []), float)
        if y.size == 0:
            ax.set_visible(False); continue
        t = np.arange(1, y.size + 1) / 12.0
        ax.plot(t, y, "-", lw=2, label="mean (50 %)")
        if key == "rutTOT_mo":                       # components of the total rut
            for k2, l2 in (("rut_tot_AC_mo", "AC"), ("rut_tot_BASE_mo", "base"), ("rut_tot_SUBG_mo", "subgrade")):
                if k2 in o:
                    ax.plot(t, np.asarray(o[k2], float), "--", lw=1, label=l2)
        if rel in o and len(o[rel]) == y.size:
            ax.plot(t, np.asarray(o[rel], float), ":", lw=2, label="at reliability")
        if key in thr:
            ax.axhline(thr[key], color="r", ls="--", lw=1.2, label="threshold %g" % thr[key])
        ax.set_xlabel("years"); ax.set_ylabel(lab); ax.grid(alpha=.3); ax.legend(fontsize=8)
    eng = o.get("engine", "?"); st = o.get("engine_stats", {})
    fig.suptitle("%s\n%s engine, %s yr, loop %.0f s, solves %s" % (o.get("name", os.path.basename(path)), eng,
                 o.get("design_life_yr", "?"), o.get("elapsed_s", 0), st.get("solves", "?")), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = out_png or os.path.splitext(path)[0] + ".png"
    fig.savefig(out_png, dpi=130)
    print("->", out_png)
    if show:
        plt.show()
    return out_png


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
    plot_output(a[0] if a else CONFIG["OUTPUT_JSON"], a[1] if len(a) > 1 else CONFIG["OUT_PNG"], CONFIG["SHOW"])
