"""Compare nartfem-apd runs (MatLEA vs nartfem engine, with/without geogrid): tables + figure.

    python scripts/apd_compare_engines.py          (CONFIG block)

Reads <RUN_DIR>/<case>-<engine>-output.json and -cases.json written by apd.run and writes
<RUN_DIR>/comparison.md and <RUN_DIR>/comparison.png.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    RUN_DIR="runs/full",
    CASES=["FL_AC3_B5_SG4k_linear", "FL_AC3_B5_SG4k_noGG", "FL_AC3_B5_SG4k_GG", "EL_AC4_B8_SG15k_GGmid"],
    ENGINES=["matlea", "nartfem"],
    YEARS=[1, 5, 10, 20],
    GEOGRID_PAIRS=[("FL_AC3_B5_SG4k_noGG", "FL_AC3_B5_SG4k_GG")],
    PLOT=True,
)
# -------------------------------------------------------------------------------------

DISTRESS = [("FCbu_mo", "Bottom-up cracking (%)"), ("FCtd_mo", "Top-down cracking (ft/mile)"),
            ("rut_tot_AC_mo", "AC rut (in)"), ("rut_tot_BASE_mo", "Base rut (in)"),
            ("rut_tot_SUBG_mo", "Subgrade rut (in)"), ("rutTOT_mo", "Total rut (in)"),
            ("IRI_mo", "IRI (in/mile)"), ("Dbu_cum", "Cum. BU damage (-)"), ("Dtd_cum", "Cum. TD damage (-)")]
STRAINS = [("eps_bu", "AC-bottom tensile strain"), ("eps_td", "Near-surface max principal strain"),
           ("eps_v_ac", "AC vertical strain (max sublayer)"), ("eps_v_base", "Base-top vertical strain"),
           ("eps_v_subg", "Subgrade-top vertical strain")]


def load(run_dir, case, eng):
    p = os.path.join(ROOT, run_dir, "%s-%s-output.json" % (case, eng))
    if not os.path.exists(p):
        return None, None
    with open(p) as f:
        out = json.load(f)
    with open(p.replace("-output.json", "-cases.json")) as f:
        cases = json.load(f)
    return out, cases


def at_year(arr, yr, per_year=12):
    """value at the end of year ``yr``; cumulative-damage arrays have 5 quintiles per month."""
    arr = np.asarray(arr)
    i = min(per_year * yr - 1, len(arr) - 1)
    return arr[i]


def fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    ax = abs(x)
    return "%.3g" % x if ax < 1e-2 or ax >= 1e4 else "%.3f" % x if ax < 10 else "%.1f" % x


def main(cfg=CONFIG):
    lines = []
    P = lines.append
    data = {(c, e): load(cfg["RUN_DIR"], c, e) for c in cfg["CASES"] for e in cfg["ENGINES"]}
    P("# MatLEA vs nartfem response engine - nartfem-apd distress comparison\n")
    for c in cfg["CASES"]:
        o = data[(c, cfg["ENGINES"][0])][0] or data[(c, cfg["ENGINES"][1])][0]
        if o:
            P("* **%s**: %s (%d yr; AC sublayers %s in; analysis depths %s in)" % (
                c, o["name"], o["design_life_yr"], o["sublayer_thickness_in"], [round(z, 2) for z in o["analysis_depths_in"]]))
    # ---- distress tables
    for c in cfg["CASES"]:
        outs = {e: data[(c, e)][0] for e in cfg["ENGINES"]}
        if not all(outs.values()):
            continue
        P("\n## %s - distresses (MatLEA | nartfem | ratio nartfem/MatLEA)\n" % c)
        P("| distress | " + " | ".join("yr %d" % y for y in cfg["YEARS"]) + " |")
        P("|---|" + "---|" * len(cfg["YEARS"]))
        for key, lab in DISTRESS:
            cells = []
            for y in cfg["YEARS"]:
                py = 60 if key.endswith("_cum") else 12
                a = at_year(outs["matlea"][key], y, py); b = at_year(outs["nartfem"][key], y, py)
                r = b / a if a else float("nan")
                cells.append("%s / %s (%.2f)" % (fmt(a), fmt(b), r))
            P("| %s | %s |" % (lab, " | ".join(cells)))
        rel = [("fcbuAtRel", "BU cracking @rel (%)"), ("fctdtAtRel", "TD cracking @rel (ft/mile)"),
               ("totrutAtRel", "Total rut @rel (in)"), ("acrutAtRel", "AC rut @rel (in)"), ("iriAtRel", "IRI @rel (in/mile)")]
        P("\n| at reliability (end of life) | MatLEA | nartfem |")
        P("|---|---|---|")
        for key, lab in rel:
            P("| %s | %s | %s |" % (lab, fmt(outs["matlea"][key][-1]), fmt(outs["nartfem"][key][-1])))
        # engine stats
        for e in cfg["ENGINES"]:
            st = outs[e]["engine_stats"]
            P("\n%s: loop %.0f s, engine %s" % (e, outs[e]["elapsed_s"], st))
    # ---- critical strains
    for c in cfg["CASES"]:
        ca = {e: data[(c, e)][1] for e in cfg["ENGINES"]}
        if not all(ca.values()):
            continue
        P("\n## %s - critical strains per climate case, nartfem / MatLEA ratio (single | tandem axle)\n" % c)
        P("| response | mean ratio | min | max | MatLEA mean (1e-6) | nartfem mean (1e-6) |")
        P("|---|---|---|---|---|---|")
        for key, lab in STRAINS:
            for ax in ("si", "ta"):
                a = np.array([r[key][ax] for r in ca["matlea"]]); b = np.array([r[key][ax] for r in ca["nartfem"]])
                m = a > 0
                rr = b[m] / a[m]
                P("| %s (%s) | %.3f | %.3f | %.3f | %.0f | %.0f |" % (lab, ax, rr.mean(), rr.min(), rr.max(), 1e6 * a[m].mean(), 1e6 * b[m].mean()))
        lm = [r["layer_moduli"] for r in ca["nartfem"] if r.get("layer_moduli")]
        if lm:
            lm = np.array(lm)
            P("\nnartfem layer moduli (geometric mean over the layer, single-axle state), psi: min / mean / max per layer:")
            for j in range(lm.shape[1]):
                P("* layer %d: %.0f / %.0f / %.0f" % (j + 1, lm[:, j].min(), lm[:, j].mean(), lm[:, j].max()))
        # damage-weighted sensitivity: fatigue ~ eps^-3.9, subgrade rut ~ eps^1.35
        a = np.array([r["eps_bu"]["si"] for r in ca["matlea"]]); b = np.array([r["eps_bu"]["si"] for r in ca["nartfem"]])
        P("\nImplied by the strain ratios alone: BU fatigue damage ratio ~ mean((eps_nf/eps_lea)^3.9) = %.2f; "
          "subgrade rut ratio ~ mean((eps_nf/eps_lea)^1.35) = %.2f" % (
              np.mean((b / a) ** 3.9), np.mean((np.array([r["eps_v_subg"]["si"] for r in ca["nartfem"]]) /
                                                 np.array([r["eps_v_subg"]["si"] for r in ca["matlea"]])) ** 1.35)))
    # ---- geogrid effect
    for c0, c1 in cfg["GEOGRID_PAIRS"]:
        P("\n## Geogrid effect: %s -> %s (ratio with/without), end of design life\n" % (c0, c1))
        P("| distress | " + " | ".join(cfg["ENGINES"]) + " |")
        P("|---|" + "---|" * len(cfg["ENGINES"]))
        for key, lab in DISTRESS:
            cells = []
            for e in cfg["ENGINES"]:
                o0, o1 = data[(c0, e)][0], data[(c1, e)][0]
                if o0 and o1:
                    a, b = np.asarray(o0[key])[-1], np.asarray(o1[key])[-1]
                    cells.append("%s -> %s (%.3f)" % (fmt(a), fmt(b), b / a if a else float("nan")))
                else:
                    cells.append("-")
            P("| %s | %s |" % (lab, " | ".join(cells)))
        for e in cfg["ENGINES"]:
            k0, k1 = data[(c0, e)][1], data[(c1, e)][1]
            if k0 and k1:
                cells = []
                for key, lab in STRAINS:
                    a = np.array([r[key]["si"] for r in k0]); b = np.array([r[key]["si"] for r in k1])
                    cells.append("%s %.3f" % (lab.split(" ")[0], np.mean(b[a > 0] / a[a > 0])))
                P("* %s strain ratios with/without geogrid (single axle, mean over cases): %s" % (e, "; ".join(cells)))
    md = "\n".join(lines)
    out = os.path.join(ROOT, cfg["RUN_DIR"], "comparison.md")
    with open(out, "w") as f:
        f.write(md)
    print(md)
    print("\n-> written", out)
    if cfg["PLOT"]:
        plot(cfg, data)


def plot(cfg, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("FCbu_mo", "Bottom-up cracking (%)"), ("FCtd_mo", "Top-down cracking (ft/mile)"),
            ("rut_tot_AC_mo", "AC rut (in)"), ("rut_tot_BASE_mo", "Base rut (in)"),
            ("rut_tot_SUBG_mo", "Subgrade rut (in)"), ("IRI_mo", "IRI (in/mile)")]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    styles = {"matlea": "--", "nartfem": "-"}
    for ax, (key, lab) in zip(axs.ravel(), keys):
        for c in cfg["CASES"]:
            for e in cfg["ENGINES"]:
                o = data[(c, e)][0]
                if o:
                    y = np.asarray(o[key]); ax.plot(np.arange(1, y.size + 1) / 12, y, styles[e], label="%s %s" % (c, e))
        ax.set_xlabel("years"); ax.set_ylabel(lab); ax.grid(alpha=.3)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(ROOT, cfg["RUN_DIR"], "comparison.png")
    fig.savefig(p, dpi=130)
    print("-> written", p)


if __name__ == "__main__":
    main()
