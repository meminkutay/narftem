"""ESALs of one or more nartfem-apd-1 (or legacy) inputs - AASHTO-93 equivalents over the design
life, as computed by the traffic preprocessor.

    python scripts/apd_calc_esal.py                       (CONFIG block)
    python scripts/apd_calc_esal.py a.apd.json b.apd.json ... [--csv out.csv]

Prints total ESALs, first-year ESALs, ESALs per axle group and axle counts; with several files
also writes a CSV.
"""
from __future__ import annotations
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    # FILES=["examples/apd/FL_AC3_B5_SG4k_noGG.apd.json"],     # paths or globs, repo-relative or absolute
    FILES=["/Users/Kutay/MEAPARuns/meapa_nartfem/tanzila/gg_tanzila/gg_AC4_B6.apd.json"],     # paths or globs, repo-relative or absolute
    CSV=None,                                                # e.g. "runs/esal.csv"
)
# -------------------------------------------------------------------------------------


def main(files, csv_path=None):
    from nartfem.apd import run as arun
    from nartfem.apd.traffic import esal
    paths = []
    for f in files:
        f = f if os.path.isabs(f) else os.path.join(ROOT, f)
        paths += sorted(glob.glob(f)) if any(c in f for c in "*?[") else [f]
    rows = []
    for p in paths:
        d = arun.load_any(p)
        e = esal(d)
        row = {"input": os.path.basename(p), "aadtt": e["aadtt"], "design_life_yr": e["ESAL_design_life_yr"],
               "ESAL_total": e["ESAL_total"], "ESAL_first_year": e["ESAL_first_year"]}
        row.update({"ESAL_" + k: v for k, v in e["ESAL_by_axle_group"].items()})
        row.update({"axles_" + k: v for k, v in e["axles_total_by_group"].items()})
        rows.append(row)
        print("%-50s AADTT %6.0f  %2d yr  ESAL total %.1f Million  (first year %.4g; si %.3g ta %.3g tr %.3g qd %.3g)" % (
            row["input"], e["aadtt"], e["ESAL_design_life_yr"], e["ESAL_total"]/1e6, e["ESAL_first_year"],
            e["ESAL_by_axle_group"]["si"], e["ESAL_by_axle_group"]["ta"], e["ESAL_by_axle_group"]["tr"], e["ESAL_by_axle_group"]["qd"]))
    if csv_path:
        csv_path = csv_path if os.path.isabs(csv_path) else os.path.join(ROOT, csv_path)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("->", csv_path)
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]; c = CONFIG["CSV"]
    if "--csv" in a:
        i = a.index("--csv"); c = a[i + 1]; a = a[:i] + a[i + 2:]
    main(a or CONFIG["FILES"], c)
