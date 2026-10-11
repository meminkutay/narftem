"""Batch MEAPA runs: every JSON input in a folder (meapa-2 or legacy), one process per run.

    python scripts/meapa_batch.py                 (CONFIG block)
    python scripts/meapa_batch.py --input-dir examples/meapa --engines nartfem --jobs 4 --years 2

Each (input, engine) pair is one job; jobs run in a process pool of JOBS workers (the Miner's loop
of one run is sequential, so parallelism comes from running several inputs at once - with JOBS
runs in flight, leave the per-run `workers` at 1).  Per run: <OUT_DIR>/<name>-<engine>-output.json,
-cases.json and <name>-<engine>.log; at the end <OUT_DIR>/batch_summary.csv (end-of-life distresses,
at-reliability values, run time) and the same table on the console.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__));
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    INPUT_DIR="/Users/Kutay/MEAPARuns/meapa_nartfem/no_gg",  # folder with the input JSONs (repo-relative or absolute)
    PATTERN="*.json",  # glob inside INPUT_DIR ("*.meapa2.json" to skip legacy files)
    ENGINES=["nartfem"],  # one run per engine per input: "nartfem", "matlea", or "auto"
    YEARS=None,  # None -> design life of each input
    OUT_DIR="/Users/Kutay/MEAPARuns/meapa_nartfem/no_gg/out",
    JOBS=4,  # runs in flight at the same time (processes)
    LOAD_SCALING=None,  # None -> input's; "linear" | "power"
    SKIP_EXISTING=True,  # skip a job whose -output.json already exists
)
# -------------------------------------------------------------------------------------

SUMMARY_COLS = [("FCbu_mo", "BU crack %"), ("FCtd_mo", "TD crack ft/mi"), ("rut_tot_AC_mo", "AC rut in"),
                ("rut_tot_BASE_mo", "base rut in"), ("rut_tot_SUBG_mo", "subg rut in"), ("rutTOT_mo", "total rut in"),
                ("IRI_mo", "IRI in/mi"), ("fcbuAtRel", "BU@rel %"), ("totrutAtRel", "rut@rel in"),
                ("iriAtRel", "IRI@rel")]


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def run_one(args):
    """One (input, engine) job in its own process; returns a summary row."""
    path, engine, years, out_dir, load_scaling = args
    from meapa import run as mrun
    base = os.path.splitext(os.path.basename(path))[0].replace(".meapa2", "")
    tag = engine if engine != "auto" else None
    log_path = os.path.join(out_dir, "%s-%s.log" % (base, engine))
    t0 = time.time()
    row = {"input": os.path.basename(path), "engine": engine, "status": "ok", "time_s": 0.0}
    with open(log_path, "w") as logf:
        def log(*a):
            logf.write(" ".join(str(x) for x in a) + "\n");
            logf.flush()

        try:
            d = mrun.load_any(path)
            out, _, _ = mrun.run(d, None if engine == "auto" else engine, years, out_dir, 1, load_scaling, True,
                                 log=log, tag=tag)
            row["engine"] = out["engine"];
            row["years"] = out["design_life_yr"]
            for key, lab in SUMMARY_COLS:
                row[lab] = float(out[key][-1])
        except Exception as e:  # keep the batch going; the log has the traceback
            logf.write(traceback.format_exc())
            row["status"] = "ERROR: %s" % e
    row["time_s"] = round(time.time() - t0, 1)
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=CONFIG["INPUT_DIR"]);
    p.add_argument("--pattern", default=CONFIG["PATTERN"])
    p.add_argument("--engines", nargs="+", default=CONFIG["ENGINES"]);
    p.add_argument("--years", type=int, default=CONFIG["YEARS"])
    p.add_argument("--out", default=CONFIG["OUT_DIR"]);
    p.add_argument("--jobs", type=int, default=CONFIG["JOBS"])
    p.add_argument("--load-scaling", default=CONFIG["LOAD_SCALING"])
    p.add_argument("--rerun", action="store_true", help="run even when the output exists")
    a = p.parse_args(argv)
    in_dir, out_dir = _abs(a.input_dir), _abs(a.out)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, a.pattern)))
    files = [f for f in files if not f.endswith("-output.json") and not f.endswith("-cases.json")]
    jobs = []
    for f in files:
        for eng in a.engines:
            base = os.path.splitext(os.path.basename(f))[0].replace(".meapa2", "")
            if CONFIG["SKIP_EXISTING"] and not a.rerun and eng != "auto" and \
                    os.path.exists(os.path.join(out_dir, "%s-%s-output.json" % (base, eng))):
                print("skip (exists):", base, eng);
                continue
            jobs.append((f, eng, a.years, out_dir, a.load_scaling))
    print("%d inputs, %d jobs, %d in parallel -> %s" % (len(files), len(jobs), a.jobs, out_dir))
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result();
            rows.append(r)
            print("  done %-45s %-8s %-6s %6.0f s  %s" % (r["input"], r["engine"], r["status"][:6], r["time_s"],
                                                          "BU %.3g%%  rut %.3f in  IRI %.1f" % (r["BU crack %"],
                                                                                                r["total rut in"],
                                                                                                r["IRI in/mi"]) if r[
                                                                                                                       "status"] == "ok" else ""))
    rows.sort(key=lambda r: (r["input"], r["engine"]))
    cols = ["input", "engine", "years", "status", "time_s"] + [lab for _, lab in SUMMARY_COLS]
    with open(os.path.join(out_dir, "batch_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore");
        w.writeheader();
        w.writerows(rows)
    print("\nwall time %.0f s; summary -> %s" % (time.time() - t0, os.path.join(out_dir, "batch_summary.csv")))
    print("%-45s %-8s %5s %-7s %10s %12s %10s %10s" % ("input", "engine", "yr", "status", "BU %", "TD ft/mi", "rut in",
                                                       "IRI"))
    for r in rows:
        if r["status"] == "ok":
            print(
                "%-45s %-8s %5s %-7s %10.3g %12.0f %10.3f %10.1f" % (r["input"], r["engine"], r.get("years", ""), "ok",
                                                                     r["BU crack %"], r["TD crack ft/mi"],
                                                                     r["total rut in"], r["IRI in/mi"]))
        else:
            print("%-45s %-8s %5s %s" % (r["input"], r["engine"], "", r["status"]))


if __name__ == "__main__":
    main()
