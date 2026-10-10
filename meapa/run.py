"""Run MEAPA (meapa-2 or legacy input) with the MatLEA or the nartfem response engine.

    python -m meapa.run                       (CONFIG block below)
    python -m meapa.run input.json [--engine matlea|nartfem|auto] [--years N] [--out DIR]
                                             [--workers N] [--load-scaling linear|power]

Steps: input -> legacy ``inps`` (legacy_bridge) -> UPDAPSAC_STEP1 (unchanged legacy
preprocessor: traffic, sublayering, climate, |E*|, thermal cracking) -> Miner's loop
(meapa.distress) with the chosen engine -> reliability (legacy f_calc_distress_atrel) ->
<name>-<engine>-output.json (+ <name>-<engine>-cases.json with the critical strains per case).
"""
from __future__ import annotations
import argparse
import copy
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

from . import schema
from .legacy_bridge import to_legacy_inps, resolve_climate_file, legacy_modules
from .convert_legacy import convert as convert_legacy
from .engine_base import context_from_step1

# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    INPUT="examples/meapa/FL_AC3_B5_SG4k_noGG.meapa2.json",
    ENGINE=None,            # None -> input's analysis.response_engine.engine; or "matlea" | "nartfem"
    YEARS=None,             # None -> input's design life
    OUT_DIR="runs",
    WORKERS=None,           # None -> input's response_engine.workers
    LOAD_SCALING=None,      # None -> input's; "linear" | "power"
    QUIET_STEP1=True,       # swallow the legacy preprocessor's console output
)
# -------------------------------------------------------------------------------------


def load_any(path):
    with open(path) as f:
        d = json.load(f)
    if schema.is_legacy(d):
        d = convert_legacy(d)
        d["_converted_from_legacy"] = True
    schema.validate(d)
    d["_source_path"] = os.path.abspath(path)
    return d


def step1(d, out_dir, quiet=True, log=print):
    legacy_modules()
    from UPDAPSAC_STEP1 import UPDAPSAC_STEP1
    from f_sublayer_MEPDGmethod import ff_sublayer_MEPDGmethod
    base_dir = os.path.dirname(d.get("_source_path", "")) or None
    clim = resolve_climate_file(d["climate"], os.path.join(out_dir, "climate_cache"), base_dir)
    inps = to_legacy_inps(d, clim)
    t0 = time.time()
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            S1 = UPDAPSAC_STEP1(inps, out_dir, "status.txt")
    else:
        S1 = UPDAPSAC_STEP1(inps, out_dir, "status.txt")
    log(" --> Step 1 done (%.1f s): %d months, %d sublayers" % (time.time() - t0, S1["numMo"], len(S1["TH_sub"])))
    TH_sub, _, LayNo_sub = ff_sublayer_MEPDGmethod(np.array(inps["layerTH"]), np.array(inps["layerType"]))
    if S1["CSMExists"] or S1["EACExists"]:
        raise NotImplementedError("meapa.run handles AC over granular base only; use the legacy UPDAPSACPY for CSM/EAC")
    return S1, inps, LayNo_sub


def make_engine(name, ctx, d, opt, log):
    if name == "matlea":
        from .engine_matlea import MatLEAEngine
        return MatLEAEngine(ctx, opt, log)
    from .engine_nartfem import NartfemEngine
    return NartfemEngine(ctx, d["structure"], opt, log)


# ----------------------------------------------------------------------------- parallel responses
def _worker(args):
    name, ctx, d, opt, Ez_rows, levels = args
    eng = make_engine(name, ctx, d, opt, lambda *a, **k: None)
    out = [eng.responses(E, levels) for E in Ez_rows]
    return out, dict(eng.stats)


class PrecomputedEngine:
    """Serves responses computed in parallel (chunks of consecutive cases per worker so that the
    nonlinear warm start still applies within a chunk)."""
    def __init__(self, name, ctx, d, opt, Ez_rows, levels, workers, log):
        from concurrent.futures import ProcessPoolExecutor
        self.name = name; self.ctx = ctx
        chunks = np.array_split(np.arange(len(Ez_rows)), workers)
        jobs = [(name, ctx, d, opt, [Ez_rows[i] for i in c], levels) for c in chunks if c.size]
        t0 = time.time()
        self.cache = {}; self.stats = {"solves": 0, "iterations": 0, "time_s": 0.0, "not_converged": 0, "workers": workers}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for c, (res, st) in zip([c for c in chunks if c.size], ex.map(_worker, jobs)):
                for i, r in zip(c, res):
                    self.cache[int(i)] = r
                for k, v in st.items():
                    self.stats[k] = self.stats.get(k, 0) + v
        self.stats["wall_s"] = time.time() - t0
        self._n = 0
        log(" --> %d cases precomputed with %d workers in %.0f s" % (len(Ez_rows), workers, self.stats["wall_s"]))

    def responses(self, E_sub, levels=(1.0,)):
        r = self.cache[self._n]; self._n += 1
        return r


def run(d, engine=None, years=None, out_dir="runs", workers=None, load_scaling=None, quiet_step1=True,
        log=print, tag=None):
    from .distress import run_miners
    if years:
        d = copy.deepcopy(d)
        d["analysis"]["design_life_yr"] = int(years)
    opt = schema.engine_options(d)
    if engine:
        opt["engine"] = engine
    if load_scaling:
        opt["load_scaling"] = load_scaling
    if workers:
        opt["workers"] = int(workers)
    d["analysis"]["response_engine"] = opt
    name = schema.engine_choice(d)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(d.get("_source_path", "model.json")))[0].replace(".meapa2", "")
    tag = tag or name
    log("=== MEAPA run: %s  [engine %s, load scaling %s, %d yr]" % (d.get("name", base), name, opt["load_scaling"], d["analysis"]["design_life_yr"]))
    S1, inps, LayNo_sub = step1(d, out_dir, quiet_step1, log)
    ctx = context_from_step1(S1, inps, LayNo_sub)
    levels = tuple(opt["power_levels"]) if opt["load_scaling"] == "power" else (1.0,)
    nw = int(opt.get("workers") or 1)
    if nw > 1 and name == "nartfem":
        Esub = S1["Esubdum_M"]
        rows = [Esub[iMo, :, jQ + 3] for iMo in range(S1["numMo"]) for jQ in range(S1["numQuint"])]
        eng = PrecomputedEngine(name, ctx, d, opt, rows, levels, nw, log)
    else:
        eng = make_engine(name, ctx, d, opt, log)
    record = []
    out = run_miners(S1, eng, inps, opt, log=log, record=record)
    # ---- reliability (legacy)
    from f_calc_distress_atrel import f_calc_distress_atrel
    ld = {"FCbu": out["FCbu_mo"], "FCtd": out["FCtd_mo"], "rut_TAC": out["rut_tot_AC_mo"], "rut_TBASE": out["rut_tot_BASE_mo"],
          "rut_TSUBG": out["rut_tot_SUBG_mo"], "rutTOT": out["rutTOT_mo"], "IRI": out["IRI_mo"],
          "TC_mo": np.asarray(S1["TC_mo"]).tolist(), "CSMExists": False, "EACExists": False}
    with contextlib.redirect_stdout(io.StringIO()):
        ld = f_calc_distress_atrel(ld, inps["stdevData"], inps["reliabilityThresholds"], plt_flag=0)
    out["TC_mo"] = ld["TC_mo"]
    for k in ("totrutAtRel", "acrutAtRel", "fcbuAtRel", "fctdtAtRel", "tc_ACAtRel", "iriAtRel"):
        out[k] = ld[k]
    out["name"] = d.get("name", ""); out["input"] = d.get("_source_path"); out["design_life_yr"] = d["analysis"]["design_life_yr"]
    out["layer_names"] = [L.get("name") for L in d["structure"]["layers"]]
    out["sublayer_thickness_in"] = np.asarray(S1["TH_sub"]).tolist()
    out["analysis_depths_in"] = np.asarray(S1["ZC_in_subM"][0]).tolist()
    fout = os.path.join(out_dir, "%s-%s-output.json" % (base, tag))
    with open(fout, "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(out_dir, "%s-%s-cases.json" % (base, tag)), "w") as f:
        json.dump(record, f)
    log(" --> saved %s  (loop %.0f s, engine stats %s)" % (fout, out["elapsed_s"], out["engine_stats"]))
    return out, record, S1


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default=CONFIG["INPUT"])
    p.add_argument("--engine", default=CONFIG["ENGINE"]); p.add_argument("--years", type=int, default=CONFIG["YEARS"])
    p.add_argument("--out", default=CONFIG["OUT_DIR"]); p.add_argument("--workers", type=int, default=CONFIG["WORKERS"])
    p.add_argument("--load-scaling", default=CONFIG["LOAD_SCALING"]); p.add_argument("--tag", default=None)
    a = p.parse_args(argv)
    d = load_any(a.input)
    run(d, a.engine, a.years, a.out, a.workers, a.load_scaling, CONFIG["QUIET_STEP1"], tag=a.tag)


if __name__ == "__main__":
    main()
