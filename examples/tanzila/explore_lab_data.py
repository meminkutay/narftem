"""
explore_lab_data.py
-------------------
Step 1 of the geogrid/geosynthetic-reinforced base calibration for MEAPA.

Reads the processed MTS outputs for the control (no geosynthetic) and the
HSR100-100 reinforced 6-in x 12-in base specimens, checks units and
geometry, plots the raw data and writes tidy CSV tables to OUT_DIR. These
tables are what fit_mr.py and fit_rutting.py read.

Nothing is fitted here, and no input file is modified.

Run from PyCharm: edit the CONFIG block if needed, then press Run.
"""
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = Path(__file__).resolve().parent          # the 'tanzila' folder
OUT_DIR = DATA_ROOT / "out"

RAW = "Raw Data Files/MR+PD"
# One entry per specimen that has raw MTS files (MR test followed by the PD
# test on the same specimen). 'label' is used in every table and figure.
SPECIMENS = [
    dict(group="Control", label="C-T1",
         mr=f"{RAW}/Without Geogrid/6 inch/MR_1/MR - (Timed)-Output.xlsx",
         pd=f"{RAW}/Without Geogrid/6 inch/MR_1/PD/PD - (Timed)-PD-Output.xlsx"),
    dict(group="Control", label="C-T2",
         mr=f"{RAW}/Without Geogrid/6 inch/MR_2/MR - (Timed)-Output.xlsx",
         pd=f"{RAW}/Without Geogrid/6 inch/MR_2/PD/PD - (Timed)-PD-Output.xlsx"),
    dict(group="HSR100-100", label="R-T1",
         mr=f"{RAW}/With Geogrid/6inch/3. HSR100-100/Trial 1/MR/MR - (Timed)-Output.xlsx",
         pd=f"{RAW}/With Geogrid/6inch/3. HSR100-100/Trial 1/PD/PD - (Timed)-PD-Output.xlsx"),
    dict(group="HSR100-100", label="R-T2",
         mr=f"{RAW}/With Geogrid/6inch/3. HSR100-100/Trial 2/MR/MR - (Timed)-Output.xlsx",
         pd=f"{RAW}/With Geogrid/6inch/3. HSR100-100/Trial 2/PD/PD - (Timed)-PD-Output.xlsx"),
    dict(group="HSR100-100", label="R-T3",
         mr=f"{RAW}/With Geogrid/6inch/3. HSR100-100/Trial 3/MR/MR - (Timed)-Output.xlsx",
         pd=f"{RAW}/With Geogrid/6inch/3. HSR100-100/Trial 3/PD/PD - (Timed)-PD-Output.xlsx"),
]

# Extra control MR sheets that exist only in the spreadsheet (no raw MTS files
# in this folder). Shown for context only; set to None to skip.
EXTRA_CONTROL_WORKBOOK = f"{RAW}/Without Geogrid/6 inch/Analysis_6 inch.xlsx"
EXTRA_CONTROL_SHEETS = ["Control (T3)", "Control (T3(2))", "Control (T3(3))",
                        "Control (T4)", "Control (T6)", "Control (T7)"]

# Appendix B workbooks, used only to cross-check the values read from the raw outputs.
APPENDIX_CONTROL = "Analyzed Files/Appendix B (MR)/0. Analysis_6 inch_Control.xlsx"
APPENDIX_HSR = "Analyzed Files/Appendix B (MR)/3. Analysis_6 inch_HSR100-100.xlsx"
APPENDIX_MAP = {"C-T1": (APPENDIX_CONTROL, "Control (T1)_new"),
                "C-T2": (APPENDIX_CONTROL, "Control (T2)_new"),
                "R-T1": (APPENDIX_HSR, "HSR100-100 (T1)_new"),
                "R-T2": (APPENDIX_HSR, "HSR100-100 (T2)_new"),
                "R-T3": (APPENDIX_HSR, "HSR100-100 (T3)_new")}

PA_PSI = 14.706              # MEAPA atmospheric pressure
KPA_PER_PSI = 6.894757
N_LAST_CYCLES = 5            # T307: Mr = mean of the last 5 cycles of a sequence
PD_DOWNSAMPLE = 200          # points per specimen (log-spaced) in pd_tidy.csv
SPEC_D_IN, SPEC_H_IN = 6.0, 12.0   # confirmed below from the MTS stress/strain ratios

# Raw PD time histories (~100 MB each). True = re-derive εp and εr per pulse from
# the raw CSV (cached in OUT_DIR after the first run); False = use the MTS outputs only.
REPROCESS_RAW_PD = True
RAW_PD_PULSE_THRESHOLD_KN = 1.0    # a pulse = axial load above this
RAW_PD_PHASE_SPLIT_KN = 2.8        # conditioning pulses ~1.9 kN, PD pulses ~3.5 kN
FIG_DPI = 200
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Grayscale-friendly styling: groups differ by fill, specimens by marker.
GROUP_STYLE = {"Control": dict(color="0.45", mfc="white", ls="--"),
               "HSR100-100": dict(color="black", mfc="black", ls="-")}
MARKERS = {"C-T1": "o", "C-T2": "s", "R-T1": "^", "R-T2": "D", "R-T3": "v"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "0.88",
                     "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight"})


# ----------------------------------------------------------------------------- readers
def _sheet_df(path, sheet, ncols):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    hdr = [str(h).strip() if h is not None else f"c{i}" for i, h in enumerate(rows[0][:ncols])]
    body = [r[:ncols] for r in rows[1:] if r[0] is not None]
    return pd.DataFrame(body, columns=hdr)


def read_mr(spec):
    """Sequence-level MR results (US and SI sheets) + per-cycle QC from Cycle_Info."""
    path = DATA_ROOT / spec["mr"]
    us = _sheet_df(path, "Seq_Info_US", 7)
    si = _sheet_df(path, "Seq_Info_SI", 7)
    cyc = _sheet_df(path, "Cycle_Info", 17)

    # --- unit cross-check between the US and SI sheets
    r_sd = (si["Deviator Stress (kPa)"] / us["Deviator Stress (psi)"]).astype(float)
    r_mr = (si["MR (MPa)"] * 1000 / us["MR (psi)"]).astype(float)       # kPa/psi
    assert np.allclose(r_sd, KPA_PER_PSI, rtol=2e-3), "deviator stress units inconsistent"
    assert np.allclose(r_mr, KPA_PER_PSI, rtol=2e-3), "Mr units inconsistent"

    # --- specimen geometry implied by the software's own stress/strain
    c = cyc.astype(float)
    H_mm = np.nanmedian(c["DispRange"] / c["Strain"])
    A_m2 = np.nanmedian(c["ForceRange"] / c["Stress_kPa"])
    D_mm = np.sqrt(4 * A_m2 / np.pi) * 1000

    # --- per-sequence scatter of the last N cycles
    last = (c.sort_values(["SequenceNum", "ActualCycle"])
             .groupby("SequenceNum").tail(N_LAST_CYCLES)
             .groupby("SequenceNum")["MR_psi"].agg(["mean", "std"]))

    df = pd.DataFrame({
        "group": spec["group"], "specimen": spec["label"],
        "seq": us["Sequence Number"].astype(int),
        "n_cycles": us["Num Cycles"].astype(int),
        "sigma3_psi": us["Confine Pressure (psi)"].astype(float),
        "sigmad_psi": us["Deviator Stress (psi)"].astype(float),
        "eps_r": us["Strain"].astype(float),
        "Mr_psi": us["MR (psi)"].astype(float),
    })
    df["Mr_last5_cv_pct"] = (100 * last["std"] / last["mean"]).reindex(df["seq"]).values
    df["theta_psi"] = df["sigmad_psi"] + 3 * df["sigma3_psi"]
    df["tau_oct_psi"] = np.sqrt(2) / 3 * df["sigmad_psi"]
    df["conditioning"] = df["seq"] == 0
    geom = dict(specimen=spec["label"], H_in=H_mm / 25.4, D_in=D_mm / 25.4)
    return df, geom


def read_pd(spec):
    """Repeated-load permanent deformation: plastic strain and per-cycle resilient response."""
    path = DATA_ROOT / spec["pd"]
    seq = _sheet_df(path, "Seq_Info_US", 7)
    ps = _sheet_df(path, "Plastic_Strain", 2).astype(float)
    cyc = _sheet_df(path, "Cycle_Info", 18).astype(float)
    main = cyc[cyc["SequenceNum"] == 1].copy()          # seq 0 = 500-cycle conditioning
    main["N"] = main["ActualCycle"]
    out = pd.DataFrame({"N": ps["Cycle_Num"], "eps_p": ps["PlasticStrn"]})
    out = out.merge(main[["N", "Stress_kPa", "Strain", "MR_psi"]], on="N", how="left")
    out = out.rename(columns={"Strain": "eps_r", "MR_psi": "Mr_psi"})
    out["sigmad_psi"] = out["Stress_kPa"] / KPA_PER_PSI
    out = out.drop(columns="Stress_kPa")
    out.insert(0, "specimen", spec["label"])
    out.insert(0, "group", spec["group"])
    conf = float(seq.loc[seq["Sequence Number"] == 1, "Confine Pressure (psi)"].iloc[0])
    cond = seq.loc[seq["Sequence Number"] == 0].iloc[0]
    out["sigma3_psi"] = conf
    info = dict(specimen=spec["label"], sigma3_psi=conf,
                cond_sigma3_psi=float(cond["Confine Pressure (psi)"]),
                cond_sigmad_psi=float(cond["Deviator Stress (psi)"]),
                cond_cycles=int(cond["Num Cycles"]))
    return out, info


def reprocess_raw_pd(spec):
    """Re-derive per-pulse permanent and resilient strain from the raw MTS time history.

    Why: the MTS post-processor mis-segmented R-T2 (its Cycle_Info shows a 90-cycle
    conditioning and a 410-cycle PD sequence at 15 psi, while the raw file holds the
    same 500 + 10 000 pulses as every other specimen). Re-deriving all five specimens
    with one transparent method keeps them comparable.

    Method (sign: MTS force and displacement are negative in compression):
      * a pulse is a contiguous run of axial load above RAW_PD_PULSE_THRESHOLD_KN;
      * pulses whose peak exceeds RAW_PD_PHASE_SPLIT_KN belong to the PD sequence;
      * peak deformation = most compressive displacement within the pulse (+ a few samples);
      * rest deformation = median displacement over the last 40 % of the rest period;
      * εr = (rest - peak)/H ; εp(N) = rest(N) - rest before the first PD pulse, /H ;
      * cyclic σd = (peak load - rest load)/A.
    """
    cache = OUT_DIR / f"pd_raw_cycles_{spec['label']}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    raw = DATA_ROOT / spec["pd"].replace("PD - (Timed)-PD-Output.xlsx", "PD - (Timed).csv")
    d = pd.read_csv(raw, skiprows=[0, 1, 2, 3, 4, 5, 7], encoding="utf-8-sig",
                    usecols=[1, 2, 3])
    d.columns = ["t", "x", "F"]
    x = -d["x"].to_numpy(float)          # mm, compression positive
    F = -d["F"].to_numpy(float)          # kN, compression positive
    on = F > RAW_PD_PULSE_THRESHOLD_KN
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts, ends = edges[on[edges + 1]] + 1, edges[~on[edges + 1]] + 1
    if ends[0] < starts[0]:
        ends = ends[1:]
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]
    peakF = np.array([F[s:e].max() for s, e in zip(starts, ends)])
    pdmask = peakF > RAW_PD_PHASE_SPLIT_KN
    s_pd, e_pd = starts[pdmask], ends[pdmask]
    nxt = np.append(s_pd[1:], min(len(x), e_pd[-1] + (e_pd[-1] - s_pd[-1]) * 8))
    H, A = SPEC_H_IN * 25.4, np.pi / 4 * (SPEC_D_IN * 0.0254) ** 2
    pre = slice(max(0, s_pd[0] - 60), s_pd[0] - 5)
    x_ref = np.median(x[pre])
    rows = []
    for i, (s, e, nx) in enumerate(zip(s_pd, e_pd, nxt)):
        w0 = e + int(0.6 * (nx - e))
        x_rest = np.median(x[w0:nx]) if nx > w0 else np.nan
        F_rest = np.median(F[w0:nx]) if nx > w0 else np.nan
        x_peak = x[s:e + 3].max()
        rows.append((i + 1, (x_rest - x_ref) / H, (x_peak - x_rest) / H,
                     (F[s:e].max() - F_rest) / A / KPA_PER_PSI, F_rest / A / KPA_PER_PSI))
    out = pd.DataFrame(rows, columns=["N", "eps_p_raw", "eps_r_raw", "sigmad_raw_psi", "contact_psi"])
    out["n_cond_pulses"] = int((~pdmask).sum())
    out.to_csv(cache, index=False)
    return out


def read_extra_controls():
    if not EXTRA_CONTROL_WORKBOOK:
        return pd.DataFrame()
    wb = openpyxl.load_workbook(DATA_ROOT / EXTRA_CONTROL_WORKBOOK, read_only=True, data_only=True)
    frames = []
    for sh in EXTRA_CONTROL_SHEETS:
        rows = list(wb[sh].iter_rows(min_row=2, max_row=16, max_col=12, values_only=True))
        d = pd.DataFrame({"sigmad_psi": [r[2] for r in rows], "sigma3_psi": [r[6] for r in rows],
                          "Mr_ksi": [r[11] for r in rows]}).apply(pd.to_numeric, errors="coerce")
        d["sigmad_psi"] /= KPA_PER_PSI
        d["sigma3_psi"] /= KPA_PER_PSI
        d["theta_psi"] = d["sigmad_psi"] + 3 * d["sigma3_psi"]
        d["Mr_psi"] = d["Mr_ksi"] * 1000
        d["sheet"] = sh
        frames.append(d.dropna(subset=["Mr_psi"]))
    wb.close()
    return pd.concat(frames, ignore_index=True)


def appendix_crosscheck(mr):
    rows = []
    for lab, (f, sh) in APPENDIX_MAP.items():
        wb = openpyxl.load_workbook(DATA_ROOT / f, read_only=True, data_only=True)
        allrows = list(wb[sh].iter_rows(min_row=1, max_row=30, max_col=9, values_only=True))
        wb.close()
        h = next(i for i, r in enumerate(allrows) if r[8] == "Mr (ksi)")   # header row varies by sheet
        vals = [r[8] for r in allrows[h + 1:h + 16]]
        mine = mr[(mr.specimen == lab) & ~mr.conditioning].sort_values("seq")["Mr_psi"].values / 1000
        app = np.array(vals, float)
        rows.append(dict(specimen=lab, appendix_sheet=sh,
                         max_abs_diff_ksi=float(np.nanmax(np.abs(mine - app)))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plots
def _style(lab, grp):
    s = GROUP_STYLE[grp]
    return dict(marker=MARKERS[lab], color=s["color"], mfc=s["mfc"], mec=s["color"], ms=5)


def plot_mr_vs_theta(mr, extra, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if len(extra):
        ax.plot(extra.theta_psi, extra.Mr_psi / 1000, "x", color="0.75", ms=4,
                label="Other control sheets (no raw files)")
    for lab, d in mr[~mr.conditioning].groupby("specimen", sort=False):
        ax.plot(d.theta_psi, d.Mr_psi / 1000, ls="none", label=f"{lab} ({d.group.iloc[0]})",
                **_style(lab, d.group.iloc[0]))
    ax.set_xlabel("Bulk stress θ (psi)")
    ax.set_ylabel("Resilient modulus Mr (ksi)")
    ax.legend(fontsize=7.5, ncol=2)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_mr_vs_sigmad(mr, path):
    s3_levels = sorted(mr.loc[~mr.conditioning, "sigma3_psi"].round().unique())
    fig, axes = plt.subplots(1, len(s3_levels), figsize=(11, 3.2), sharey=True)
    for ax, s3 in zip(axes, s3_levels):
        sub = mr[(~mr.conditioning) & (mr.sigma3_psi.round() == s3)]
        for lab, d in sub.groupby("specimen", sort=False):
            st = _style(lab, d.group.iloc[0])
            ax.plot(d.sigmad_psi, d.Mr_psi / 1000, ls=GROUP_STYLE[d.group.iloc[0]]["ls"], lw=1,
                    label=lab, **st)
        ax.set_title(f"σ3 = {s3:.0f} psi")
        ax.set_xlabel("σd (psi)")
    axes[0].set_ylabel("Mr (ksi)")
    axes[-1].legend(fontsize=7)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_mr_ratio(mr, path):
    m = mr[~mr.conditioning]
    piv = m.pivot_table(index="seq", columns="specimen", values="Mr_psi")
    th = m.groupby("seq")["theta_psi"].mean()
    c = piv[[s for s in piv if s.startswith("C")]]
    r = piv[[s for s in piv if s.startswith("R")]]
    ratio_mean = r.mean(axis=1) / c.mean(axis=1)
    # all reinforced/control pairings -> range
    pairs = np.array([r[a] / c[b] for a in r for b in c]).T
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.vlines(th, pairs.min(axis=1), pairs.max(axis=1), color="0.6", lw=3, alpha=0.6,
              label="Range over all R/C specimen pairings")
    ax.plot(th, ratio_mean, "ko", ms=5, label="Mean(R) / mean(C)")
    ax.axhline(1, color="0.3", lw=0.8)
    ax.set_xlabel("Bulk stress θ (psi)")
    ax.set_ylabel("Mr ratio, reinforced / control")
    ax.legend(fontsize=7.5)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return pd.DataFrame({"theta_psi": th, "ratio_mean": ratio_mean,
                         "ratio_min": pairs.min(axis=1), "ratio_max": pairs.max(axis=1)})


def plot_pd(pd_all, path):
    col = "eps_p_raw" if "eps_p_raw" in pd_all else "eps_p"
    pd_all = pd_all.assign(eps_p=pd_all[col])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Permanent strain " + ("re-derived from raw displacement" if col == "eps_p_raw"
                                        else "from MTS PlasticStrn"), fontsize=9)
    for lab, d in pd_all.groupby("specimen", sort=False):
        g = d.group.iloc[0]
        st = _style(lab, g)
        idx = np.unique(np.geomspace(1, len(d), 25).astype(int)) - 1
        for ax in axes:
            ax.plot(d.N, d.eps_p * 100, color=st["color"], ls=GROUP_STYLE[g]["ls"], lw=0.9)
            ax.plot(d.N.iloc[idx], d.eps_p.iloc[idx] * 100, ls="none", label=lab, **st)
    # group means on a common N grid
    for g, d in pd_all.groupby("group"):
        mean = d.pivot_table(index="N", columns="specimen", values="eps_p").mean(axis=1)
        axes[0].plot(mean.index, mean * 100, color=GROUP_STYLE[g]["color"], lw=2.4,
                     alpha=0.5, label=f"{g} mean")
    axes[0].set_xlabel("Load cycles N")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Load cycles N (log)")
    for ax in axes:
        ax.set_ylabel("Permanent axial strain εp (%)")
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_pd_resilient(pd_all, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for lab, d in pd_all.groupby("specimen", sort=False):
        g = d.group.iloc[0]
        st = _style(lab, g)
        if "eps_r_raw" in d:      # raw re-derivation is complete for every specimen
            dd = d.dropna(subset=["eps_r_raw"]).assign(
                eps_r=lambda v: v.eps_r_raw, sigmad_psi=lambda v: v.sigmad_raw_psi,
                Mr_psi=lambda v: v.sigmad_raw_psi / v.eps_r_raw)
        else:
            dd = d.dropna(subset=["eps_r"])
        roll = dd.set_index("N")[["eps_r", "Mr_psi", "sigmad_psi"]].rolling(100, center=True).median()
        for ax, col, sc in zip(axes, ["sigmad_psi", "eps_r", "Mr_psi"], [1, 1e6, 1e-3]):
            ax.plot(roll.index, roll[col] * sc, color=st["color"], ls=GROUP_STYLE[g]["ls"], lw=1.1,
                    label=lab, marker=st["marker"], mfc=st["mfc"], ms=4, markevery=1500)
    for ax, yl in zip(axes, ["Cyclic deviator σd (psi)", "Resilient strain εr (µε)", "Mr in PD test (ksi)"]):
        ax.set_xscale("log"); ax.set_xlabel("N"); ax.set_ylabel(yl)
    axes[2].legend(fontsize=7)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_pd_raw_vs_mts(pd_all, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for lab, d in pd_all.groupby("specimen", sort=False):
        g = d.group.iloc[0]
        st = _style(lab, g)
        idx = np.unique(np.geomspace(1, len(d), 25).astype(int)) - 1
        axes[0].plot(d.N, d.eps_p_raw * 100, color=st["color"], ls=GROUP_STYLE[g]["ls"], lw=0.9)
        axes[0].plot(d.N.iloc[idx], d.eps_p_raw.iloc[idx] * 100, ls="none", label=lab, **st)
        axes[1].plot(d.N, d.eps_p * 100, color=st["color"], ls=GROUP_STYLE[g]["ls"], lw=0.9, label=lab)
    axes[0].set_title("Re-derived from raw actuator displacement", fontsize=9)
    axes[1].set_title("MTS post-processor 'PlasticStrn'", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Load cycles N"); ax.set_ylabel("Permanent axial strain εp (%)")
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(exist_ok=True)
    mr_frames, geoms, pd_frames, pd_info = [], [], [], []
    for spec in SPECIMENS:
        df, geom = read_mr(spec)
        mr_frames.append(df); geoms.append(geom)
        p, info = read_pd(spec)
        pd_frames.append(p); pd_info.append(info)
    mr = pd.concat(mr_frames, ignore_index=True)
    pd_all = pd.concat(pd_frames, ignore_index=True)
    if REPROCESS_RAW_PD:
        raw = []
        for spec in SPECIMENS:
            r = reprocess_raw_pd(spec)
            r.insert(0, "specimen", spec["label"])
            raw.append(r)
        raw = pd.concat(raw, ignore_index=True)
        pd_all = pd_all.merge(raw, on=["specimen", "N"], how="left")
    extra = read_extra_controls()

    print("\n=== UNITS / CONVENTIONS ===")
    print("Stresses in psi, compression positive. Strains dimensionless (in/in), "
          "US and SI sheets agree within 0.2 %.")
    print("θ = σd + 3σ3 ; τoct = √2/3·σd (triaxial, σ2 = σ3) ; Pa =", PA_PSI, "psi")
    print("\n=== SPECIMEN GEOMETRY implied by MTS stress/strain (should be 6 x 12 in) ===")
    print(pd.DataFrame(geoms).round(2).to_string(index=False))

    print("\n=== CROSS-CHECK against Appendix B workbook (Mr, ksi) ===")
    print(appendix_crosscheck(mr).round(3).to_string(index=False))

    m = mr[~mr.conditioning]
    print("\n=== MR TEST: stress ranges and modulus by specimen (sequences 1-15) ===")
    summ = m.groupby(["group", "specimen"], sort=False).agg(
        n_seq=("seq", "size"),
        s3_min=("sigma3_psi", "min"), s3_max=("sigma3_psi", "max"),
        sd_min=("sigmad_psi", "min"), sd_max=("sigmad_psi", "max"),
        theta_min=("theta_psi", "min"), theta_max=("theta_psi", "max"),
        Mr_min_ksi=("Mr_psi", lambda x: x.min() / 1000), Mr_max_ksi=("Mr_psi", lambda x: x.max() / 1000),
        Mr_mean_ksi=("Mr_psi", lambda x: x.mean() / 1000),
        cv_last5_max_pct=("Mr_last5_cv_pct", "max"))
    print(summ.round(2).to_string())

    ratio = plot_mr_ratio(mr, OUT_DIR / "fig_mr_ratio_vs_theta.png")
    print("\n=== Mr ratio reinforced/control by sequence ===")
    print(ratio.round(3).to_string())
    cm = m[m.group == "Control"].groupby("seq")["Mr_psi"]
    rm = m[m.group != "Control"].groupby("seq")["Mr_psi"]
    print(f"Replicate CV of Mr (mean over sequences): control {np.mean(cm.std() / cm.mean()) * 100:.1f} %, "
          f"reinforced {np.mean(rm.std() / rm.mean()) * 100:.1f} %")
    print(f"Overall Mr gain (mean of per-sequence group-mean ratios): {(ratio.ratio_mean.mean() - 1) * 100:.1f} %")

    print("\n=== PD TEST ===")
    rows = []
    for (g, lab), d in pd_all.groupby(["group", "specimen"], sort=False):
        dd = d.dropna(subset=["eps_r"])
        ep = d.set_index("N")["eps_p"]
        rows.append(dict(group=g, specimen=lab, sigma3_psi=d.sigma3_psi.iloc[0],
                         sigmad_mean_psi=dd.sigmad_psi.mean(), sigmad_cv_pct=100 * dd.sigmad_psi.std() / dd.sigmad_psi.mean(),
                         N_max=int(d.N.max()), eps_r_mean_ue=dd.eps_r.mean() * 1e6,
                         Mr_PD_ksi=dd.Mr_psi.median() / 1000,
                         ep_100_pct=ep.loc[100] * 100, ep_1000_pct=ep.loc[1000] * 100,
                         ep_10000_pct=ep.iloc[-1] * 100))
    pdsum = pd.DataFrame(rows)
    if REPROCESS_RAW_PD:
        rr = pd_all.groupby("specimen", sort=False).agg(
            n_cond_pulses=("n_cond_pulses", "first"),
            sigmad_raw_psi=("sigmad_raw_psi", "median"), contact_psi=("contact_psi", "median"),
            eps_r_raw_ue=("eps_r_raw", lambda v: v.median() * 1e6))
        for n in (100, 1000, 10000):
            rr[f"ep_raw_{n}_pct"] = pd_all[pd_all.N == n].set_index("specimen")["eps_p_raw"] * 100
        pdsum = pdsum.merge(rr.reset_index(), on="specimen")
    print(pdsum.round(3).to_string(index=False))
    print("Conditioning before PD:", pd.DataFrame(pd_info).to_string(index=False))
    for g, d in pdsum.groupby("group"):
        print(f"  {g}: εp(10k) mean {d.ep_10000_pct.mean():.2f} %  sd {d.ep_10000_pct.std():.2f} %  "
              f"CV {100 * d.ep_10000_pct.std() / d.ep_10000_pct.mean():.1f} %")
    cg, rg = pdsum[pdsum.group == "Control"], pdsum[pdsum.group != "Control"]
    cols = ["ep_100_pct", "ep_1000_pct", "ep_10000_pct"]
    if REPROCESS_RAW_PD:
        cols += ["ep_raw_100_pct", "ep_raw_1000_pct", "ep_raw_10000_pct"]
        print("  Raw re-derivation: εp(10k) MTS / raw =",
              (pdsum.ep_10000_pct / pdsum.ep_raw_10000_pct).round(1).tolist(),
              " (εr MTS vs raw agree:", (pdsum.eps_r_mean_ue / pdsum.eps_r_raw_ue).round(2).tolist(), ")")
    for col in cols:
        print(f"  ratio R/C {col}: {rg[col].mean() / cg[col].mean():.3f}")

    if len(extra):
        # sensitivity: Mr gain if the extra control sheets are pooled with C-T1/C-T2 (same 15 sequences)
        ex_seq = extra.assign(seq=extra.groupby("sheet").cumcount() + 1)
        pooled = pd.concat([m[m.group == "Control"][["seq", "Mr_psi"]], ex_seq[["seq", "Mr_psi"]]])
        g_pool = rm.mean() / pooled.groupby("seq")["Mr_psi"].mean()
        print(f"Sensitivity - Mr gain vs pooled 8 control sheets: mean {(g_pool.mean() - 1) * 100:.1f} %, "
              f"at seq 15 {(g_pool.loc[15] - 1) * 100:.1f} %, seq 7-15 mean {(g_pool.loc[7:].mean() - 1) * 100:.1f} %")
        print("\n=== Other control sheets in", EXTRA_CONTROL_WORKBOOK, "(no raw files here) ===")
        ex = extra.groupby("sheet").agg(n=("Mr_psi", "size"), Mr_at_max_theta_ksi=("Mr_ksi", "last"),
                                        theta_max=("theta_psi", "max"))
        print(ex.round(2).to_string())

    # --- figures
    plot_mr_vs_theta(mr, extra, OUT_DIR / "fig_mr_vs_theta.png")
    plot_mr_vs_sigmad(mr, OUT_DIR / "fig_mr_vs_sigmad_by_sigma3.png")
    plot_pd(pd_all, OUT_DIR / "fig_pd_eps_p_vs_N.png")
    plot_pd_resilient(pd_all, OUT_DIR / "fig_pd_resilient_vs_N.png")
    if REPROCESS_RAW_PD:
        plot_pd_raw_vs_mts(pd_all, OUT_DIR / "fig_pd_raw_vs_mts.png")

    # --- tidy tables for the fitting scripts
    mr.to_csv(OUT_DIR / "mr_tidy.csv", index=False)
    keep = []
    for lab, d in pd_all.groupby("specimen", sort=False):
        idx = np.unique(np.geomspace(1, len(d), PD_DOWNSAMPLE).astype(int)) - 1
        keep.append(d.iloc[idx])
    pd.concat(keep).to_csv(OUT_DIR / "pd_tidy.csv", index=False)
    pd_all.to_csv(OUT_DIR / "pd_full.csv", index=False)
    pdsum.to_csv(OUT_DIR / "pd_summary.csv", index=False)
    summ.to_csv(OUT_DIR / "mr_summary.csv")
    print("\nWrote figures and tables to", OUT_DIR)


if __name__ == "__main__":
    main()
