"""
fit_rutting.py
--------------
Step 3: permanent-deformation calibration and translation into MEAPA's base-rutting input.

  1. MEPDG time model  εp(N) = ε0'·exp(-(ρ/N)^β)  per specimen and per group (group-mean
     curve), with parameter correlations (identifiability) and specimen-level scatter.
  2. Reinforced/control permanent-strain ratio vs N (one stress state was tested:
     σ3 = 10 psi, cyclic σd ≈ 27.8 psi), with all specimen pairings and a specimen bootstrap.
  3. Translation into MEAPA. MEAPA's base model (f_njit_functions.ff_calc_eps_BASE_with_N):
        εp = briB · Re · exp(-(ρ/N)^β) · εv
     with Re, ρ, β from Wc, Wc from CBR = (E/2555)^(1/0.64) and the GWT (a1 = 0.15, a9 = 20);
     there is no separate ε0/εr input, so briB is the only lever. With the equivalent
     reinforced base, MEAPA already lowers εv (stiffer base) and changes Re, ρ, β (higher E).
     briB_R is chosen so that MEAPA reproduces the lab ratio at the lab stress state:
        briB_R = briB_C · R_lab(N) / R_MEAPA(N),
        R_MEAPA = [Re·exp(-(ρ/N)^β)·εv]_R / [ ... ]_C   with εv ∝ 1/Mr from the k's.

Reads OUT_DIR/pd_full.csv (explore_lab_data.py) and OUT_DIR/mr_fit_results.json (fit_mr.py).
"""
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = Path(__file__).resolve().parent
OUT_DIR = DATA_ROOT / "out"
PD_FULL = OUT_DIR / "pd_full.csv"
MR_RESULTS = OUT_DIR / "mr_fit_results.json"

CONTROL, REINF = "Control", "HSR100-100"
# Permanent strain used for the fits: "mts" = MTS post-processor PlasticStrn (chosen by
# E. Kutay); "raw" = re-derived from actuator displacement. The other one is reported as
# a sensitivity.
EPS_P_SOURCE = "mts"
N_FIT_MIN, N_FIT_MAX, N_FIT_POINTS = 5, 10000, 150     # log-spaced points used in the fits
N_REPORT = [100, 1000, 10000]                          # ratio reported at these N
N_MATCH = 10000                                        # N at which briB is matched (end of data)

# MEAPA base rutting
BRIB_CONTROL = 0.965            # current MEAPA calibration "briB" for the base (example inputs)
GWT_FT = 5.0                    # depth to ground water table (template design)
A1, A9 = 0.15, 20.0             # MEPDG constants as coded in ff_calc_eps_BASE_with_N
FIELD_E_RANGE_KSI = [15, 20, 25, 30, 40]    # sensitivity of briB_R to the base modulus level
FIELD_N = [1e4, 1e5, 1e6, 1e7]               # sensitivity of briB_R to the N used for matching
N_BOOT = 2000
SEED = 2
FIG_DPI = 200
# =============================================================================

import json
import math
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight"})
MARKERS = {"C-T1": "o", "C-T2": "s", "R-T1": "^", "R-T2": "D", "R-T3": "v"}
GSTYLE = {CONTROL: dict(color="0.45", mfc="white", ls="--"), REINF: dict(color="black", mfc="black", ls="-")}
PA_PSI = 14.706


def mr_universal(k, s3, sd):
    sz = s3 + sd
    return k[0] * PA_PSI * ((2 * s3 + sz) / PA_PSI) ** k[1] * (math.sqrt(2) / 3 * abs(sz - s3) / PA_PSI + 1) ** k[2]


# ----------------------------------------------------------------------------- time model
def time_model(p, N):
    e0, rho, beta = math.exp(p[0]), math.exp(p[1]), p[2]
    return e0 * np.exp(-(rho / N) ** beta)


def fit_time_model(N, ep):
    m = (ep > 0) & np.isfinite(ep)
    N, ep = N[m], ep[m]
    f = lambda p: np.log(time_model(p, N)) - np.log(ep)
    best = None
    for b0 in (0.2, 0.4, 0.7):
        for r0 in (1e2, 1e3, 1e4):
            sol = least_squares(f, [math.log(ep.max() * 1.5), math.log(r0), b0],
                                bounds=([math.log(ep.max() * 0.5), math.log(1e-3), 0.02], [math.log(10.0), math.log(1e12), 2.0]))
            if best is None or sol.cost < best.cost:
                best = sol
    J = best.jac
    n, p = len(N), 3
    s2 = 2 * best.cost / (n - p)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov)); corr = cov / np.outer(se, se)
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan); corr = np.full((3, 3), np.nan)
    pred = time_model(best.x, N)
    return dict(eps0=math.exp(best.x[0]), rho=math.exp(best.x[1]), beta=best.x[2], x=best.x,
                rmse_ln=float(np.sqrt(np.mean((np.log(pred) - np.log(ep)) ** 2))),
                R2=float(1 - np.sum((ep - pred) ** 2) / np.sum((ep - ep.mean()) ** 2)),
                se_ln_eps0=float(se[0]), se_ln_rho=float(se[1]), se_beta=float(se[2]),
                corr_lneps0_lnrho=float(corr[0, 1]), corr_lnrho_beta=float(corr[1, 2]),
                eps0_over_epN=float(math.exp(best.x[0]) / ep[-1]))


# ----------------------------------------------------------------------------- MEAPA rutting functions
def meapa_base_terms(E_psi, N, gwt=GWT_FT):
    """Re, ρ, β and Re·exp(-(ρ/N)^β) exactly as in ff_calc_eps_BASE_with_N."""
    cbr = (E_psi / 2555.0) ** (1 / 0.64)
    wc = 51.712 * cbr ** (-0.3586 * gwt ** 0.1192)
    beta = 10 ** (-0.61119 - 0.017638 * wc)
    co = math.log(A1 / A9)
    rho = 1e9 * (co / (1 - 1e9 ** beta)) ** (1 / beta)
    re = (math.exp(rho ** beta) * A1 + math.exp((rho / 1e9) ** beta) * A9) / 2
    return dict(CBR=cbr, Wc=wc, beta=beta, rho=rho, Re=re, timefn=re * math.exp(-((rho / N) ** beta)))


def meapa_ratio(E_C, E_R, N):
    """R_MEAPA = [Re·exp()·εv]_R / [Re·exp()·εv]_C at equal stress (εv ∝ 1/E)."""
    tC, tR = meapa_base_terms(E_C, N), meapa_base_terms(E_R, N)
    return (tR["timefn"] / E_R) / (tC["timefn"] / E_C), tC, tR


# ----------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(PD_FULL)
    col = "eps_p" if EPS_P_SOURCE == "mts" else "eps_p_raw"
    alt = "eps_p_raw" if col == "eps_p" else "eps_p"
    print(f"Permanent strain used: {col} ({EPS_P_SOURCE}); sensitivity: {alt}")
    st = d.groupby("specimen").agg(group=("group", "first"), s3=("sigma3_psi", "first"),
                                   sd=("sigmad_raw_psi", "median"), er=("eps_r_raw", "median"))
    print("PD stress state per specimen (σ3, cyclic σd from the raw re-derivation, εr median):")
    print(st.round(4).to_string())
    print("One stress state only: the ratio cannot be resolved as a function of stress.")

    Ngrid = np.unique(np.concatenate([np.geomspace(N_FIT_MIN, N_FIT_MAX, N_FIT_POINTS).round(),
                                      N_REPORT, [N_MATCH]]).astype(int))
    curves = {}
    for s, g in d.groupby("specimen"):
        gi = g.set_index("N")
        curves[s] = dict(group=g.group.iloc[0], ep=gi[col].reindex(Ngrid).values,
                         ep_alt=gi[alt].reindex(Ngrid).values,
                         er=gi["eps_r_raw"].rolling(101, center=True, min_periods=1).median().reindex(Ngrid).values)

    # ---------------------------------------------------------------- 1. time model fits
    print("\n=== 1. MEPDG time model  εp = ε0'·exp(-(ρ/N)^β)  (fit in ln εp, N = %d-%d) ===" % (N_FIT_MIN, N_FIT_MAX))
    rows = []
    for s, c in curves.items():
        f = fit_time_model(Ngrid.astype(float), c["ep"])
        rows.append(dict(fit=s, group=c["group"], **{k: v for k, v in f.items() if k != "x"}))
    gm = {}
    for grp in (CONTROL, REINF):
        mean = np.mean([c["ep"] for c in curves.values() if c["group"] == grp], axis=0)
        gm[grp] = mean
        f = fit_time_model(Ngrid.astype(float), mean)
        rows.append(dict(fit=f"{grp} (group mean)", group=grp, **{k: v for k, v in f.items() if k != "x"}))
    tm = pd.DataFrame(rows)
    print(tm.round(4).to_string(index=False))
    tm.to_csv(OUT_DIR / "pd_time_model_fits.csv", index=False)
    print("Note: |corr(ln ε0', ln ρ)| near 1 means ε0' and ρ are not separately identified by 10 000 cycles;"
          " the curve inside the data range is well defined, the asymptote ε0' is not.")

    # ---------------------------------------------------------------- 2. ratios
    print("\n=== 2. Reinforced / control permanent-strain ratio (equal stress state) ===")
    Cs = [s for s, c in curves.items() if c["group"] == CONTROL]
    Rs = [s for s, c in curves.items() if c["group"] == REINF]

    def ratio_of(key, specsR, specsC):
        return np.mean([curves[s][key] for s in specsR], axis=0) / np.mean([curves[s][key] for s in specsC], axis=0)

    ratio = ratio_of("ep", Rs, Cs)
    ratio_alt = ratio_of("ep_alt", Rs, Cs)
    pairs = np.array([curves[r]["ep"] / curves[c]["ep"] for r in Rs for c in Cs])
    boot = []
    for _ in range(N_BOOT):
        bR = list(rng.choice(Rs, len(Rs))); bC = list(rng.choice(Cs, len(Cs)))
        boot.append(ratio_of("ep", bR, bC))
    boot = np.array(boot)
    er_ratio = ratio_of("er", Rs, Cs)
    norm_ratio = ratio / er_ratio                       # (εp/εr)_R / (εp/εr)_C
    rr = pd.DataFrame({"N": Ngrid, "ratio": ratio, "ratio_boot_lo": np.percentile(boot, 2.5, axis=0),
                       "ratio_boot_hi": np.percentile(boot, 97.5, axis=0), "ratio_pair_min": pairs.min(0),
                       "ratio_pair_max": pairs.max(0), f"ratio_{alt}": ratio_alt,
                       "er_ratio": er_ratio, "ratio_ep_over_er": norm_ratio}).set_index("N")
    rr.to_csv(OUT_DIR / "pd_ratio_vs_N.csv")
    rep = rr.loc[[n for n in N_REPORT if n in rr.index]]
    print(rep.round(3).to_string())
    fC = tm.set_index("fit").loc[f"{CONTROL} (group mean)"]; fR = tm.set_index("fit").loc[f"{REINF} (group mean)"]
    print(f"Asymptotic ratio from group-mean time models (ε0'_R/ε0'_C, extrapolated): {fR.eps0 / fC.eps0:.3f}"
          f"  <- not usable: ρ runs to its bound for the reinforced curves (no plateau within 10 000 cycles)")
    print(f"Replicate scatter at N = 10 000: control CV {np.std([curves[s]['ep'][-1] for s in Cs], ddof=1) / np.mean([curves[s]['ep'][-1] for s in Cs]) * 100:.1f} %, "
          f"reinforced CV {np.std([curves[s]['ep'][-1] for s in Rs], ddof=1) / np.mean([curves[s]['ep'][-1] for s in Rs]) * 100:.1f} %")

    # ---------------------------------------------------------------- 3. MEAPA translation
    print("\n=== 3. Translation into MEAPA base rutting (briB) ===")
    mrres = json.load(open(MR_RESULTS))
    kC, kR = mrres["control"]["k"], mrres["reinforced_equivalent"]["k"]
    s3, sd = 10.0, float(st.sd.median())
    E_C, E_R = mr_universal(kC, s3, sd), mr_universal(kR, s3, sd)
    print(f"Lab PD state σ3 = {s3} psi, σd = {sd:.1f} psi: Mr control {E_C:.0f} psi, equivalent reinforced {E_R:.0f} psi "
          f"(εv ratio from k's {E_C / E_R:.3f}; measured εr ratio in the PD test {er_ratio[-1]:.3f})")
    R_lab = float(rr.ratio.loc[N_MATCH])
    R_m, tC, tR = meapa_ratio(E_C, E_R, N_MATCH)
    briB_R = BRIB_CONTROL * R_lab / R_m
    print(f"MEAPA terms at N = {N_MATCH}: control  CBR {tC['CBR']:.1f}  Wc {tC['Wc']:.2f} %  β {tC['beta']:.4f}  ρ {tC['rho']:.4g}  Re {tC['Re']:.4g}")
    print(f"                          reinforced CBR {tR['CBR']:.1f}  Wc {tR['Wc']:.2f} %  β {tR['beta']:.4f}  ρ {tR['rho']:.4g}  Re {tR['Re']:.4g}")
    print(f"R_lab({N_MATCH}) = {R_lab:.3f};  R_MEAPA = εv ratio {E_C / E_R:.3f} × Re·exp() ratio {tR['timefn'] / tC['timefn']:.3f} = {R_m:.3f}")
    print(f"=> briB_R = {BRIB_CONTROL} × {R_lab:.3f} / {R_m:.3f} = {briB_R:.3f}   (briB_R / briB_C = {briB_R / BRIB_CONTROL:.3f})")

    # uncertainty: specimen bootstrap of R_lab, and the k bootstrap envelope of the Mr gain
    iN = list(rr.index).index(N_MATCH)
    bri_boot = BRIB_CONTROL * boot[:, iN] / R_m
    lo, hi = np.percentile(bri_boot, [2.5, 97.5])
    pr_lo, pr_hi = BRIB_CONTROL * rr.ratio_pair_min.loc[N_MATCH] / R_m, BRIB_CONTROL * rr.ratio_pair_max.loc[N_MATCH] / R_m
    print(f"   specimen bootstrap 95 %: {lo:.3f}-{hi:.3f};  range over the 6 R/C specimen pairings: {pr_lo:.3f}-{pr_hi:.3f}")

    sens = []
    for n_m in [n for n in N_REPORT if n in rr.index]:
        Rm_n = meapa_ratio(E_C, E_R, n_m)[0]
        sens.append(dict(case=f"match at N = {n_m}", R_lab=rr.ratio.loc[n_m], R_MEAPA=Rm_n, briB_R=BRIB_CONTROL * rr.ratio.loc[n_m] / Rm_n))
    sens.append(dict(case="asymptote ε0'_R/ε0'_C (NOT identified - shown only to make the point)", R_lab=fR.eps0 / fC.eps0, R_MEAPA=meapa_ratio(E_C, E_R, 1e12)[0],
                     briB_R=BRIB_CONTROL * (fR.eps0 / fC.eps0) / meapa_ratio(E_C, E_R, 1e12)[0]))
    sens.append(dict(case=f"{alt} instead of {col}", R_lab=rr[f"ratio_{alt}"].loc[N_MATCH], R_MEAPA=R_m,
                     briB_R=BRIB_CONTROL * rr[f"ratio_{alt}"].loc[N_MATCH] / R_m))
    gain = E_R / E_C
    for e in FIELD_E_RANGE_KSI:
        for nf in FIELD_N:
            Rm_f = meapa_ratio(e * 1e3, e * 1e3 * gain, nf)[0]
            sens.append(dict(case=f"MEAPA ratio at E_C = {e} ksi, N = {nf:.0e} (same gain)", R_lab=R_lab, R_MEAPA=Rm_f,
                             briB_R=BRIB_CONTROL * R_lab / Rm_f))
    sens = pd.DataFrame(sens)
    print("\nSensitivity of briB_R:")
    print(sens.round(3).to_string(index=False))
    sens.to_csv(OUT_DIR / "rutting_briB_sensitivity.csv", index=False)

    # Alternative that brackets the result: geosynthetic only in the rutting model (control k's)
    briB_rut_only = BRIB_CONTROL * R_lab
    print(f"\nBracketing alternative (control k's for both, benefit only through briB): briB_R = {BRIB_CONTROL} × {R_lab:.3f} = {briB_rut_only:.3f}")

    out = dict(eps_p_source=EPS_P_SOURCE, N_match=N_MATCH, lab_state=dict(sigma3_psi=s3, sigmad_psi=sd),
               R_lab=R_lab, R_lab_boot95=[float(np.percentile(boot[:, iN], 2.5)), float(np.percentile(boot[:, iN], 97.5))],
               R_lab_alt_source=float(rr[f"ratio_{alt}"].loc[N_MATCH]),
               ratio_ep_over_er=float(rr.ratio_ep_over_er.loc[N_MATCH]),
               E_C_psi=E_C, E_R_psi=E_R, R_MEAPA=R_m, meapa_terms=dict(control=tC, reinforced=tR),
               briB_control=BRIB_CONTROL, briB_reinforced=briB_R, briB_reinforced_boot95=[lo, hi],
               briB_reinforced_pair_range=[pr_lo, pr_hi], briB_rut_only=briB_rut_only, gwt_ft=GWT_FT,
               time_model=tm.to_dict(orient="records"))
    with open(OUT_DIR / "rutting_results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    Nf = np.geomspace(1, 1e5, 200)
    for s, c in curves.items():
        g = c["group"]
        axes[0].plot(Ngrid, c["ep"] * 100, ls="none", marker=MARKERS[s], ms=3.5, mec=GSTYLE[g]["color"],
                     mfc=GSTYLE[g]["mfc"], markevery=6, label=s)
        p = fit_time_model(Ngrid.astype(float), c["ep"])["x"]
        axes[0].plot(Nf, time_model(p, Nf) * 100, color=GSTYLE[g]["color"], ls=GSTYLE[g]["ls"], lw=0.9)
    axes[0].axvspan(1e4, 1e5, color="0.94", lw=0)
    axes[0].text(1.3e4, axes[0].get_ylim()[1] * 0.05, "extrapolation", fontsize=7, color="0.4")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Load cycles N")
    axes[0].set_ylabel(f"Permanent strain εp (%), {'MTS PlasticStrn' if EPS_P_SOURCE == 'mts' else 'raw'}")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].fill_between(rr.index, rr.ratio_boot_lo, rr.ratio_boot_hi, color="0.88", label="Specimen bootstrap 95 %")
    axes[1].plot(rr.index, rr.ratio_pair_min, color="0.6", lw=0.8, ls=":", label="Min / max over R/C pairings")
    axes[1].plot(rr.index, rr.ratio_pair_max, color="0.6", lw=0.8, ls=":")
    axes[1].plot(rr.index, rr.ratio, "k-", lw=1.8, label=f"Mean R / mean C ({EPS_P_SOURCE})")
    axes[1].plot(rr.index, rr[f"ratio_{alt}"], "k--", lw=1.1, label=f"Mean R / mean C ({'raw' if alt == 'eps_p_raw' else 'mts'})")
    axes[1].plot(rr.index, rr.ratio_ep_over_er, color="0.4", ls="-.", lw=1.1, label="(εp/εr)_R / (εp/εr)_C")
    axes[1].axhline(1, color="0.3", lw=0.8)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Load cycles N"); axes[1].set_ylabel("Ratio, reinforced / control")
    axes[1].set_ylim(0.4, 1.4); axes[1].legend(fontsize=7, loc="upper right")
    fig.savefig(OUT_DIR / "fig_pd_fits_and_ratio.png", dpi=FIG_DPI); plt.close(fig)
    print("\nWrote", OUT_DIR / "rutting_results.json", "and figures.")


if __name__ == "__main__":
    main()
