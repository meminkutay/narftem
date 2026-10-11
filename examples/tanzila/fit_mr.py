"""
fit_mr.py
---------
Step 2: resilient-modulus calibration for the control and HSR100-100 reinforced base.

  1. NCHRP 1-37A universal-model fits (k1, k2, k3) per specimen and per group, with
     R², RMSE, asymptotic 95 % CIs and a bootstrap (points resampled within specimen).
  2. Feasibility of the MEAPA/NCHRP 1-50 geosynthetic confinement model: the same
     equations as nartfem/solid3d/geogrid.py, applied to each lab stress state with the
     control k's, then averaged over the specimen height (series springs: the actuator
     measures the whole 12-in specimen). Gives the largest Mr gain the model can produce
     and a least-squares fit of M and delta to the reinforced data (identifiability).
  3. Equivalent reinforced base: separate k's for the reinforced material, constrained so
     Mr_reinforced >= Mr_control over the tested stress range.

Reads OUT_DIR/mr_tidy.csv written by explore_lab_data.py. Needs scipy.
"""
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = Path(__file__).resolve().parent
OUT_DIR = DATA_ROOT / "out"
MR_TIDY = OUT_DIR / "mr_tidy.csv"

CONTROL, REINF = "Control", "HSR100-100"
# (specimen, sequence): reason. Agreed with E. Kutay before fitting.
EXCLUDE_POINTS = {("R-T3", 1): "seating error: Mr = 3.5 ksi, 7 % CV over the last 5 cycles"}

PA_PSI = 14.706                  # MEAPA atmospheric pressure
FIT_IN_LOG = True                # residuals ln(Mr_pred) - ln(Mr_meas) (equal relative weight)
N_BOOT = 1000
SEED = 1

# --- geosynthetic confinement model (MEAPA inputs). Defaults = nartfem example base/geogrid.
SPEC_H_IN = 12.0                 # specimen height
GRID_Z_IN = 6.0                  # geosynthetic at mid-height
NU_BASE = 0.30                   # base Poisson's ratio used in the confinement equation
GG_DEFAULT = dict(delta_in=6.0, M_kNm=427.8, vg=0.35, phi_deg=51.32, c_psi=2.93,
                  eps0=0.149, rho=72.4, beta=0.247, m=1.72323, n=-2.16, N=1.0e6)
M_SWEEP_KNM = [1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 1e6]
DELTA_SWEEP_IN = [3.0, 6.0, 12.0, 24.0]

# --- equivalent reinforced base: Mr_R >= Mr_C enforced on a grid spanning the tested states
CONSTRAINT_SIGMA3_PSI = [3, 4, 5, 7.5, 10, 12.5, 15, 17.5, 20]
CONSTRAINT_N_SIGMAD = 8          # sigma_d points per sigma3 between the tested min and max
# Also enforce Mr_R >= Mr_C below the tested envelope (field points far from the load and the
# at-rest seed state have θ < 11.7 psi). Agreed with E. Kutay; False = tested envelope only.
EXTEND_CONSTRAINT_LOW = True
LOW_SIGMA3_PSI = [1.0, 1.5, 2.0, 2.5, 3.0]
LOW_SIGMAD_PSI = [0.3, 0.84, 1.38, 1.92, 2.46, 3.0]      # θ down to 3.3 psi
FIG_DPI = 200
# =============================================================================

import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize, brentq
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PSI_TO_KPA = 6.89                # as in nartfem geogrid.py
IN_TO_M = 0.0254
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight"})
MARKERS = {"C-T1": "o", "C-T2": "s", "R-T1": "^", "R-T2": "D", "R-T3": "v"}
GSTYLE = {CONTROL: dict(color="0.45", mfc="white"), REINF: dict(color="black", mfc="black")}


# ----------------------------------------------------------------------------- universal model
def mr_universal(k, s3, sd, ds=0.0, Pa=PA_PSI):
    """Mr (psi) for a triaxial state: sx = sy = s3 + ds, sz = s3 + sd (compression positive)."""
    sx = np.asarray(s3, float) + ds
    sz = np.asarray(s3, float) + np.asarray(sd, float)
    theta = 2 * sx + sz
    toct = np.sqrt(2.0) / 3.0 * np.abs(sz - sx)
    return k[0] * Pa * (theta / Pa) ** k[1] * (toct / Pa + 1.0) ** k[2]


def _resid(p, s3, sd, mr):
    k = (math.exp(p[0]), p[1], p[2])
    pred = mr_universal(k, s3, sd)
    return np.log(pred) - np.log(mr) if FIT_IN_LOG else (pred - mr) / mr.mean()


def fit_universal(d, p0=(math.log(600), 0.6, 0.3)):
    s3, sd, mr = d.sigma3_psi.values, d.sigmad_psi.values, d.Mr_psi.values
    sol = least_squares(_resid, p0, args=(s3, sd, mr), method="lm")
    k = np.array([math.exp(sol.x[0]), sol.x[1], sol.x[2]])
    pred = mr_universal(k, s3, sd)
    n, p = len(mr), 3
    s2 = np.sum(sol.fun ** 2) / (n - p)
    J = sol.jac
    cov = s2 * np.linalg.inv(J.T @ J)                    # in (ln k1, k2, k3)
    tq = stats.t.ppf(0.975, n - p)
    se = np.sqrt(np.diag(cov))
    ci = [(math.exp(sol.x[0] - tq * se[0]), math.exp(sol.x[0] + tq * se[0])),
          (sol.x[1] - tq * se[1], sol.x[1] + tq * se[1]),
          (sol.x[2] - tq * se[2], sol.x[2] + tq * se[2])]
    corr = cov / np.outer(se, se)
    return dict(k=k.tolist(), ci95=[list(c) for c in ci], n=n,
                R2=1 - np.sum((mr - pred) ** 2) / np.sum((mr - mr.mean()) ** 2),
                R2_log=1 - np.sum((np.log(mr) - np.log(pred)) ** 2) / np.sum((np.log(mr) - np.log(mr).mean()) ** 2),
                RMSE_psi=float(np.sqrt(np.mean((mr - pred) ** 2))),
                Se_Sy=float(np.std(mr - pred, ddof=p) / np.std(mr, ddof=1)),
                corr=corr.round(3).tolist(), _x=sol.x)


def bootstrap(d, rng, n=N_BOOT):
    """Points resampled with replacement within each specimen; returns (n,3) k arrays."""
    out = []
    groups = [g for _, g in d.groupby("specimen")]
    x0 = fit_universal(d)["_x"]
    for _ in range(n):
        b = pd.concat([g.sample(len(g), replace=True, random_state=rng.integers(1 << 31)) for g in groups])
        sol = least_squares(_resid, x0, args=(b.sigma3_psi.values, b.sigmad_psi.values, b.Mr_psi.values), method="lm")
        out.append([math.exp(sol.x[0]), sol.x[1], sol.x[2]])
    return np.array(out)


# ----------------------------------------------------------------------------- confinement model
def _gg_consts(g):
    phi = math.radians(g["phi_deg"])
    alpha = 2 * math.sin(phi) / (math.sqrt(3) * (3 - math.sin(phi)))
    K_kPa = g["c_psi"] * PSI_TO_KPA * 6 * math.cos(phi) / (math.sqrt(3) * (3 - math.sin(phi)))
    kGG = (2 * g["M_kNm"] / ((1 - g["vg"]) * g["delta_in"] * IN_TO_M * alpha)
           if np.isfinite(g["M_kNm"]) else np.inf)
    return alpha, K_kPa, kGG


def _strain_terms(ds_kPa, s3_kPa, sz_kPa, Mr_kPa, nu, g, alpha, K_kPa):
    """Elastic lateral extension + 0.85*plastic term of nartfem solid3d/geogrid._residual."""
    sx = sy = s3_kPa + ds_kPa
    sz = sz_kPa
    I1 = sx + sy + sz
    J2 = ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) / 6.0
    base = alpha * I1 + K_kPa
    ep = 0.0
    if base > 0 and sz > max(sx, sy):
        ep = g["eps0"] * math.exp(-((g["rho"] / g["N"]) ** g["beta"])) * math.sqrt(J2) ** g["m"] * base ** g["n"]
        if not math.isfinite(ep):
            ep = 0.0
    lat = nu * sz / Mr_kPa - (1 - nu) * (sx + sy) / (2 * Mr_kPa)
    return lat + 0.85 * ep


def delta_sigma_max_psi(s3, sd, k, g, nu=NU_BASE):
    """Δσ3,max (psi) at the grid plane for one triaxial state (MEAPA equation; Mr from the
    no-geogrid k's, as in MEAPA). M = inf gives the zero-lateral-strain limit."""
    alpha, K_kPa, kGG = _gg_consts(g)
    s3k, szk = s3 * PSI_TO_KPA, (s3 + sd) * PSI_TO_KPA
    Mr = mr_universal(k, s3, sd) * PSI_TO_KPA
    h = lambda ds: _strain_terms(ds, s3k, szk, Mr, nu, g, alpha, K_kPa)
    if h(0.0) <= 0:
        return 0.0                                   # grid not mobilized (no lateral extension)
    f = (lambda ds: -h(ds)) if np.isinf(kGG) else (lambda ds: ds - kGG * h(ds))
    hi = 1.0
    while f(hi) < 0 and hi < 1e6:
        hi *= 2
    return brentq(f, 0.0, hi, xtol=1e-8) / PSI_TO_KPA


def composite_gain(s3, sd, k, g, nu=NU_BASE, H=SPEC_H_IN, zg=GRID_Z_IN, uniform=False, nz=241):
    """Specimen-average Mr (series springs over the height) with confinement / without."""
    dmax = delta_sigma_max_psi(s3, sd, k, g, nu)
    z = np.linspace(0, H, nz)
    if uniform:
        ds = np.full(nz, dmax)
    else:
        ds = dmax * np.maximum(0.0, 1 - np.abs(z - zg) / (0.5 * g["delta_in"]))
    mr_z = mr_universal(k, s3, sd, ds)
    comp = H / np.trapezoid(1.0 / mr_z, z)
    return comp / mr_universal(k, s3, sd), dmax


# ----------------------------------------------------------------------------- equivalent reinforced base
def constraint_states(d):
    """Grid of (s3, sd) inside the tested envelope (sd range interpolated from the test)."""
    env = d.groupby(d.sigma3_psi.round()).sigmad_psi.agg(["min", "max"])
    pts = []
    for s3 in CONSTRAINT_SIGMA3_PSI:
        lo = np.interp(s3, env.index, env["min"]); hi = np.interp(s3, env.index, env["max"])
        for sd in np.linspace(lo, hi, CONSTRAINT_N_SIGMAD):
            pts.append((s3, sd))
    return np.array(pts)


def fit_equivalent(dR, kC, states, x0):
    s3, sd, mr = dR.sigma3_psi.values, dR.sigmad_psi.values, dR.Mr_psi.values
    lnC = np.log(mr_universal(kC, states[:, 0], states[:, 1]))
    obj = lambda p: np.sum(_resid(p, s3, sd, mr) ** 2)
    cons = {"type": "ineq",
            "fun": lambda p: np.log(mr_universal((math.exp(p[0]), p[1], p[2]), states[:, 0], states[:, 1])) - lnC}
    best = None
    for start in (x0, [math.log(kC[0]), kC[1], kC[2]], [x0[0], x0[1], kC[2]]):
        sol = minimize(obj, start, method="SLSQP", constraints=[cons], options=dict(ftol=1e-12, maxiter=500))
        if sol.success and (best is None or sol.fun < best.fun):
            best = sol
    k = np.array([math.exp(best.x[0]), best.x[1], best.x[2]])
    pred = mr_universal(k, s3, sd)
    active = int(np.sum(cons["fun"](best.x) < 1e-6))
    return dict(k=k.tolist(), n=len(mr), R2=1 - np.sum((mr - pred) ** 2) / np.sum((mr - mr.mean()) ** 2),
                RMSE_psi=float(np.sqrt(np.mean((mr - pred) ** 2))), active_constraints=active,
                min_ratio_on_grid=float(np.exp(cons["fun"](best.x)).min()), _x=best.x)


# ----------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    mr = pd.read_csv(MR_TIDY)
    mr = mr[~mr.conditioning]
    excl = mr.apply(lambda r: (r.specimen, r.seq) in EXCLUDE_POINTS, axis=1)
    print("Excluded points:")
    for (s, q), why in EXCLUDE_POINTS.items():
        print(f"  {s} seq {q}: {why}")
    d = mr[~excl].copy()
    dC, dR = d[d.group == CONTROL], d[d.group == REINF]

    # ---------------------------------------------------------------- 1. universal fits
    print(f"\n=== 1. Universal model fits (Pa = {PA_PSI} psi, {'ln' if FIT_IN_LOG else 'linear'} residuals) ===")
    rows, fits = [], {}
    for name, sub in [(s, g) for s, g in d.groupby("specimen")] + [(CONTROL + " (pooled)", dC), (REINF + " (pooled)", dR)]:
        f = fit_universal(sub)
        fits[name] = f
        rows.append(dict(fit=name, n=f["n"], k1=f["k"][0], k2=f["k"][1], k3=f["k"][2],
                         k1_lo=f["ci95"][0][0], k1_hi=f["ci95"][0][1], k2_lo=f["ci95"][1][0], k2_hi=f["ci95"][1][1],
                         k3_lo=f["ci95"][2][0], k3_hi=f["ci95"][2][1], R2=f["R2"], R2_log=f["R2_log"],
                         RMSE_psi=f["RMSE_psi"], Se_Sy=f["Se_Sy"], corr_k2k3=f["corr"][1][2]))
    tab = pd.DataFrame(rows)
    print(tab.round(3).to_string(index=False))
    tab.to_csv(OUT_DIR / "mr_universal_fits.csv", index=False)
    kC = np.array(fits[CONTROL + " (pooled)"]["k"])
    kRu = np.array(fits[REINF + " (pooled)"]["k"])

    bC, bR = bootstrap(dC, rng), bootstrap(dR, rng)
    print(f"\nBootstrap 95 % intervals (points resampled within specimen, {N_BOOT} draws):")
    for nm, b in [("control", bC), ("reinforced (unconstrained)", bR)]:
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        print(f"  {nm:28s} k1 {lo[0]:.0f}-{hi[0]:.0f}  k2 {lo[1]:.3f}-{hi[1]:.3f}  k3 {lo[2]:.3f}-{hi[2]:.3f}")

    # measured gain per sequence (group means), fitted gains at the same states
    seqs = d.groupby("seq").agg(s3=("sigma3_psi", "mean"), sd=("sigmad_psi", "mean"), theta=("theta_psi", "mean"))
    seqs["meas_ratio"] = dR.groupby("seq").Mr_psi.mean() / dC.groupby("seq").Mr_psi.mean()
    seqs["fit_ratio_unconstrained"] = mr_universal(kRu, seqs.s3, seqs.sd) / mr_universal(kC, seqs.s3, seqs.sd)
    boot_ratio = np.array([mr_universal(r, seqs.s3, seqs.sd) / mr_universal(c, seqs.s3, seqs.sd) for c, r in zip(bC, bR)])
    seqs["fit_ratio_lo"], seqs["fit_ratio_hi"] = np.percentile(boot_ratio, [2.5, 97.5], axis=0)

    # ---------------------------------------------------------------- 2. confinement-model feasibility
    print("\n=== 2. Geosynthetic confinement model (MEAPA / NCHRP 1-50) applied to the lab states ===")
    print(f"control k = {kC.round(4).tolist()}  (k3 {'> 0: stress hardening in τoct' if kC[2] > 0 else '<= 0'})")
    g0 = dict(GG_DEFAULT)
    feas = []
    for q, r in seqs.iterrows():
        dinf = delta_sigma_max_psi(r.s3, r.sd, kC, dict(g0, M_kNm=np.inf))
        g_def, d_def = composite_gain(r.s3, r.sd, kC, g0)
        g_inf_tri, _ = composite_gain(r.s3, r.sd, kC, dict(g0, M_kNm=np.inf, delta_in=SPEC_H_IN))
        g_inf_uni, _ = composite_gain(r.s3, r.sd, kC, dict(g0, M_kNm=np.inf), uniform=True)
        # same, elastic term only (plastic strain term switched off)
        g_el, _ = composite_gain(r.s3, r.sd, kC, dict(g0, M_kNm=np.inf, eps0=0.0), uniform=True)
        feas.append(dict(seq=q, s3=r.s3, sd=r.sd, theta=r.theta, dsig_default_psi=d_def, dsig_Minf_psi=dinf,
                         gain_default=g_def, gain_Minf_delta12=g_inf_tri, gain_upper_bound=g_inf_uni,
                         gain_upper_elastic_only=g_el, meas_ratio=r.meas_ratio))
    feas = pd.DataFrame(feas).set_index("seq")
    print(feas.round(3).to_string())
    ub = feas.gain_upper_bound
    print(f"\nLargest specimen-average gain the model can give (M → ∞, Δσ3,max over the whole height): "
          f"{(ub.max() - 1) * 100:.1f} % (seq {ub.idxmax()}), mean over sequences {(ub.mean() - 1) * 100:.1f} %.")
    print(f"With the MEAPA example sheet (M = {g0['M_kNm']} kN/m, δ = {g0['delta_in']} in): mean "
          f"{(feas.gain_default.mean() - 1) * 100:.2f} %, max {(feas.gain_default.max() - 1) * 100:.2f} %.")
    print(f"Measured gain: mean {(feas.meas_ratio.mean() - 1) * 100:.1f} %, seq 7-15 mean "
          f"{(feas.meas_ratio.loc[7:].mean() - 1) * 100:.1f} %, max {(feas.meas_ratio.max() - 1) * 100:.1f} %.")

    # Mr response to an arbitrary uniform Δσ3 (independent of the confinement equation)
    ds_axis = np.linspace(0, 20, 81)
    resp = {q: mr_universal(kC, r.s3, r.sd, ds_axis) / mr_universal(kC, r.s3, r.sd) for q, r in seqs.iterrows()}

    # M and delta sweeps at the seq with the largest measured gain
    q_star = int(feas.meas_ratio.idxmax())
    sweep = {dl: [composite_gain(seqs.s3[q_star], seqs.sd[q_star], kC, dict(g0, M_kNm=M, delta_in=dl))[0]
                  for M in M_SWEEP_KNM] for dl in DELTA_SWEEP_IN}

    # least-squares fit of M, delta (and an eps0 multiplier) to the reinforced data
    def gg_resid(p, free_eps0):
        g = dict(g0, M_kNm=math.exp(p[0]), delta_in=p[1], eps0=g0["eps0"] * (math.exp(p[2]) if free_eps0 else 1.0))
        out = []
        for _, r in dR.iterrows():
            gain, _ = composite_gain(r.sigma3_psi, r.sigmad_psi, kC, g, nz=121)
            out.append(math.log(mr_universal(kC, r.sigma3_psi, r.sigmad_psi) * gain) - math.log(r.Mr_psi))
        return np.array(out)

    sse0 = float(np.sum((np.log(mr_universal(kC, dR.sigma3_psi, dR.sigmad_psi)) - np.log(dR.Mr_psi)) ** 2))
    gg_fits = {}
    for free in (False, True):
        p0 = [math.log(g0["M_kNm"]), g0["delta_in"]] + ([0.0] if free else [])
        lb = [math.log(1e-2), 1.0] + ([-10.0] if free else [])
        ub_ = [math.log(1e7), 2 * SPEC_H_IN] + ([10.0] if free else [])
        sol = least_squares(gg_resid, p0, args=(free,), bounds=(lb, ub_), diff_step=1e-3)
        J = sol.jac
        sv = np.linalg.svd(J, compute_uv=False)
        gg_fits["M, δ, ε0" if free else "M, δ"] = dict(
            M_kNm=math.exp(sol.x[0]), delta_in=sol.x[1], eps0_mult=math.exp(sol.x[2]) if free else 1.0,
            SSE=float(np.sum(sol.fun ** 2)), SSE_no_grid=sse0,
            jac_singular_values=sv.tolist(), condition=float(sv[0] / max(sv[-1], 1e-30)),
            at_bound=[bool(abs(x - l) < 1e-6 or abs(x - u) < 1e-6) for x, l, u in zip(sol.x, lb, ub_)])
    print("\nLeast-squares fit of the confinement model to the reinforced data (control k's fixed):")
    for nm, f in gg_fits.items():
        print(f"  free {nm:8s}: M = {f['M_kNm']:.3g} kN/m, δ = {f['delta_in']:.2f} in, ε0 x {f['eps0_mult']:.3g}; "
              f"SSE(ln) {f['SSE']:.4f} vs {f['SSE_no_grid']:.4f} with no grid; "
              f"Jacobian singular values {np.round(f['jac_singular_values'], 4).tolist()} ; at bound {f['at_bound']}")
    sse_equiv = None

    # ---------------------------------------------------------------- 3. equivalent reinforced base
    print("\n=== 3. Equivalent reinforced base (separate k's, constrained Mr_R >= Mr_C) ===")
    states_tested = constraint_states(d)
    eq_tested = fit_equivalent(dR, kC, states_tested, fits[REINF + " (pooled)"]["_x"])
    print(f"Tested envelope only ({len(states_tested)} states): k = {np.round(eq_tested['k'], 4).tolist()}  "
          f"R² {eq_tested['R2']:.3f}  RMSE {eq_tested['RMSE_psi']:.0f} psi; Mr_R/Mr_C at θ = 8 psi (σ3 = 2, σd = 2): "
          f"{mr_universal(eq_tested['k'], 2.0, 2.0) / mr_universal(kC, 2.0, 2.0):.3f}")
    states = states_tested
    if EXTEND_CONSTRAINT_LOW:
        low = np.array([(a, b) for a in LOW_SIGMA3_PSI for b in LOW_SIGMAD_PSI])
        states = np.vstack([states_tested, low])
    eq = fit_equivalent(dR, kC, states, fits[REINF + " (pooled)"]["_x"]) if EXTEND_CONSTRAINT_LOW else eq_tested
    print(f"RECOMMENDED ({'tested envelope + low-stress states, θ ≥ 3.3 psi' if EXTEND_CONSTRAINT_LOW else 'tested envelope'}, "
          f"{len(states)} states):")
    print(f"k_R,eq = {np.round(eq['k'], 4).tolist()}  R² {eq['R2']:.3f}  RMSE {eq['RMSE_psi']:.0f} psi  "
          f"active constraints {eq['active_constraints']} of {len(states)}  min Mr_R/Mr_C on grid {eq['min_ratio_on_grid']:.4f}")
    u = fits[REINF + " (pooled)"]
    print(f"(unconstrained reinforced fit: k = {np.round(u['k'], 4).tolist()}  R² {u['R2']:.3f}  RMSE {u['RMSE_psi']:.0f} psi; "
          f"min Mr_R/Mr_C on grid {np.min(mr_universal(kRu, states[:, 0], states[:, 1]) / mr_universal(kC, states[:, 0], states[:, 1])):.3f})")
    kRe = np.array(eq["k"])
    seqs["fit_ratio_equivalent"] = mr_universal(kRe, seqs.s3, seqs.sd) / mr_universal(kC, seqs.s3, seqs.sd)
    sse_equiv = float(np.sum((np.log(mr_universal(kRe, dR.sigma3_psi, dR.sigmad_psi)) - np.log(dR.Mr_psi)) ** 2))
    print(f"SSE(ln) of the reinforced data: no-grid control model {sse0:.4f}, best confinement-model fit "
          f"{min(f['SSE'] for f in gg_fits.values()):.4f}, equivalent base {sse_equiv:.4f}")

    # equivalent-base bootstrap of k and of the gain
    bE = []
    for c, r in zip(bC[:300], range(300)):
        b = pd.concat([g.sample(len(g), replace=True, random_state=rng.integers(1 << 31)) for _, g in dR.groupby("specimen")])
        try:
            bE.append(fit_equivalent(b, c, states, eq["_x"])["k"])
        except Exception:
            pass
    bE = np.array(bE)
    lo, hi = np.percentile(bE, [2.5, 97.5], axis=0)
    print(f"Bootstrap 95 % (n = {len(bE)}): k1 {lo[0]:.0f}-{hi[0]:.0f}  k2 {lo[1]:.3f}-{hi[1]:.3f}  k3 {lo[2]:.3f}-{hi[2]:.3f}")

    print("\nGain by sequence (measured vs fitted):")
    print(seqs.round(3).to_string())
    seqs.to_csv(OUT_DIR / "mr_gain_by_sequence.csv")
    feas.to_csv(OUT_DIR / "mr_geogrid_feasibility.csv")

    # Mr at the PD stress state and at a typical field base state, for the rutting step
    pd_state = dict(s3=10.0, sd=27.8)
    res = dict(
        excluded_points={f"{s} seq {q}": w for (s, q), w in EXCLUDE_POINTS.items()},
        Pa_psi=PA_PSI, residuals="ln" if FIT_IN_LOG else "linear",
        control=dict(k=kC.tolist(), ci95=fits[CONTROL + " (pooled)"]["ci95"],
                     boot95=np.percentile(bC, [2.5, 97.5], axis=0).tolist(),
                     R2=fits[CONTROL + " (pooled)"]["R2"], RMSE_psi=fits[CONTROL + " (pooled)"]["RMSE_psi"]),
        reinforced_unconstrained=dict(k=kRu.tolist(), ci95=u["ci95"], R2=u["R2"], RMSE_psi=u["RMSE_psi"],
                                      boot95=np.percentile(bR, [2.5, 97.5], axis=0).tolist()),
        reinforced_equivalent=dict(k=kRe.tolist(), R2=eq["R2"], RMSE_psi=eq["RMSE_psi"],
                                   boot95=[lo.tolist(), hi.tolist()], active_constraints=eq["active_constraints"],
                                   constraint="Mr_R >= Mr_C on %d (sigma3, sigma_d) states: tested envelope%s" % (
                                       len(states), " + low-stress states (theta >= 3.3 psi)" if EXTEND_CONSTRAINT_LOW else "")),
        reinforced_equivalent_tested_only=dict(k=eq_tested["k"], R2=eq_tested["R2"], RMSE_psi=eq_tested["RMSE_psi"]),
        per_specimen={s: dict(k=fits[s]["k"], R2=fits[s]["R2"]) for s in d.specimen.unique()},
        feasibility=dict(upper_bound_gain_max=float(ub.max()), upper_bound_gain_mean=float(ub.mean()),
                         default_gain_mean=float(feas.gain_default.mean()), default_gain_max=float(feas.gain_default.max()),
                         measured_gain_mean=float(feas.meas_ratio.mean()), measured_gain_max=float(feas.meas_ratio.max()),
                         gg_fits=gg_fits, gg_defaults=g0, nu=NU_BASE, sse_ln=dict(no_grid=sse0, equivalent=sse_equiv)),
        pd_state=dict(pd_state, Mr_control_psi=float(mr_universal(kC, **pd_state)),
                      Mr_reinf_equiv_psi=float(mr_universal(kRe, **pd_state))),
    )
    with open(OUT_DIR / "mr_fit_results.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    # ---------------------------------------------------------------- figures
    # (a) measured vs fitted
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (title, kfun) in zip(axes, [("Group universal fits (reinforced unconstrained)", lambda g: kC if g == CONTROL else kRu),
                                         ("Control fit + equivalent reinforced base", lambda g: kC if g == CONTROL else kRe)]):
        for s, sub in d.groupby("specimen"):
            g = sub.group.iloc[0]
            ax.plot(sub.Mr_psi / 1e3, mr_universal(kfun(g), sub.sigma3_psi, sub.sigmad_psi) / 1e3, ls="none",
                    marker=MARKERS[s], ms=5, mec=GSTYLE[g]["color"], mfc=GSTYLE[g]["mfc"], label=s)
        ax.plot([0, 50], [0, 50], color="0.3", lw=0.8)
        ax.set_xlim(0, 50); ax.set_ylim(0, 50); ax.set_aspect("equal")
        ax.set_xlabel("Measured Mr (ksi)"); ax.set_ylabel("Predicted Mr (ksi)"); ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.savefig(OUT_DIR / "fig_mr_measured_vs_fitted.png", dpi=FIG_DPI); plt.close(fig)

    # (b) gain vs theta
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.fill_between(seqs.theta.sort_values(), seqs.sort_values("theta").fit_ratio_lo,
                    seqs.sort_values("theta").fit_ratio_hi, color="0.88", label="Unconstrained fit, bootstrap 95 %")
    ax.plot(seqs.theta, seqs.meas_ratio, "ko", ms=5, label="Measured, mean(R)/mean(C)")
    o = seqs.sort_values("theta")
    ax.plot(o.theta, o.fit_ratio_unconstrained, "k:", lw=1.2, marker="x", ms=4, label="Unconstrained reinforced fit / control fit")
    ax.plot(o.theta, o.fit_ratio_equivalent, "k-", lw=1.6, marker="s", ms=4, mfc="white", label="Equivalent base (Mr_R ≥ Mr_C), recommended")
    th_lo = np.linspace(3.3, 12, 30)
    s3_lo = th_lo / 4.0                                       # σd = σ3 along this low-stress path
    ax.plot(th_lo, mr_universal(kRe, s3_lo, s3_lo) / mr_universal(kC, s3_lo, s3_lo), "k-", lw=0.9)
    ax.plot(th_lo, mr_universal(eq_tested["k"], s3_lo, s3_lo) / mr_universal(kC, s3_lo, s3_lo), color="0.3", ls=(0, (1, 1)), lw=1.2,
            label="Equivalent base, tested envelope only (σd = σ3 below θ = 12)")
    ax.axvspan(0, 11.6, color="0.95", lw=0)
    of = feas.loc[o.index]
    ax.plot(o.theta, of.gain_upper_bound, color="0.45", ls="--", lw=1.4, marker="^", ms=4, label="Confinement model, upper bound (M → ∞)")
    ax.plot(o.theta, of.gain_default, color="0.6", ls="-.", lw=1.2, label="Confinement model, M = %.0f kN/m, δ = %.0f in" % (g0["M_kNm"], g0["delta_in"]))
    ax.axhline(1, color="0.3", lw=0.8)
    ax.set_xlabel("Bulk stress θ (psi)"); ax.set_ylabel("Mr ratio, reinforced / control")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(OUT_DIR / "fig_mr_gain_vs_theta.png", dpi=FIG_DPI); plt.close(fig)

    # (c) feasibility
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ls_cycle = ["-", "--", "-.", ":"]
    for (dl, gains), ls in zip(sweep.items(), ls_cycle):
        axes[0].plot(M_SWEEP_KNM, (np.array(gains) - 1) * 100, color="black", ls=ls, marker="o", ms=3, label=f"δ = {dl:g} in")
    axes[0].axhline((feas.meas_ratio[q_star] - 1) * 100, color="0.5", lw=2, alpha=0.6, label=f"Measured (seq {q_star})")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Sheet stiffness M (kN/m)")
    axes[0].set_ylabel("Specimen-average Mr gain (%)")
    axes[0].set_title(f"Seq {q_star}: σ3 = {seqs.s3[q_star]:.0f} psi, σd = {seqs.sd[q_star]:.0f} psi", fontsize=9)
    axes[0].legend(fontsize=7)
    shades = {3: "0.75", 5: "0.6", 10: "0.45", 15: "0.25", 20: "0.0"}
    for q, r in seqs.iterrows():
        s3r = int(round(r.s3))
        axes[1].plot(ds_axis, (resp[q] - 1) * 100, color=shades[s3r], lw=1,
                     label=f"σ3 = {s3r} psi" if q == seqs[seqs.s3.round() == s3r].index[0] else None)
        axes[1].plot(feas.dsig_Minf_psi[q], (mr_universal(kC, r.s3, r.sd, feas.dsig_Minf_psi[q]) / mr_universal(kC, r.s3, r.sd) - 1) * 100,
                     "k^", ms=4)
    axes[1].set_xlabel("Added horizontal stress Δσ3, uniform (psi)")
    axes[1].set_ylabel("Mr change, control k's (%)")
    axes[1].set_title("Universal model response; ▲ = Δσ3,max at M → ∞", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.savefig(OUT_DIR / "fig_geogrid_feasibility.png", dpi=FIG_DPI); plt.close(fig)
    print("\nWrote", OUT_DIR / "mr_fit_results.json", "and figures.")


if __name__ == "__main__":
    main()
