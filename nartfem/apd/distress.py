"""Miner's-law damage accumulation (bottom-up / top-down fatigue, AC / base / subgrade rutting,
IRI) over months x temperature quintiles - the loop of ``f_struc_resp_ACGB`` (legacy
UPDAPSAC_STEP2.py) with two generalizations:

  1. the structural responses come from a ResponseEngine (MatLEA or nartfem);
  2. the critical strain of each axle-load bin is a function ``strain_by_load`` of the
     engine's results instead of the fixed linear scaling  eps(P) = eps_ref * P / P_ref:
        "linear"  - nartfem-apd (one solution at the reference load, strains proportional to load)
        "power"   - solutions at several load levels (pressure x level), log-log interpolation
                    in P between the levels, log-log extrapolation outside.

With the MatLEA engine, linear scaling and ``rut_monotone = False`` the numbers are identical
to the legacy loop (the legacy njit transfer functions are called unchanged for Nf, plastic
strain, wander and IRI).  ``rut_monotone = True`` (default, option analysis.response_engine.
rut_monotone) adds the guard that plastic strains never decrease - see ``_no_decrease``.

Only the AC-over-granular-base structure (no CSM, no existing AC) is implemented here; CSM /
EAC structures keep using the legacy f_struc_resp_ACCSM / f_struc_resp_ACEAC.
"""
from __future__ import annotations
import time
from math import exp, log10
import numpy as np

from .legacy_bridge import legacy_modules
from .engine_base import AXLE_KEYS, REF_LOAD_LB

legacy_modules()
from f_njit_functions import (ff_calc_Nf, ff_apply_wander, ff_calc_nEQ_AC, ff_calc_eps_plastic_AC,  # noqa: E402
                              ff_calc_nEQ_BASE, ff_calc_eps_plastic_BASE, f_calc_IRI,
                              f_BUFC_CritStrains, f_TDFC_CritStrains, f_rut_CritStrains)

AX_LOADS = {"si": "siLoadsLb", "ta": "taLoadsLb", "tr": "trLoadsLb", "qd": "qdLoadsLb"}


# ----------------------------------------------------------------------------- load scaling
def strain_by_load(ec_levels, levels, loads_lb, ref_lb, mode="linear"):
    """ec_levels: (nlev, nX) critical strain profile per solved load level (level = load /
    reference load); returns (nloads, nX)."""
    ec_levels = np.asarray(ec_levels, float);
    levels = np.asarray(levels, float)
    f = np.asarray(loads_lb, float) / ref_lb
    if mode == "linear" or ec_levels.shape[0] == 1:
        i1 = int(np.argmin(np.abs(levels - 1.0)))
        return ec_levels[i1][None, :] * (f / levels[i1])[:, None]
    # power: log-log interpolation per X position (zero strains stay zero)
    lf = np.log(f);
    ll = np.log(levels)
    out = np.zeros((f.size, ec_levels.shape[1]))
    for j in range(ec_levels.shape[1]):
        col = ec_levels[:, j]
        if np.all(col > 0):
            lc = np.log(col)
            slope_lo = (lc[1] - lc[0]) / (ll[1] - ll[0]);
            slope_hi = (lc[-1] - lc[-2]) / (ll[-1] - ll[-2])
            v = np.interp(lf, ll, lc)
            v = np.where(lf < ll[0], lc[0] + slope_lo * (lf - ll[0]), v)
            v = np.where(lf > ll[-1], lc[-1] + slope_hi * (lf - ll[-1]), v)
            out[:, j] = np.exp(v)
        else:  # clipped (compressive) positions: linear in load
            i1 = int(np.argmin(np.abs(levels - 1.0)))
            out[:, j] = col[i1] * f / levels[i1]
    return out


# ----------------------------------------------------------------------------- transfer functions
def damage_at_loads(ec_by_load, X, E, N_mo, Bi, ki, C, CH, X_Wander):
    """Fatigue damage per load bin (max over X after wander) - f_calc_damage_at_different_loads
    for one axle type with the strain profile of each load given explicitly."""
    NL = ec_by_load.shape[0]
    numWander = len(X_Wander)
    D_mx = np.zeros(NL)
    for k in range(NL):
        Nf = ff_calc_Nf(Bi, ki, C, CH, ec_by_load[k], E)
        D = N_mo[k] / Nf
        Di = np.zeros((numWander, D.size))
        for m in range(numWander):
            Di[m, :] = ff_apply_wander(X, X_Wander[m], D / numWander)
        D_mx[k] = np.amax(np.sum(Di, axis=0))
    return D_mx


RUT_MONOTONE = True   # plastic strain never decreases (see _no_decrease); False = legacy behaviour


def _no_decrease(ep_new, ep_old):
    """Permanent deformation cannot heal: where the strain-hardening update gives a smaller value
    or NaN/inf, keep the previous plastic strain.  In the legacy functions a position whose
    critical strain is zero in this climate case (vertical strain clipped - e.g. between the
    tires in cold months - or kzrut = 0) gets denom = 0 -> nEQ = inf -> eps_p = NaN, and the NaN
    is reset to 0 at the next case, so the accumulated strain at that position restarted from
    zero every year (the seasonal 'dips' of the AC/base rut)."""
    if not RUT_MONOTONE:
        return ep_new
    bad = ~np.isfinite(ep_new) | (ep_new < ep_old)
    return np.where(bad, ep_old, ep_new)


def eps_AC_with_N(ec_by_load, N_mo, epLAST, kzrut, Briz, kriz, T_Qnti):
    """ff_calc_eps_AC_with_N with explicit strain per load (numWander = 1 as in the legacy code)."""
    NL = ec_by_load.shape[0]
    e_p = np.zeros((NL, ec_by_load.shape[1]))
    epLAST = np.where(np.isfinite(epLAST), epLAST, 0.0)
    for j in range(NL):
        e_c1O = ec_by_load[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            nEQ = ff_calc_nEQ_AC(e_c1O, epLAST, kzrut, Briz, kriz, T_Qnti)
            nCOR = nEQ + N_mo[j]
            ep_new = ff_calc_eps_plastic_AC(e_c1O, kzrut, Briz, kriz, T_Qnti, nCOR)
        epLAST = _no_decrease(ep_new, epLAST)
        e_p[j] = epLAST
    return e_p


def eps_BASE_with_N(ec_by_load, N_mo, epLAST, BriBSsub, Erut, GWT):
    """ff_calc_eps_BASE_with_N (Tseng-Lytton, MEPDG water-content form) with explicit strain per load."""
    NL = ec_by_load.shape[0]
    e_p = np.zeros((NL, ec_by_load.shape[1]))
    a1, a9 = 0.15, 20
    CBRzi = (Erut / 2555) ** (1 / 0.64)
    Wczi = 51.712 * (CBRzi ** (-0.3586 * (GWT ** 0.1192)))
    Betazi = 10 ** (-0.61119 - (0.017638 * Wczi))
    Cozi = np.log(a1 / a9)
    rhozi = (10 ** 9) * (Cozi / (1 - 10 ** (9 * Betazi))) ** (1 / Betazi)
    Re = ((np.exp(rhozi ** Betazi) * a1) + (np.exp((rhozi / 10 ** 9) ** Betazi) * a9)) / 2
    epLAST = np.where(np.isfinite(epLAST), epLAST, 0.0)
    for j in range(NL):
        e_c1O = ec_by_load[j]
        epsMaxAtInf = BriBSsub * Re * e_c1O
        with np.errstate(divide="ignore", invalid="ignore"):
            nEQ = ff_calc_nEQ_BASE(epLAST, epsMaxAtInf, rhozi, Betazi)
            nCOR = nEQ + N_mo[j]
            epLASTnext = ff_calc_eps_plastic_BASE(e_c1O, BriBSsub, Re, rhozi, Betazi, nCOR)
        Iinf = np.where(nCOR > 1e12)[0]
        if len(Iinf) > 0:
            epLASTnext[Iinf] = epLAST[Iinf]
        epLASTnext = _no_decrease(epLASTnext, epLAST)
        e_p[j] = epLASTnext
        epLAST = epLASTnext
    return e_p


# ----------------------------------------------------------------------------- the loop
def run_miners(S1, engine, inps, opt, log=print, record=None):
    """S1: Step1Output (legacy); engine: ResponseEngine; inps: legacy inps (for loads);
    opt: engine options (load_scaling, power_levels).  Returns the legacy out_resp dict plus
    'cases' (per-case critical strains) when ``record`` is a list."""
    global RUT_MONOTONE
    RUT_MONOTONE = bool(opt.get("rut_monotone", True))
    mode = opt.get("load_scaling", "linear")
    levels = tuple(opt.get("power_levels", [0.5, 1.0, 2.0])) if mode == "power" else (1.0,)
    al = S1["analysisLoc"];
    cal = S1["calibCoeffs"]
    Esub = S1["Esubdum_M"];
    Tsub = S1["Tsubdum_M"]
    numMo, numQuint = S1["numMo"], S1["numQuint"]
    NL_sub = len(S1["TH_sub"]) + 1
    Ez = np.zeros((numQuint * numMo, NL_sub + 2));
    T_Qnt = np.zeros_like(Ez)
    n = 0
    for iMo in range(numMo):
        for jQ in range(numQuint):
            Ez[n, :] = np.hstack((iMo, jQ, Esub[iMo, :, jQ + 3]));
            T_Qnt[n, :] = np.hstack((iMo, jQ, Tsub[iMo, :, jQ + 3]));
            n += 1
    NumEz = len(Ez)
    loads = {ax: np.asarray(S1[AX_LOADS[ax]], float) for ax in AXLE_KEYS}
    xyz = {ax: al["xyz_" + ax] for ax in AXLE_KEYS}
    SI_X, SJ = al["SI_X"], {"si": 1, "ta": al["SJ2_Y"], "tr": al["SJ3_Y"], "qd": al["SJ4_Y"]}
    zi_bu, zi_td = S1["zi_AC_bu_fat"], S1["zi_AC_td_fat"]
    zi_AC, zi_BASE, zi_SUBG = S1["zi_sub_inAC"], S1["zi_sub_inBASE"], S1["zi_sub_inSUBG"]
    TH_subAC, TH_subBASE = S1["TH_subAC"], S1["TH_subBASE"]
    IBASEsub = S1["IBASEsub"];
    X_Wander = S1["X_Wander"];
    GWT = S1["GWT"];
    hbedrock = S1["hbedrock"]
    tr = S1["traf_NCor"];
    TC_mo = S1["TC_mo"]
    Ind1 = np.where(S1["ZC_in_subM"][0] == zi_bu)[0][0];
    Ifcbu = S1["ZC_in_subM"][2][Ind1]
    numACsub, numBASEsub, numSUBGsub = len(TH_subAC), len(TH_subBASE), 2
    numX = len(np.unique(xyz["si"][:, 0])) * 2 - 1
    epLAST_AC = np.zeros((numACsub, numX));
    epLAST_BASE = np.zeros((numBASEsub, numX));
    epLAST_SUBG = np.zeros((numSUBGsub, numX))
    Dbu_ax = {ax: 0.0 for ax in AXLE_KEYS};
    Dtd_ax = {ax: 0.0 for ax in AXLE_KEYS}
    FCbu = np.zeros(NumEz);
    FCtd = np.zeros(NumEz);
    rut_AC = np.zeros((NumEz, numACsub));
    rut_BASE = np.zeros((NumEz, numBASEsub))
    rut_tot_AC = np.zeros(NumEz);
    rut_tot_BASE = np.zeros(NumEz);
    rut_tot_SUBG = np.zeros(NumEz);
    rutTOT = np.zeros(NumEz);
    IRI = np.zeros(NumEz)
    Dbu_cum = np.zeros(NumEz);
    Dtd_cum = np.zeros(NumEz)
    mo = {k: np.zeros(numMo) for k in ("FCbu", "FCtd", "rut_tot_AC", "rut_tot_BASE", "rut_tot_SUBG", "rutTOT", "IRI")}
    C_TF_bu, Cp1, Cp2, C_TF_td = cal["C_TF_bu"], cal["Cprime_1"], cal["Cprime_2"], cal["C_TF_td"]
    IRIo = float(cal["IRIo"]);
    C_IRI = np.array(cal["C_IRI"])
    t0 = time.time();
    iMnt = 0;
    iQ = 0
    log(" --> Starting structural response computations (%s engine, %s load scaling)..." % (engine.name, mode))
    for nE in range(NumEz):
        Ezi = Ez[nE];
        iMo, jQ = int(Ezi[0]), int(Ezi[1]);
        E_sub = Ezi[2:]
        resp = engine.responses(E_sub, levels)
        EbuAC = Ezi[Ifcbu + 1];
        EtdAC = Ezi[2];
        ErutBASE = Ezi[IBASEsub + 2];
        ErutSUBG = Ezi[-1]
        # critical strain profiles per level
        ebu, etd, erutc, erutb, eruts, Xs = {}, {}, {}, {}, {}, {}
        for li, lev in enumerate(levels):
            r = {ax: resp[ax][li] for ax in AXLE_KEYS}
            e1, e2, e3, e4, X1, X2, X3, X4 = f_BUFC_CritStrains(r["si"]["eps_xx"], r["si"]["eps_yy"], r["ta"]["eps_xx"],
                                                                r["ta"]["eps_yy"],
                                                                r["tr"]["eps_xx"], r["tr"]["eps_yy"], r["qd"]["eps_xx"],
                                                                r["qd"]["eps_yy"],
                                                                xyz["si"], xyz["ta"], xyz["tr"], xyz["qd"], zi_bu, SI_X,
                                                                SJ["ta"], SJ["tr"], SJ["qd"])
            for ax, e in zip(AXLE_KEYS, (e1, e2, e3, e4)):
                ebu.setdefault(ax, []).append(e)
            Xs["bu"] = (X1, X2, X3, X4)
            e1, e2, e3, e4, X1, X2, X3, X4 = f_TDFC_CritStrains(r["si"]["eps_princ"], r["ta"]["eps_princ"],
                                                                r["tr"]["eps_princ"], r["qd"]["eps_princ"],
                                                                xyz["si"], xyz["ta"], xyz["tr"], xyz["qd"], zi_td)
            for ax, e in zip(AXLE_KEYS, (e1, e2, e3, e4)):
                etd.setdefault(ax, []).append(e)
            Xs["td"] = (X1, X2, X3, X4)
            for key, zi, store in (("ac", zi_AC, erutc), ("base", zi_BASE, erutb), ("subg", zi_SUBG, eruts)):
                e1, e2, e3, e4, X1, X2, X3, X4 = f_rut_CritStrains(r["si"]["eps_zz"], r["ta"]["eps_zz"],
                                                                   r["tr"]["eps_zz"], r["qd"]["eps_zz"],
                                                                   xyz["si"], xyz["ta"], xyz["tr"], xyz["qd"], zi)
                for ax, e in zip(AXLE_KEYS, (e1, e2, e3, e4)):
                    store.setdefault(ax, []).append(e)
                Xs[key] = (X1, X2, X3, X4)
        # ---- rutting (sequential over axle types within the quintile, as legacy)
        for i in range(numACsub):
            ep = epLAST_AC[i]
            for ax in AXLE_KEYS:
                ecl = strain_by_load(np.array([erutc[ax][li][i] for li in range(len(levels))]), levels, loads[ax],
                                     REF_LOAD_LB[ax], mode)
                e_p = eps_AC_with_N(ecl, tr["Nc_acrut_" + ax][iMnt][i], ep, cal["kzrut"][i], cal["Briz"][i],
                                    cal["kriz"][i], T_Qnt[nE][i + 2])
                ep = e_p[-1]
            epLAST_AC[i] = ep;
            rut_AC[nE, i] = np.nanmax(ep) * TH_subAC[i]
        rut_tot_AC[nE] = rut_AC[nE].sum()
        for i in range(numBASEsub):
            ep = epLAST_BASE[i]
            for ax in AXLE_KEYS:
                ecl = strain_by_load(np.array([erutb[ax][li][i] for li in range(len(levels))]), levels, loads[ax],
                                     REF_LOAD_LB[ax], mode)
                e_p = eps_BASE_with_N(ecl, tr["Nc_baserut_" + ax][iMnt][i], ep, cal["BriBsub"][i], ErutBASE[i], GWT)
                ep = e_p[-1]
            epLAST_BASE[i] = ep;
            rut_BASE[nE, i] = np.nanmax(ep) * TH_subBASE[i]
        rut_tot_BASE[nE] = rut_BASE[nE].sum()
        e_pMaxS = np.zeros(numSUBGsub)
        for i in range(numSUBGsub):
            ep = epLAST_SUBG[i]
            for ax in AXLE_KEYS:
                ecl = strain_by_load(np.array([eruts[ax][li][i] for li in range(len(levels))]), levels, loads[ax],
                                     REF_LOAD_LB[ax], mode)
                e_p = eps_BASE_with_N(ecl, tr["Nc_subgrut_" + ax][iMnt][i], ep, cal["BriS"], ErutSUBG, GWT)
                ep = e_p[-1]
            epLAST_SUBG[i] = ep;
            e_pMaxS[i] = np.nanmax(ep)
        if e_pMaxS[1] <= 0 or e_pMaxS[0] <= 0:
            rut_tot_SUBG[nE] = 0.0
        else:
            ks = max((1 / 6) * np.log(e_pMaxS[0] / e_pMaxS[1]), 1e-6)
            rut_tot_SUBG[nE] = (1 / ks) * e_pMaxS[0] * (1 - np.exp(-ks * hbedrock))
        rutTOT[nE] = rut_tot_AC[nE] + rut_tot_BASE[nE] + rut_tot_SUBG[nE]
        # ---- fatigue damage
        for k, ax in enumerate(AXLE_KEYS):
            ecl = strain_by_load(np.array(ebu[ax]), levels, loads[ax], REF_LOAD_LB[ax], mode)
            Dbu_ax[ax] += damage_at_loads(ecl, Xs["bu"][k], EbuAC, tr["Nc_bufc_" + ax][iMnt][0], cal["Bi_bu"],
                                          cal["ki_bu"], cal["C_bu"], cal["CH_bu"], X_Wander).sum()
            ecl = strain_by_load(np.array(etd[ax]), levels, loads[ax], REF_LOAD_LB[ax], mode)
            Dtd_ax[ax] += damage_at_loads(ecl, Xs["td"][k], EtdAC, tr["Nc_tdfc_" + ax][iMnt][0], cal["Bi_td"],
                                          cal["ki_td"], cal["C_td"], cal["CH_td"], X_Wander).sum()
        Dbu_cum[nE] = sum(Dbu_ax.values());
        Dtd_cum[nE] = sum(Dtd_ax.values())
        FCbu[nE] = 0.0 if Dbu_cum[nE] <= 0 else (1 / 60) * (
                    C_TF_bu[2] / (1 + exp(Cp1 * C_TF_bu[0] + Cp2 * C_TF_bu[1] * log10(Dbu_cum[nE] * 100))))
        FCtd[nE] = 0.0 if Dtd_cum[nE] <= 0 else 10.56 * (
                    C_TF_td[2] / (1 + exp(C_TF_td[0] - C_TF_td[1] * log10(Dtd_cum[nE] * 100))))
        FCtd_perc = FCtd[nE] / 5280 * 3 / 12 * 100
        FCtot = (FCbu[nE] / 100 + FCtd_perc / 100 - (FCbu[nE] / 100 * FCtd_perc / 100) / 100) * 100
        IRI[nE] = f_calc_IRI(IRIo, FCtot, rutTOT[nE], TC_mo[iMnt], iMnt, S1["FreezeIndex"], S1["Prec_in_an_avg"],
                             S1["P200_subg"], S1["P4_subg"], C_IRI)
        if record is not None:
            i1 = int(np.argmin(np.abs(np.asarray(levels) - 1.0)))
            record.append({"iMo": iMo, "jQ": jQ, "E_AC_top": float(Ezi[2]), "E_AC_bot": float(EbuAC),
                           "eps_bu": {ax: float(np.max(ebu[ax][i1])) for ax in AXLE_KEYS},
                           "eps_td": {ax: float(np.max(etd[ax][i1])) for ax in AXLE_KEYS},
                           "eps_v_subg": {ax: float(np.max(eruts[ax][i1][0])) for ax in AXLE_KEYS},
                           "eps_v_base": {ax: float(np.max(erutb[ax][i1][0])) for ax in AXLE_KEYS},
                           "eps_v_ac": {ax: float(np.max(erutc[ax][i1])) for ax in AXLE_KEYS},
                           "layer_moduli": engine.layer_moduli(("si", 1.0)) if hasattr(engine,
                                                                                       "layer_moduli") else None})
        iQ += 1
        if iQ > 4:
            iQ = 0
            for k, arr in (("FCbu", FCbu), ("FCtd", FCtd), ("rut_tot_AC", rut_tot_AC), ("rut_tot_BASE", rut_tot_BASE),
                           ("rut_tot_SUBG", rut_tot_SUBG), ("rutTOT", rutTOT), ("IRI", IRI)):
                mo[k][iMnt] = arr[nE]
            log(" ---> FCbu[%d] = %.3g%%, FCtd[%d] = %.3gft/mile, ACrut[%d] = %.3gin, BASErut[%d] = %.3gin, SUBGrut[%d] = %.3gin, TOTALrut[%d] = %.3gin, IRI[%d] = %.5gin/mile  (%.0f s)"
                % (iMnt, FCbu[nE], iMnt, FCtd[nE], iMnt, rut_tot_AC[nE], iMnt, rut_tot_BASE[nE], iMnt, rut_tot_SUBG[nE],
                   iMnt, rutTOT[nE], iMnt, IRI[nE], time.time() - t0))
            iMnt += 1
    out = {k + "_mo": v.tolist() for k, v in mo.items()}
    out["Dbu_cum"] = Dbu_cum.tolist();
    out["Dtd_cum"] = Dtd_cum.tolist()
    out["engine"] = engine.name;
    out["engine_stats"] = dict(engine.stats);
    out["load_scaling"] = mode
    out["elapsed_s"] = time.time() - t0
    return out
