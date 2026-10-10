

# from scipy.interpolate import griddata
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import savemat
# import scipy as sp
from numpy import zeros, sin, cos, sum, vstack, array, meshgrid, pi, linalg, matmul, round, \
    isnan, amax, amin, where, abs, argwhere, all, append, flipud, fliplr, argmax, interp, unique,\
    log, exp, isreal, sign, linspace, moveaxis, ndarray, nanmax
# import math
from numba import njit
# from matplotlib import pyplot as plt
# import scipy.io as sio

@njit
def f_calc_IRI(IRIo, FCtot, rutTOT, TC, iMntCum, FreezeIndex, Prec_in_an_avg, P200, P4, C_IRI):

    Age = (iMntCum + 1) / 12 # months to years

    Frost = log((Prec_in_an_avg + 1) * (FreezeIndex + 1) * P4)
    Swell = log((Prec_in_an_avg + 1) * (FreezeIndex + 1) * P200)
    SF = (Frost + Swell) * Age ** 1.5

    # C_IRI = [40, 0.4, 0.008, 0.015] # default coefficients
    IRI = IRIo + C_IRI[0] * rutTOT + C_IRI[1] * FCtot + C_IRI[2] * TC + C_IRI[3] * SF

    return IRI

@njit
def ff_calc_Nf(Bi, ki, C, CH, epscrit, E):
    Nf = Bi[0] * ki[0] * C * CH * epscrit**(-Bi[1] * ki[1]) * E**(-Bi[2] * ki[2])
    return Nf

@njit
def ff_calc_nEQ_AC(e_c, epLAST, kzrut, Briz, kriz, T_Qnti):
    denom = e_c * kzrut * Briz[0] * 10 ** kriz[0] * T_Qnti ** (Briz[1] * kriz[1])
    nEQ = (epLAST / denom) ** (1 / (Briz[2] * kriz[2]))
    nEQ[isnan(nEQ)] = 0

    return nEQ


@njit
def ff_calc_eps_plastic_AC(e_c, kzrut, Briz, kriz, T_Qnti, nCOR):
    e_p = e_c * kzrut * Briz[0] * 10 ** kriz[0] * \
            T_Qnti ** (Briz[1] * kriz[1]) * \
            nCOR ** (Briz[2] * kriz[2])

    return e_p

@njit
def ff_calc_nEQ_BASE(epLAST, epsMaxAtInf, rhozi, Betazi):

    epdiff = epsMaxAtInf - epLAST
    Inege = where(epdiff < 0)[0]

    epLASTneq = epLAST.copy()
    if len(Inege) > 0:
        epLASTneq[Inege] = epsMaxAtInf[Inege]

    A = log(epsMaxAtInf / epLASTneq)

    denom = A ** (1 / Betazi)
    nEQ = rhozi / denom

    return nEQ


@njit
def ff_calc_eps_plastic_BASE(e_c, BriBSsub, Re, rhozi, Betazi, nCOR):
    expVal = -((rhozi / nCOR) ** Betazi)

    e_p = e_c * BriBSsub * Re * exp(expVal)

    return e_p

@njit
def ff_calc_eps_BASE_with_N(Pax, Pref, e_rutb, Xrut1, X_Wander, NAij_AxtwMoQnt, epLAST, BriBSsub, ErutBASE, GWT):
    # numWander = 1  # len(X_Wander) # this is because the wheel wander is not working properly at the moment
    NloadsAx = len(Pax)
    e_p = zeros((NloadsAx, len(e_rutb)))
    a1 = 0.15
    # b1 = 0
    a9 = 20
    # b9 = 0
    # for each load level
    for j in range(NloadsAx):
        # critical strain extrapolation for this load level
        e_c1O = e_rutb / Pref * Pax[j]
        CBRzi = (ErutBASE / 2555) ** (1 / 0.64)
        Wczi = 51.712 * (CBRzi ** (-0.3586 * (GWT ** 0.1192)))
        Betazi = 10 ** (-0.61119 - (0.017638 * Wczi))
        Cozi = log(a1 / a9)
        rhozi = (10 ** 9) * (Cozi / (1 - 10 ** (9* Betazi))) ** (1 / Betazi)
        Re = ((exp(rhozi ** Betazi) * a1) + (exp((rhozi / 10 ** 9) ** Betazi) * a9)) / 2

        # maximum possible plastic strain, based on the form of the equation
        epsMaxAtInf = BriBSsub * Re * e_c1O

        # x = 10**(linspace(0,7,100))
        # y = ff_calc_eps_plastic_BASE(e_c1O[0], BriBSsub, Re, rhozi, Betazi, x)
        # plt.plot(x, y, marker='o', markerfacecolor='blue', markersize=3, color='skyblue', linewidth=1)
        # plt.show()

        nEQ = ff_calc_nEQ_BASE(epLAST, epsMaxAtInf, rhozi, Betazi)

        # corrected cycle for this month/quintile
        nCUR = NAij_AxtwMoQnt[j]
        nCOR = nEQ + nCUR

        epLASTnext = ff_calc_eps_plastic_BASE(e_c1O, BriBSsub, Re, rhozi, Betazi, nCOR)

        # When nCOR is infinity for a point, don't try to update the plastic strain, keep the same as last
        Iinf = where(nCOR > 1e12)[0]
        if len(Iinf) > 0:
            epLASTnext[Iinf] = epLAST[Iinf] # take it from the previous plastic strain.
            epLASTnext[Iinf] = epLAST[Iinf] # take it from the previous plastic strain.

        e_p[j] = epLASTnext
        epLAST = epLASTnext

    return e_p

@njit
def f_cal_SUBGRut(e_ruts1, e_ruts2, e_ruts3, e_ruts4, Xrut1, Xrut2, Xrut3, Xrut4,
                Psi, Pta, Ptr, Pqd, NAij_SitwMoQnt, NAij_TatwMoQnt, NAij_TrtwMoQnt, NAij_QdtwMoQnt,
                BriSsub, X_Wander, epLASTsbg, numSUBGsub, ErutSUBG, GWT, hbedrock):

    e_pMaxS = zeros(numSUBGsub)
    # for each point (top of subgrade and 6" below the top of subgrade)
    for i in range(numSUBGsub):

        # sublayer temperature
        epLASTsbgi = epLASTsbg[i]
        e_p1 = ff_calc_eps_BASE_with_N(Psi, 18000, e_ruts1[i], Xrut1, X_Wander, NAij_SitwMoQnt[i], epLASTsbgi, BriSsub, ErutSUBG, GWT)

        epLASTsbgi = e_p1[-1]
        e_p2 = ff_calc_eps_BASE_with_N(Pta, 36000, e_ruts2[i], Xrut2, X_Wander, NAij_TatwMoQnt[i], epLASTsbgi, BriSsub, ErutSUBG, GWT)

        epLASTsbgi = e_p2[-1]
        e_p3 = ff_calc_eps_BASE_with_N(Ptr, 54000, e_ruts3[i], Xrut3, X_Wander, NAij_TrtwMoQnt[i], epLASTsbgi, BriSsub, ErutSUBG, GWT)

        epLASTsbgi = e_p3[-1]
        e_p4 = ff_calc_eps_BASE_with_N(Pqd, 72000, e_ruts4[i], Xrut4, X_Wander, NAij_QdtwMoQnt[i], epLASTsbgi, BriSsub, ErutSUBG, GWT)

        epLASTsbg[i] = e_p4[-1]
        e_pMaxS[i] = nanmax(epLASTsbg[i])
        e_pMaxS

    if (e_pMaxS[1] <=0 ) or (e_pMaxS[0] <=0 ):
        rut_SUBG = 0
    else:
        ksubg = (1 / 6) * log(e_pMaxS[0] / e_pMaxS[1])

        if ksubg < 1e-6:
            ksubg = 1e-6

        rut_SUBG =  (1 / ksubg) * e_pMaxS[0] * (1 - exp(-ksubg * hbedrock))

    return rut_SUBG, epLASTsbg


@njit
def f_cal_BASERut(e_rutb1, e_rutb2, e_rutb3, e_rutb4, Xrut1, Xrut2, Xrut3, Xrut4,
                Psi, Pta, Ptr, Pqd, NAij_SitwMoQnt, NAij_TatwMoQnt, NAij_TrtwMoQnt, NAij_QdtwMoQnt,
                BriBsub, TH_insubBASE, X_Wander, epLAST, numBASEsub, ErutBASE, GWT):

    # mat_contents = sio.loadmat('e_pB.MAT', squeeze_me=True)
    # e_pB = mat_contents['e_pB']
    # e_pB = moveaxis(e_pB, 0, -1)
    # e_pB = moveaxis(e_pB, 0, -1)

    rut_BASE = zeros(numBASEsub)
    # for each BASE sublayer
    for i in range(numBASEsub):
        # e_pBzi = e_pB[i]

        # sublayer temperature
        epLASTi = epLAST[i]
        e_p1 = ff_calc_eps_BASE_with_N(Psi, 18000, e_rutb1[i], Xrut1, X_Wander, NAij_SitwMoQnt[i], epLASTi, BriBsub[i], ErutBASE[i], GWT)

        # diff1 = e_p1 - e_pBzi[:39, :]

        epLASTi = e_p1[-1]
        e_p2 = ff_calc_eps_BASE_with_N(Pta, 36000, e_rutb2[i], Xrut2, X_Wander, NAij_TatwMoQnt[i], epLASTi, BriBsub[i], ErutBASE[i], GWT)

        # diff2 = e_p2 - e_pBzi[40:79, :]

        epLASTi = e_p2[-1]
        e_p3 = ff_calc_eps_BASE_with_N(Ptr, 54000, e_rutb3[i], Xrut3, X_Wander, NAij_TrtwMoQnt[i], epLASTi, BriBsub[i], ErutBASE[i], GWT)

        epLASTi = e_p3[-1]
        e_p4 = ff_calc_eps_BASE_with_N(Pqd, 72000, e_rutb4[i], Xrut4, X_Wander, NAij_QdtwMoQnt[i], epLASTi, BriBsub[i], ErutBASE[i], GWT)

        epLAST[i] = e_p4[-1]
        epMAX = nanmax(epLAST[i])

        rut_BASE[i] =  epMAX * TH_insubBASE[i]

        # Inan = argwhere(isnan(epLAST[i]))
        # if Inan.size > 0:
        #     epLAST[i]

    return rut_BASE, epLAST


@njit
def ff_calc_eps_AC_with_N(Pax, Pref, e_rutc1, Xrut1, X_Wander, NAij_AxtwMoQnt, epLAST, kzrut, Briz, kriz, T_Qnti):

    numWander = 1 #len(X_Wander) # this is because the wheel wander is not working properly at the moment

    NloadsAx = len(Pax)
    e_p = zeros((NloadsAx, len(e_rutc1)))
    
    # for each load level
    for j in range(NloadsAx):
        if numWander > 1:
            # critical strain extrapolation for this load level
            e_c1O = e_rutc1 / Pref * Pax[j] #/ numWander
            nEQ = ff_calc_nEQ_AC(e_c1O, epLAST, kzrut, Briz, kriz, T_Qnti)
            # corrected cycle for this month/quintile
            nCUR = NAij_AxtwMoQnt[j]
            nCOR = nEQ + nCUR

            epLAST = ff_calc_eps_plastic_AC(e_c1O, kzrut, Briz, kriz, T_Qnti, nCOR)
            # Apply wheel wander
            epLASTw = zeros((numWander, len(epLAST)))
            for m in range(numWander):
                epLASTw[m, :] = ff_apply_wander(Xrut1, X_Wander[m], epLAST)

            epLAST = sum(epLASTw, axis=0)
            e_p[j] = epLAST

        else:
            # critical strain extrapolation for this load level
            e_c1O = e_rutc1 / Pref * Pax[j]
            nEQ = ff_calc_nEQ_AC(e_c1O, epLAST, kzrut, Briz, kriz, T_Qnti)
            # corrected cycle for this month/quintile
            nCUR = NAij_AxtwMoQnt[j]
            nCOR = nEQ + nCUR

            epLAST = ff_calc_eps_plastic_AC(e_c1O, kzrut, Briz, kriz, T_Qnti, nCOR)
            e_p[j] = epLAST

        epLAST

    return e_p



@njit
def f_cal_ACRut(e_rutc1, e_rutc2, e_rutc3, e_rutc4, Xrut1, Xrut2, Xrut3, Xrut4,
                Psi, Pta, Ptr, Pqd, NAij_SitwMoQnt, NAij_TatwMoQnt, NAij_TrtwMoQnt, NAij_QdtwMoQnt,
                kriz, Briz, kzrut, T_Qnt, TH_insubAC, X_Wander, epLAST, numACsub):

    rut_AC = zeros(numACsub)
    # for each AC sublayer
    for i in range(numACsub):
        # sublayer temperature
        T_Qnti = T_Qnt[i+2]
        e_p1 = ff_calc_eps_AC_with_N(Psi, 18000, e_rutc1[i], Xrut1, X_Wander, NAij_SitwMoQnt[i], epLAST[i], kzrut[i], Briz[i], kriz[i], T_Qnti)
        e_p2 = ff_calc_eps_AC_with_N(Pta, 36000, e_rutc2[i], Xrut2, X_Wander, NAij_TatwMoQnt[i], e_p1[-1], kzrut[i], Briz[i], kriz[i], T_Qnti)
        e_p3 = ff_calc_eps_AC_with_N(Ptr, 54000, e_rutc3[i], Xrut3, X_Wander, NAij_TrtwMoQnt[i], e_p2[-1], kzrut[i], Briz[i], kriz[i], T_Qnti)
        e_p4 = ff_calc_eps_AC_with_N(Pqd, 72000, e_rutc4[i], Xrut4, X_Wander, NAij_QdtwMoQnt[i], e_p3[-1], kzrut[i], Briz[i], kriz[i], T_Qnti)
        epLAST[i] = e_p4[-1]
        epMAX = nanmax(epLAST[i])
        rut_AC[i] =  epMAX * TH_insubAC[i]


    return rut_AC, epLAST
@njit
def ff_calc_NfCSM(CSMBeta_c1_sub, CSMBeta_c2_sub, s_c, CSMMoR_sub):
    Nf = 10**((0.972 * CSMBeta_c1_sub - s_c/CSMMoR_sub)/(0.0825*CSMBeta_c2_sub))
    return Nf


@njit
def f_calc_damage_at_different_loadsCSM(sc1, sc2, sc3, sc4, X1, X2, X3, X4,
                                     CSMMoR_sub, Psi, Pta, Ptr, Pqd,
                                     NAij_SitwMoQnt, NAij_TatwMoQnt, NAij_TrtwMoQnt, NAij_QdtwMoQnt,
                                     CSMBeta_c1_sub, CSMBeta_c2_sub, X_Wander):

    NloadsSi = len(Psi)
    NloadsTa = len(Pta)
    NloadsTr = len(Ptr)
    NloadsQd = len(Pqd)

    D_csm1_mx = zeros((NloadsSi))
    D_csm2_mx = zeros((NloadsTa))
    D_csm3_mx = zeros((NloadsTr))
    D_csm4_mx = zeros((NloadsQd))

    NloadsMax = max([NloadsSi, NloadsTa, NloadsTr, NloadsQd])
    numWander = len(X_Wander)

    for k in range(NloadsMax):

        if k < NloadsSi:
            s_c1O = sc1 / 18000 * Psi[k]
            NfSi = ff_calc_NfCSM(CSMBeta_c1_sub, CSMBeta_c2_sub, s_c1O, CSMMoR_sub)
            D_si = NAij_SitwMoQnt[k] / NfSi
            Di_si = zeros((len(X_Wander), len(D_si)))

        if k < NloadsTa:
            s_c2O = sc2 / 36000 * Pta[k]
            NfTa = ff_calc_NfCSM(CSMBeta_c1_sub, CSMBeta_c2_sub, s_c2O, CSMMoR_sub)
            D_ta = NAij_TatwMoQnt[k] / NfTa
            Di_ta = zeros((len(X_Wander), len(D_ta)))

        if k < NloadsTr:
            s_c3O = sc3 / 54000 * Ptr[k]
            NfTr = ff_calc_NfCSM(CSMBeta_c1_sub, CSMBeta_c2_sub, s_c3O, CSMMoR_sub)
            D_tr = NAij_TrtwMoQnt[k] / NfTr
            Di_tr = zeros((len(X_Wander), len(D_tr)))

        if k < NloadsQd:
            s_c4O = sc4 / 72000 * Pqd[k]
            NfQd = ff_calc_NfCSM(CSMBeta_c1_sub, CSMBeta_c2_sub, s_c4O, CSMMoR_sub)
            D_qd = NAij_QdtwMoQnt[k] / NfQd
            Di_qd = zeros((len(X_Wander), len(D_qd)))

        # Apply wheel wander
        for m in range(numWander):
            if k < NloadsSi:
                Di_si[m, :] = ff_apply_wander(X1, X_Wander[m], D_si/numWander)
            if k < NloadsTa:
                Di_ta[m, :] = ff_apply_wander(X2, X_Wander[m], D_ta/numWander)
            if k < NloadsTr:
                Di_tr[m, :] = ff_apply_wander(X3, X_Wander[m], D_tr/numWander)
            if k < NloadsQd:
                Di_qd[m, :] = ff_apply_wander(X4, X_Wander[m], D_qd/numWander)

        if k < NloadsSi:
            Di_sicum = sum(Di_si, axis=0)
            D_csm1_mx[k] = amax(Di_sicum)
        if k < NloadsTa:
            Di_tacum = sum(Di_ta, axis=0)
            D_csm2_mx[k] = amax(Di_tacum)
        if k < NloadsTr:
            Di_trcum = sum(Di_tr, axis=0)
            D_csm3_mx[k] = amax(Di_trcum)
        if k < NloadsQd:
            Di_qdcum = sum(Di_qd, axis=0)
            D_csm4_mx[k] = amax(Di_qdcum)

    # D_csm1_mx = Bottom-up damage due to each load level for single axle, [1x39]
    # D_csm2_mx = Bottom-up damage due to each load level for tandem axle, [1x39]
    # D_csm3_mx = Bottom-up damage due to each load level for tridem axle, [1x31]
    # D_csm4_mx = Bottom-up damage due to each load level for quad axle, [1x31]
    # Dcsm = sum(D_csm1_mx) + sum(D_csm2_mx) + sum(D_csm3_mx) + sum(D_csm4_mx)
    # out_resp = {}
    # out_resp['D_csm1_mx'] = D_csm1_mx
    # out_resp['D_csm2_mx'] = D_csm2_mx
    # out_resp['D_csm3_mx'] = D_csm3_mx
    # out_resp['D_csm4_mx'] = D_csm4_mx
    # savemat("outputs_Python_CSMFC.mat", out_resp)
    return D_csm1_mx, D_csm2_mx, D_csm3_mx, D_csm4_mx

@njit
def f_calc_damage_at_different_loads(ec1, ec2, ec3, ec4, X1, X2, X3, X4,
                                     EbuEAC, Psi, Pta, Ptr, Pqd,
                                     NAij_SitwMoQnt, NAij_TatwMoQnt, NAij_TrtwMoQnt, NAij_QdtwMoQnt,
                                     Bi, ki, C, CH, X_Wander):

    NloadsSi = len(Psi)
    NloadsTa = len(Pta)
    NloadsTr = len(Ptr)
    NloadsQd = len(Pqd)

    D1_mx = zeros((NloadsSi))
    D2_mx = zeros((NloadsTa))
    D3_mx = zeros((NloadsTr))
    D4_mx = zeros((NloadsQd))

    NloadsMax = max([NloadsSi, NloadsTa, NloadsTr, NloadsQd])
    numWander = len(X_Wander)

    for k in range(NloadsMax):

        if k < NloadsSi:
            e_c1O = ec1 / 18000 * Psi[k]
            NfSi = ff_calc_Nf(Bi, ki, C, CH, e_c1O, EbuEAC)
            D_si = NAij_SitwMoQnt[k] / NfSi
            Di_si = zeros((len(X_Wander), len(D_si)))

        if k < NloadsTa:
            e_c2O = ec2 / 36000 * Pta[k]
            NfTa = ff_calc_Nf(Bi, ki, C, CH, e_c2O, EbuEAC)
            D_ta = NAij_TatwMoQnt[k] / NfTa
            Di_ta = zeros((len(X_Wander), len(D_ta)))

        if k < NloadsTr:
            e_c3O = ec3 / 54000 * Ptr[k]
            NfTr = ff_calc_Nf(Bi, ki, C, CH, e_c3O, EbuEAC)
            D_tr = NAij_TrtwMoQnt[k] / NfTr
            Di_tr = zeros((len(X_Wander), len(D_tr)))

        if k < NloadsQd:
            e_c4O = ec4 / 72000 * Pqd[k]
            NfQd = ff_calc_Nf(Bi, ki, C, CH, e_c4O, EbuEAC)
            D_qd = NAij_QdtwMoQnt[k] / NfQd
            Di_qd = zeros((len(X_Wander), len(D_qd)))

        # Apply wheel wander
        for m in range(numWander):
            if k < NloadsSi:
                Di_si[m, :] = ff_apply_wander(X1, X_Wander[m], D_si/numWander)
            if k < NloadsTa:
                Di_ta[m, :] = ff_apply_wander(X2, X_Wander[m], D_ta/numWander)
            if k < NloadsTr:
                Di_tr[m, :] = ff_apply_wander(X3, X_Wander[m], D_tr/numWander)
            if k < NloadsQd:
                Di_qd[m, :] = ff_apply_wander(X4, X_Wander[m], D_qd/numWander)

        if k < NloadsSi:
            Di_sicum = sum(Di_si, axis=0)
            D1_mx[k] = amax(Di_sicum)
        if k < NloadsTa:
            Di_tacum = sum(Di_ta, axis=0)
            D2_mx[k] = amax(Di_tacum)
        if k < NloadsTr:
            Di_trcum = sum(Di_tr, axis=0)
            D3_mx[k] = amax(Di_trcum)
        if k < NloadsQd:
            Di_qdcum = sum(Di_qd, axis=0)
            D4_mx[k] = amax(Di_qdcum)

    # D1_mx = Bottom-up damage due to each load level for single axle, [1x39]
    # D2_mx = Bottom-up damage due to each load level for tandem axle, [1x39]
    # D3_mx = Bottom-up damage due to each load level for tridem axle, [1x31]
    # D4_mx = Bottom-up damage due to each load level for quad axle, [1x31]

    return D1_mx, D2_mx, D3_mx, D4_mx

@njit
def ff_apply_wander(X1, X_Wander, D_si):
    X1_shift = X1 - X_Wander

    # Eliminate damage at the points outside the range of X1 after shift
    X1dum = X1.copy()
    X1_shiftmin = amin(X1_shift)
    Ilower = where(X1 < X1_shiftmin)[0]
    X1dum[Ilower] = X1_shiftmin

    X1_shiftmax= amax(X1_shift)
    Ihigher = where(X1 > X1_shiftmax)[0]
    X1dum[Ihigher] = X1_shiftmax

    # if the sequence xp = X1_shift is non-increasing, interpolation results are meaningless.
    Di_sidum = interp(X1dum, X1_shift, D_si)
    Di_sidum[Ilower] = 0
    Di_sidum[Ihigher] = 0
    return Di_sidum

@njit
def ff_find_eps_at_ZC(xyz_ax, eps_xx_ax, eps_yy_ax, zi):
    Ival1 = where(xyz_ax[:, 2] == zi)[0]
    e_xm1 = eps_xx_ax[Ival1] # xx: transverse direction (perpendicular to traffic)
    e_ym1 = eps_yy_ax[Ival1] # yy: longitudinal direction (traffic movement direction)
    pos_xm1 = xyz_ax[Ival1, 0]
    return pos_xm1, e_xm1, e_ym1

@njit
def ff_find_epscrit_multaxleBU(xyz_ax, eps_xx_ax, eps_yy_ax, zi_AC_bu_fat, SI, SJ):
    pos_xm2, e_xm2, e_ym2 = ff_find_eps_at_ZC(xyz_ax, eps_xx_ax, eps_yy_ax, zi_AC_bu_fat)
    e_xm2   = e_xm2.reshape((SJ, SI)).T
    e_ym2   = e_ym2.reshape((SJ, SI)).T
    pos_xm2 = pos_xm2.reshape((SJ, SI)).T

    numYcoords = len(e_xm2[0])
    mx_e_xmYdir = zeros((numYcoords))
    mx_e_ymYdir = zeros((numYcoords))
    for i in range(numYcoords):
        mx_e_xmYdir[i] = amax(e_xm2[:, i])
        mx_e_ymYdir[i] = amax(e_ym2[:, i])

    if amax(mx_e_xmYdir) > amax(mx_e_ymYdir):
        IndY  = argmax(mx_e_xmYdir)  # find the Y coordinate where the max is located
        e_cta = e_xm2[:, IndY]
    else:
        IndY  = argmax(mx_e_ymYdir)  # find the Y coordinate where the max is located
        e_cta = e_ym2[:, IndY]

    pos_xm2 = pos_xm2[:, IndY]

    X2 = append(-flipud(pos_xm2[1:]), pos_xm2)
    e_c2 = append(flipud(e_cta[1:]), e_cta)

    # Make compressive strains zero
    e_c2[where(e_c2 < 0)[0]] = 0

    return X2, e_c2

@njit
def f_BUFC_CritStrains(eps_xx_si, eps_yy_si,
                      eps_xx_ta, eps_yy_ta,
                      eps_xx_tr, eps_yy_tr,
                      eps_xx_qd, eps_yy_qd,
                      xyz_si, xyz_ta, xyz_tr, xyz_qd,
                      zi_AC_bu_fat, SI, SJ2, SJ3, SJ4):

    # Find the strain profile, 'X versus strain', where 'X versus strain' is maximum along Y direction
    X1, e_c1 = ff_find_epscrit_multaxleBU(xyz_si, eps_xx_si, eps_yy_si, zi_AC_bu_fat, SI, 1)  # SINGLE axle
    X2, e_c2 = ff_find_epscrit_multaxleBU(xyz_ta, eps_xx_ta, eps_yy_ta, zi_AC_bu_fat, SI, SJ2)  # TANDEM axle
    X3, e_c3 = ff_find_epscrit_multaxleBU(xyz_tr, eps_xx_tr, eps_yy_tr, zi_AC_bu_fat, SI, SJ3)  # TRIDEM axle
    X4, e_c4 = ff_find_epscrit_multaxleBU(xyz_qd, eps_xx_qd, eps_yy_qd, zi_AC_bu_fat, SI, SJ4)  # QUAD axle

    return e_c1, e_c2, e_c3, e_c4, X1, X2, X3, X4

@njit
def ff_find_epscrit_multaxleTD(xyz_ax, eps_princ_ax, zi_AC_td_fat):

    Ival1 = where(xyz_ax[:, 2] <= zi_AC_td_fat)[0]
    eps_princ_ax1 = eps_princ_ax[Ival1] # principal direction
    xyz_ax1 = xyz_ax[Ival1]

    Imax = where(eps_princ_ax1==max(eps_princ_ax1))[0]
    xyz_ax2 = xyz_ax1[Imax[0]]

    # Find the y coordinate where the max strain is located (applies to tridem and above)
    Iy = (xyz_ax1[:, 1] == xyz_ax2[1]) & (xyz_ax1[:, 2] == xyz_ax2[2])

    x_ax3 = xyz_ax1[Iy, 0]
    eps_princ_ax3 = eps_princ_ax1[Iy]

    X2 = append(-flipud(x_ax3[1:]), x_ax3)
    e_c2 = append(flipud(eps_princ_ax3[1:]), eps_princ_ax3)

    # Make compressive strains zero
    e_c2[where(e_c2 < 0)[0]] = 0
    return X2, e_c2


@njit
def f_TDFC_CritStrains(eps_princ_si,
                      eps_princ_ta,
                      eps_princ_tr,
                      eps_princ_qd,
                      xyz_si, xyz_ta, xyz_tr, xyz_qd,
                      zi_AC_td_fat):

    # Find the strain profile, 'X versus strain', where 'X versus strain' is maximum along Y direction
    X1, e_c1 = ff_find_epscrit_multaxleTD(xyz_si, eps_princ_si, zi_AC_td_fat)  # SINGLE axle
    X2, e_c2 = ff_find_epscrit_multaxleTD(xyz_ta, eps_princ_ta, zi_AC_td_fat)  # TANDEM axle
    X3, e_c3 = ff_find_epscrit_multaxleTD(xyz_tr, eps_princ_tr, zi_AC_td_fat)  # TRIDEM axle
    X4, e_c4 = ff_find_epscrit_multaxleTD(xyz_qd, eps_princ_qd, zi_AC_td_fat)  # QUAD axle


    return e_c1, e_c2, e_c3, e_c4, X1, X2, X3, X4


@njit
def ff_find_epscrit_multaxleRUT(xyz_ax, eps_zz_ax, zi_AC_rut):

    Ival1 = where(xyz_ax[:, 2] == zi_AC_rut)[0]
    eps_zz_ax1 = eps_zz_ax[Ival1]
    xyz_ax1 = xyz_ax[Ival1]

    Imax = where(eps_zz_ax1==max(eps_zz_ax1))[0]
    xyz_ax2 = xyz_ax1[Imax[0]]

    # Find the y coordinate where the max strain is located (applies to tridem and above)
    Iy = (xyz_ax1[:, 1] == xyz_ax2[1]) & (xyz_ax1[:, 2] == xyz_ax2[2])

    x_ax3 = xyz_ax1[Iy, 0]
    eps_zz_ax3 = eps_zz_ax1[Iy]

    X2 = append(-flipud(x_ax3[1:]), x_ax3)
    e_c2 = append(flipud(eps_zz_ax3[1:]), eps_zz_ax3)

    # Make tensile strains zero
    e_c2[where(e_c2 < 0)[0]] = 0
    return X2, e_c2


@njit
def f_rut_CritStrains(eps_zz_si,
                      eps_zz_ta,
                      eps_zz_tr,
                      eps_zz_qd,
                      xyz_si, xyz_ta, xyz_tr, xyz_qd,
                      zi_PV_in):
    numX_si = len(unique(xyz_si[:, 0]))*2-1
    numX_ta = len(unique(xyz_ta[:, 0]))*2-1
    numX_tr = len(unique(xyz_tr[:, 0]))*2-1
    numX_qd = len(unique(xyz_qd[:, 0]))*2-1
    numZ = len(zi_PV_in)
    e_c1M = zeros((numZ, numX_si))
    e_c2M = zeros((numZ, numX_ta))
    e_c3M = zeros((numZ, numX_tr))
    e_c4M = zeros((numZ, numX_qd))


    for i in range(numZ):
        # Find the strain profile, 'X versus strain', where 'X versus strain' is maximum along Y direction
        X1, e_c1M[i] = ff_find_epscrit_multaxleRUT(xyz_si, -eps_zz_si, zi_PV_in[i])  # SINGLE axle
        X2, e_c2M[i] = ff_find_epscrit_multaxleRUT(xyz_ta, -eps_zz_ta, zi_PV_in[i])  # TANDEM axle
        X3, e_c3M[i] = ff_find_epscrit_multaxleRUT(xyz_tr, -eps_zz_tr, zi_PV_in[i])  # TRIDEM axle
        X4, e_c4M[i] = ff_find_epscrit_multaxleRUT(xyz_qd, -eps_zz_qd, zi_PV_in[i])  # QUAD axle

    return e_c1M, e_c2M, e_c3M, e_c4M, X1, X2, X3, X4

@njit
def f_lin_interp1(x, y, xo):
    # Linear interpolation.
    dxo     = x-xo
    Idxo    = where(dxo <= 0)[0]
    n       = Idxo[-1]+1
    lenx    = len(x)
    yo: float64 = 0

    if n==lenx:
        yo = y[-1]
        # print("True")
    else:
        dy   = y[n] - y[n-1]
        dx   = x[n] - x[n-1]

        yo = y[n-1] + dy/dx*(xo-x[n-1])

    return yo

# njit is used to accelerate the algorithm
@njit
def ff_superpose_strains_stresses(RM_ax, theta_ax, ZC_in, strain_r, strain_t, strain_z, strain_rz, stress_r, stress_t, stress_z, stress_rz, rz):
    Ntires  = len(RM_ax[0])
    Ncoords = len(RM_ax) * len(ZC_in)

    # xx: transverse direction (perpendicular to traffic)
    # yy: longitudinal direction (traffic movement direction)
    # zz: depth
    # (+): tension,     # (-): compression

    eps_xx_tire= zeros((Ntires, Ncoords))
    eps_xy_tire= zeros((Ntires, Ncoords))
    eps_xz_tire= zeros((Ntires, Ncoords))
    eps_yx_tire= zeros((Ntires, Ncoords))
    eps_yy_tire= zeros((Ntires, Ncoords))
    eps_yz_tire= zeros((Ntires, Ncoords))
    eps_zx_tire= zeros((Ntires, Ncoords))
    eps_zy_tire= zeros((Ntires, Ncoords))
    eps_zz_tire= zeros((Ntires, Ncoords))

    sig_xx_tire= zeros((Ntires, Ncoords))
    sig_xy_tire= zeros((Ntires, Ncoords))
    sig_xz_tire= zeros((Ntires, Ncoords))
    sig_yx_tire= zeros((Ntires, Ncoords))
    sig_yy_tire= zeros((Ntires, Ncoords))
    sig_yz_tire= zeros((Ntires, Ncoords))
    sig_zx_tire= zeros((Ntires, Ncoords))
    sig_zy_tire= zeros((Ntires, Ncoords))
    sig_zz_tire= zeros((Ntires, Ncoords))



    RZTheta = zeros((Ncoords, 3))
    numZ = len(ZC_in)
    numR = len(RM_ax[:, 0])

    RotM = zeros((3, 3))
    STRN_rtz = zeros((3, 3))
    STRS_rtz = zeros((3, 3))

    for nn in range(Ntires):
        for j in range(numZ):
            for k in range(numR):
                RZTheta[j * numR + k, 0] = RM_ax[k, nn]
                RZTheta[j * numR + k, 1] = ZC_in[j]
                RZTheta[j * numR + k, 2] = theta_ax[k, nn]/180*pi

        for mm in range(Ncoords):
            # CONVERT TO CARTESIAN COORDINATE SYSTEM
            # RotM0 = vstack((
            #     [cos(ZthetaMrad[mm, 1]) , -sin(ZthetaMrad[mm, 1]) , 0],
            #     [sin(ZthetaMrad[mm, 1]) ,  cos(ZthetaMrad[mm, 1]) , 0],
            #     [0                      ,  0                      , 1]))
            # STRN_rtz0 = vstack((
            #     [eps_r[nn, mm]    , 0                   , eps_rz[nn, mm]],
            #     [0               , eps_t[nn, mm]        , 0],
            #     [eps_rz[nn, mm]   , 0                   , eps_z[nn, mm]]))

            RotM[0, 0] =  cos(RZTheta[mm, 2])
            RotM[0, 1] = -sin(RZTheta[mm, 2])
            RotM[1, 0] =  sin(RZTheta[mm, 2])
            RotM[1, 1] =  cos(RZTheta[mm, 2])
            RotM[2, 2] = 1

            ind = where(rz[:, 1] == RZTheta[mm, 1])[0]
            r2 = rz[ind, 0]
            strain_r2 = strain_r[ind]
            strain_t2 = strain_t[ind]
            strain_z2 = strain_z[ind]
            strain_rz2 = strain_rz[ind]

            stress_r2 = stress_r[ind]
            stress_t2 = stress_t[ind]
            stress_z2 = stress_z[ind]
            stress_rz2 = stress_rz[ind]


            r0 = RZTheta[mm, 0]
            eps_r2 = f_lin_interp1(r2, strain_r2, r0)
            eps_t2 = f_lin_interp1(r2, strain_t2, r0)
            eps_z2 = f_lin_interp1(r2, strain_z2, r0)
            eps_rz2 = f_lin_interp1(r2, strain_rz2, r0)

            sig_r2 = f_lin_interp1(r2, stress_r2, r0)
            sig_t2 = f_lin_interp1(r2, stress_t2, r0)
            sig_z2 = f_lin_interp1(r2, stress_z2, r0)
            sig_rz2 = f_lin_interp1(r2, stress_rz2, r0)

            STRN_rtz[0, 0] = eps_r2
            STRN_rtz[0, 2] = eps_rz2
            STRN_rtz[1, 1] = eps_t2
            STRN_rtz[2, 0] = eps_rz2
            STRN_rtz[2, 2] = eps_z2

            STRS_rtz[0, 0] = sig_r2
            STRS_rtz[0, 2] = sig_rz2
            STRS_rtz[1, 1] = sig_t2
            STRS_rtz[2, 0] = sig_rz2
            STRS_rtz[2, 2] = sig_z2


            STRN_xyz: array = RotM @ STRN_rtz @ RotM.T
            STRS_xyz: array = RotM @ STRS_rtz @ RotM.T


            eps_xx_tire[nn, mm] = STRN_xyz[0, 0]
            eps_xy_tire[nn, mm] = STRN_xyz[0, 1]
            eps_xz_tire[nn, mm] = STRN_xyz[0, 2]
            eps_yx_tire[nn, mm] = STRN_xyz[1, 0]
            eps_yy_tire[nn, mm] = STRN_xyz[1, 1]
            eps_yz_tire[nn, mm] = STRN_xyz[1, 2]
            eps_zx_tire[nn, mm] = STRN_xyz[2, 0]
            eps_zy_tire[nn, mm] = STRN_xyz[2, 1]
            eps_zz_tire[nn, mm] = STRN_xyz[2, 2]

            sig_xx_tire[nn, mm] = STRS_xyz[0, 0]
            sig_xy_tire[nn, mm] = STRS_xyz[0, 1]
            sig_xz_tire[nn, mm] = STRS_xyz[0, 2]
            sig_yx_tire[nn, mm] = STRS_xyz[1, 0]
            sig_yy_tire[nn, mm] = STRS_xyz[1, 1]
            sig_yz_tire[nn, mm] = STRS_xyz[1, 2]
            sig_zx_tire[nn, mm] = STRS_xyz[2, 0]
            sig_zy_tire[nn, mm] = STRS_xyz[2, 1]
            sig_zz_tire[nn, mm] = STRS_xyz[2, 2]

    # Actual superposition, i.e., adding the strains and stresses due to the multiple tires
    eps_xx = sum(eps_xx_tire, axis=0)
    eps_xy = sum(eps_xy_tire, axis=0)
    eps_xz = sum(eps_xz_tire, axis=0)
    eps_yx = sum(eps_yx_tire, axis=0)
    eps_yy = sum(eps_yy_tire, axis=0)
    eps_yz = sum(eps_yz_tire, axis=0)
    eps_zx = sum(eps_zx_tire, axis=0)
    eps_zy = sum(eps_zy_tire, axis=0)
    eps_zz = sum(eps_zz_tire, axis=0)

    sig_xx = sum(sig_xx_tire, axis=0)
    sig_xy = sum(sig_xy_tire, axis=0)
    sig_xz = sum(sig_xz_tire, axis=0)
    sig_yx = sum(sig_yx_tire, axis=0)
    sig_yy = sum(sig_yy_tire, axis=0)
    sig_yz = sum(sig_yz_tire, axis=0)
    sig_zx = sum(sig_zx_tire, axis=0)
    sig_zy = sum(sig_zy_tire, axis=0)
    sig_zz = sum(sig_zz_tire, axis=0)

    # Find the principal strains, strains at directions where shear is zero
    STRN_xyz3D = zeros((Ncoords, 3, 3))
    STRN_xyz3D[:, 0, 0] = eps_xx.copy()
    STRN_xyz3D[:, 0, 1] = eps_xy.copy()
    STRN_xyz3D[:, 0, 2] = eps_xz.copy()

    STRN_xyz3D[:, 1, 0] = eps_yx.copy()
    STRN_xyz3D[:, 1, 1] = eps_yy.copy()
    STRN_xyz3D[:, 1, 2] = eps_yz.copy()

    STRN_xyz3D[:, 2, 0] = eps_zx.copy()
    STRN_xyz3D[:, 2, 1] = eps_zy.copy()
    STRN_xyz3D[:, 2, 2] = eps_zz.copy()

    #  Compute the principal strains
    eps_princ = zeros((Ncoords))
    for i in range(Ncoords):
        eig1 = linalg.eigvals(STRN_xyz3D[i])
        eps_princ[i] = max(eig1) # maximum principal strain

    return eps_xx, eps_xy, eps_xz, eps_yx, eps_yy, eps_yz, eps_zx, eps_zy, eps_zz, sig_xx, sig_xy, sig_xz, sig_yx, sig_yy, sig_yz, sig_zx, sig_zy, sig_zz, eps_princ

@njit
def ff_superpose_strains(RM_ax, theta_ax, ZC_in, strain_r, strain_t, strain_z, strain_rz, rz):


    Ntires  = len(RM_ax[0])
    Ncoords = len(RM_ax) * len(ZC_in)

    # eps_r = zeros((Ntires, Ncoords))
    # eps_t = zeros((Ntires, Ncoords))
    # eps_z = zeros((Ntires, Ncoords))
    # eps_rz= zeros((Ntires, Ncoords))

    # xx: transverse direction (perpendicular to traffic)
    # yy: longitudinal direction (traffic movement direction)
    # zz: depth
    # (+): tension
    # (-): compression

    eps_xx_tire= zeros((Ntires, Ncoords))
    eps_xy_tire= zeros((Ntires, Ncoords))
    eps_xz_tire= zeros((Ntires, Ncoords))
    eps_yx_tire= zeros((Ntires, Ncoords))
    eps_yy_tire= zeros((Ntires, Ncoords))
    eps_yz_tire= zeros((Ntires, Ncoords))
    eps_zx_tire= zeros((Ntires, Ncoords))
    eps_zy_tire= zeros((Ntires, Ncoords))
    eps_zz_tire= zeros((Ntires, Ncoords))

    RZTheta = zeros((Ncoords, 3))
    numZ = len(ZC_in)
    numR = len(RM_ax[:, 0])

    RotM = zeros((3, 3))
    STRN_rtz = zeros((3, 3))



    for nn in range(Ntires):

        # GENERATE AN R-Z COMBINATION FOR EACH TIRE
        # ZRM0 = array(meshgrid(ZC_in, RM_ax[:, nn])).T.reshape(-1, 2)
        # ZThetaM0 = array(meshgrid(ZC_in, theta_ax[:, nn])).T.reshape(-1, 2)
        # ZthetaMrad0 = ZThetaM0/180*pi

        for j in range(numZ):
            for k in range(numR):
                RZTheta[j * numR + k, 0] = RM_ax[k, nn]
                RZTheta[j * numR + k, 1] = ZC_in[j]
                RZTheta[j * numR + k, 2] = theta_ax[k, nn]/180*pi


        # # INTERPOLATE TO FIND STRAINS AT EACH POINT DUE TO EACH TIRE nn
        # eps_r[nn] = griddata(rz, strain_r,  (RZTheta[:, 0], RZTheta[:, 1]), method='linear')
        # eps_t[nn] = griddata(rz, strain_t,  (RZTheta[:, 0], RZTheta[:, 1]), method='linear')
        # eps_z[nn] = griddata(rz, strain_z,  (RZTheta[:, 0], RZTheta[:, 1]), method='linear')
        # eps_rz[nn]= griddata(rz, strain_rz, (RZTheta[:, 0], RZTheta[:, 1]), method='linear')


        for mm in range(Ncoords):
            # CONVERT TO CARTESIAN COORDINATE SYSTEM
            # RotM0 = vstack((
            #     [cos(ZthetaMrad[mm, 1]) , -sin(ZthetaMrad[mm, 1]) , 0],
            #     [sin(ZthetaMrad[mm, 1]) ,  cos(ZthetaMrad[mm, 1]) , 0],
            #     [0                      ,  0                      , 1]))
            # STRN_rtz0 = vstack((
            #     [eps_r[nn, mm]    , 0                   , eps_rz[nn, mm]],
            #     [0               , eps_t[nn, mm]        , 0],
            #     [eps_rz[nn, mm]   , 0                   , eps_z[nn, mm]]))

            RotM[0, 0] =  cos(RZTheta[mm, 2])
            RotM[0, 1] = -sin(RZTheta[mm, 2])
            RotM[1, 0] =  sin(RZTheta[mm, 2])
            RotM[1, 1] =  cos(RZTheta[mm, 2])
            RotM[2, 2] = 1

            # STRN_rtz[0, 0] = eps_r[nn, mm]
            # STRN_rtz[0, 2] = eps_rz[nn, mm]
            # STRN_rtz[1, 1] = eps_t[nn, mm]
            # STRN_rtz[2, 0] = eps_rz[nn, mm]
            # STRN_rtz[2, 2] = eps_z[nn, mm]

            ind = where(rz[:, 1] == RZTheta[mm, 1])[0]
            r2 = rz[ind, 0]
            strain_r2 = strain_r[ind]
            strain_t2 = strain_t[ind]
            strain_z2 = strain_z[ind]
            strain_rz2 = strain_rz[ind]

            r0 = RZTheta[mm, 0]
            eps_r2 = f_lin_interp1(r2, strain_r2, r0)
            eps_t2 = f_lin_interp1(r2, strain_t2, r0)
            eps_z2 = f_lin_interp1(r2, strain_z2, r0)
            eps_rz2 = f_lin_interp1(r2, strain_rz2, r0)

            STRN_rtz[0, 0] = eps_r2
            STRN_rtz[0, 2] = eps_rz2
            STRN_rtz[1, 1] = eps_t2
            STRN_rtz[2, 0] = eps_rz2
            STRN_rtz[2, 2] = eps_z2

            # print(f'rtruth = {eps_r[nn, mm]}, rnew = {eps_r2}')
            # print(f'rtruth = {eps_t[nn, mm]}, rnew = {eps_t2}')
            # print(f'rtruth = {eps_z[nn, mm]}, rnew = {eps_z2}')
            # print(f'rtruth = {eps_rz[nn, mm]}, rnew = {eps_rz2}')

            STRN_xyz: array = RotM @ STRN_rtz @ RotM.T

            eps_xx_tire[nn, mm] = STRN_xyz[0, 0]
            eps_xy_tire[nn, mm] = STRN_xyz[0, 1]
            eps_xz_tire[nn, mm] = STRN_xyz[0, 2]
            eps_yx_tire[nn, mm] = STRN_xyz[1, 0]
            eps_yy_tire[nn, mm] = STRN_xyz[1, 1]
            eps_yz_tire[nn, mm] = STRN_xyz[1, 2]
            eps_zx_tire[nn, mm] = STRN_xyz[2, 0]
            eps_zy_tire[nn, mm] = STRN_xyz[2, 1]
            eps_zz_tire[nn, mm] = STRN_xyz[2, 2]

            # inan = isnan(STRN_xyz).any()
            # if inan:
            #     print(f'ERROR!!! ONE OF THE STRAIN VALUES IS NAN: {eps_r[nn, mm]}')
            #     ZRM[mm,nn]
            #     maxrz = amax(rz, axis=0)
            #     maxZRM = amax(ZRM, axis=0)
            #     inan

    # Actual superposition, i.e., adding the strains due to the multiple tires
    eps_xx = sum(eps_xx_tire, axis=0)
    eps_xy = sum(eps_xy_tire, axis=0)
    eps_xz = sum(eps_xz_tire, axis=0)
    eps_yx = sum(eps_yx_tire, axis=0)
    eps_yy = sum(eps_yy_tire, axis=0)
    eps_yz = sum(eps_yz_tire, axis=0)
    eps_zx = sum(eps_zx_tire, axis=0)
    eps_zy = sum(eps_zy_tire, axis=0)
    eps_zz = sum(eps_zz_tire, axis=0)

    # Find the principal strains, strains at directions where shear is zero
    STRN_xyz3D = zeros((Ncoords, 3, 3))
    STRN_xyz3D[:, 0, 0] = eps_xx.copy()
    STRN_xyz3D[:, 0, 1] = eps_xy.copy()
    STRN_xyz3D[:, 0, 2] = eps_xz.copy()

    STRN_xyz3D[:, 1, 0] = eps_yx.copy()
    STRN_xyz3D[:, 1, 1] = eps_yy.copy()
    STRN_xyz3D[:, 1, 2] = eps_yz.copy()

    STRN_xyz3D[:, 2, 0] = eps_zx.copy()
    STRN_xyz3D[:, 2, 1] = eps_zy.copy()
    STRN_xyz3D[:, 2, 2] = eps_zz.copy()

    #  Compute the principal strains
    eps_princ = zeros((Ncoords, 1))
    for i in range(Ncoords):
        eps_princ[i] = max(linalg.eigvals(STRN_xyz3D[i]))  # maximum principal strain


    return eps_xx, eps_xy, eps_xz, eps_yx, eps_yy, eps_yz, eps_zx, eps_zy, eps_zz, eps_princ