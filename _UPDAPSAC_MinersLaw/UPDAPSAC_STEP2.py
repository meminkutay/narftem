# import numpy as np
from numpy import zeros, vstack, hstack, amax, round, where, array, unique, isnan, arange
import numpy as np
from MatLEA_MAIN_CODEMOD import MatLEA
import time
from f_njit_functions import ff_superpose_strains_stresses, ff_superpose_strains, f_BUFC_CritStrains, \
    f_calc_damage_at_different_loads, f_TDFC_CritStrains, f_rut_CritStrains, f_cal_ACRut, f_cal_BASERut, \
    f_cal_SUBGRut, f_calc_IRI, f_calc_damage_at_different_loadsCSM
from math import exp, log10
import updapsutil as utl


def sigdigit(num, sigd):
    return round(num * 10 ** sigd) / (10 ** sigd)


def ff_process_superposition1(RM_ax, theta_ax, ZC_in, strain_r, strain_t, strain_z, strain_rz, rz):
    eps_xx, eps_xy, eps_xz, eps_yx, eps_yy, eps_yz, eps_zx, eps_zy, eps_zz, eps_princ = ff_superpose_strains(RM_ax,
                                                                                                             theta_ax,
                                                                                                             ZC_in,
                                                                                                             strain_r,
                                                                                                             strain_t,
                                                                                                             strain_z,
                                                                                                             strain_rz,
                                                                                                             rz)

    out_resp = {'eps_xx': eps_xx, 'eps_xy': eps_xy, 'eps_xz': eps_xz,
                'eps_yx': eps_yx, 'eps_yy': eps_yy, 'eps_yz': eps_yz,
                'eps_zx': eps_zx, 'eps_zy': eps_zy, 'eps_zz': eps_zz, 'eps_princ': eps_princ}

    return out_resp


def ff_process_superposition2(RM_ax, theta_ax, ZC_in, strain_r, strain_t, strain_z, strain_rz, stress_r, stress_t,
                              stress_z, stress_rz, rz):
    eps_xx, eps_xy, eps_xz, eps_yx, eps_yy, eps_yz, eps_zx, eps_zy, eps_zz, \
        sig_xx, sig_xy, sig_xz, sig_yx, sig_yy, sig_yz, sig_zx, sig_zy, sig_zz, eps_princ = ff_superpose_strains_stresses(
        RM_ax,
        theta_ax,
        ZC_in,
        strain_r,
        strain_t,
        strain_z,
        strain_rz,
        stress_r,
        stress_t,
        stress_z,
        stress_rz,
        rz)

    out_resp = {'sig_xx': sig_xx, 'sig_xy': sig_xy, 'sig_xz': sig_xz,
                'sig_yx': sig_yx, 'sig_yy': sig_yy, 'sig_yz': sig_yz,
                'sig_zx': sig_zx, 'sig_zy': sig_zy, 'sig_zz': sig_zz,
                'eps_xx': eps_xx, 'eps_xy': eps_xy, 'eps_xz': eps_xz,
                'eps_yx': eps_yx, 'eps_yy': eps_yy, 'eps_yz': eps_yz,
                'eps_zx': eps_zx, 'eps_zy': eps_zy, 'eps_zz': eps_zz,
                'eps_princ': eps_princ}

    return out_resp


def f_struc_resp_ACGB(Step1Output):
    analysisLoc = Step1Output['analysisLoc']
    calibCoeffs = Step1Output['calibCoeffs']
    CR_in = Step1Output['CR_in']
    Esubdum_M = Step1Output['Esubdum_M']
    FreezeIndex = Step1Output['FreezeIndex']
    GWT = Step1Output['GWT']
    hbedrock = Step1Output['hbedrock']
    IBASEsub = Step1Output['IBASEsub']
    LoadsLb_qd = Step1Output['qdLoadsLb']
    LoadsLb_si = Step1Output['siLoadsLb']
    LoadsLb_ta = Step1Output['taLoadsLb']
    LoadsLb_tr = Step1Output['trLoadsLb']
    numMo = Step1Output['numMo']
    numQuint = Step1Output['numQuint']
    P200_subg = Step1Output['P200_subg']
    P4_subg = Step1Output['P4_subg']
    PR_sub = Step1Output['PR_sub']
    Prec_in_an_avg = Step1Output['Prec_in_an_avg']
    RC = Step1Output['RC']
    TC_mo = Step1Output['TC_mo']
    TH_sub = Step1Output['TH_sub']
    TH_subAC = Step1Output['TH_subAC']
    TH_subBASE = Step1Output['TH_subBASE']
    tirePressure = Step1Output['tirePressure']
    traf_NCor = Step1Output['traf_NCor']
    Tsubdum_M = Step1Output['Tsubdum_M']
    X_Wander = Step1Output['X_Wander']
    ZC_in_subM = Step1Output['ZC_in_subM']
    zi_AC_bu_fat = Step1Output['zi_AC_bu_fat']
    zi_AC_td_fat = Step1Output['zi_AC_td_fat']
    zi_sub_inAC = Step1Output['zi_sub_inAC']
    zi_sub_inBASE = Step1Output['zi_sub_inBASE']
    zi_sub_inSUBG = Step1Output['zi_sub_inSUBG']

    ZC_in = ZC_in_subM[0]
    NL_sub = len(TH_sub) + 1

    # Visualization setting.
    PLTSTR = None  # PLTSTR is the visualization variable which could be 'STRAINS', 'STRESSES', 'DISPLACEMENTS',
    #   or an empty string for disable the visualization part.

    Ez = zeros((numQuint * numMo, NL_sub + 2))
    T_Qnt = zeros((numQuint * numMo, NL_sub + 2))

    # Create a big 2D matrix where columns are the layers and rows are Month/Quintile combination.
    n = 0
    for iMo in range(numMo):
        for jQnt in range(numQuint):
            colno = jQnt + 3
            Ez[n, :] = hstack((iMo, jQnt, Esubdum_M[iMo, :, colno]))
            T_Qnt[n, :] = hstack((iMo, jQnt, Tsubdum_M[iMo, :, colno]))
            n = n + 1

    def run_MatLEA(ZC_in_subM, RC, tirePressure, CR_in, Ezi, PR_sub, TH_sub, PLTSTR, analysisLoc, ZC_in):
        iMo = int(Ezi[0])
        jQnt = int(Ezi[1])
        Ez2 = Ezi[2:]
        ttt = time.time()
        outAxisymm = MatLEA(ZC_in_subM, RC, tirePressure, CR_in, Ez2, PR_sub, TH_sub, PLTSTR)

        strain_r = outAxisymm['strain_r']
        strain_t = outAxisymm['strain_t']
        strain_z = outAxisymm['strain_z']
        strain_rz = outAxisymm['strain_rz']

        # stress_r = outAxisymm['stress_r']
        # stress_t = outAxisymm['stress_t']
        # stress_z = outAxisymm['stress_z']
        # stress_rz = outAxisymm['stress_rz']

        rz = vstack((outAxisymm['r'], outAxisymm['z'])).T

        # Perform superposition. Note that the def 'ff_process_superposition' is used so that
        # 'ff_superpose_strains_stresses' can be used in @njit to speed up processing time
        # out_resp_si = ff_process_superposition2(analysisLoc['RM_si'], analysisLoc['theta_si'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz,
        #                                         stress_r, stress_t, stress_z, stress_rz, rz)
        # out_resp_ta = ff_process_superposition2(analysisLoc['RM_ta'], analysisLoc['theta_ta'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz,
        #                                         stress_r, stress_t, stress_z, stress_rz, rz)
        # out_resp_tr = ff_process_superposition2(analysisLoc['RM_tr'], analysisLoc['theta_tr'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz,
        #                                         stress_r, stress_t, stress_z, stress_rz, rz)
        # out_resp_qd = ff_process_superposition2(analysisLoc['RM_qd'], analysisLoc['theta_qd'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz,
        #                                         stress_r, stress_t, stress_z, stress_rz, rz)

        out_resp_si = ff_process_superposition1(analysisLoc['RM_si'], analysisLoc['theta_si'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz, rz)
        out_resp_ta = ff_process_superposition1(analysisLoc['RM_ta'], analysisLoc['theta_ta'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz, rz)
        out_resp_tr = ff_process_superposition1(analysisLoc['RM_tr'], analysisLoc['theta_tr'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz, rz)
        out_resp_qd = ff_process_superposition1(analysisLoc['RM_qd'], analysisLoc['theta_qd'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz, rz)

        # out_resp = {'out_resp_si': out_resp_si, 'out_resp_ta': out_resp_ta, 'out_resp_tr': out_resp_tr,
        #             'out_resp_qd': out_resp_qd}
        # savemat("out_resp.mat", out_resp)
        # print(f'MatLEA has been ran for Mo: {Ezi[0]} - Qnt: {Ezi[1]}')
        return outAxisymm, out_resp_si, out_resp_ta, out_resp_tr, out_resp_qd, iMo, jQnt

    # location of bottom up cracking starting layer
    Ind1 = where(ZC_in_subM[0] == zi_AC_bu_fat)[0][0]
    Ifcbu = ZC_in_subM[2][Ind1]

    # Without parallel processing
    start = time.perf_counter()
    print(' --> Starting structural response computations...')
    iQnt = 0
    iMntCum = 0  # month no, cumulative, starting 0
    NumEz = len(Ez)
    D_bu_si_cum = zeros((NumEz))
    D_bu_ta_cum = zeros((NumEz))
    D_bu_tr_cum = zeros((NumEz))
    D_bu_qd_cum = zeros((NumEz))
    Dbu_cum = zeros((NumEz))
    FCbu = zeros((NumEz))

    D_td_si_cum = zeros((NumEz))
    D_td_ta_cum = zeros((NumEz))
    D_td_tr_cum = zeros((NumEz))
    D_td_qd_cum = zeros((NumEz))
    Dtd_cum = zeros((NumEz))
    FCtd = zeros((NumEz))

    numACsub = len(TH_subAC)
    numBASEsub = len(TH_subBASE)
    numSUBGsub = 2

    rut_AC = zeros((NumEz, numACsub))

    xyz_si = analysisLoc['xyz_si']
    xyz_ta = analysisLoc['xyz_ta']
    xyz_tr = analysisLoc['xyz_tr']
    xyz_qd = analysisLoc['xyz_qd']

    SI_X = analysisLoc['SI_X']  # Number of unique X coordinates in the tire/axle coordinate matrix
    SJ2_Y = analysisLoc['SJ2_Y']  # Number of unique Y coordinates in the tire/axle coordinate matrix
    SJ3_Y = analysisLoc['SJ3_Y']  # Number of unique Y coordinates in the tire/axle coordinate matrix
    SJ4_Y = analysisLoc['SJ4_Y']  # Number of unique Y coordinates in the tire/axle coordinate matrix

    numX_si = len(unique(
        xyz_si[:, 0])) * 2 - 1  # number of X coordinates (perpendicular to traffic direction, same for all axles)

    epLAST_AC = zeros((numACsub, numX_si))  # plastic strain from the last (previous) cycle, for each AC sublayer
    rut_tot_AC = zeros(NumEz)

    rut_BASE = zeros((NumEz, numBASEsub))
    epLAST_BASE = zeros((numBASEsub, numX_si))  # plastic strain from the last (previous) cycle, for each BASE sublayer
    rut_tot_BASE = zeros(NumEz)

    epLAST_SUBG = zeros(
        (numSUBGsub, numX_si))  # plastic strain from the last (previous) cycle, for each point in Subgrade
    rut_tot_SUBG = zeros(NumEz)

    rutTOT = zeros(NumEz)
    IRI = zeros(NumEz)

    FCbu_mo = zeros(numMo)
    FCtd_mo = zeros(numMo)
    rut_tot_AC_mo = zeros(numMo)
    rut_tot_BASE_mo = zeros(numMo)
    rut_tot_SUBG_mo = zeros(numMo)
    rutTOT_mo = zeros(numMo)
    IRI_mo = zeros(numMo)

    for n in range(NumEz):  # row number for Ez, corresponding to each month and each quintile
        Ezi = Ez[n]
        outAxisymm, out_resp_si, out_resp_ta, \
            out_resp_tr, out_resp_qd, iMo, jQnt = run_MatLEA(ZC_in_subM, RC, tirePressure, CR_in, Ezi, PR_sub, TH_sub,
                                                             PLTSTR, analysisLoc, ZC_in)
        eps_xx_si = out_resp_si['eps_xx']
        eps_xx_ta = out_resp_ta['eps_xx']
        eps_xx_tr = out_resp_tr['eps_xx']
        eps_xx_qd = out_resp_qd['eps_xx']

        eps_yy_si = out_resp_si['eps_yy']
        eps_yy_ta = out_resp_ta['eps_yy']
        eps_yy_tr = out_resp_tr['eps_yy']
        eps_yy_qd = out_resp_qd['eps_yy']

        eps_zz_si = out_resp_si['eps_zz']
        eps_zz_ta = out_resp_ta['eps_zz']
        eps_zz_tr = out_resp_tr['eps_zz']
        eps_zz_qd = out_resp_qd['eps_zz']

        eps_princ_si = out_resp_si['eps_princ']
        eps_princ_ta = out_resp_ta['eps_princ']
        eps_princ_tr = out_resp_tr['eps_princ']
        eps_princ_qd = out_resp_qd['eps_princ']

        EbuAC = Ezi[Ifcbu + 1]
        EtdAC = Ezi[2]
        ErutBASE = Ezi[IBASEsub + 2]
        ErutSUBG = Ezi[-1]

        # AC bottom-up cracking critical strains
        e_buc1, e_buc2, e_buc3, e_buc4, Xbu1, Xbu2, Xbu3, Xbu4 = f_BUFC_CritStrains(eps_xx_si, eps_yy_si,
                                                                                    eps_xx_ta, eps_yy_ta,
                                                                                    eps_xx_tr, eps_yy_tr,
                                                                                    eps_xx_qd, eps_yy_qd,
                                                                                    xyz_si, xyz_ta, xyz_tr, xyz_qd,
                                                                                    zi_AC_bu_fat, SI_X, SJ2_Y, SJ3_Y,
                                                                                    SJ4_Y)
        # AC top-down cracking critical strains
        e_tdc1, e_tdc2, e_tdc3, e_tdc4, Xtd1, Xtd2, Xtd3, Xtd4 = f_TDFC_CritStrains(eps_princ_si,
                                                                                    eps_princ_ta,
                                                                                    eps_princ_tr,
                                                                                    eps_princ_qd,
                                                                                    xyz_si, xyz_ta, xyz_tr, xyz_qd,
                                                                                    zi_AC_td_fat)
        # AC rut critical strains
        e_rutc1, e_rutc2, e_rutc3, e_rutc4, Xrutc1, Xrutc2, Xrutc3, Xrutc4 = f_rut_CritStrains(eps_zz_si,
                                                                                               eps_zz_ta,
                                                                                               eps_zz_tr,
                                                                                               eps_zz_qd,
                                                                                               xyz_si, xyz_ta, xyz_tr,
                                                                                               xyz_qd,
                                                                                               zi_sub_inAC)
        # Base rut critical strains
        e_rutb1, e_rutb2, e_rutb3, e_rutb4, Xrutb1, Xrutb2, Xrutb3, Xrutb4 = f_rut_CritStrains(eps_zz_si,
                                                                                               eps_zz_ta,
                                                                                               eps_zz_tr,
                                                                                               eps_zz_qd,
                                                                                               xyz_si, xyz_ta, xyz_tr,
                                                                                               xyz_qd,
                                                                                               zi_sub_inBASE)
        # Subrade rut critical strains
        e_ruts1, e_ruts2, e_ruts3, e_ruts4, Xruts1, Xruts2, Xruts3, Xruts4 = f_rut_CritStrains(eps_zz_si,
                                                                                               eps_zz_ta,
                                                                                               eps_zz_tr,
                                                                                               eps_zz_qd,
                                                                                               xyz_si, xyz_ta, xyz_tr,
                                                                                               xyz_qd,
                                                                                               zi_sub_inSUBG)

        # AC rutting computation
        rut_AC[n], epLAST_AC = f_cal_ACRut(e_rutc1, e_rutc2, e_rutc3, e_rutc4, Xrutc1, Xrutc2, Xrutc3, Xrutc4,
                                           LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                           traf_NCor['Nc_acrut_si'][iMntCum],
                                           traf_NCor['Nc_acrut_ta'][iMntCum],
                                           traf_NCor['Nc_acrut_tr'][iMntCum],
                                           traf_NCor['Nc_acrut_qd'][iMntCum],
                                           calibCoeffs['kriz'], calibCoeffs['Briz'], calibCoeffs['kzrut'],
                                           T_Qnt[n], TH_subAC, X_Wander, epLAST_AC, numACsub)

        rut_tot_AC[n] = sum(rut_AC[n])

        # Base rutting computation
        rut_BASE[n], epLAST_BASE = f_cal_BASERut(e_rutb1, e_rutb2, e_rutb3, e_rutb4, Xrutb1, Xrutb2, Xrutb3, Xrutb4,
                                                 LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                 traf_NCor['Nc_baserut_si'][iMntCum],
                                                 traf_NCor['Nc_baserut_ta'][iMntCum],
                                                 traf_NCor['Nc_baserut_tr'][iMntCum],
                                                 traf_NCor['Nc_baserut_qd'][iMntCum],
                                                 calibCoeffs['BriBsub'], TH_subBASE, X_Wander, epLAST_BASE, numBASEsub,
                                                 ErutBASE, GWT)
        rut_tot_BASE[n] = sum(rut_BASE[n])

        # subgrade rutting computation
        rut_tot_SUBG[n], epLAST_SUBG = f_cal_SUBGRut(e_ruts1, e_ruts2, e_ruts3, e_ruts4, Xruts1, Xruts2, Xruts3, Xruts4,
                                                     LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                     traf_NCor['Nc_subgrut_si'][iMntCum],
                                                     traf_NCor['Nc_subgrut_ta'][iMntCum],
                                                     traf_NCor['Nc_subgrut_tr'][iMntCum],
                                                     traf_NCor['Nc_subgrut_qd'][iMntCum],
                                                     calibCoeffs['BriS'], X_Wander, epLAST_SUBG, numSUBGsub, ErutSUBG,
                                                     GWT, hbedrock)

        rutTOT[n] = rut_tot_AC[n] + rut_tot_BASE[n] + rut_tot_SUBG[n]

        # Compute bottom-up damage
        D_bu1_mx, D_bu2_mx, \
            D_bu3_mx, D_bu4_mx = f_calc_damage_at_different_loads(e_buc1, e_buc2, e_buc3, e_buc4,
                                                                  Xbu1, Xbu2, Xbu3, Xbu4, EbuAC,
                                                                  LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                                  traf_NCor['Nc_bufc_si'][iMntCum][0],
                                                                  traf_NCor['Nc_bufc_ta'][iMntCum][0],
                                                                  traf_NCor['Nc_bufc_tr'][iMntCum][0],
                                                                  traf_NCor['Nc_bufc_qd'][iMntCum][0],
                                                                  calibCoeffs['Bi_bu'], calibCoeffs['ki_bu'],
                                                                  calibCoeffs['C_bu'], calibCoeffs['CH_bu'], X_Wander)
        # Compute top-down damage
        D_td1_mx, D_td2_mx, \
            D_td3_mx, D_td4_mx = f_calc_damage_at_different_loads(e_tdc1, e_tdc2, e_tdc3, e_tdc4,
                                                                  Xtd1, Xtd2, Xtd3, Xtd4, EtdAC,
                                                                  LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                                  traf_NCor['Nc_tdfc_si'][iMntCum][0],
                                                                  traf_NCor['Nc_tdfc_ta'][iMntCum][0],
                                                                  traf_NCor['Nc_tdfc_tr'][iMntCum][0],
                                                                  traf_NCor['Nc_tdfc_qd'][iMntCum][0],
                                                                  calibCoeffs['Bi_td'], calibCoeffs['ki_td'],
                                                                  calibCoeffs['C_td'], calibCoeffs['CH_td'], X_Wander)

        # Cumulative damage at each Month - Quintile

        if n == 0:
            D_bu_si_cum[n] = sum(D_bu1_mx)
            D_bu_ta_cum[n] = sum(D_bu2_mx)
            D_bu_tr_cum[n] = sum(D_bu3_mx)
            D_bu_qd_cum[n] = sum(D_bu4_mx)

            D_td_si_cum[n] = sum(D_td1_mx)
            D_td_ta_cum[n] = sum(D_td2_mx)
            D_td_tr_cum[n] = sum(D_td3_mx)
            D_td_qd_cum[n] = sum(D_td4_mx)
        else:
            D_bu_si_cum[n] = D_bu_si_cum[n - 1] + sum(D_bu1_mx)
            D_bu_ta_cum[n] = D_bu_ta_cum[n - 1] + sum(D_bu2_mx)
            D_bu_tr_cum[n] = D_bu_tr_cum[n - 1] + sum(D_bu3_mx)
            D_bu_qd_cum[n] = D_bu_qd_cum[n - 1] + sum(D_bu4_mx)

            D_td_si_cum[n] = D_td_si_cum[n - 1] + sum(D_td1_mx)
            D_td_ta_cum[n] = D_td_ta_cum[n - 1] + sum(D_td2_mx)
            D_td_tr_cum[n] = D_td_tr_cum[n - 1] + sum(D_td3_mx)
            D_td_qd_cum[n] = D_td_qd_cum[n - 1] + sum(D_td4_mx)

        Dbu_cum[n] = D_bu_si_cum[n] + D_bu_ta_cum[n] + D_bu_tr_cum[n] + D_bu_qd_cum[n]
        C_TF_bu = calibCoeffs['C_TF_bu']
        Cprime_1 = calibCoeffs['Cprime_1']
        Cprime_2 = calibCoeffs['Cprime_2']
        if Dbu_cum[n] <= 0:
            FCbu[n] = 0
        else:
            FCbu[n] = ((1 / 60) * (C_TF_bu[2] / (
                    1 + exp(Cprime_1 * C_TF_bu[0] + Cprime_2 * C_TF_bu[1] * log10(Dbu_cum[n] * 100)))))

        Dtd_cum[n] = D_td_si_cum[n] + D_td_ta_cum[n] + D_td_tr_cum[n] + D_td_qd_cum[n]
        C_TF_td = calibCoeffs['C_TF_td']

        if Dtd_cum[n] <= 0:
            FCtd[n] = 0
        else:
            FCtd[n] = (10.56 * (C_TF_td[2] / (1 + exp(C_TF_td[0] - C_TF_td[1] * log10(Dtd_cum[n] * 100)))))

        # Compute the IRI
        IRIo = float(calibCoeffs['IRIo'])
        C_IRI = array(calibCoeffs['C_IRI'])
        FCtd_perc = FCtd[
                        n] / 5280 * 3 / 12 * 100  # from ft/mile to percent (%)--> TD(%) = TD(ft/mile) / 5280 ft/mile * 3 ft (wheel path width) / 12 ft (lane width) * 100
        # FCtot = FCbu[n] + FCtd_perc
        FCbu_perc = FCbu[n]
        FCtot = (FCbu_perc / 100 + FCtd_perc / 100 -
                 (FCbu_perc / 100 * FCtd_perc / 100) / 100) * 100

        IRI[n] = f_calc_IRI(IRIo, FCtot, rutTOT[n], TC_mo[iMntCum], iMntCum, FreezeIndex, Prec_in_an_avg, P200_subg,
                            P4_subg, C_IRI)
        time_measured: float = time.perf_counter() - start
        # print(f'Distresses ran for Mo: {Ezi[0]} - Qnt: {Ezi[1]}, cumulative Month = {iMntCum}, Elapsed time = {time_measured} sec')
        start = time.perf_counter()

        iQnt = iQnt + 1  # Quintile no
        if iQnt > 4:
            iQnt = 0
            FCbu_mo[iMntCum] = FCbu[n]
            FCtd_mo[iMntCum] = FCtd[n]
            rut_tot_AC_mo[iMntCum] = rut_tot_AC[n]
            rut_tot_BASE_mo[iMntCum] = rut_tot_BASE[n]
            rut_tot_SUBG_mo[iMntCum] = rut_tot_SUBG[n]
            rutTOT_mo[iMntCum] = rutTOT[n]
            IRI_mo[iMntCum] = IRI[n]

            txttoadd = f' ---> FCbu[{iMntCum}] = {FCbu[n]:.3}%, FCtd[{iMntCum}] = {FCtd[n]:.3}ft/mile, ACrut[{iMntCum}] = {rut_tot_AC[n]:.3}in, ' \
                       f'BASErut[{iMntCum}] = {rut_tot_BASE[n]:.3}in, SUBGrut[{iMntCum}] = {rut_tot_SUBG[n]:.3}in,' \
                       f'TOTALrut[{iMntCum}] = {rutTOT[n]:.3}in, IRI[{iMntCum}] = {IRI[n]:.5}in/mile \n'
            print(txttoadd)
            utl.appendtxt(Step1Output['fld_out'], Step1Output['StatusTxtFile'], txttoadd)

            iMntCum = iMntCum + 1  # cumulative month no
            #  Create a dictionary of outputs
            # out_resp = {'FCbu_mo': FCbu_mo.tolist(),
            #             'FCtd_mo': FCtd_mo.tolist(),
            #             'rut_tot_AC_mo': rut_tot_AC_mo.tolist(),
            #             'rut_tot_BASE_mo': rut_tot_BASE_mo.tolist(),
            #             'rut_tot_SUBG_mo': rut_tot_SUBG_mo.tolist(),
            #             'rutTOT_mo': rutTOT_mo.tolist(),
            #             'IRI_mo': IRI_mo.tolist()}

            # utl.save_to_output(out_resp, Step1Output['fld_out'], Step1Output['json_output_file'])

    #  Create a dictionary of outputs
    out_resp = {'FCbu_mo': FCbu_mo.tolist(),
                'FCtd_mo': FCtd_mo.tolist(),
                'rut_tot_AC_mo': rut_tot_AC_mo.tolist(),
                'rut_tot_BASE_mo': rut_tot_BASE_mo.tolist(),
                'rut_tot_SUBG_mo': rut_tot_SUBG_mo.tolist(),
                'rutTOT_mo': rutTOT_mo.tolist(),
                'IRI_mo': IRI_mo.tolist()}
    
    # out_resp['Dbu_cum'] =Dbu_cum
    # out_resp['Dtd_cum'] =Dtd_cum

    # savemat("outputs_Python_fFC_rut_MatLEA.mat", out_resp)
    # # With parallel processing
    # start = time.perf_counter()
    # num_cores = multiprocessing.cpu_count()
    # outputPythi = Parallel(n_jobs=num_cores, verbose=50)(
    #     delayed(run_MatLEA)(ZC_in_subM, RC, tirePressure, CR_in, Ezi, PR_sub, TH_sub, PLTSTR,
    #                         RM_si, RM_ta, RM_tr, RM_qd, theta_si, theta_ta, theta_tr, theta_qd, ZC_in) for Ezi in Ez)
    #
    # time_measured: float = time.perf_counter() - start
    # print(f'Elapsed time Parallel processing = {time_measured} sec')

    return out_resp


def f_struc_resp_ACCSM(Step1Output):
    analysisLoc = Step1Output['analysisLoc']
    calibCoeffs = Step1Output['calibCoeffs']
    CR_in = Step1Output['CR_in']
    Esubdum_M = Step1Output['Esubdum_M']
    FreezeIndex = Step1Output['FreezeIndex']
    GWT = Step1Output['GWT']
    hbedrock = Step1Output['hbedrock']
    IBASEsub = Step1Output['IBASEsub']
    LoadsLb_qd = Step1Output['qdLoadsLb']
    LoadsLb_si = Step1Output['siLoadsLb']
    LoadsLb_ta = Step1Output['taLoadsLb']
    LoadsLb_tr = Step1Output['trLoadsLb']
    numMo = Step1Output['numMo']
    numQuint = Step1Output['numQuint']
    P200_subg = Step1Output['P200_subg']
    P4_subg = Step1Output['P4_subg']
    PR_sub = Step1Output['PR_sub']
    Prec_in_an_avg = Step1Output['Prec_in_an_avg']
    RC = Step1Output['RC']
    TC_mo = Step1Output['TC_mo']
    TH_sub = Step1Output['TH_sub']
    TH_subAC = Step1Output['TH_subAC']
    TH_subBASE = Step1Output['TH_subBASE']
    tirePressure = Step1Output['tirePressure']
    traf_NCor = Step1Output['traf_NCor']
    Tsubdum_M = Step1Output['Tsubdum_M']
    X_Wander = Step1Output['X_Wander']
    ZC_in_subM = Step1Output['ZC_in_subM']
    zi_AC_bu_fat = Step1Output['zi_AC_bu_fat']
    zi_AC_td_fat = Step1Output['zi_AC_td_fat']
    zi_sub_inAC = Step1Output['zi_sub_inAC']
    zi_sub_inBASE = Step1Output['zi_sub_inBASE']
    zi_sub_inSUBG = Step1Output['zi_sub_inSUBG']

    CSMEmax_sub = Step1Output['CSMEmax_sub']
    CSMEmin_sub = Step1Output['CSMEmin_sub']
    CSMMoR_sub = Step1Output['CSMMoR_sub']
    ICSMsub = int(Step1Output['ICSMsub'][0])
    zi_CSM_fat = Step1Output['zi_CSM_fat']

    ZC_in = ZC_in_subM[0]
    NL_sub = len(TH_sub) + 1

    # Visualization setting.
    PLTSTR = None  # PLTSTR is the visualization variable which could be 'STRAINS', 'STRESSES', 'DISPLACEMENTS',
    #   or an empty string for disable the visualization part.

    Ez = zeros((numQuint * numMo, NL_sub + 2))
    T_Qnt = zeros((numQuint * numMo, NL_sub + 2))

    # Create a big 2D matrix where columns are the layers and rows are Month/Quintile combination.
    n = 0
    for iMo in range(numMo):
        for jQnt in range(numQuint):
            colno = jQnt + 3
            Ez[n, :] = hstack((iMo, jQnt, Esubdum_M[iMo, :, colno]))
            T_Qnt[n, :] = hstack((iMo, jQnt, Tsubdum_M[iMo, :, colno]))
            n = n + 1

    def run_MatLEA(ZC_in_subM, RC, tirePressure, CR_in, Ezi, PR_sub, TH_sub, PLTSTR, analysisLoc, ZC_in):
        iMo = int(Ezi[0])
        jQnt = int(Ezi[1])
        Ez2 = Ezi[2:]
        outAxisymm = MatLEA(ZC_in_subM, RC, tirePressure, CR_in, Ez2, PR_sub, TH_sub, PLTSTR)

        strain_r = outAxisymm['strain_r']
        strain_t = outAxisymm['strain_t']
        strain_z = outAxisymm['strain_z']
        strain_rz = outAxisymm['strain_rz']

        stress_r = outAxisymm['stress_r']
        stress_t = outAxisymm['stress_t']
        stress_z = outAxisymm['stress_z']
        stress_rz = outAxisymm['stress_rz']

        rz = vstack((outAxisymm['r'], outAxisymm['z'])).T

        # Perform superposition. Note that the def 'ff_process_superposition' is used so that
        # 'ff_superpose_strains_stresses' can be used in @njit to speed up processing time
        out_resp_si = ff_process_superposition2(analysisLoc['RM_si'], analysisLoc['theta_si'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz,
                                                stress_r, stress_t, stress_z, stress_rz, rz)
        out_resp_ta = ff_process_superposition2(analysisLoc['RM_ta'], analysisLoc['theta_ta'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz,
                                                stress_r, stress_t, stress_z, stress_rz, rz)
        out_resp_tr = ff_process_superposition2(analysisLoc['RM_tr'], analysisLoc['theta_tr'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz,
                                                stress_r, stress_t, stress_z, stress_rz, rz)
        out_resp_qd = ff_process_superposition2(analysisLoc['RM_qd'], analysisLoc['theta_qd'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz,
                                                stress_r, stress_t, stress_z, stress_rz, rz)

        # out_resp_si = ff_process_superposition1(analysisLoc['RM_si'], analysisLoc['theta_si'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz,rz)
        # out_resp_ta = ff_process_superposition1(analysisLoc['RM_ta'], analysisLoc['theta_ta'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz, rz)
        # out_resp_tr = ff_process_superposition1(analysisLoc['RM_tr'], analysisLoc['theta_tr'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz, rz)
        # out_resp_qd = ff_process_superposition1(analysisLoc['RM_qd'], analysisLoc['theta_qd'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz, rz)

        # out_resp = {'out_resp_si': out_resp_si, 'out_resp_ta': out_resp_ta, 'out_resp_tr': out_resp_tr,
        #             'out_resp_qd': out_resp_qd}
        # savemat("out_resp.mat", out_resp)
        # print(f'MatLEA has been ran for Mo: {Ezi[0]} - Qnt: {Ezi[1]}')
        return outAxisymm, out_resp_si, out_resp_ta, out_resp_tr, out_resp_qd, iMo, jQnt

    # location of bottom up cracking starting layer
    Ind1 = where(ZC_in_subM[0] == zi_AC_bu_fat)[0][0]
    Ifcbu = ZC_in_subM[2][Ind1]

    # Without parallel processing
    start = time.perf_counter()
    print(' --> Starting structural response computations...')
    iQnt = 0
    iMntCum = 0  # month no, cumulative, starting 0
    NumEz = len(Ez)
    D_bu_si_cum = zeros(NumEz)
    D_bu_ta_cum = zeros(NumEz)
    D_bu_tr_cum = zeros(NumEz)
    D_bu_qd_cum = zeros(NumEz)
    Dbu_cum = zeros(NumEz)
    FCbu = zeros(NumEz)

    D_td_si_cum = zeros(NumEz)
    D_td_ta_cum = zeros(NumEz)
    D_td_tr_cum = zeros(NumEz)
    D_td_qd_cum = zeros(NumEz)
    Dtd_cum = zeros(NumEz)
    FCtd = zeros(NumEz)

    numACsub = len(TH_subAC)
    numBASEsub = len(TH_subBASE)
    numSUBGsub = 2

    rut_AC = zeros((NumEz, numACsub))

    xyz_si = analysisLoc['xyz_si']
    xyz_ta = analysisLoc['xyz_ta']
    xyz_tr = analysisLoc['xyz_tr']
    xyz_qd = analysisLoc['xyz_qd']

    SI_X = analysisLoc['SI_X']  # Number of unique X coordinates in the tire/axle coordinate matrix
    SJ2_Y = analysisLoc['SJ2_Y']  # Number of unique Y coordinates in the tire/axle coordinate matrix
    SJ3_Y = analysisLoc['SJ3_Y']  # Number of unique Y coordinates in the tire/axle coordinate matrix
    SJ4_Y = analysisLoc['SJ4_Y']  # Number of unique Y coordinates in the tire/axle coordinate matrix

    numX_si = len(unique(
        xyz_si[:, 0])) * 2 - 1  # number of X coordinates (perpendicular to traffic direction, same for all axles)

    epLAST_AC = zeros((numACsub, numX_si))  # plastic strain from the last (previous) cycle, for each AC sublayer
    rut_tot_AC = zeros(NumEz)

    rut_BASE = zeros((NumEz, numBASEsub))
    epLAST_BASE = zeros((numBASEsub, numX_si))  # plastic strain from the last (previous) cycle, for each BASE sublayer
    rut_tot_BASE = zeros(NumEz)

    epLAST_SUBG = zeros(
        (numSUBGsub, numX_si))  # plastic strain from the last (previous) cycle, for each point in Subgrade
    rut_tot_SUBG = zeros(NumEz)

    rutTOT = zeros(NumEz)
    IRI = zeros(NumEz)

    FCbu_mo = zeros(numMo)
    FCtd_mo = zeros(numMo)
    rut_tot_AC_mo = zeros(numMo)
    rut_tot_BASE_mo = zeros(numMo)
    rut_tot_SUBG_mo = zeros(numMo)
    rutTOT_mo = zeros(numMo)
    IRI_mo = zeros(numMo)
    FCcsm_mo = zeros(numMo)

    # CSM RELATED VARIABLES
    D_csm_si_cum = zeros(NumEz)
    D_csm_ta_cum = zeros(NumEz)
    D_csm_tr_cum = zeros(NumEz)
    D_csm_qd_cum = zeros(NumEz)
    Dcsm_cum = zeros(NumEz)
    FCcsm = zeros(NumEz)  # Fatigue Cracking - CSM
    ECSMD = zeros(NumEz)  # MODULUS - CSM
    t_yr = arange(1, numMo + 1) / 12
    # RCac_mi = zeros(numMo) # reflective cracking unit response function (like creep compliance for AC)
    TRCac_mo = zeros(numMo)  # reflective cracking
    delFCcsm = zeros(numMo)
    RFCa_rc = calibCoeffs['RFCa_rc']
    RFCb_rc = calibCoeffs['RFCb_rc']

    for n in range(NumEz):  # row number for Ez, corresponding to each month and each quintile
        Ezi = Ez[n]
        # if n>0:
        #     DCSMcum[i] = DCSMcum[i-1] + DCsmcum
        # DCSMcum is a fraction, enter 0.5 for 50 %
        if n == 0:
            ECSMD[n] = CSMEmax_sub
        else:
            ECSMD[n] = CSMEmin_sub + (CSMEmax_sub - CSMEmin_sub) / (1 + exp(-4 + 14 * Dcsm_cum[n - 1]))
        Ezi[ICSMsub + 2] = ECSMD[n]

        outAxisymm, out_resp_si, out_resp_ta, \
            out_resp_tr, out_resp_qd, iMo, jQnt = run_MatLEA(ZC_in_subM, RC, tirePressure, CR_in, Ezi, PR_sub, TH_sub,
                                                             PLTSTR, analysisLoc, ZC_in)
        eps_xx_si = out_resp_si['eps_xx']
        eps_xx_ta = out_resp_ta['eps_xx']
        eps_xx_tr = out_resp_tr['eps_xx']
        eps_xx_qd = out_resp_qd['eps_xx']

        eps_yy_si = out_resp_si['eps_yy']
        eps_yy_ta = out_resp_ta['eps_yy']
        eps_yy_tr = out_resp_tr['eps_yy']
        eps_yy_qd = out_resp_qd['eps_yy']

        eps_zz_si = out_resp_si['eps_zz']
        eps_zz_ta = out_resp_ta['eps_zz']
        eps_zz_tr = out_resp_tr['eps_zz']
        eps_zz_qd = out_resp_qd['eps_zz']

        eps_princ_si = out_resp_si['eps_princ']
        eps_princ_ta = out_resp_ta['eps_princ']
        eps_princ_tr = out_resp_tr['eps_princ']
        eps_princ_qd = out_resp_qd['eps_princ']

        sig_xx_si = out_resp_si['sig_xx']
        sig_xx_ta = out_resp_ta['sig_xx']
        sig_xx_tr = out_resp_tr['sig_xx']
        sig_xx_qd = out_resp_qd['sig_xx']

        sig_yy_si = out_resp_si['sig_yy']
        sig_yy_ta = out_resp_ta['sig_yy']
        sig_yy_tr = out_resp_tr['sig_yy']
        sig_yy_qd = out_resp_qd['sig_yy']

        EbuAC = Ezi[Ifcbu + 1]
        EtdAC = Ezi[2]
        ErutBASE = Ezi[IBASEsub + 2]
        ErutSUBG = Ezi[-1]

        # CSM cracking critical stresses
        s_buc1, s_buc2, s_buc3, s_buc4, Xbu1csm, Xbu2csm, Xbu3csm, Xbu4csm = f_BUFC_CritStrains(sig_xx_si, sig_yy_si,
                                                                                                sig_xx_ta, sig_yy_ta,
                                                                                                sig_xx_tr, sig_yy_tr,
                                                                                                sig_xx_qd, sig_yy_qd,
                                                                                                xyz_si, xyz_ta, xyz_tr,
                                                                                                xyz_qd,
                                                                                                zi_CSM_fat, SI_X, SJ2_Y,
                                                                                                SJ3_Y, SJ4_Y)

        # AC bottom-up cracking critical strains
        e_buc1, e_buc2, e_buc3, e_buc4, Xbu1, Xbu2, Xbu3, Xbu4 = f_BUFC_CritStrains(eps_xx_si, eps_yy_si,
                                                                                    eps_xx_ta, eps_yy_ta,
                                                                                    eps_xx_tr, eps_yy_tr,
                                                                                    eps_xx_qd, eps_yy_qd,
                                                                                    xyz_si, xyz_ta, xyz_tr, xyz_qd,
                                                                                    zi_AC_bu_fat, SI_X, SJ2_Y, SJ3_Y,
                                                                                    SJ4_Y)
        # AC top-down cracking critical strains
        e_tdc1, e_tdc2, e_tdc3, e_tdc4, Xtd1, Xtd2, Xtd3, Xtd4 = f_TDFC_CritStrains(eps_princ_si,
                                                                                    eps_princ_ta,
                                                                                    eps_princ_tr,
                                                                                    eps_princ_qd,
                                                                                    xyz_si, xyz_ta, xyz_tr, xyz_qd,
                                                                                    zi_AC_td_fat)
        # AC rut critical strains
        e_rutc1, e_rutc2, e_rutc3, e_rutc4, Xrutc1, Xrutc2, Xrutc3, Xrutc4 = f_rut_CritStrains(eps_zz_si,
                                                                                               eps_zz_ta,
                                                                                               eps_zz_tr,
                                                                                               eps_zz_qd,
                                                                                               xyz_si, xyz_ta, xyz_tr,
                                                                                               xyz_qd,
                                                                                               zi_sub_inAC)
        # Base rut critical strains
        e_rutb1, e_rutb2, e_rutb3, e_rutb4, Xrutb1, Xrutb2, Xrutb3, Xrutb4 = f_rut_CritStrains(eps_zz_si,
                                                                                               eps_zz_ta,
                                                                                               eps_zz_tr,
                                                                                               eps_zz_qd,
                                                                                               xyz_si, xyz_ta, xyz_tr,
                                                                                               xyz_qd,
                                                                                               zi_sub_inBASE)
        # Subrade rut critical strains
        e_ruts1, e_ruts2, e_ruts3, e_ruts4, Xruts1, Xruts2, Xruts3, Xruts4 = f_rut_CritStrains(eps_zz_si,
                                                                                               eps_zz_ta,
                                                                                               eps_zz_tr,
                                                                                               eps_zz_qd,
                                                                                               xyz_si, xyz_ta, xyz_tr,
                                                                                               xyz_qd,
                                                                                               zi_sub_inSUBG)
        # calibCoeffs['CSMBeta_c2_sub'] = 1.2
        # Compute fatigue damage in the CSM layer
        D_csm1_mx, D_csm2_mx, \
            D_csm3_mx, D_csm4_mx = f_calc_damage_at_different_loadsCSM(s_buc1, s_buc2, s_buc3, s_buc4,
                                                                       Xbu1csm, Xbu2csm, Xbu3csm, Xbu4csm, CSMMoR_sub,
                                                                       LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                                       traf_NCor['Nc_csm_si'][iMntCum][0],
                                                                       traf_NCor['Nc_csm_ta'][iMntCum][0],
                                                                       traf_NCor['Nc_csm_tr'][iMntCum][0],
                                                                       traf_NCor['Nc_csm_qd'][iMntCum][0],
                                                                       calibCoeffs['CSMBeta_c1_sub'],
                                                                       calibCoeffs['CSMBeta_c2_sub'], X_Wander)

        # D_csm1_mx, D_csm2_mx, \
        # D_csm3_mx, D_csm4_mx = f_calc_damage_at_different_loadsCSM(s_buc1, s_buc2, s_buc3, s_buc4,
        #                                                         Xbu1csm, Xbu2csm, Xbu3csm, Xbu4csm, CSMMoR_sub,
        #                                                         LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
        #                                                         traf_NCor['Nc_bufc_si'][iMntCum][0],
        #                                                         traf_NCor['Nc_bufc_ta'][iMntCum][0],
        #                                                         traf_NCor['Nc_bufc_tr'][iMntCum][0],
        #                                                         traf_NCor['Nc_bufc_qd'][iMntCum][0],
        #                                                         calibCoeffs['CSMBeta_c1_sub'],
        #                                                         calibCoeffs['CSMBeta_c2_sub'], X_Wander*0)
        # AC rutting computation
        rut_AC[n], epLAST_AC = f_cal_ACRut(e_rutc1, e_rutc2, e_rutc3, e_rutc4, Xrutc1, Xrutc2, Xrutc3, Xrutc4,
                                           LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                           traf_NCor['Nc_acrut_si'][iMntCum],
                                           traf_NCor['Nc_acrut_ta'][iMntCum],
                                           traf_NCor['Nc_acrut_tr'][iMntCum],
                                           traf_NCor['Nc_acrut_qd'][iMntCum],
                                           calibCoeffs['kriz'], calibCoeffs['Briz'], calibCoeffs['kzrut'],
                                           T_Qnt[n], TH_subAC, X_Wander, epLAST_AC, numACsub)

        rut_tot_AC[n] = sum(rut_AC[n])

        # Base rutting computation
        rut_BASE[n], epLAST_BASE = f_cal_BASERut(e_rutb1, e_rutb2, e_rutb3, e_rutb4, Xrutb1, Xrutb2, Xrutb3, Xrutb4,
                                                 LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                 traf_NCor['Nc_baserut_si'][iMntCum],
                                                 traf_NCor['Nc_baserut_ta'][iMntCum],
                                                 traf_NCor['Nc_baserut_tr'][iMntCum],
                                                 traf_NCor['Nc_baserut_qd'][iMntCum],
                                                 calibCoeffs['BriBsub'], TH_subBASE, X_Wander, epLAST_BASE, numBASEsub,
                                                 ErutBASE, GWT)
        rut_tot_BASE[n] = sum(rut_BASE[n])

        # subgrade rutting computation
        rut_tot_SUBG[n], epLAST_SUBG = f_cal_SUBGRut(e_ruts1, e_ruts2, e_ruts3, e_ruts4, Xruts1, Xruts2, Xruts3, Xruts4,
                                                     LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                     traf_NCor['Nc_subgrut_si'][iMntCum],
                                                     traf_NCor['Nc_subgrut_ta'][iMntCum],
                                                     traf_NCor['Nc_subgrut_tr'][iMntCum],
                                                     traf_NCor['Nc_subgrut_qd'][iMntCum],
                                                     calibCoeffs['BriS'], X_Wander, epLAST_SUBG, numSUBGsub, ErutSUBG,
                                                     GWT, hbedrock)

        rutTOT[n] = rut_tot_AC[n] + rut_tot_BASE[n] + rut_tot_SUBG[n]

        # Compute bottom-up damage
        D_bu1_mx, D_bu2_mx, \
            D_bu3_mx, D_bu4_mx = f_calc_damage_at_different_loads(e_buc1, e_buc2, e_buc3, e_buc4,
                                                                  Xbu1, Xbu2, Xbu3, Xbu4, EbuAC,
                                                                  LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                                  traf_NCor['Nc_bufc_si'][iMntCum][0],
                                                                  traf_NCor['Nc_bufc_ta'][iMntCum][0],
                                                                  traf_NCor['Nc_bufc_tr'][iMntCum][0],
                                                                  traf_NCor['Nc_bufc_qd'][iMntCum][0],
                                                                  calibCoeffs['Bi_bu'], calibCoeffs['ki_bu'],
                                                                  calibCoeffs['C_bu'], calibCoeffs['CH_bu'], X_Wander)
        # Compute top-down damage
        D_td1_mx, D_td2_mx, \
            D_td3_mx, D_td4_mx = f_calc_damage_at_different_loads(e_tdc1, e_tdc2, e_tdc3, e_tdc4,
                                                                  Xtd1, Xtd2, Xtd3, Xtd4, EtdAC,
                                                                  LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                                  traf_NCor['Nc_tdfc_si'][iMntCum][0],
                                                                  traf_NCor['Nc_tdfc_ta'][iMntCum][0],
                                                                  traf_NCor['Nc_tdfc_tr'][iMntCum][0],
                                                                  traf_NCor['Nc_tdfc_qd'][iMntCum][0],
                                                                  calibCoeffs['Bi_td'], calibCoeffs['ki_td'],
                                                                  calibCoeffs['C_td'], calibCoeffs['CH_td'], X_Wander)

        # Cumulative damage at each Month - Quintile

        if n == 0:
            D_bu_si_cum[n] = sum(D_bu1_mx)
            D_bu_ta_cum[n] = sum(D_bu2_mx)
            D_bu_tr_cum[n] = sum(D_bu3_mx)
            D_bu_qd_cum[n] = sum(D_bu4_mx)

            D_td_si_cum[n] = sum(D_td1_mx)
            D_td_ta_cum[n] = sum(D_td2_mx)
            D_td_tr_cum[n] = sum(D_td3_mx)
            D_td_qd_cum[n] = sum(D_td4_mx)

            D_csm_si_cum[n] = sum(D_csm1_mx)
            D_csm_ta_cum[n] = sum(D_csm2_mx)
            D_csm_tr_cum[n] = sum(D_csm3_mx)
            D_csm_qd_cum[n] = sum(D_csm4_mx)


        else:
            D_bu_si_cum[n] = D_bu_si_cum[n - 1] + sum(D_bu1_mx)
            D_bu_ta_cum[n] = D_bu_ta_cum[n - 1] + sum(D_bu2_mx)
            D_bu_tr_cum[n] = D_bu_tr_cum[n - 1] + sum(D_bu3_mx)
            D_bu_qd_cum[n] = D_bu_qd_cum[n - 1] + sum(D_bu4_mx)

            D_td_si_cum[n] = D_td_si_cum[n - 1] + sum(D_td1_mx)
            D_td_ta_cum[n] = D_td_ta_cum[n - 1] + sum(D_td2_mx)
            D_td_tr_cum[n] = D_td_tr_cum[n - 1] + sum(D_td3_mx)
            D_td_qd_cum[n] = D_td_qd_cum[n - 1] + sum(D_td4_mx)

            D_csm_si_cum[n] = D_csm_si_cum[n - 1] + sum(D_csm1_mx)
            D_csm_ta_cum[n] = D_csm_ta_cum[n - 1] + sum(D_csm2_mx)
            D_csm_tr_cum[n] = D_csm_tr_cum[n - 1] + sum(D_csm3_mx)
            D_csm_qd_cum[n] = D_csm_qd_cum[n - 1] + sum(D_csm4_mx)

        Dcsm_cum[n] = D_csm_si_cum[n] + D_csm_ta_cum[n] + D_csm_tr_cum[n] + D_csm_qd_cum[n]
        # FCcsm[n] = 100/(1+exp(6-6*Dcsm_cum[n]))
        CSMC1 = 0
        CSMC2 = 68
        CSMC3 = 2
        CSMC4 = 2

        # CSMC1 = calibCoeffs['CSMC1']
        # CSMC2 = calibCoeffs['CSMC2']
        # CSMC3 = calibCoeffs['CSMC3']
        # CSMC4 = calibCoeffs['CSMC4']
        if Dcsm_cum[n] <= 0:
            FCcsm[n] = 0
        else:
            FCcsm[n] = CSMC1 + CSMC2 / (1 + exp(CSMC3 - CSMC4 * log10(Dcsm_cum[n] * 100.0)))

        Dbu_cum[n] = D_bu_si_cum[n] + D_bu_ta_cum[n] + D_bu_tr_cum[n] + D_bu_qd_cum[n]
        C_TF_bu = calibCoeffs['C_TF_bu']
        Cprime_1 = calibCoeffs['Cprime_1']
        Cprime_2 = calibCoeffs['Cprime_2']

        if Dbu_cum[n] <= 0:
            FCbu[n] = 0
        else:
            FCbu[n] = ((1 / 60) * (C_TF_bu[2] / (
                    1 + exp(Cprime_1 * C_TF_bu[0] + Cprime_2 * C_TF_bu[1] * log10(Dbu_cum[n] * 100)))))

        Dtd_cum[n] = D_td_si_cum[n] + D_td_ta_cum[n] + D_td_tr_cum[n] + D_td_qd_cum[n]
        C_TF_td = calibCoeffs['C_TF_td']
        if Dtd_cum[n] <= 0:
            FCtd[n] = 0
        else:
            FCtd[n] = (10.56 * (C_TF_td[2] / (1 + exp(C_TF_td[0] - C_TF_td[1] * log10(Dtd_cum[n] * 100)))))

        # Compute the IRI
        IRIo = float(calibCoeffs['IRIo'])
        C_IRI = array(calibCoeffs['C_IRI'])
        FCtd_perc = FCtd[
                        n] / 5280 * 3 / 12 * 100  # from ft/mile to percent (%)--> TD(%) = TD(ft/mile) / 5280 ft/mile * 3 ft (wheel path width) / 12 ft (lane width) * 100
        FCtot = FCbu[n] + FCtd_perc
        IRI[n] = f_calc_IRI(IRIo, FCtot, rutTOT[n], TC_mo[iMntCum], iMntCum, FreezeIndex, Prec_in_an_avg, P200_subg,
                            P4_subg, C_IRI)
        # time_measured: float = time.perf_counter() - start
        # print(f'Distresses ran for Mo: {Ezi[0]} - Qnt: {Ezi[1]}, cumulative Month = {iMntCum}, Elapsed time = {time_measured} sec')
        # start = time.perf_counter()

        iQnt = iQnt + 1  # Quintile no
        if iQnt > 4:
            iQnt = 0

            FCcsm_mo[iMntCum] = FCcsm[n]
            # Compute reflective cracking
            tminust = (iMntCum + 1) / 12 - t_yr[:(iMntCum + 1)]
            RCac_mi = 100. / (1 + np.exp(RFCa_rc + RFCb_rc * tminust))
            if iMntCum > 0:
                delFCcsm[iMntCum - 1] = FCcsm_mo[iMntCum] - FCcsm_mo[iMntCum - 1]
                TRCac_mo[iMntCum] = RCac_mi[:iMntCum] @ (delFCcsm[:iMntCum] / 100)

            FCbu_mo[iMntCum] = FCbu[n]
            FCtd_mo[iMntCum] = FCtd[n]
            rut_tot_AC_mo[iMntCum] = rut_tot_AC[n]
            rut_tot_BASE_mo[iMntCum] = rut_tot_BASE[n]
            rut_tot_SUBG_mo[iMntCum] = rut_tot_SUBG[n]
            rutTOT_mo[iMntCum] = rutTOT[n]
            IRI_mo[iMntCum] = IRI[n]

            txttoadd = f' ---> TRCac_mo[{iMntCum}] = {TRCac_mo[iMntCum]:.3}%, FCcsm[{iMntCum}] = {FCcsm[n]:.3}%, ECSMD [{iMntCum}] = {ECSMD[n]:.5} psi, ' \
                       f'FCbu[{iMntCum}] = {FCbu[n]:.3}%, FCtd[{iMntCum}] = {FCtd[n]:.3}ft/mile, ACrut[{iMntCum}] = {rut_tot_AC[n]:.3}in, ' \
                       f'BASErut[{iMntCum}] = {rut_tot_BASE[n]:.3}in, SUBGrut[{iMntCum}] = {rut_tot_SUBG[n]:.3}in,' \
                       f'TOTALrut[{iMntCum}] = {rutTOT[n]:.3}in, IRI[{iMntCum}] = {IRI[n]:.5}in/mile \n'
            print(txttoadd)
            utl.appendtxt(Step1Output['fld_out'], Step1Output['StatusTxtFile'], txttoadd)

            iMntCum = iMntCum + 1  # cumulative month no
            iMntCum

    #  Create a dictionary of outputs
    out_resp = {}
    # out_resp['Dbu_cum'] =Dbu_cum

    out_resp['TRCac_mo'] = TRCac_mo.tolist()
    out_resp['FCcsm_mo'] = FCcsm_mo.tolist()
    out_resp['FCbu_mo'] = FCbu_mo.tolist()
    out_resp['ECSMD'] = ECSMD.tolist()
    # out_resp['Dtd_cum'] =Dtd_cum
    out_resp['FCtd_mo'] = FCtd_mo.tolist()
    out_resp['rut_tot_AC_mo'] = rut_tot_AC_mo.tolist()
    out_resp['rut_tot_BASE_mo'] = rut_tot_BASE_mo.tolist()
    out_resp['rut_tot_SUBG_mo'] = rut_tot_SUBG_mo.tolist()
    out_resp['rutTOT_mo'] = rutTOT_mo.tolist()
    out_resp['IRI_mo'] = IRI_mo.tolist()

    # savemat("outputs_Python_fFC_rut_MatLEA.mat", out_resp)
    # # With parallel processing
    # start = time.perf_counter()
    # num_cores = multiprocessing.cpu_count()
    # outputPythi = Parallel(n_jobs=num_cores, verbose=50)(
    #     delayed(run_MatLEA)(ZC_in_subM, RC, tirePressure, CR_in, Ezi, PR_sub, TH_sub, PLTSTR,
    #                         RM_si, RM_ta, RM_tr, RM_qd, theta_si, theta_ta, theta_tr, theta_qd, ZC_in) for Ezi in Ez)
    #
    # time_measured: float = time.perf_counter() - start
    # print(f'Elapsed time Parallel processing = {time_measured} sec')

    return out_resp


def f_struc_resp_ACEAC(Step1Output):
    analysisLoc = Step1Output['analysisLoc']
    calibCoeffs = Step1Output['calibCoeffs']
    CR_in = Step1Output['CR_in']
    Esubdum_M = Step1Output['Esubdum_M']
    FreezeIndex = Step1Output['FreezeIndex']
    GWT = Step1Output['GWT']
    hbedrock = Step1Output['hbedrock']
    IBASEsub = Step1Output['IBASEsub']
    LoadsLb_qd = Step1Output['qdLoadsLb']
    LoadsLb_si = Step1Output['siLoadsLb']
    LoadsLb_ta = Step1Output['taLoadsLb']
    LoadsLb_tr = Step1Output['trLoadsLb']
    numMo = Step1Output['numMo']
    numQuint = Step1Output['numQuint']
    P200_subg = Step1Output['P200_subg']
    P4_subg = Step1Output['P4_subg']
    PR_sub = Step1Output['PR_sub']
    Prec_in_an_avg = Step1Output['Prec_in_an_avg']
    RC = Step1Output['RC']
    TC_mo = Step1Output['TC_mo']
    TH_sub = Step1Output['TH_sub']
    TH_subAC = Step1Output['TH_subAC']
    TH_subBASE = Step1Output['TH_subBASE']
    tirePressure = Step1Output['tirePressure']
    traf_NCor = Step1Output['traf_NCor']
    Tsubdum_M = Step1Output['Tsubdum_M']
    X_Wander = Step1Output['X_Wander']
    ZC_in_subM = Step1Output['ZC_in_subM']
    zi_AC_bu_fat = Step1Output['zi_AC_bu_fat']
    zi_AC_td_fat = Step1Output['zi_AC_td_fat']
    zi_sub_inAC = Step1Output['zi_sub_inAC']
    zi_sub_inBASE = Step1Output['zi_sub_inBASE']
    zi_sub_inSUBG = Step1Output['zi_sub_inSUBG']

    IEACsub = array(Step1Output['IEACsub'])
    zi_EAC_bu_fat = Step1Output['zi_EAC_bu_fat']

    P10c1 = Step1Output['P10c1']
    EACEstarDamage = Step1Output['EACEstarDamage']

    ZC_in = ZC_in_subM[0]
    NL_sub = len(TH_sub) + 1

    # Visualization setting.
    PLTSTR = None  # PLTSTR is the visualization variable which could be 'STRAINS', 'STRESSES', 'DISPLACEMENTS',
    #   or an empty string for disable the visualization part.

    Ez = zeros((numQuint * numMo, NL_sub + 2))
    T_Qnt = zeros((numQuint * numMo, NL_sub + 2))

    # Create a big 2D matrix where columns are the layers and rows are Month/Quintile combination.
    n = 0
    for iMo in range(numMo):
        for jQnt in range(numQuint):
            colno = jQnt + 3
            Ez[n, :] = hstack((iMo, jQnt, Esubdum_M[iMo, :, colno]))
            T_Qnt[n, :] = hstack((iMo, jQnt, Tsubdum_M[iMo, :, colno]))
            n = n + 1

    def run_MatLEA(ZC_in_subM, RC, tirePressure, CR_in, Ezi, PR_sub, TH_sub, PLTSTR, analysisLoc, ZC_in):
        iMo = int(Ezi[0])
        jQnt = int(Ezi[1])
        Ez2 = Ezi[2:]
        outAxisymm = MatLEA(ZC_in_subM, RC, tirePressure, CR_in, Ez2, PR_sub, TH_sub, PLTSTR)

        strain_r = outAxisymm['strain_r']
        strain_t = outAxisymm['strain_t']
        strain_z = outAxisymm['strain_z']
        strain_rz = outAxisymm['strain_rz']

        # stress_r = outAxisymm['stress_r']
        # stress_t = outAxisymm['stress_t']
        # stress_z = outAxisymm['stress_z']
        # stress_rz = outAxisymm['stress_rz']

        rz = vstack((outAxisymm['r'], outAxisymm['z'])).T

        # Perform superposition. Note that the def 'ff_process_superposition' is used so that
        # 'ff_superpose_strains_stresses' can be used in @njit to speed up processing time
        # out_resp_si = ff_process_superposition2(analysisLoc['RM_si'], analysisLoc['theta_si'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz,
        #                                         stress_r, stress_t, stress_z, stress_rz, rz)
        # out_resp_ta = ff_process_superposition2(analysisLoc['RM_ta'], analysisLoc['theta_ta'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz,
        #                                         stress_r, stress_t, stress_z, stress_rz, rz)
        # out_resp_tr = ff_process_superposition2(analysisLoc['RM_tr'], analysisLoc['theta_tr'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz,
        #                                         stress_r, stress_t, stress_z, stress_rz, rz)
        # out_resp_qd = ff_process_superposition2(analysisLoc['RM_qd'], analysisLoc['theta_qd'], ZC_in,
        #                                         strain_r, strain_t, strain_z, strain_rz,
        #                                         stress_r, stress_t, stress_z, stress_rz, rz)

        out_resp_si = ff_process_superposition1(analysisLoc['RM_si'], analysisLoc['theta_si'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz, rz)
        out_resp_ta = ff_process_superposition1(analysisLoc['RM_ta'], analysisLoc['theta_ta'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz, rz)
        out_resp_tr = ff_process_superposition1(analysisLoc['RM_tr'], analysisLoc['theta_tr'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz, rz)
        out_resp_qd = ff_process_superposition1(analysisLoc['RM_qd'], analysisLoc['theta_qd'], ZC_in,
                                                strain_r, strain_t, strain_z, strain_rz, rz)

        # out_resp = {'out_resp_si': out_resp_si, 'out_resp_ta': out_resp_ta, 'out_resp_tr': out_resp_tr,
        #             'out_resp_qd': out_resp_qd}
        # savemat("out_resp.mat", out_resp)
        # print(f'MatLEA has been ran for Mo: {Ezi[0]} - Qnt: {Ezi[1]}')
        return outAxisymm, out_resp_si, out_resp_ta, out_resp_tr, out_resp_qd, iMo, jQnt

    # location of bottom up cracking starting layer
    Ind1 = where(ZC_in_subM[0] == zi_AC_bu_fat)[0][0]
    Ifcbu = ZC_in_subM[2][Ind1]

    # Without parallel processing
    start = time.perf_counter()
    print(' --> Starting structural response computations...')
    iQnt = 0
    iMntCum = 0  # month no, cumulative, starting 0
    NumEz = len(Ez)
    D_bu_si_cum = zeros((NumEz))
    D_bu_ta_cum = zeros((NumEz))
    D_bu_tr_cum = zeros((NumEz))
    D_bu_qd_cum = zeros((NumEz))
    Dbu_cum = zeros((NumEz))
    FCbu = zeros((NumEz))

    D_td_si_cum = zeros((NumEz))
    D_td_ta_cum = zeros((NumEz))
    D_td_tr_cum = zeros((NumEz))
    D_td_qd_cum = zeros((NumEz))
    Dtd_cum = zeros((NumEz))
    FCtd = zeros((NumEz))

    D_bueac_si_cum = zeros((NumEz))
    D_bueac_ta_cum = zeros((NumEz))
    D_bueac_tr_cum = zeros((NumEz))
    D_bueac_qd_cum = zeros((NumEz))
    Dbueac_cum = zeros((NumEz))
    FCbueac = zeros((NumEz))
    TRCac_mo = zeros(numMo)  # reflective cracking
    delFCbueac = zeros(numMo)
    RFCa_rc = calibCoeffs['RFCa_rc']
    RFCb_rc = calibCoeffs['RFCb_rc']
    t_yr = arange(1, numMo + 1) / 12

    numACsub = len(TH_subAC)
    numBASEsub = len(TH_subBASE)
    numSUBGsub = 2

    rut_AC = zeros((NumEz, numACsub))

    xyz_si = analysisLoc['xyz_si']
    xyz_ta = analysisLoc['xyz_ta']
    xyz_tr = analysisLoc['xyz_tr']
    xyz_qd = analysisLoc['xyz_qd']

    SI_X = analysisLoc['SI_X']  # Number of unique X coordinates in the tire/axle coordinate matrix
    SJ2_Y = analysisLoc['SJ2_Y']  # Number of unique Y coordinates in the tire/axle coordinate matrix
    SJ3_Y = analysisLoc['SJ3_Y']  # Number of unique Y coordinates in the tire/axle coordinate matrix
    SJ4_Y = analysisLoc['SJ4_Y']  # Number of unique Y coordinates in the tire/axle coordinate matrix

    numX_si = len(unique(
        xyz_si[:, 0])) * 2 - 1  # number of X coordinates (perpendicular to traffic direction, same for all axles)

    epLAST_AC = zeros((numACsub, numX_si))  # plastic strain from the last (previous) cycle, for each AC sublayer
    rut_tot_AC = zeros(NumEz)

    rut_BASE = zeros((NumEz, numBASEsub))
    epLAST_BASE = zeros((numBASEsub, numX_si))  # plastic strain from the last (previous) cycle, for each BASE sublayer
    rut_tot_BASE = zeros(NumEz)

    epLAST_SUBG = zeros(
        (numSUBGsub, numX_si))  # plastic strain from the last (previous) cycle, for each point in Subgrade
    rut_tot_SUBG = zeros(NumEz)

    rutTOT = zeros(NumEz)
    IRI = zeros(NumEz)

    FCbu_mo = zeros(numMo)
    FCtd_mo = zeros(numMo)
    rut_tot_AC_mo = zeros(numMo)
    rut_tot_BASE_mo = zeros(numMo)
    rut_tot_SUBG_mo = zeros(numMo)
    rutTOT_mo = zeros(numMo)
    IRI_mo = zeros(numMo)
    FCbueac_mo = zeros(numMo)

    for n in range(NumEz):  # row number for Ez, corresponding to each month and each quintile
        Ezi = Ez[n]
        # Apply damage to the existing AC and its effect on |E*|
        if n > 0:
            E_EACsub = Ezi[IEACsub + 2]
            if Dbueac_cum[n - 1] > 0:
                E_EAC_dam = P10c1 + (E_EACsub - P10c1) / (1 + exp(-0.3 + 5 * log10(Dbueac_cum[n - 1])))
            else:
                E_EAC_dam = E_EACsub
            Ezi[IEACsub + 2] = E_EAC_dam

        outAxisymm, out_resp_si, out_resp_ta, \
            out_resp_tr, out_resp_qd, iMo, jQnt = run_MatLEA(ZC_in_subM, RC, tirePressure, CR_in, Ezi, PR_sub, TH_sub,
                                                             PLTSTR, analysisLoc, ZC_in)
        eps_xx_si = out_resp_si['eps_xx']
        eps_xx_ta = out_resp_ta['eps_xx']
        eps_xx_tr = out_resp_tr['eps_xx']
        eps_xx_qd = out_resp_qd['eps_xx']

        eps_yy_si = out_resp_si['eps_yy']
        eps_yy_ta = out_resp_ta['eps_yy']
        eps_yy_tr = out_resp_tr['eps_yy']
        eps_yy_qd = out_resp_qd['eps_yy']

        eps_zz_si = out_resp_si['eps_zz']
        eps_zz_ta = out_resp_ta['eps_zz']
        eps_zz_tr = out_resp_tr['eps_zz']
        eps_zz_qd = out_resp_qd['eps_zz']

        eps_princ_si = out_resp_si['eps_princ']
        eps_princ_ta = out_resp_ta['eps_princ']
        eps_princ_tr = out_resp_tr['eps_princ']
        eps_princ_qd = out_resp_qd['eps_princ']

        EbuAC = Ezi[Ifcbu + 1]
        EtdAC = Ezi[2]
        ErutBASE = Ezi[IBASEsub + 2]
        ErutSUBG = Ezi[-1]

        EbuEAC = Ezi[IEACsub + 2][-1]
        # AC bottom-up cracking critical strains
        e_buc1, e_buc2, e_buc3, e_buc4, Xbu1, Xbu2, Xbu3, Xbu4 = f_BUFC_CritStrains(eps_xx_si, eps_yy_si,
                                                                                    eps_xx_ta, eps_yy_ta,
                                                                                    eps_xx_tr, eps_yy_tr,
                                                                                    eps_xx_qd, eps_yy_qd,
                                                                                    xyz_si, xyz_ta, xyz_tr, xyz_qd,
                                                                                    zi_AC_bu_fat, SI_X, SJ2_Y, SJ3_Y,
                                                                                    SJ4_Y)
        # EAC bottom-up cracking critical strains
        e_bueacc1, e_bueacc2, e_bueacc3, e_bueacc4, Xbueac1, Xbueac2, Xbueac3, Xbueac4 = f_BUFC_CritStrains(eps_xx_si,
                                                                                                            eps_yy_si,
                                                                                                            eps_xx_ta,
                                                                                                            eps_yy_ta,
                                                                                                            eps_xx_tr,
                                                                                                            eps_yy_tr,
                                                                                                            eps_xx_qd,
                                                                                                            eps_yy_qd,
                                                                                                            xyz_si,
                                                                                                            xyz_ta,
                                                                                                            xyz_tr,
                                                                                                            xyz_qd,
                                                                                                            zi_EAC_bu_fat,
                                                                                                            SI_X, SJ2_Y,
                                                                                                            SJ3_Y,
                                                                                                            SJ4_Y)

        # AC top-down cracking critical strains
        e_tdc1, e_tdc2, e_tdc3, e_tdc4, Xtd1, Xtd2, Xtd3, Xtd4 = f_TDFC_CritStrains(eps_princ_si,
                                                                                    eps_princ_ta,
                                                                                    eps_princ_tr,
                                                                                    eps_princ_qd,
                                                                                    xyz_si, xyz_ta, xyz_tr, xyz_qd,
                                                                                    zi_AC_td_fat)
        # AC rut critical strains
        e_rutc1, e_rutc2, e_rutc3, e_rutc4, Xrutc1, Xrutc2, Xrutc3, Xrutc4 = f_rut_CritStrains(eps_zz_si,
                                                                                               eps_zz_ta,
                                                                                               eps_zz_tr,
                                                                                               eps_zz_qd,
                                                                                               xyz_si, xyz_ta, xyz_tr,
                                                                                               xyz_qd,
                                                                                               zi_sub_inAC)
        # Base rut critical strains
        e_rutb1, e_rutb2, e_rutb3, e_rutb4, Xrutb1, Xrutb2, Xrutb3, Xrutb4 = f_rut_CritStrains(eps_zz_si,
                                                                                               eps_zz_ta,
                                                                                               eps_zz_tr,
                                                                                               eps_zz_qd,
                                                                                               xyz_si, xyz_ta, xyz_tr,
                                                                                               xyz_qd,
                                                                                               zi_sub_inBASE)
        # Subrade rut critical strains
        e_ruts1, e_ruts2, e_ruts3, e_ruts4, Xruts1, Xruts2, Xruts3, Xruts4 = f_rut_CritStrains(eps_zz_si,
                                                                                               eps_zz_ta,
                                                                                               eps_zz_tr,
                                                                                               eps_zz_qd,
                                                                                               xyz_si, xyz_ta, xyz_tr,
                                                                                               xyz_qd,
                                                                                               zi_sub_inSUBG)

        # AC rutting computation
        rut_AC[n], epLAST_AC = f_cal_ACRut(e_rutc1, e_rutc2, e_rutc3, e_rutc4, Xrutc1, Xrutc2, Xrutc3, Xrutc4,
                                           LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                           traf_NCor['Nc_acrut_si'][iMntCum],
                                           traf_NCor['Nc_acrut_ta'][iMntCum],
                                           traf_NCor['Nc_acrut_tr'][iMntCum],
                                           traf_NCor['Nc_acrut_qd'][iMntCum],
                                           calibCoeffs['kriz'], calibCoeffs['Briz'], calibCoeffs['kzrut'],
                                           T_Qnt[n], TH_subAC, X_Wander, epLAST_AC, numACsub)

        rut_tot_AC[n] = sum(rut_AC[n])

        # Base rutting computation
        rut_BASE[n], epLAST_BASE = f_cal_BASERut(e_rutb1, e_rutb2, e_rutb3, e_rutb4, Xrutb1, Xrutb2, Xrutb3, Xrutb4,
                                                 LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                 traf_NCor['Nc_baserut_si'][iMntCum],
                                                 traf_NCor['Nc_baserut_ta'][iMntCum],
                                                 traf_NCor['Nc_baserut_tr'][iMntCum],
                                                 traf_NCor['Nc_baserut_qd'][iMntCum],
                                                 calibCoeffs['BriBsub'], TH_subBASE, X_Wander, epLAST_BASE, numBASEsub,
                                                 ErutBASE, GWT)
        rut_tot_BASE[n] = sum(rut_BASE[n])

        # subgrade rutting computation
        rut_tot_SUBG[n], epLAST_SUBG = f_cal_SUBGRut(e_ruts1, e_ruts2, e_ruts3, e_ruts4, Xruts1, Xruts2, Xruts3, Xruts4,
                                                     LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                     traf_NCor['Nc_subgrut_si'][iMntCum],
                                                     traf_NCor['Nc_subgrut_ta'][iMntCum],
                                                     traf_NCor['Nc_subgrut_tr'][iMntCum],
                                                     traf_NCor['Nc_subgrut_qd'][iMntCum],
                                                     calibCoeffs['BriS'], X_Wander, epLAST_SUBG, numSUBGsub, ErutSUBG,
                                                     GWT, hbedrock)

        rutTOT[n] = rut_tot_AC[n] + rut_tot_BASE[n] + rut_tot_SUBG[n]

        # Compute bottom-up damage
        D_bu1_mx, D_bu2_mx, \
            D_bu3_mx, D_bu4_mx = f_calc_damage_at_different_loads(e_buc1, e_buc2, e_buc3, e_buc4,
                                                                  Xbu1, Xbu2, Xbu3, Xbu4, EbuAC,
                                                                  LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                                  traf_NCor['Nc_bufc_si'][iMntCum][0],
                                                                  traf_NCor['Nc_bufc_ta'][iMntCum][0],
                                                                  traf_NCor['Nc_bufc_tr'][iMntCum][0],
                                                                  traf_NCor['Nc_bufc_qd'][iMntCum][0],
                                                                  calibCoeffs['Bi_bu'], calibCoeffs['ki_bu'],
                                                                  calibCoeffs['C_bu'], calibCoeffs['CH_bu'], X_Wander)
        # Compute top-down damage
        D_td1_mx, D_td2_mx, \
            D_td3_mx, D_td4_mx = f_calc_damage_at_different_loads(e_tdc1, e_tdc2, e_tdc3, e_tdc4,
                                                                  Xtd1, Xtd2, Xtd3, Xtd4, EtdAC,
                                                                  LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                                  traf_NCor['Nc_tdfc_si'][iMntCum][0],
                                                                  traf_NCor['Nc_tdfc_ta'][iMntCum][0],
                                                                  traf_NCor['Nc_tdfc_tr'][iMntCum][0],
                                                                  traf_NCor['Nc_tdfc_qd'][iMntCum][0],
                                                                  calibCoeffs['Bi_td'], calibCoeffs['ki_td'],
                                                                  calibCoeffs['C_td'], calibCoeffs['CH_td'], X_Wander)

        # Compute EAC bottom-up damage
        D_bueac1_mx, D_bueac2_mx, \
            D_bueac3_mx, D_bueac4_mx = f_calc_damage_at_different_loads(e_bueacc1, e_bueacc2, e_bueacc3, e_bueacc4,
                                                                        Xbueac1, Xbueac2, Xbueac3, Xbueac4, EbuEAC,
                                                                        LoadsLb_si, LoadsLb_ta, LoadsLb_tr, LoadsLb_qd,
                                                                        traf_NCor['Nc_eac_si'][iMntCum][0],
                                                                        traf_NCor['Nc_eac_ta'][iMntCum][0],
                                                                        traf_NCor['Nc_eac_tr'][iMntCum][0],
                                                                        traf_NCor['Nc_eac_qd'][iMntCum][0],
                                                                        calibCoeffs['Bi_bueac'],
                                                                        calibCoeffs['ki_bueac'],
                                                                        calibCoeffs['C_bueac'], calibCoeffs['CH_bueac'],
                                                                        X_Wander)

        # Cumulative damage at each Month - Quintile

        if n == 0:
            D_bu_si_cum[n] = sum(D_bu1_mx)
            D_bu_ta_cum[n] = sum(D_bu2_mx)
            D_bu_tr_cum[n] = sum(D_bu3_mx)
            D_bu_qd_cum[n] = sum(D_bu4_mx)

            D_bueac_si_cum[n] = sum(D_bueac1_mx)
            D_bueac_ta_cum[n] = sum(D_bueac2_mx)
            D_bueac_tr_cum[n] = sum(D_bueac3_mx)
            D_bueac_qd_cum[n] = sum(D_bueac4_mx)

            D_td_si_cum[n] = sum(D_td1_mx)
            D_td_ta_cum[n] = sum(D_td2_mx)
            D_td_tr_cum[n] = sum(D_td3_mx)
            D_td_qd_cum[n] = sum(D_td4_mx)


        else:
            D_bu_si_cum[n] = D_bu_si_cum[n - 1] + sum(D_bu1_mx)
            D_bu_ta_cum[n] = D_bu_ta_cum[n - 1] + sum(D_bu2_mx)
            D_bu_tr_cum[n] = D_bu_tr_cum[n - 1] + sum(D_bu3_mx)
            D_bu_qd_cum[n] = D_bu_qd_cum[n - 1] + sum(D_bu4_mx)

            D_bueac_si_cum[n] = D_bueac_si_cum[n - 1] + sum(D_bueac1_mx)
            D_bueac_ta_cum[n] = D_bueac_ta_cum[n - 1] + sum(D_bueac2_mx)
            D_bueac_tr_cum[n] = D_bueac_tr_cum[n - 1] + sum(D_bueac3_mx)
            D_bueac_qd_cum[n] = D_bueac_qd_cum[n - 1] + sum(D_bueac4_mx)

            D_td_si_cum[n] = D_td_si_cum[n - 1] + sum(D_td1_mx)
            D_td_ta_cum[n] = D_td_ta_cum[n - 1] + sum(D_td2_mx)
            D_td_tr_cum[n] = D_td_tr_cum[n - 1] + sum(D_td3_mx)
            D_td_qd_cum[n] = D_td_qd_cum[n - 1] + sum(D_td4_mx)

        Dbu_cum[n] = D_bu_si_cum[n] + D_bu_ta_cum[n] + D_bu_tr_cum[n] + D_bu_qd_cum[n]
        C_TF_bu = calibCoeffs['C_TF_bu']
        Cprime_1 = calibCoeffs['Cprime_1']
        Cprime_2 = calibCoeffs['Cprime_2']
        if Dbu_cum[n] <= 0:
            FCbu[n] = 0
        else:
            FCbu[n] = ((1 / 60) * (C_TF_bu[2] / (
                    1 + exp(Cprime_1 * C_TF_bu[0] + Cprime_2 * C_TF_bu[1] * log10(Dbu_cum[n] * 100)))))

        Dbueac_cum[n] = EACEstarDamage + D_bueac_si_cum[n] + D_bueac_ta_cum[n] + D_bueac_tr_cum[n] + D_bueac_qd_cum[n]
        # CEAC1 = 0
        # CEAC2 = 68
        # CEAC3 = 2
        # CEAC4 = 2
        #
        # # CEAC1 = calibCoeffs['CEAC1']
        # # CEAC2 = calibCoeffs['CEAC2']
        # # CEAC3 = calibCoeffs['CEAC3']
        # # CEAC4 = calibCoeffs['CEAC4']
        #
        # FCbueac[n] = CEAC1 + CEAC2 / (1 + exp(CEAC3 - CEAC4 * log10(Dbueac_cum[n] * 100.0)))
        if Dbueac_cum[n] <= 0:
            FCbueac[n] = 0
        else:
            FCbueac[n] = 100 / (1 + exp(6 - 6 * Dbueac_cum[n]))

        Dtd_cum[n] = D_td_si_cum[n] + D_td_ta_cum[n] + D_td_tr_cum[n] + D_td_qd_cum[n]
        C_TF_td = calibCoeffs['C_TF_td']
        if Dtd_cum[n] <= 0:
            FCtd[n] = 0
        else:
            FCtd[n] = (10.56 * (C_TF_td[2] / (1 + exp(C_TF_td[0] - C_TF_td[1] * log10(Dtd_cum[n] * 100)))))

        # Compute the IRI
        IRIo = float(calibCoeffs['IRIo'])
        C_IRI = array(calibCoeffs['C_IRI'])
        FCtd_perc = FCtd[
                        n] / 5280 * 3 / 12 * 100  # from ft/mile to percent (%)--> TD(%) = TD(ft/mile) / 5280 ft/mile * 3 ft (wheel path width) / 12 ft (lane width) * 100
        FCtot = FCbu[n] + FCtd_perc

        IRI[n] = f_calc_IRI(IRIo, FCtot, rutTOT[n], TC_mo[iMntCum], iMntCum, FreezeIndex, Prec_in_an_avg, P200_subg,
                            P4_subg, C_IRI)
        time_measured: float = time.perf_counter() - start
        # print(f'Distresses ran for Mo: {Ezi[0]} - Qnt: {Ezi[1]}, cumulative Month = {iMntCum}, Elapsed time = {time_measured} sec')
        start = time.perf_counter()

        iQnt = iQnt + 1  # Quintile no
        if iQnt > 4:
            iQnt = 0
            FCbueac_mo[iMntCum] = FCbueac[n]
            # Compute reflective cracking
            tminust = (iMntCum + 1) / 12 - t_yr[:(iMntCum + 1)]
            RCac_mi = 100. / (1 + np.exp(RFCa_rc + RFCb_rc * tminust))
            if iMntCum > 0:
                delFCbueac[iMntCum - 1] = FCbueac_mo[iMntCum] - FCbueac_mo[iMntCum - 1]
                TRCac_mo[iMntCum] = RCac_mi[:iMntCum] @ (delFCbueac[:iMntCum] / 100)

            FCbu_mo[iMntCum] = FCbu[n]
            FCtd_mo[iMntCum] = FCtd[n]
            rut_tot_AC_mo[iMntCum] = rut_tot_AC[n]
            rut_tot_BASE_mo[iMntCum] = rut_tot_BASE[n]
            rut_tot_SUBG_mo[iMntCum] = rut_tot_SUBG[n]
            rutTOT_mo[iMntCum] = rutTOT[n]
            IRI_mo[iMntCum] = IRI[n]

            txttoadd = f' ---> TRCac_mo[{iMntCum}] = {TRCac_mo[iMntCum]:.3}%, FCbueac[{iMntCum}] = {FCbueac[n]:.3}%, FCbu[{iMntCum}] = {FCbu[n]:.3}%, FCtd[{iMntCum}] = {FCtd[n]:.3}ft/mile, ACrut[{iMntCum}] = {rut_tot_AC[n]:.3}in,' \
                       f'BASErut[{iMntCum}] = {rut_tot_BASE[n]:.3}in, SUBGrut[{iMntCum}] = {rut_tot_SUBG[n]:.3}in,' \
                       f'TOTALrut[{iMntCum}] = {rutTOT[n]:.3}in, IRI[{iMntCum}] = {IRI[n]:.5}in/mile \n'
            print(txttoadd)
            utl.appendtxt(Step1Output['fld_out'], Step1Output['StatusTxtFile'], txttoadd)

            iMntCum = iMntCum + 1  # cumulative month no
            iMntCum

    #  Create a dictionary of outputs
    out_resp = {}
    # out_resp['Dbu_cum'] =Dbu_cum
    out_resp['FCbu_mo'] = FCbu_mo.tolist()
    # out_resp['Dtd_cum'] =Dtd_cum
    out_resp['FCtd_mo'] = FCtd_mo.tolist()
    out_resp['rut_tot_AC_mo'] = rut_tot_AC_mo.tolist()
    out_resp['rut_tot_BASE_mo'] = rut_tot_BASE_mo.tolist()
    out_resp['rut_tot_SUBG_mo'] = rut_tot_SUBG_mo.tolist()
    out_resp['rutTOT_mo'] = rutTOT_mo.tolist()
    out_resp['IRI_mo'] = IRI_mo.tolist()

    # savemat("outputs_Python_fFC_rut_MatLEA.mat", out_resp)
    # # With parallel processing
    # start = time.perf_counter()
    # num_cores = multiprocessing.cpu_count()
    # outputPythi = Parallel(n_jobs=num_cores, verbose=50)(
    #     delayed(run_MatLEA)(ZC_in_subM, RC, tirePressure, CR_in, Ezi, PR_sub, TH_sub, PLTSTR,
    #                         RM_si, RM_ta, RM_tr, RM_qd, theta_si, theta_ta, theta_tr, theta_qd, ZC_in) for Ezi in Ez)
    #
    # time_measured: float = time.perf_counter() - start
    # print(f'Elapsed time Parallel processing = {time_measured} sec')

    return out_resp
