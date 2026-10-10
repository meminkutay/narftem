from numpy import zeros, array, cumsum, where, hstack, vstack, amax, \
    logical_and, log10, unique
from math import pi
from updapsutil import editjson, makedir, loadjson, savejson, \
    sigdigit, strcmp, f_analysis_locations, f_process_calib_coeffs, \
    mlogspace, mlinspace, f_get_Ncorr_ax, f_process_traf_N, appendtxt
from f_rawTraf_to_AXL2 import processTrafficData
import time
from f_construct_EstarMC import f_construct_EstarMC
from f_sublayer_MEPDGmethod import ff_sublayer_MEPDGmethod
from MatLEA_Sub01_Find_LayNo import Modify_coordinates
from f_calc_freqMatrix import f_calc_freqMatrix
from MCLIM_FiniteDiffDempsey import FiniteDiffDempsey
from f_calculate_Estar_atQuintiles import f_calculate_Estar_atQuintiles
from f_calc_TCcrack import f_calc_TCcrack
from scipy.stats import norm


def UPDAPSAC_STEP1(inps, fld_out, StatusTxtFile):
    # Load the JSON file
    inps['skipGstar'] = 0  # 0 = dont skip Gstar
    numQuint = 5  # Number of quintiles in a month --> FIXED. DO NOT CHANGE
    if 'AClayers' in inps:
        AClayersDum = inps['AClayers'].copy()
        # I did this below because of json was loading the number indices as text and converting into text key
        AClayers = {}
        for i in range(len(AClayersDum)):
            layerNo = f'Layer{i}'
            AClayers[i] = AClayersDum[layerNo].copy()
        del AClayersDum
    else:
        # __________________________ Decided not to generate mastercurve here. __________________________
        # __________________________ The master curve coefficients are input now __________________________
        # Generate |E*| master curve for each AC layer
        AClayers = f_construct_EstarMC(inps['estarJSON'], inps['gstarJSON'], inps['layerTref'], inps['skipGstar'])

        # Some example inputs here
        # aTF1 = array([7.1899e-07,  -0.067657])
        # dDeg1 = array([31.9066,   -1.7489,   3.3750])
        # cpsi1 = array([2.6065,    4.3224,   -0.8770,    0.2816])

    inps['hbedrock'] = 1000  # in --> Make this an input later on...
    tDesign = int(inps['tDesign'])

    try:
        grType = inps['grType']
    except:
        grType = inps['crType']

    try:
        monthStart = inps['monthStart']
    except:
        monthStart = inps['monthText']

    # Depending on where the als data is coming from, do some processing
    # if 'alsInterp' key exists in inps, it means it is coming from NAPCOMPlus code
    # Else, it is coming from the earlier UPDAPS I project.
    if 'als' in inps:
        # alsData = inps['alsInterp'].copy()
        # alsData['DataType'] = 1 # NAPCOMPlus data
        # truckClassNames = alsData['rownames']
        # siLoadsLb = array(alsData['siLoadsKips']) * 1000
        # taLoadsLb = array(alsData['taLoadsKips']) * 1000
        # trLoadsLb = array(alsData['trLoadsKips']) * 1000
        # qdLoadsLb = array(alsData['qdLoadsKips']) * 1000

        als = inps['als'].copy()
        alsData = {}
        alsData['DataType'] = 1  # NAPCOMPlus data
        truckClassNames = als['truckClassNames']
        alsData['alsSingle'] = array(als['alsSi'])
        alsData['alsTandem'] = array(als['alsTa'])
        alsData['alsTridem'] = array(als['alsTr'])
        alsData['alsQuad'] = array(als['alsQd'])
        siLoadsLb = array(als['siLoadsKips']) * 1000
        taLoadsLb = array(als['taLoadsKips']) * 1000
        trLoadsLb = array(als['trLoadsKips']) * 1000
        qdLoadsLb = array(als['qdLoadsKips']) * 1000
        alsData['siLoadsLb'] = siLoadsLb
        alsData['taLoadsLb'] = taLoadsLb
        alsData['trLoadsLb'] = trLoadsLb
        alsData['qdLoadsLb'] = qdLoadsLb

    else:
        alsData = {}
        alsData['alsSingle'] = inps['alsSingle']
        alsData['alsTandem'] = inps['alsTandem']
        alsData['alsTridem'] = inps['alsTridem']
        alsData['alsQuad'] = inps['alsQuad']
        alsData['siLoadsLb'] = inps['siLoadsLb']
        alsData['taLoadsLb'] = inps['taLoadsLb']
        alsData['trLoadsLb'] = inps['trLoadsLb']
        alsData['qdLoadsLb'] = inps['qdLoadsLb']
        alsData['DataType'] = 2  # Earlier method, from UPDAPS-I project
        truckClassNames = inps['truckClass']
        siLoadsLb = array(inps['siLoadsLb'])
        taLoadsLb = array(inps['taLoadsLb'])
        trLoadsLb = array(inps['trLoadsLb'])
        qdLoadsLb = array(inps['qdLoadsLb'])

    # _________________________________________________________________________________________
    # Process traffic inps to compute number of applications of single, tandem, tridem and quad axles.
    tic = time.time()
    # _________________________________________________________________________________________
    traf_N = processTrafficData(inps['aadtt'], inps['aptQd'], inps['aptSi'],
                                inps['aptTa'], inps['aptTr'], grType,
                                inps['ddf'], inps['ldf'], inps['mdf'], truckClassNames,
                                inps['classPercent'], inps['growthPercent'], monthStart, tDesign, alsData)

    #     traf_N = {'ESALtot': ESALtot, 'NAij_Sitw': NAij_Sitw, 'NAij_Tatw': NAij_Tatw, 'NAij_Trtw': NAij_Trtw,
    #               'NAij_Qdtw': NAij_Qdtw, 'NoDays': NoDays}
    # _________________________________________________________________________________________
    toc = time.time() - tic
    print(f'Traffic processed in {toc} seconds')
    txttoadd = f'ESAL total = {sigdigit(traf_N["ESALtot"] / 1e6, 3)} million \n '
    print(txttoadd)
    appendtxt(fld_out, StatusTxtFile, txttoadd)

    # Sublayer the pavement layers
    layerType = array(inps['layerType'])
    layerTH = array(inps['layerTH'])
    # _________________________________________________________________________________________
    TH_sub, Edum_sub, LayNo_sub = ff_sublayer_MEPDGmethod(layerTH, layerType)
    # _________________________________________________________________________________________

    layerPR = array(inps['layerPR'])  # Poisson's ratio of each original layer
    layerMR = array(inps['layerMR'])  # Resilient modulus of each original layer

    PR_sub = layerPR[LayNo_sub]  # Poisson's ratio of each sublayer
    E_sub = layerMR[LayNo_sub]  # Sublayer elastic modulus (AC moduli here are zero, to be entered later)
    LayerType_sub = layerType[LayNo_sub]

    layerTH = array(layerTH)
    layerTHcum = cumsum(layerTH)

    zi_sub_in = cumsum(TH_sub) - TH_sub / 2  # Depth from the surface to the center of each sublayer

    IAC = array(strcmp(layerType, 'AC'))
    IAClast = IAC[-1]  # Last AC layer

    IEAC = array(strcmp(layerType, 'EAC'))

    if IEAC.size == 0:
        EACExists = False
        P10c1 = []
        EACEstarDamage = []

    else:
        EACExists = True
        P10c1 = 10 ** AClayers[IEAC[-1]]['EstarCoeff'][
            0]  # P10c1 = 10^c1, where c1 = coefficient of the |E*| master curve, for the EAC
        EACEstarDamage = inps['layerEstarDamage'][IEAC[-1]]

    ICSM = array(strcmp(layerType, 'CSM'))
    if ICSM.size == 0:
        CSMExists = False
    else:
        CSMExists = True

    ICSMsub = strcmp(LayerType_sub, 'CSM')  # indices of the CSM sublayers
    IACsub = strcmp(LayerType_sub, 'AC')  # indices of the AC sublayers
    IEACsub = strcmp(LayerType_sub, 'EAC')  # indices of the EAC sublayers

    TH_subAC = TH_sub[IACsub]
    layerTHcumACEAC = layerTHcum[(layerType[:len(layerTHcum)] == 'AC') | (layerType[:len(layerTHcum)] == 'EAC')]
    layerTHcumAC = layerTHcum[(layerType[:len(layerTHcum)] == 'AC')]

    # Define the Z coordinates
    zi_sub_inAC = zi_sub_in[zi_sub_in <= layerTHcumACEAC[-1]]
    numzi_inAC = len(zi_sub_inAC)
    IBASEsub = where(logical_and(zi_sub_in > layerTHcumACEAC[-1], zi_sub_in < layerTHcum[-1]))[0]
    TH_subBASE = TH_sub[IBASEsub]
    zi_sub_inBASE = zi_sub_in[IBASEsub]
    zi_sub_inSUBG = array([layerTHcum[-1] * 1.001, layerTHcum[-1] + 6])

    zi_AC_bu_fat = 0.99 * layerTHcum[IAClast]
    zi_AC_td_fat = 0.5  # inches

    ZC_in_sub = hstack((0, zi_sub_in, zi_AC_bu_fat, zi_AC_td_fat, zi_sub_inSUBG))

    if EACExists:
        zi_EAC_bu_fat = 0.99 * layerTHcum[IEAC[-1]]
        ZC_in_sub = hstack((ZC_in_sub, zi_EAC_bu_fat))
    else:
        zi_EAC_bu_fat = []

    if CSMExists:
        zi_CSM_fat = 0.99 * layerTHcum[ICSM[-1]]
        ZC_in_sub = hstack((ZC_in_sub, zi_CSM_fat))

        CSM_Emax = array(inps['csmEmax'])
        CSM_Emin = array(inps['csmEmin'])
        CSM_MoR = array(inps['csmMoR'])
        CSM_Beta_c1 = array(inps['csmBetaC1'])
        CSM_Beta_c2 = array(inps['csmBetaC2'])

        CSMEmax_sub = CSM_Emax[LayNo_sub[ICSMsub]]
        CSMEmin_sub = CSM_Emin[LayNo_sub[ICSMsub]]
        CSMMoR_sub = CSM_MoR[LayNo_sub[ICSMsub]]
        CSMBeta_c1_sub = CSM_Beta_c1[LayNo_sub[ICSMsub]]
        CSMBeta_c2_sub = CSM_Beta_c2[LayNo_sub[ICSMsub]]

        E_sub[ICSMsub] = CSM_Emax[LayNo_sub[ICSMsub]]

    else:
        zi_CSM_fat = []
        CSMEmax_sub = []
        CSMEmin_sub = []
        CSMMoR_sub = []
        CSMBeta_c1_sub = []
        CSMBeta_c2_sub = []

    ZC_in_sub = unique(ZC_in_sub)
    # _________________________________________________________________________________________
    ZC_in_subM = Modify_coordinates(ZC_in_sub,
                                    TH_sub)  # Pre-processing of Z-coordinates, to find which layer each Z coordinate is located.
    # _________________________________________________________________________________________
    #  ZC_in_subM[0] = DISTANCE TO SURFACE
    #  ZC_in_subM[1] = DISTANCE TO THE TOP OF THE CURRENT LAYER
    #  ZC_in_subM[2] = LAYER NO WHERE THE Z COORDINATE IS
    ZC_in_subM[2] = array(ZC_in_subM[2])

    # Calculate loading frequencies at different depths and load levels from vehicle speed

    taAxleSpecin = inps['taAxleSpec']
    trAxleSpecin = inps['trAxleSpec']
    qdAxleSpecin = inps['qdAxleSpec']

    vehicleSpeed = inps['vehicleSpeed']
    tirePressure = inps['tirePressure']
    axleSpacingin = hstack((taAxleSpecin, trAxleSpecin, qdAxleSpecin))
    # _________________________________________________________________________________________
    freq_Ncorr_CR = f_calc_freqMatrix(Edum_sub, TH_sub, ZC_in_sub, vehicleSpeed, axleSpacingin,
                                      siLoadsLb, taLoadsLb, trLoadsLb, qdLoadsLb, tirePressure)
    # _________________________________________________________________________________________

    # Define the structural analysis location and superposition variables
    # for the purpose of defining analysis locations, contact radius is computed using 18 kip load.
    # when we use '3 point' loading approach, this will change.
    CR_in = (18000 / 4 / (pi * tirePressure)) ** 0.5
    dualTireSpec = inps['dualTireSpec']
    # _________________________________________________________________________________________
    analysisLoc = f_analysis_locations(dualTireSpec, taAxleSpecin, trAxleSpecin, qdAxleSpecin, CR_in, ZC_in_subM)
    # _________________________________________________________________________________________
    maxRC = amax(analysisLoc['RM_qd'])
    RC = hstack((mlinspace(0.0, CR_in, 5), mlogspace(log10(CR_in * 1.1), log10(maxRC), 10)))

    # Calculate the layer temperatures using mclim (mechanistic climatic) finite difference model
    layerAbsorp = inps['layerAbsorp'][0]

    # # _________________________________________________________________________________________
    txttoadd = f'--> Running the climatic model \n'
    print(txttoadd)
    appendtxt(fld_out, StatusTxtFile, txttoadd)

    # This is the starting month of the simulation
    if 'monthText' in inps:
        startingMonth = inps['monthText']
    else:
        startingMonth = 'AUGUST'

    Tsubdum_M, MAAT_F, Yr, Mo, Tpave_mek_F, Tsurf_mek_F, Prec_in, \
        FreezeIndex, Prec_in_an_avg, yrstart, Dy, Hr = FiniteDiffDempsey(inps['fnameAirTemp'], inps['fldAirTemp'],
                                                                         inps['latitude'], inps['longitude'],
                                                                         inps['daylightSavingSelected'], inps['utc'],
                                                                         startingMonth,
                                                                         tDesign, array(inps['layerGD']),
                                                                         array(inps['layerK']), array(inps['layerC']),
                                                                         layerTHcum, zi_sub_in, layerAbsorp,
                                                                         inps['climateType'], inps['climateModel'])

    # ClimateRunOutput = {'Tsubdum_M': Tsubdum_M, 'MAAT_F': MAAT_F, 'Yr': Yr, 'Mo': Mo, 'Tpave_mek_F': Tpave_mek_F,'Tsurf_mek_F': Tsurf_mek_F, 'Prec_in': Prec_in,
    #                'FreezeIndex': FreezeIndex, 'Prec_in_an_avg': Prec_in_an_avg, 'yrstart': yrstart, 'Dy': Dy, 'Hr': Hr}

    # savemat("outputs_Python_FiniteDiffDempsey.mat", ClimateRunOutput)
    txttoadd = f' --> Climatic model done \n'
    print(txttoadd)
    appendtxt(fld_out, StatusTxtFile, txttoadd)
    # _________________________________________________________________________________________

    # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # __  below is to skip FiniteDiffDempsey function during development stage. To be deleted...
    # print(f'!!! Reading climatic data from from the "outputs_Python_FiniteDiffDempsey.mat" file ')
    # mat_contents = loadmat('outputs_Python_FiniteDiffDempsey.mat', squeeze_me=True)
    # Tsubdum_M = mat_contents['Tsubdum_M']
    # MAAT_F = mat_contents['MAAT_F']
    # Tpave_mek_F = mat_contents['Tpave_mek_F']
    # Tsurf_mek_F = mat_contents['Tsurf_mek_F']
    # FreezeIndex = mat_contents['FreezeIndex']
    # Prec_in_an_avg = mat_contents['Prec_in_an_avg']
    # print(f' !!! Done reading...')
    #
    # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    layerVa = array(inps['layerVa'])
    layerVbeff = array(inps['layerVbeff'])
    layerRTFO: float = inps['layerRTFO'][0]
    layerTref = array(inps['layerTref'])
    # _________________________________________________________________________________________
    Esubdum_M, Va_PV_inAC, Vbeff_PV_inAC = f_calculate_Estar_atQuintiles(Tsubdum_M, siLoadsLb, layerTHcumACEAC,
                                                                         freq_Ncorr_CR['freq_si'],
                                                                         layerRTFO, zi_sub_inAC, AClayers, layerTref,
                                                                         MAAT_F, layerType, layerVa, layerVbeff,
                                                                         layerTH, E_sub)
    # _________________________________________________________________________________________
    # Step1Output = {}
    # Step1Output['Esubdum_M'] = Esubdum_M
    # savemat("outputs_Python_Esubdum_M.mat", Step1Output)

    IDT = array(inps['layerIDT'])
    TCcoef_K = inps['tCoefK']

    # TCcoef_K = 10.0
    # _________________________________________________________________________________________
    thermalCrack = f_calc_TCcrack(LayNo_sub, zi_sub_inAC, layerTHcumAC, IDT[0], MAAT_F, tDesign, TCcoef_K,
                                  AClayers[0], layerTref[0], Tsurf_mek_F, Tpave_mek_F)
    # _________________________________________________________________________________________

    txttoadd = f' --> Thermal cracking computations are done \n'
    print(txttoadd)
    appendtxt(fld_out, StatusTxtFile, txttoadd)

    # For the lateral wander
    z_quint = norm.ppf([0.10, 0.30, 0.50, 0.70, 0.90])
    # z_quint = [-1.2816, -0.5244, 0, 0.5244, 1.2816] # standard normal distribution, divided into 5 equal intervals
    X_Wander = inps['wheelWander'] * z_quint
    P200_subg = inps['p200Subg']
    P4_subg = inps['p4Subg']
    numMo = int(len(Esubdum_M))
    GWT = inps['gwt']
    hbedrock = inps['hbedrock']
    # _________________________________________________________________________________________
    calibCoeffs = f_process_calib_coeffs(inps, Va_PV_inAC, Vbeff_PV_inAC, TH_subAC, TH_subBASE, zi_sub_inAC)
    # _________________________________________________________________________________________
    if CSMExists:
        calibCoeffs['CSMBeta_c1_sub'] = CSMBeta_c1_sub
        calibCoeffs['CSMBeta_c2_sub'] = CSMBeta_c2_sub

    # _________________________________________________________________________________________
    traf_NCor = f_process_traf_N(traf_N, freq_Ncorr_CR, zi_AC_bu_fat, zi_AC_td_fat,
                                 zi_sub_inAC, zi_sub_inBASE, zi_sub_inSUBG, zi_CSM_fat, zi_EAC_bu_fat)
    # _________________________________________________________________________________________

    # analysisLoc, calibCoeffs, CR_in, Esubdum_M, FreezeIndex, freq_Ncorr_CR, GWT, \
    # hbedrock, IBASEsub, inps['qdLoadsLb'], inps['siLoadsLb'], inps['taLoadsLb'], inps['trLoadsLb'], \
    # numMo, numQuint, P200_subg, P4_subg, PR_sub, Prec_in_an_avg, RC, thermalCrack['TC_mo'], \
    # TH_sub, TH_subAC, TH_subBASE, tirePressure, traf_NCor, Tsubdum_M, X_Wander, \
    # ZC_in_subM, zi_AC_bu_fat, zi_AC_td_fat, zi_sub_inAC, zi_sub_inBASE, zi_sub_inSUBG
    #
    Step1Output = {'analysisLoc': analysisLoc, 'calibCoeffs': calibCoeffs, 'CR_in': CR_in, 'Esubdum_M': Esubdum_M,
                   'FreezeIndex': FreezeIndex, 'GWT': GWT, 'hbedrock': hbedrock,
                   'IBASEsub': IBASEsub, 'qdLoadsLb': qdLoadsLb, 'siLoadsLb': siLoadsLb,
                   'taLoadsLb': taLoadsLb, 'trLoadsLb': trLoadsLb, 'numMo': numMo,
                   'numQuint': numQuint, 'P200_subg': P200_subg, 'P4_subg': P4_subg,
                   'PR_sub': PR_sub, 'Prec_in_an_avg': Prec_in_an_avg, 'RC': RC, 'TC_mo': thermalCrack['TC_mo'],
                   'TH_sub': TH_sub, 'TH_subAC': TH_subAC, 'TH_subBASE': TH_subBASE, 'tirePressure': tirePressure,
                   'traf_NCor': traf_NCor, 'Tsubdum_M': Tsubdum_M, 'X_Wander': X_Wander, 'ZC_in_subM': ZC_in_subM,
                   'zi_AC_bu_fat': zi_AC_bu_fat, 'zi_AC_td_fat': zi_AC_td_fat, 'zi_sub_inAC': zi_sub_inAC,
                   'zi_sub_inBASE': zi_sub_inBASE, 'zi_sub_inSUBG': zi_sub_inSUBG, 'EACExists': EACExists,
                   'CSMExists': CSMExists,
                   'ICSMsub': ICSMsub, 'IEACsub': IEACsub, 'zi_CSM_fat': zi_CSM_fat, 'zi_EAC_bu_fat': zi_EAC_bu_fat,
                   'CSMEmax_sub': CSMEmax_sub, 'CSMEmin_sub': CSMEmin_sub, 'CSMMoR_sub': CSMMoR_sub, 'P10c1': P10c1,
                   'EACEstarDamage': EACEstarDamage}

    txttoadd = f' --> STEP I is done \n'
    print(txttoadd)
    appendtxt(fld_out, StatusTxtFile, txttoadd)
    return Step1Output
