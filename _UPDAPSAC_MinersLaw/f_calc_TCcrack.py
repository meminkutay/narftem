import numpy as np
from f_lin_interp1 import f_lin_interp1
from f_calc_ThermalStress import f_calc_ThermalStress
from updapsutil import f_Et_from_Estar
from scipy.interpolate import interp1d

import warnings

warnings.filterwarnings("ignore")


def normpdf(x, mu, sigma):
    y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    return y


def f_calc_TCcrack(LayNo_sub, zi_sub_inAC, layerTHcumAC, IDT, MAAT_F, tdesign, TCcoef_K, AClayer, layerTref,
                   Tsurf_mek_F, Tpave_mek_F):
    IfirstAC = np.where(LayNo_sub == 0)[0]
    zi_TC_AC = zi_sub_inAC[IfirstAC]  # do the thermal cracking analysis using the top layer only.

    # Only top AClayer is considered
    numzinTopAC = len(zi_TC_AC)
    Tpave_mek_F1 = Tpave_mek_F[:, 0:numzinTopAC].T
    Tpave_mek_F2 = np.vstack((Tsurf_mek_F, Tpave_mek_F1))

    zi_TC_AC = np.hstack((0, zi_TC_AC))
    numzinTopAC = len(zi_TC_AC)

    TH_in_cumAC = np.hstack((0, layerTHcumAC))

    asphaltCTE = 3e-5  # 1/C - Asphalt mixture coefficient of thermal contraction / expansion
    aTF0 = AClayer['aTF']
    cpsi0 = AClayer['EstarCoeff']
    dDeg0 = AClayer['PhaseAngCoeff']

    # Convert |E*| to E(t), only for the top layer.
    # Actually computes the prony series coefficients Einf, Ei, taui
    pronyCoeffsAC0 = f_Et_from_Estar(cpsi0, dDeg0, 33.0)
    # pronyCoeffsAC = {'Einf': Einf, 'Ei': Ei, 'm': m, 't': t, 'Et': Et}

    Einf = pronyCoeffsAC0['Einf']
    Ei = pronyCoeffsAC0['Ei']
    taui = pronyCoeffsAC0['taui']
    slopem = pronyCoeffsAC0['m']
    print(f'Performing the convolution integral to compute thermal stresses...')
    # _________________________________________________________________________________________
    t, stressSV = f_calc_ThermalStress(asphaltCTE, aTF0, Einf, Ei, taui, layerTref, Tpave_mek_F2, zi_TC_AC, numzinTopAC)
    # _________________________________________________________________________________________
    # output_save = {'t': t, 'stressSV': stressSV}
    # savemat('outputs_Python_stressSV2.mat' ,output_save)
    print(f'Thermal stresses computation is done...')

    coef_Bt = 1  # Calibration coefficient - to be eliminated later. Not used since it is 1

    stressSV = -stressSV.T  # Make all tension positive
    stressSV[stressSV < 0] = 0
    stressSV[0, :] = stressSV[1, :]  # the surface and the mid depth of the first layer should have the same stress

    # compute the daily max and min stress values
    len_t = len(t)
    # t_hr = t/3600
    Iday = [i for i in range(1, len_t, 24)]
    len_Iday = len(Iday)
    stressSVmax = np.zeros((len_Iday - 1, len(zi_TC_AC)))
    stressSVmin = np.zeros((len_Iday - 1, len(zi_TC_AC)))

    t_day = [i for i in range(1, len_Iday, 1)]

    for n in range(0, len_Iday - 1):
        Icurr = [i for i in range(Iday[n], Iday[n + 1])]
        stressSVmax[n, :] = np.max(stressSV[Icurr, :], 0)
        stressSVmin[n, :] = np.min(stressSV[Icurr, :], 0)

    # Calculate thermal crack depth propagation
    nn = 0.8 * (1 + 1 / slopem)
    print(f'MAAT = {str(MAAT_F)}oF, K = {str(TCcoef_K)}')
    Co = np.array(np.zeros(len(t_day)))
    Co[0] = 0  # crack depth (inches)

    TH_in_cumAC = np.hstack(([0, TH_in_cumAC]))

    hAC = TH_in_cumAC[-1]

    len_tday = len(t_day)
    sigmm = np.zeros((len_tday, 1))

    for i in range(0, len_tday - 1):
        if Co[i] > hAC:
            idx = -(len(Co) - i + 1)
            Co[idx:] = Co[i]
            print('Crack length reached the entire thickness of the AC layers, stopping...')
            break
        else:
            # find the sigma at the tip of the crack
            # sigm = interp1(zi_TC_AC, stressSV(i+1,:), Co(i)) % super slow the one below is much faster
            # yo = f_lin_interp1(x, y, xo)

            if np.isnan(Co[i]):
                print(Co[i])
                print(stressSVmax[i, :])
                raise ValueError(['NaN in C[i]', Co[i]])
            else:

                [sigmmax, temp] = f_lin_interp1(zi_TC_AC, stressSVmax[i, :], Co[i])
                K_2 = sigmmax * (0.45 + 1.99 * Co[i] ** 0.56)  # stress intensity factor

                [sigmmin, temp] = f_lin_interp1(zi_TC_AC, stressSVmin[i, :], Co[i])
                K_1 = sigmmin * (0.45 + 1.99 * Co[i] ** 0.56)  # stress intensity factor
                deltaK = K_2 - K_1

                logA = (4.389 - 2.52 * np.log10(10000 * IDT * nn))
                A = 10 ** logA
                deltaC = (TCcoef_K * coef_Bt) ** (nn + 1) * A * deltaK ** nn
                Co[i + 1] = Co[i] + deltaC

            sigmm[i] = sigmmax - sigmmin

    x = np.log10(Co / hAC) / 0.769
    # print('x = ',x)
    TC = 2 / 5 * 5280 * normpdf(x, 0, 0.769 / 2)
    # print('TC',TC)

    t_mo = [i for i in range(1, tdesign * 12 + 1)]
    t_mo_day = np.array(t_mo) * 30.4167

    # print('t_mo_day = ',t_mo_day)
    t_mo_day[-1] = t_day[-1]

    TC_mo = interp1d(t_day, TC)(t_mo_day)
    Co_mo = interp1d(t_day, Co)(t_mo_day)

    thermalCrack = {'TC_mo': TC_mo, 'Co_mo': Co_mo}

    return thermalCrack
