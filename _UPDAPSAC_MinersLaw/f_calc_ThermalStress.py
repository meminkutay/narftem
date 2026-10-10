# from scipy import integrate, io
import numpy as np
# import time
# from pathlib import Path
# from matplotlib import pyplot as plt
from numba import njit

@njit
def mlinspace(minVal, maxVal, numvals):
    minV: float = minVal
    maxV: float = maxVal
    stepp = (maxV - minV) / (numvals-1)
    outArray: float = np.arange(minV, maxV+stepp, stepp)
    return outArray

@njit
def mlogspace(logminVal, logmaxVal, numvals):
    logmin: float = logminVal
    logmax: float = logmaxVal
    stepp = (logmax - logmin) / (numvals-1)
    logOutArray: float = np.arange(logmin, logmax+stepp, stepp)
    outArray = 10 ** logOutArray
    return outArray

@njit
def mcumtrapz(Y, X):
    lenX = len(X)
    integOut = np.zeros(lenX-1)
    for i in range(lenX-1):
        Ai = (Y[i] + Y[i+1]) * (X[i+1] - X[i]) / 2
        if i==0:
            integOut[i] = Ai
        else:
            integOut[i] = integOut[i-1] + Ai

    return integOut

@njit
def f_calc_ThermalStress(CTE, aTF, Einf, Ei, taui, layerTrefi, Tpave_mek_F, zi_TC_AC, numzinTopAC):

    numHrs = len(Tpave_mek_F[0])
    stressSV = np.zeros((numzinTopAC, numHrs))
    tR = np.zeros((numzinTopAC, numHrs))
    ei = np.zeros((numzinTopAC, numHrs))

    for i in range(numzinTopAC):
        # print('---> Running the "f_calc_ThermalStress" for the layer ' + str(i)) # slows down numba by 1 second
        print('---> Running the "f_calc_ThermalStress" for the next layer ... ')

        # Temperature when the pavement is in zero stress
        # To_C = T_C[0, 0]
        To_C = 0
        # To_F = To_C * 1.8 + 32

        T_F = Tpave_mek_F[i]

        T_C = (T_F - 32) / 1.8

        logaT = aTF[0] * (T_F ** 2 - layerTrefi ** 2) + aTF[1] * (T_F - layerTrefi)

        aT = 10 ** logaT

        t0 = np.linspace(0, numHrs-1, numHrs) * 3600
        t0[0] = 3.6e-12

        Y = aT ** -1
        # reduced time = tR = integral[0 to t](dt / aT)
        # tR0 = integrate.cumtrapz(Y, t0)
        tR0 = mcumtrapz(Y, t0)
        # difff = sum(tR02-tR0)

        tR01 = np.zeros(len(tR0)+1)
        tR01[0] = 1e-20
        tR01[1:] = tR0.copy()
        tR0 = tR01.copy()

        ei0 = CTE * (T_C - To_C) # * np.ones(nts, 1)

        tR[i] = tR0
        ei[i] = ei0
        t = t0

        stressSV[i]  = fVE_State_Variable_Et2(tR[i], ei[i], Ei, taui, Einf, 3, zi_TC_AC[i])

    return t, stressSV


@njit
def fVE_State_Variable_Et2(tR, ei, Ei, taui, Einf, npts, zi):

    len_ei = len(ei)
    stress = np.zeros(len_ei)
    sigmaj_prev = np.zeros(len(Ei))

    for i in range(len_ei - 1):
        deR = ei[i + 1] - ei[i]
        dtR = tR[i + 1] - tR[i]
        if dtR > 0:
            if i == 0:
                # tt = np.logspace(-20, np.log10(tR[i + 1]), 30)
                tt = mlogspace(-20, np.log10(tR[i + 1]), 30)

            else:
                ddtR = dtR / npts
                # tt = np.linspace(tR[i] + ddtR, tR[i + 1], npts - 1)
                tt = mlinspace(tR[i] + ddtR, tR[i + 1], npts - 1)

            # linear interpolation.Assumes linear increase from previous point to the next.
            eit = ei[i] + deR / dtR * (tt - tt[0])
            stresst = eit * 0

            for j in range(len(tt) - 1):
                dt = tt[j + 1] - tt[j]
                de = eit[j + 1] - eit[j]

                # stress in each maxwell element
                sigmaj = np.exp(-dt / taui) * sigmaj_prev + de / dt * Ei * taui * (1 - np.exp(-dt / taui))
                stresst[j] = Einf * eit[j] + np.sum(sigmaj)
                sigmaj_prev = sigmaj

            stress[i + 1] = stresst[j]

        else:
            stress[i + 1] = stress[i]

    return stress

