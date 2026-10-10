import logging

# Configure logging


import json
import warnings

import numpy as np
import pandas as pd
from scipy import optimize


# from matplotlib import pyplot as plt

def f_construct_EstarMC(estarJSON, gstarJSON, layerTref, skipGstar):
    layer = {}
    layers = {}
    # fig, axs = plt.subplots(2, len(layerTref))
    cinp0 = [1, 2.8, -1, 0.2, 2.42000569e-04, -9.93952685e-02]
    cmin = [1.147, 2.761, -2, 0.1, 0.0000041, -0.132]
    cmax = [3.805, 5.587, -0.4, 0.6, 0.0004074, -0.059]
    bnds = np.array([cmin, cmax]).transpose()
    bnds = tuple(bnds)
    for l in range(0, len(estarJSON)):  # loop through each layer (multiple AC layers)
        if estarJSON[l] != 'N/A':  # skip if the layer is not an AC layer
            print('Computing the |E*| master curve of layer ' + str(l + 1))
            Estar = json.loads(estarJSON[l])
            numrowsE = len(Estar['estar1M'])
            EstarM = np.zeros(numrowsE * 4).reshape(numrowsE, 4)
            for i in range(0, numrowsE):
                EstarM[i][0] = Estar['estar1M'][i]['temp']
                EstarM[i][1] = Estar['estar1M'][i]['freq']
                EstarM[i][2] = Estar['estar1M'][i]['estar']
                EstarM[i][3] = Estar['estar1M'][i]['phaseAngle']

            Gstar = json.loads(gstarJSON[l])
            numrowsG = len(Gstar['gstar1M'])
            GstarM = np.zeros(numrowsG * 3).reshape(numrowsG, 3)
            for i in range(0, numrowsG):
                GstarM[i][0] = Gstar['gstar1M'][i]['temp']
                GstarM[i][1] = Gstar['gstar1M'][i]['gstar']
                GstarM[i][2] = Gstar['gstar1M'][i]['phaseAngle']
            # Compute |E*| master curve coefficients
            Tref = layerTref[l]
            optout = optimize.minimize(f_MC_Estar_err, cinp0, args=(EstarM, Tref), bounds=bnds, method='SLSQP')
            cinp = optout['x']
            print('EstarCoeff = ' + str(cinp))

            # Compute |E*| fit from the newly computed coefficients
            logEsvfit, logtR = f_MC_Estar(cinp, EstarM, Tref)
            Esvfit = 10 ** logEsvfit

            # Compute the percent error between measured and fit |E*|
            errM = np.abs(logEsvfit - np.log10(EstarM[:, 2])) / np.log10(EstarM[:, 2]) * 100
            err = np.average(errM)
            print('Percent err = ' + str(err))

            # Concatenate Esvfit and errM to the right side of EstarM
            EstarM = np.hstack((EstarM, Esvfit.reshape(-1, 1), errM.reshape(-1, 1)))

            # Convert to a pandas DataFrame for better display
            column_names = ['Temp_F', 'Freq_Hz', 'Estar_psi', 'PhaseAngle_deg', 'Estarfit_psi', 'PercentError']
            EstarM_df = pd.DataFrame(EstarM, columns=column_names)

            # Format the DataFrame to display full numbers (not scientific notation)
            pd.set_option('display.float_format', '{:.2f}'.format)

            # Print the entire EstarM DataFrame
            print("Updated EstarM matrix:")
            print(EstarM_df)

            if err > 1:
                warnings.warn("Warning: Error in |E*| is larger than 1%")

            # Compute the phase angle fit
            dinp0 = [33.8940, - 1.0267, 3.5487]
            logfR = -logtR
            optout2 = optimize.minimize(f_MC_Mix_PhaseAngle_err, dinp0, args=(logfR, EstarM[:, 3]), method='SLSQP')
            dinp = optout2['x']
            DeltaFit = f_calc_PhaseAngleFit(dinp, logfR)

            # uncomment below to visualize the data, if needed (also uncomment # fig, axs = plt.subplots(2, len(layerTref)) above)
            # axs[0, l].plot(logtR, logEsvfit, label='Master curve fit')
            # axs[0, l].plot(logtR, np.log10(EstarM[:,2]),marker='o', markerfacecolor='blue', markersize=9, color='skyblue', linewidth=0, label='Measured')
            # axs[0, l].set(ylabel = "log(|E*|)")
            # axs[0, l].set(xlabel = "log tR")
            # axs[1, l].plot(logtR, DeltaFit, label='Master curve fit')
            # axs[1, l].plot(logtR, EstarM[:,3], marker='o', markerfacecolor='blue', markersize=9, color='skyblue', linewidth=0, label='Measured')
            # axs[1, l].set(ylabel="Phase Angle")
            # axs[1, l].set(xlabel="log tR")

            if skipGstar == 0:
                # Compute the A-VTS relationship of the binder
                A, VTS = f_compute_A_VTS_binder(GstarM)
            else:
                A = 0
                VTS = 0

            layer['Estar'] = EstarM
            layer['Gstar'] = GstarM
            layer['EstarCoeff'] = cinp[:4]
            layer['aTF'] = cinp[4:6]
            layer['PhaseAngCoeff'] = dinp
            layer['Estarfit'] = Esvfit
            layer['PhaseAngFit'] = DeltaFit
            layer['logReducedTime'] = logtR
            layer['EstarMCPercErr'] = err
            layer['Abinder'] = A
            layer['VTSbinder'] = VTS
            layers[l] = layer

    # plt.legend()
    # plt.show()

    return layers


def f_compute_A_VTS_binder(GstarM):
    TRankine = GstarM[:, 0] + 459.67
    GstarBind = GstarM[:, 1]
    PhaseAnBind = GstarM[:, 2]
    nu = 1000 * (GstarBind / 10) * np.reciprocal(np.sin(np.radians(PhaseAnBind))) ** 4.8628  # centiPoise (cP),
    nu_MAX = 2.7e12
    nu[nu > nu_MAX] = 2.7e12  # maximum possible nu is 2.7e12.
    loglog_nu = np.log10(np.log10(nu))
    logT = np.log10(TRankine)
    p = np.polyfit(logT, loglog_nu, 1)
    A = p[1]
    VTS = p[0]
    return A, VTS


def f_MC_Estar_err(cinp, EstarM, Tref):
    logEsvfit, logtR = f_MC_Estar(cinp, EstarM, Tref)
    err = np.sum(np.abs(logEsvfit - np.log10(EstarM[:, 2])))
    return err


def f_MC_Estar(cinp, EstarM, Tref):
    T = EstarM[:, 0]
    f = EstarM[:, 1]

    logaT = cinp[4] * (T ** 2 - Tref ** 2) + cinp[5] * (T - Tref)
    t = np.reciprocal(f)
    logtR = np.log10(t) - logaT
    logEsvfit = cinp[0] + cinp[1] * np.reciprocal(1 + np.exp(cinp[2] + cinp[3] * logtR))
    return logEsvfit, logtR


def f_MC_Mix_PhaseAngle_err(dDeg, logfR, DeltaMeasured):
    DeltaFit = f_calc_PhaseAngleFit(dDeg, logfR)
    err = sum(abs(DeltaFit - DeltaMeasured))
    return err


def f_calc_PhaseAngleFit(dDeg, logfR):
    DeltaFit = dDeg[0] * np.exp(-(dDeg[1] - logfR) ** 2 / (2 * dDeg[2] ** 2))
    return DeltaFit
