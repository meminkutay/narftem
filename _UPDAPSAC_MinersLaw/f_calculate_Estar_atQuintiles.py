from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import savemat
import numpy as np
from MatLEA_Sub01_Find_LayNo import Modify_coordinates

def f_calculate_Estar_atQuintiles(TQsub, siLoadsLb, TH_in_cumAC, freq_si, RTFO, zi_PV_inAC, layers,
                                  Tref, MAAT_F, layerType, Va, Vbeff, layerTH, E_sub):
    P_si = siLoadsLb / 4

    freqs = freq_si[:, P_si == 4500]

    zi_PV_inACM = Modify_coordinates(zi_PV_inAC, layerTH)
    # TH_in_cumAC = TH_in_cum[(layerType[:len(TH_in_cum)] =='AC') | (layerType[:len(TH_in_cum)] == 'EAC')]
    # zi_PV_inAC  = zi_PV_in[zi_PV_in <= TH_in_cumAC[-1]]
    numzi_inAC = len(zi_PV_inAC)

    Va_PV_inAC      = zi_PV_inAC*0
    Vbeff_PV_inAC   = Va_PV_inAC.copy()

    for i in range(0, len(TH_in_cumAC)):
        if i==0:
            Va_PV_inAC[zi_PV_inAC<=TH_in_cumAC[0]] = Va[0]
            Vbeff_PV_inAC[zi_PV_inAC<=TH_in_cumAC[0]] = Vbeff[0]
        else:
            Va_PV_inAC[(zi_PV_inAC<=TH_in_cumAC[i]) & (zi_PV_inAC>TH_in_cumAC[i-1])] = Va[i]
            Vbeff_PV_inAC[(zi_PV_inAC<=TH_in_cumAC[i]) & (zi_PV_inAC>TH_in_cumAC[i-1])] = Vbeff[i]

    EQaged = np.zeros(TQsub.shape)
    numMo = TQsub.shape[0]
    for i in range(numMo):

        EQaged[i, :, :3] = TQsub[i, :, :3].copy()
        for j in range(numzi_inAC):
            T_F = TQsub[i, j, 3:]
            t_mo = i

            z_in_sub = zi_PV_inAC[j].copy()
            nn = zi_PV_inACM[2][j] - 1
            Va_sub = Va_PV_inAC[j].copy()
            freq = freqs[j][0].copy()


            if layerType[nn] == 'AC' or layerType[nn] == 'EAC':
                Esv_fit_aged = ff_Estar_from_Mastercurve_aged(layers[nn], T_F, freq, Tref[nn], RTFO, t_mo, z_in_sub, MAAT_F,Va_sub)
                EQaged[i, j, 3:] = Esv_fit_aged
            else:
                raise Exception('Layer type is incorrect...')

        for k in range(5):
            EQaged[i, numzi_inAC:, k+3] = E_sub[numzi_inAC:].T


    return EQaged, Va_PV_inAC, Vbeff_PV_inAC


def ff_Estar_from_Mastercurve_aged(layer, T_F, freq, Tref, RTFO, t_mo, z_in_sub, MAAT_F, Va_sub):

    aTF = layer['aTF']
    cpsi = layer['EstarCoeff']
    A  = layer['Abinder']
    VTS = layer['VTSbinder']

    nu_MAX = 2.7 * 10**12
    t = 1/freq

    if str(np.nan) == str(A) or A==0:
        print('No Gstar entered, so no aging is considered')
        logaT = aTF[0] * (T_F**2 - Tref**2) + aTF[1] * (T_F - Tref)
        logtR = np.log10(t) - logaT
        Esv_fit_aged = 10**(cpsi[0]+cpsi[1]/(1+np.exp(cpsi[2]+cpsi[3]*logtR)))
    else:
        TGR = T_F + 459.67
        [nu, nu_tz] = ff_GAS(RTFO, t_mo, TGR, nu_MAX, z_in_sub, MAAT_F, Va_sub, A, VTS)

        logT_R_aged = (np.log10(np.log10(nu_tz)) - A)/VTS
        T_R_aged    = 10**(logT_R_aged)
        Teq_aged_F  = T_R_aged -  459.67
        logaTaged2  = aTF[0] * (Teq_aged_F**2-Tref**2) + aTF[1]*(Teq_aged_F-Tref)
        logtRaged2  = np.log10(t) - logaTaged2
        Esv_fit_aged = 10**(cpsi[0]+cpsi[1]/(1+np.exp(cpsi[2]+cpsi[3]*logtRaged2)))

    return Esv_fit_aged


def ff_GAS(RTFO, t_mo, TR, nu_MAX, z_in_sub, MAAT_F, Va_sub, A, VTS):
    lognu = 10 ** (A + VTS * np.log10(TR))
    nu = 10**(lognu)

    if (True in list(nu > nu_MAX)):
        nu[nu > nu_MAX] = 2.7*10**12
        lognu = np.log10(nu)
    code = 0

    if RTFO == 1:
        a0 = 0
        a1 = 1
    else:
        a0 =  0.054405 + 0.004082*code
        a1 =  0.972035 + 0.010886*code

    loglognu = np.log10(lognu)
    loglognt0 = a0 + a1*loglognu
    nt0 = 10**(10**(loglognt0))

    if (True in list(nt0 > nu_MAX)):
        nt0[nt0 > nu_MAX] = 2.7*10**12
        loglognt0 = np.log10(np.log10(nt0))

    DD = -14.5521 + 10.47662*np.log10(TR) - 1.88161*np.log10(TR)**2
    CC = 10**(274.4946-193.831*np.log10(TR)+33.9366*np.log10(TR)**2)
    BB = 0.197725+0.068384*np.log10(CC)
    AA = -0.004166+1.41213*CC+CC*np.log10(MAAT_F)+DD*loglognt0
    t_mo = t_mo +1
    loglognuaged = (loglognt0 + AA*t_mo) / (1+BB*(t_mo))
    nuaged = 10**(10**(loglognuaged))

    if (True in list(nuaged > nu_MAX)):
        nuaged[nuaged > nu_MAX] = 2.7*10**12
        loglognuaged = np.log10(np.log10(nuaged))

    TR77 = 77+ 459.67
    nuorig77 = 10**(10**(A + VTS*np.log10(TR77)))
    nuorig77 = nuorig77*10

    Va_t = (Va_sub + 0.011*t_mo - 2) / (1 + (4.24 * 10**(-4))*t_mo*MAAT_F + (1.169* 10**(-3))*(t_mo/nuorig77))  +2 
    Fv   = 1

    loglognuaged2 = Fv * loglognuaged
    nuaged2 = 10**(10**(loglognuaged2))

    if (True in list(nuaged2 > nu_MAX)):
        nuaged2[nuaged2 > nu_MAX] = 2.7*10**12
        loglognuaged2 = np.log10(np.log10(nuaged2))

    EE = 23.83*np.exp(-0.0308*MAAT_F)
    nuaged2_MP = 10**(10**(loglognuaged2))*10
    nt0_MP     = 10**(10**(loglognt0))*10
    nu_tz_MP = (nuaged2_MP*(4+EE) - EE*(nt0_MP)*(1-4*z_in_sub))/(4*(1+EE*z_in_sub))
    nu_tz = nu_tz_MP/10

    if (True in list(nu_tz > nu_MAX)):
        nu_tz[nu_tz > nu_MAX] = 2.7*10**12
        loglognu_tz = np.log10(np.log10(nu_tz))

    loglognu_tz = np.log10(np.log10(nu_tz))
    return [nu, nu_tz]
