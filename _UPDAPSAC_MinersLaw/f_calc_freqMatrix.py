
import numpy as np
from math import pi
from MatLEA_Sub01_Find_LayNo import Modify_coordinates

def f_freq_MEPDG(E, hi, ZC_in_subM2, Vmph, CR_in, St, Naxles):
    # z0 = ZC_in_subM2[0] # distance to surface
    z1 = ZC_in_subM2[1] # distance to top of the layer where the point is located
    Nz = ZC_in_subM2[2] # index of the layer where the point is located

    Esubg = E[-1]
    numZ = len(z1)
    Ncorr = np.zeros(numZ)
    freq = np.zeros(numZ)
    for i in range(numZ):
        Nzcur = Nz[i]
        Ei = E[:Nzcur]
        if Nzcur == 1:
            # if point is in the top sublayer
            Zeff = float(np.sum(((Ei / Esubg) ** (1 / 3)) * z1[i]))  # [compat numpy>=2.3: 1-elem array -> scalar]
        else:
            # otherwise
            Zeff = np.sum(((Ei[:Nzcur-1] / Esubg) ** (1 / 3)) * hi[:Nzcur-1]) \
                   + ((Ei[Nzcur-1] / Esubg) ** (1 / 3)) * z1[i]

        if Zeff <= (St/2 - CR_in):
            Leff = 2 * (CR_in + Zeff)
            freq[i] = 17.6 * Vmph / Leff
            Ncorr[i] = Naxles

        elif ((Zeff > St/2 - CR_in) and (Zeff < 2 * St * (Naxles-1))):
            a = np.log((5*St*(Naxles-1)+2*CR_in)/St)/np.log((2*St*(Naxles-1))/(St/2-CR_in))
            b = np.log(St)-a*np.log(St/2-CR_in)
            Leff = np.exp((a*np.log(Zeff)+b))
            freq[i] = 17.6*Vmph/Leff

            a_N = np.log(1/Naxles)/np.log((2*St*(Naxles-1))/(St/2-CR_in))
            b_N = np.log(Naxles)-a_N*np.log(St/2-CR_in)
            Ncorr[i] = np.exp((a_N*np.log(Zeff)+b_N))

        else:
            Leff = St * (Naxles - 1) + 2 * CR_in + 2 * Zeff
            freq[i] = 17.6 * Vmph / Leff
            Ncorr[i] = 1

    return freq, Ncorr

def f_calc_freqMatrix(Edum_sub, TH_insub, ZC_in_sub, Vmph, Stm, siLoadsLb, taLoadsLb, trLoadsLb, qdLoadsLb, tirePressure):

    # The calculations below assume each axle have 4 tires, i.e., two dual tires
    CR_in_si = (siLoadsLb / 4 / (pi * tirePressure)) ** 0.5
    CR_in_ta = (taLoadsLb / 8 / (pi * tirePressure)) ** 0.5
    CR_in_tr = (trLoadsLb / 12 / (pi * tirePressure)) ** 0.5
    CR_in_qd = (qdLoadsLb / 16 / (pi * tirePressure)) ** 0.5

    n_Ps_si = len(CR_in_si)
    n_Ps_ta = len(CR_in_ta)
    n_Ps_tr = len(CR_in_tr)
    n_Ps_qd = len(CR_in_qd)

    freq_si = np.zeros([len(ZC_in_sub), n_Ps_si])
    freq_ta = np.zeros([len(ZC_in_sub), n_Ps_ta])
    freq_tr = np.zeros([len(ZC_in_sub), n_Ps_tr])
    freq_qd = np.zeros([len(ZC_in_sub), n_Ps_qd])

    Ncorr_ta = np.zeros([len(ZC_in_sub), n_Ps_ta])
    Ncorr_tr = np.zeros([len(ZC_in_sub), n_Ps_tr])
    Ncorr_qd = np.zeros([len(ZC_in_sub), n_Ps_qd])

    ZC_in_subM2 = Modify_coordinates(ZC_in_sub, TH_insub)

    for i in range(n_Ps_si):
        freq_si[:, i], _ = f_freq_MEPDG(Edum_sub, TH_insub, ZC_in_subM2, Vmph, CR_in_si[i], 1e10, 1)

    for i in range(n_Ps_ta):
        freq_ta[:, i], Ncorr_ta[:, i] = f_freq_MEPDG(Edum_sub, TH_insub, ZC_in_subM2, Vmph, CR_in_ta[i], Stm[0], 2)

    for i in range(n_Ps_tr):
        freq_tr[:, i], Ncorr_tr[:, i] = f_freq_MEPDG(Edum_sub, TH_insub, ZC_in_subM2, Vmph, CR_in_tr[i], Stm[1], 3)

    for i in range(n_Ps_qd):
        freq_qd[:, i], Ncorr_qd[:, i] = f_freq_MEPDG(Edum_sub, TH_insub, ZC_in_subM2, Vmph, CR_in_qd[i], Stm[2], 4)

    freq_Ncorr_CR = {'ZC_in_sub': ZC_in_sub, 'freq_si': freq_si, 'freq_ta': freq_ta, 'freq_tr': freq_tr,
                  'freq_qd': freq_qd, 'Ncorr_ta': Ncorr_ta, 'Ncorr_tr': Ncorr_tr, 'Ncorr_qd': Ncorr_qd,
                  'CR_in_si':CR_in_si, 'CR_in_ta':CR_in_ta, 'CR_in_tr': CR_in_tr,'CR_in_qd': CR_in_qd}

    return freq_Ncorr_CR