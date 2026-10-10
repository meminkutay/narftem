"""
@author: meminkutay
"""

# import json
import numpy as np
import calendar
# from scipy.io import loadmat
from updapsutil import savejson

def processTrafficData(aadtt ,	aptQd ,	aptSi ,	aptTa ,	aptTr ,	growthType , direcDistFac ,	laneDistFac ,
                       monthlyDistFac ,	classnames , classPercent ,	growthPercent ,	openmonth ,	tdesign ,
                       alsData):


    NoDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # number of days in each month

    numvehicleclasses = len(classPercent)

    if alsData['DataType'] == 1: # Coming from the NAPCOMPlus input processing module

        alsSingle = alsData['alsSingle'] * 100 # fraction to percentage
        alsTandem = alsData['alsTandem'] * 100 # fraction to percentage
        alsTridem = alsData['alsTridem'] * 100 # fraction to percentage
        alsQuad   = alsData['alsQuad'] * 100 # fraction to percentage

        siLoadsLb = alsData['siLoadsLb']
        taLoadsLb = alsData['taLoadsLb']
        trLoadsLb = alsData['trLoadsLb']
        qdLoadsLb = alsData['qdLoadsLb']

        # The code below is to convert the ALS into 3D array to make month shifting (due to the starting month) easier
        mon = 0
        vehclassid = 0
        numweightssi = len(alsSingle[0])
        numweightsta = len(alsTandem[0])
        numweightstr = len(alsTridem[0])
        numweightsqd = len(alsQuad[0])
        alsSingle3d = np.zeros(12 * numvehicleclasses * numweightssi).reshape((12, numvehicleclasses, numweightssi))
        alsTandem3d = np.zeros(12 * numvehicleclasses * numweightsta).reshape((12, numvehicleclasses, numweightsta))
        alsTridem3d = np.zeros(12 * numvehicleclasses * numweightstr).reshape((12, numvehicleclasses, numweightstr))
        alsQuad3d = np.zeros(12 * numvehicleclasses * numweightsqd).reshape((12, numvehicleclasses, numweightsqd))

        monthClassNames = {}
        for i in range(0, numvehicleclasses * 12):
            alsSingle3d[mon][vehclassid] = alsSingle[vehclassid]
            alsTandem3d[mon][vehclassid] = alsTandem[vehclassid]
            alsTridem3d[mon][vehclassid] = alsTridem[vehclassid]
            alsQuad3d[mon][vehclassid] = alsQuad[vehclassid]

            monthClassNames[i] = [mon+1, classnames[vehclassid][0], classnames[vehclassid][1]]

            vehclassid += 1
            if vehclassid > numvehicleclasses - 1:
                vehclassid = 0
                mon += 1
                if mon > 11:
                    mon = 0
        # ---

        monthClassNames

    elif alsData['DataType'] == 2: # Coming from the UPDAPS I project (earlier method)
        alsSingle = alsData['alsSingle']
        alsTandem = alsData['alsTandem']
        alsTridem = alsData['alsTridem']
        alsQuad = alsData['alsQuad']

        siLoadsLb = alsData['siLoadsLb']
        taLoadsLb = alsData['taLoadsLb']
        trLoadsLb = alsData['trLoadsLb']
        qdLoadsLb = alsData['qdLoadsLb']

        # The code below is to convert the ALS into 3D array to make month shifting (due to the starting month) easier
        mon = 0
        vehclassid = 0
        numweightssi = len(alsSingle[0])
        numweightsta = len(alsTandem[0])
        numweightstr = len(alsTridem[0])
        numweightsqd = len(alsQuad[0])
        alsSingle3d = np.zeros(12 * numvehicleclasses * numweightssi).reshape((12, numvehicleclasses, numweightssi))
        alsTandem3d = np.zeros(12 * numvehicleclasses * numweightsta).reshape((12, numvehicleclasses, numweightsta))
        alsTridem3d = np.zeros(12 * numvehicleclasses * numweightstr).reshape((12, numvehicleclasses, numweightstr))
        alsQuad3d = np.zeros(12 * numvehicleclasses * numweightsqd).reshape((12, numvehicleclasses, numweightsqd))
        monthClassNames = {}

        for i in range(0, len(alsSingle)):
            alsSingle3d[mon][vehclassid] = alsSingle[i]
            alsTandem3d[mon][vehclassid] = alsTandem[i]
            alsTridem3d[mon][vehclassid] = alsTridem[i]
            alsQuad3d[mon][vehclassid] = alsQuad[i]
            monthClassNames[i] = [mon+1, vehclassid+1, classnames[vehclassid]]

            vehclassid += 1
            if vehclassid > numvehicleclasses - 1:
                vehclassid = 0
                mon += 1
                if mon > 11:
                    mon = 0
        # ---

        monthClassNames

    else:
        raise Exception("alsData DataType is incorrect")

    # Next, the number of single, tandem, tridem and quad axles are calculated for year 1, for each class for each month:
    tdesign = int(tdesign)
    NTij_1 = np.zeros(12 * numvehicleclasses).reshape((12, numvehicleclasses))
    NAij_Si = np.zeros(12 * numvehicleclasses).reshape((12, numvehicleclasses))
    NAij_Ta = np.zeros(12 * numvehicleclasses).reshape((12, numvehicleclasses))
    NAij_Tr = np.zeros(12 * numvehicleclasses).reshape((12, numvehicleclasses))
    NAij_Qd = np.zeros(12 * numvehicleclasses).reshape((12, numvehicleclasses))
    for m in range(0, 12):
        for c in range(0, numvehicleclasses):
            # First, the number of trucks for each month for each class is calculated for year 1:
            NTij_1[m][c] = (
                        aadtt * direcDistFac * laneDistFac * monthlyDistFac[m][c] * NoDays[m] * classPercent[c] / 100)
            # Next, the number of single, tandem, tridem and quad axles are calculated for year 1, for each class for each month:
            NAij_Si[m][c] = NTij_1[m][c] * aptSi[c]
            NAij_Ta[m][c] = NTij_1[m][c] * aptTa[c]
            NAij_Tr[m][c] = NTij_1[m][c] * aptTr[c]
            NAij_Qd[m][c] = NTij_1[m][c] * aptQd[c]

    # ---

    # Shift data upwards so that starting month is at the top of the matrix
    def monthnametonum(openmonth):
        abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
        openmonth3 = openmonth[:3]
        openmonth3 = openmonth3.capitalize()
        monthnum = abbr_to_num[openmonth3]
        return monthnum

    monthstart = monthnametonum(openmonth)

    NAij_Si = np.roll(NAij_Si, -monthstart + 1, axis=0)
    NAij_Ta = np.roll(NAij_Ta, -monthstart + 1, axis=0)
    NAij_Tr = np.roll(NAij_Tr, -monthstart + 1, axis=0)
    NAij_Qd = np.roll(NAij_Qd, -monthstart + 1, axis=0)

    alsSingle3d = np.roll(alsSingle3d, -monthstart + 1, axis=0)
    alsTandem3d = np.roll(alsTandem3d, -monthstart + 1, axis=0)
    alsTridem3d = np.roll(alsTridem3d, -monthstart + 1, axis=0)
    alsQuad3d = np.roll(alsQuad3d, -monthstart + 1, axis=0)

    NoDays = np.roll(NoDays, -monthstart + 1, axis=0)
    NAij_Sit = np.zeros(12 * numvehicleclasses * tdesign).reshape((tdesign, 12, numvehicleclasses))
    NAij_Tat = np.zeros(12 * numvehicleclasses * tdesign).reshape((tdesign, 12, numvehicleclasses))
    NAij_Trt = np.zeros(12 * numvehicleclasses * tdesign).reshape((tdesign, 12, numvehicleclasses))
    NAij_Qdt = np.zeros(12 * numvehicleclasses * tdesign).reshape((tdesign, 12, numvehicleclasses))
    # Next, the growth factor is computed for each year.
    # There are two options for growth of traffic; (i) compound and (ii) linear.
    n = -1
    yrmonthlist = np.zeros(12 * tdesign * 3).reshape((12 * tdesign, 3))
    for t in range(0, tdesign):
        for m in range(0, 12):  # here month 0 corresponds to the openmonth, for example March, if openmonth==March
            n += 1
            yrmonthlist[n][0] = t + 1           # first row is the year
            yrmonthlist[n][1] = m + monthstart  # second row is month (first month should correspond to openmonth)
            yrmonthlist[n][2] = m               # third row is also month but it starts 0
            if yrmonthlist[n][1] > 12:
                yrmonthlist[n][1] = yrmonthlist[n][1] - 12
            # print(yrmonthlist[n][:])
            for c in range(0, numvehicleclasses):
                if 'compound' in growthType[c]:
                    GFt = (1 + growthPercent[c] / 100) ** (t)
                elif 'linear' in growthType[c]:
                    GFt = 1 + (growthPercent[c] / 100) * (t)
                else:
                    GFt = 0
                #The number of single, tandem, tridem and quad axles are calculated for each year t,
                # for each month m, for each class c:
                NAij_Sit[t][m][c] = NAij_Si[m][c] * GFt
                NAij_Tat[t][m][c] = NAij_Ta[m][c] * GFt
                NAij_Trt[t][m][c] = NAij_Tr[m][c] * GFt
                NAij_Qdt[t][m][c] = NAij_Qd[m][c] * GFt



    # Next, the number of axles corresponding to each axle weight category is computed:
    def calc_NAijtw(tdesign, alsSingle3d, NAij_Sit, numweightssi, NoDays, monthstart, numvehicleclasses):
        NAij_Sitdum = np.zeros(12 * numvehicleclasses * tdesign * numweightssi).reshape(
            (numweightssi, tdesign, 12, numvehicleclasses))

        for i in range(0, len(yrmonthlist)):
            t = int(yrmonthlist[i][0] - 1)
            m = int(yrmonthlist[i][2])
            for c in range(0,numvehicleclasses):

                for w in range(0, numweightssi):
                    prc = alsSingle3d[m][c][w]
                    NAij_Sitdum[w][t][m][c] = round(NAij_Sit[t][m][c] * prc / 100 / NoDays[m])

        NAij_SitwDay = np.sum(NAij_Sitdum, axis=3)  # sum along the axis of classes, c
        NAij_SitwDay = NAij_SitwDay.reshape(numweightssi, tdesign * 12)
        NAij_SitwDay = np.transpose(NAij_SitwDay)
        return NAij_SitwDay

    #Finally, the number of axles are summed over j (i.e., classes) to compute the total number of applications of
    # single, tandem, tridem and quad axles, regardless of the class:
    NAij_SitwDay = calc_NAijtw(tdesign, alsSingle3d, NAij_Sit, numweightssi, NoDays, monthstart, numvehicleclasses)
    NAij_TatwDay = calc_NAijtw(tdesign, alsTandem3d, NAij_Tat, numweightsta, NoDays, monthstart, numvehicleclasses)
    NAij_TrtwDay = calc_NAijtw(tdesign, alsTridem3d, NAij_Trt, numweightstr, NoDays, monthstart, numvehicleclasses)
    NAij_QdtwDay = calc_NAijtw(tdesign, alsQuad3d, NAij_Qdt, numweightsqd, NoDays, monthstart, numvehicleclasses)
    # The code below is to save and compare against the original MATLAB codes
    # np.savetxt("NAij_SitwDay.csv", NAij_SitwDay, delimiter=",")
    # np.savetxt("NAij_TatwDay.csv", NAij_TatwDay, delimiter=",")
    # np.savetxt("NAij_TrtwDay.csv", NAij_TrtwDay, delimiter=",")
    # np.savetxt("NAij_QdtwDay.csv", NAij_QdtwDay, delimiter=",")

    # This part to compute the ESAL based on AASHTO 93 method, for reference purposes. Not used in UPDAPS
    def calc_ESAL(NAij_AxtwDay, ALSLoads, StdAxl_lb, expon, NoDays):
        ESAL = 0.0
        for tmo in range(0, NAij_AxtwDay.shape[0]):  # loop through months
            for w in range(0, NAij_AxtwDay.shape[1]):  # loop through weights
                ESAL = ESAL + (NAij_AxtwDay[tmo][w]) * NoDays * (ALSLoads[w] / StdAxl_lb) ** expon
        return ESAL

    # Standard Axle Load (lb) - FOR AASHTO 93 ESAL CALCULATIOn
    StdAxl_lb = [18464.6, 33737.0, 47951.2, 62869.7]  # lb
    expon = [3.962, 3.907, 3.924, 3.9155]
    NoDaysApprox = 30.4166
    ESALSi = calc_ESAL(NAij_SitwDay, siLoadsLb, StdAxl_lb[0], expon[0], NoDaysApprox)
    ESALTa = calc_ESAL(NAij_TatwDay, taLoadsLb, StdAxl_lb[1], expon[1], NoDaysApprox)
    ESALTr = calc_ESAL(NAij_TrtwDay, trLoadsLb, StdAxl_lb[2], expon[2], NoDaysApprox)
    ESALQd = calc_ESAL(NAij_QdtwDay, qdLoadsLb, StdAxl_lb[3], expon[3], NoDaysApprox)
    ESALtot = ESALSi + ESALTa + ESALTr + ESALQd

    # Compute the number of cycles to apply for each Month-Quintile, for each weight, for each axle
    NAij_SitwMoQnt = np.zeros(NAij_SitwDay.shape)
    NAij_TatwMoQnt = np.zeros(NAij_TatwDay.shape)
    NAij_TrtwMoQnt = np.zeros(NAij_TrtwDay.shape)
    NAij_QdtwMoQnt = np.zeros(NAij_QdtwDay.shape)

    # compute the number of load applications for each quintile of each month
    n = 0
    for i in range(len(NAij_SitwDay)):
        NAij_SitwMoQnt[i, :] = float(NoDays[n]) * NAij_SitwDay[i, :] / 5.0
        NAij_TatwMoQnt[i, :] = float(NoDays[n]) * NAij_TatwDay[i, :] / 5.0  # * Ncorr_ta_zbu
        NAij_TrtwMoQnt[i, :] = float(NoDays[n]) * NAij_TrtwDay[i, :] / 5.0  # * Ncorr_tr_zbu
        NAij_QdtwMoQnt[i, :] = float(NoDays[n]) * NAij_QdtwDay[i, :] / 5.0  # * Ncorr_qd_zbu
        n = n + 1
        if n > 11:
            n = 0


    traf_N = {'ESALtot' : ESALtot, 'NAij_SitwDay' : NAij_SitwDay, 'NAij_TatwDay' : NAij_TatwDay, 'NAij_TrtwDay' : NAij_TrtwDay,
              'NAij_QdtwDay': NAij_QdtwDay, 'NoDays': NoDays, 'NAij_SitwMoQnt' : NAij_SitwMoQnt, 'NAij_TatwMoQnt' : NAij_TatwMoQnt,
              'NAij_TrtwMoQnt' : NAij_TrtwMoQnt, 'NAij_QdtwMoQnt' : NAij_QdtwMoQnt}

    return traf_N
