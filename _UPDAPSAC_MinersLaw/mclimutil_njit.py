from time import time

import numpy as np
# from scipy.stats import norm
from numba import njit

@njit
def f_get_Q(PSun_k, climateModel, Qx, sigm,Tair_R, vp_mb, Nbart,Rhr_eng, A, B, N, G, J, rhoo, vp, a):
    if climateModel.lower() == 'revised':
        # revised model - Idso/Barton.
        Qe      = Qx                                # (1 + N *(1-PSun(k))); % heat flux resulting from long-wave
        #   radiation emitted by the pavement surface, (with cloud
        #   correction) Btu/ft2-hr
        Qz      = sigm * Tair_R ** 4 * (0.74 + 0.0049 * vp_mb)   # Qz= heat flux resulting from long-wave radiation
        #   emitted by the atmosphere, Btu/ft2?hr;
        Qa      = Qz * (1 + Nbart *(1 - PSun_k))  # Qa= heat flux resulting from long-wave radiation emitted by
        #   the atmosphere, (with cloud correction) Btu/ft2?hr;
        Qi = Rhr_eng * (A + B * PSun_k)  # shortwave radiation, with cloud cover correction - Btu/ft2-h
        Qs = a * Qi  # net shortwave radiation - pavement surface Btu/ft2-h

    elif climateModel.lower() == 'original2' or climateModel == 'case-1':
        # This is what is supposed to be in Pavement ME.
        # ----------- This section was Revised by Prof. Kutay at 05/30/2020 ----------
        Qe = Qx * (1 - N * (1 - PSun_k))  # heat flux resulting from long-wave radiation emitted by the
        #   pavement surface, (with cloud correction) Btu/ft2?hr;
        Qz = sigm * Tair_R ** 4 * (G - J * 10 ** (-rhoo ** vp))  # Qz= heat flux resulting from long-wave
        #   radiation emitted by the atmosphere, Btu/ft2?hr;
        Qa = Qz * (1 - N * (1 - PSun_k))  # Qa= heat flux resulting from long-wave radiation emitted by
        #   the atmosphere, (with cloud correction) Btu/ft2?hr;
        Qi = Rhr_eng * (A + B * PSun_k)  # heat flux resulting from shortwave radiation, with cloud cover
        #   correction - Btu/ft2-h
        Qs = a * Qi  # net heat flux resulting from shortwave radiation - pavement
        #   surface Btu/ft2-h

    elif climateModel.lower() == 'original':
        # THis is what is in Pavement ME
        Qe = Qx * (1 - N * (1 - PSun_k))  # heat flux resulting from long-wave radiation emitted by the
        # pavement surface, (with cloud correction) Btu/ft2?hr;
        Qz = sigm * Tair_R ** 4 * (G - J * 10 ** (-rhoo ** vp))  # Qz= heat flux resulting from long-wave
        #   radiation emitted by the atmosphere, Btu/ft2?hr;
        Qa = Qz  # (1 - N *(1-PSun(k))); % Qa= heat flux resulting from long-wave
        #   radiation emitted by the atmosphere, (with cloud correction)
        #   Btu/ft2?hr;
        # I know above ( Qa = Qz;) doesn't make sense but this is the only way it matches Pavement ME
        Qi = Rhr_eng * (A + B * PSun_k)  # shortwave radiation, with cloud cover correction - Btu/ft2-h
        Qs = a * Qi  # net shortwave radiation - pavement surface Btu/ft2-h

    else:
        raise Exception
    # The balance.
    # =====================
    Qrad = Qs + Qa - Qe  # =
    # =====================
    return Qrad

@njit
def SolarRadiation(Lat, day_cur):
    """
    This function calculates the daily solar radiation in Btu/ft^2-day
    :param Lat: The latitute of the project site.
    :param day_cur: the number of the current day.
    :return: The
    """

    # Solar constant (= 442 Btu/ft^2-hr)
    #       1 kJ/hr = 0.947817 Btu/hr
    #       Isc_eng = Isc *  0.947817 / (3.28084^2) ; % Btu/ft^2-hr
    Isc         = 4871                      # In kJ/m2-hr

    # Day angle in radians.
    dayang      = (2 * np.pi * day_cur) / 365

    # Eccentricity factor:
    Eo          = 1.000110 + 0.034221 * np.cos(dayang)     + \
                             0.001280 * np.sin(dayang)     + \
                             0.000719 * np.cos(dayang * 2) + \
                             0.000077 * np.sin(dayang * 2)

    # Solar declination (degrees)
    delta       = 180 / np.pi * (0.006918 - 0.399912 * np.cos(dayang)     +
                                            0.070257 * np.sin(dayang)     -
                                            0.006758 * np.cos(dayang * 2) +
                                            0.000907 * np.sin(dayang * 2) -
                                            0.002697 * np.cos(dayang * 3) +
                                            0.001480 * np.sin(dayang * 3))              # In degrees

    ws          = np.arccos(-np.tan(np.deg2rad(Lat)) * np.tan(np.deg2rad(delta)))       # In radians
    R           = 24 / np.pi * Isc * Eo * np.sin(np.deg2rad(Lat)) *np.sin(np.deg2rad(delta))*(ws - np.tan(ws))#kJ/m2-day
    R_eng       = R * 0.947817 / (3.28084 ** 2)                                         # In Btu/ft^2-day

    #return section.
    return R_eng

@njit
def SunUpDown(year, month, day, latitude, longitude, daylight_saving, UTC):
    """
    This function calculates the sunrise and sunset times during the year based on the date. More details are provided
    as follows:
      	Standard Time                               Daylight Saving
      UTC	Guam	HI	AK	PST	MST	CST	EST	AST	 	PDT	MDT	CDT	EDT
      Diff	+10     -10	-9	-8	-7	-6	-5	-4      -7	-6	-5	-4
    Note that:
        Enter date in yyyy-mm-dd format <<<minus between day month and year is important>>>
        First it is important to specify two variables for Matlab to write on as shown in the title. Otherwise you will
        only see the sunrise time. The date has to be enter in 'yyyy-mm-dd' format, otherwise it wont work. This date is
        used to calculate the Julian day. Afterwards the Julian day is used to calculate the sunrise and sunset times.
        Latitude and longitude has to be entered in full degrees. If theres daylight saving active at the specified date
        and location enter the full hour value your location uses for daylight savings (usually 1 but can be different)
        for this value, otherwise enter zero. For UTC enter the plus or minus value for your location without daylight
        saving if it is active. This function calculates the times for sunrise and sunset and writes them on the
        variables in the format 'hh:mm'.
    The parameters are:
    :param date:
    :param latitude:
    :param longitude:
    :param daylight_saving:
    :param UTC:
    :return:
    """

    # Calculates Julian Day Number
    #   Note: Julian date is not identical with search in Google: however, the number of days from 2000-1-1 is correct.
    # date_str    = date.split('-')
    # year        = date[0] #int(date_str[0])
    # month       = date[1] #int(date_str[1])
    # day         = date[2] #int(date_str[2])
    a           = np.floor((14 - month) / 12)
    y           = year + 4800 - a
    m           = month + 12 * a - 3
    jdn         = day + np.floor((153 * m + 2) / 5) + 365 * y + np.floor(y / 4) - \
                  np.floor(y / 100) + np.floor(y / 400) - 32045

    # Calculate days since 1st Jan 2000
    n           = jdn - 2451545 + 0.0008

    # Calculate the sunrise and sunset (Reference: Kutay and Lanotte (2019), documentation of MEAPA.
    J_star      = n - longitude / 360
    M           = np.mod(357.5291 + 0.98560028 * J_star, 360)
    C           = 1.9148 * np.sin(np.deg2rad(M)) + 0.0200 * np.sin(np.deg2rad(M) * 2) + 0.0003 * np.sin(np.deg2rad(M)*3)
    Lambda      = np.mod(M + C + 180 + 102.9372, 360)
    J_transit   = 2451545.5 + J_star + 0.0053 * np.sin(np.deg2rad(M)) - 0.0069 * np.sin(np.deg2rad(Lambda) * 2)
    delta_sin   = np.sin(np.deg2rad(Lambda)) * np.sin(np.deg2rad(23.44))
    omega_0_cos = (np.sin(np.deg2rad(-0.83)) - np.sin(np.deg2rad(latitude)) * delta_sin) / \
                  (np.cos(np.deg2rad(latitude)) * np.cos(np.arcsin(delta_sin)))
    J_set       = J_transit + np.rad2deg(np.arccos(omega_0_cos)) / 360
    J_rise      = J_transit - np.rad2deg(np.arccos(omega_0_cos)) / 360
    rise_time   = (J_rise - jdn) * 24 + daylight_saving + UTC
    set_time    = (J_set  - jdn) * 24 + daylight_saving + UTC

    # Return section.
    return rise_time, set_time


@njit
def SolarRadiation_Daily2Hourly(R, tSR, tSS, t):
    """
    This function distribute the amount of solar radiation during the day time within the day based on the time of the
        sunrise and sunset.
    :param R: The amount of
    :param tSR: Time of the sunrise during that specific day.
    :param tSS: Time of the sunset during that specific day.
    :param t: Current time.
    :return: The amount of solar radiation.
    """

    Dayduration = tSS - tSR         # Day time duration.
    W           = Dayduration / 2   # Mid day time (most radiation occurrs here).
    S           = tSR + W           # Peak time (Mid day time)
    G           = 3 * R / (2 * Dayduration)

    if t < tSR or t > tSS:
        # In the night times: there is no radiation.
        Rhr     = 0

    elif (t - S) < 0:
        # Before the peak point: morning.
        WW      = W - (t - tSR)
        Rhr     = G / W ** 2 * (W ** 2 - WW ** 2)

    else:
        # After the peak point: After noon.
        WW      = t - S
        Rhr     = G / W ** 2 * (W ** 2 - WW ** 2)

    # Return the hourly radiation.
    return Rhr

@njit
def f_iterate_throug_time(t, PSun, U_mph, Tair_F, gd, C, dz, K, dt, zi_in, TH_in_cum, zi_PV_in, n, climateModel,
                          absorp, alph, Tcon,NoDays, latitude, longitude, daylight_saving, UTC,
                          Yr, Mo, Dy, Imonth_start, yrstart):
    dTdt        = np.zeros(n)
    Tair        = Tair_F[0]
    Tpave_mek_F = np.zeros((len(Tair_F), len(zi_PV_in)))
    Tsurf_mek_F = np.zeros(len(Tair_F))
    Tp_F_mek    = zi_PV_in * 0
    # Initialize section.
    zi_ft       = zi_in / 12                # convert in to ft
    T           = Tair_F[0] + 0 * zi_ft
    # Defining the coefficients from literature.
    A           = 0.202                         # (Dempsey's dissertation)
    B           = 0.539                         # (Dempsey's dissertation)
    a           = absorp                        # 0.85
                                                # a = absorptivity of pavement surface = 0.85-0.9 for asphalt,
                                                #       0.6-0.7 for concrete (Dempsey's dissertation)
    N           = 0.8                           # Cloud base factor (Dempsey's dissertation);0.8-0.9
    sigm        = 0.172 * 10 ** -8              # Btu/hr-ft^2-R^4 (Stefan-Boltzmann constant)
    epsl        = 0.95                          # emissivity.  Emissivity values obtained in this study were all between
                                                #  0.93 and 0.98 (Marchetti et al., 2004)
    G           = 0.77                          # (Dempsey's dissertation)
    J           = 0.28                          # (Dempsey's dissertation)
    rhoo        = 0.074                         # (Dempsey's dissertation)
    vp          = 10                            # vapor pressure = 1-10 mmHg
    vp_mb       = 1.33322 * vp

    # tocp        = time()                        # Update the Run time.
    k           = 0                     # hour count
    m           = 0
    Nbart       = 0.17
    time_hr     = 0
    tprev       = 0
    time_day    = 1
    time_month  = Imonth_start
    time_year   = yrstart


    # calculate the sunrise and sunset times for the very first day (updated later)
    # date = str(int(time_year)) + '-' + str(int(time_month)) + '-' + str(int(time_day))
    # date = np.array([time_year, time_month, time_day]).astype(int)
    # tSR, tSS    = SunUpDown(date, latitude, longitude, daylight_saving, UTC)
    tSR, tSS    = SunUpDown(Yr[0], Mo[0], Dy[0], latitude, longitude, daylight_saving, UTC)


    day_cur     = np.sum(NoDays[:time_month-1]) + time_day          # Current day number

    # Calculate daily solar radiation.R_eng=Btu/ft^2-day
    Rday_eng    = SolarRadiation(latitude, day_cur)

    # # convert some parameters to list to speed up the calculations. numpy is slow in for loops
    # T           = T.tolist()
    # dz          = dz.tolist()
    # zi_in       = zi_in.tolist()
    # NoDays      = NoDays.transpose()
    # NoDays      = NoDays.tolist()[0]
    # PSun        = PSun.tolist()
    # U_mph       = U_mph.tolist()
    # dTdt        = dTdt.tolist()
    # Tair_F      = Tair_F.tolist()
    # Dy          = Dy.tolist()
    # Yr          = Yr.tolist()
    # Mo          = Mo.tolist()
    # t           = t.tolist()

    numTimeSteps = len(t)
    # Start iteration for each hour during the design year.
    for j in range(1, numTimeSteps):

        # Providing some how a progress bar.
        # if time() - tocp > 2:
        #     tocp = time()
        #     print('Climatic model: %.2f%% complete.' % (j / len(t) * 100) + ' working on climatic record: %s.' % date)


        Tsur_R = T[0] + 459.67  # Surface temperature from F to rankine
        Qx = sigm * epsl * Tsur_R ** 4  # The flux.
        Tair_R = Tair + 459.67  # Conversion from F to rankine


        # Calculate the solar radiation at different hours
        Rhr_eng = SolarRadiation_Daily2Hourly(Rday_eng, tSR, tSS, time_hr)

        # compute the net radiation flux (Qrad)
        Qrad = f_get_Q(PSun[k], climateModel, Qx, sigm, Tair_R, vp_mb, Nbart, Rhr_eng, A, B, N, G, J, rhoo, vp, a)

        # Calculating these parameters:
        #   - Vair  = air temperature (oC)
        #   - V1    = pavement surface temperature (oC)
        #   - Vm    = average air temperature and pavement surface temperature in Kelvin
        #   - U     = average daily wind velocity in m/sec
        Vair = (Tair - 32) / 1.8
        V1 = (T[0] - 32) / 1.8
        Vm = 273 + (V1 + Vair) / 2
        U_ms = U_mph[k] * 0.44704  # Conversion from mph to m/s
        H_1 = 0.00144 * Vm ** 0.3 * U_ms ** 0.7 + 0.00097 * abs(V1 - Vair) ** 0.3  # Convection coefficient
        #   (gm-cal/cm2-sec-C)
        H = 122.93 * H_1  # convert to btu/ft^2-hr-F

        # Re-calculate and check the dtmax.
        dtmax = gd[0] * C[0] * dz[0] / (2 * (H + K[0] / dz[0]))  # for stability.
        if dt > dtmax:
            # print('dt=%.2f' % str(dt) + ' > dtmax = %.2f' % dtmax)
            # raise Exception('dt exceeds dtmax for the surface-pavementheat transfer')
            # print(f'dt= {dt} > dtmax = {dtmax}')
            # print(f'dt exceeds dtmax for the surface-pavement heat transfer')
            dt = 0.95 * dtmax
            # print(f'new dt = {dt}')

            print(' --> dt exceeds dtmax for the surface-pavement heat transfer')

        # Update surface temperature T[0]
        T[0] = T[0] * (1 - 2 * K[0] * dt / (gd[0] * C[0] * dz[0] ** 2) - 2 * H * dt / (gd[0] * C[0] * dz[0])) + \
               T[1] * 2 * K[0] * dt / (gd[0] * C[0] * dz[0] ** 2) + Tair * 2 * H * dt / (gd[0] * C[0] * dz[0]) + \
               Qrad * 2 * dt / (gd[0] * C[0] * dz[0])

        # Iteration on each depth for updating the T marix.
        for i in range(1, n - 1):
            # Note that "n" is the len(zi_ft)
            if zi_in[i] in TH_in_cum:  # if at the interface
                m = list(TH_in_cum).index(zi_in[i])  # Index of the interface which is equal to zi_in[i]
                dtmax2 = dz[i] ** 2 * (C[m] * gd[m] + C[m + 1] * gd[m + 1]) / (2 * (K[m] + K[m + 1]))  # for stability.
                if dt > dtmax2:
                    # print('dt=%.2f' % str(dt) + ' > dtmax = %.2f' % dtmax)
                    dt = 0.95 * dtmax2
                    # raise Exception('dt exceeds dtmax for the interface')

                # Calculate the incremental temperature.
                dTdt[i] = 1 / dz[i] ** 2 / (C[m] * gd[m] + C[m + 1] * gd[m + 1]) * (T[i - 1] * 2 * K[m] - 2 * T[i] *
                                                                                    (K[m] + K[m + 1]) + 2 * T[i + 1] *
                                                                                    K[m + 1])
                # Update the 'm' value, in the case it was the last iteration.
                if zi_in[i + 1] in TH_in_cum:
                    m = list(TH_in_cum).index(zi_in[i + 1])
                elif m < len(TH_in_cum):
                    m += 1
            else:  # Not at the interface, but within the layer.
                # Specify the index of layer.
                if zi_in[i] < TH_in_cum[0]:
                    m = 0
                elif zi_in[i] > TH_in_cum[-1]:
                    m = len(TH_in_cum) - 1
                else:
                    # m = np.where(np.array(TH_in_cum) - zi_in[i] < 0)[0][-1]
                    m = np.where(TH_in_cum - zi_in[i] < 0)[0][-1]

                # Calculate the incremental temperature.
                dTdt[i] = alph[m] / dz[i] ** 2 * (T[i - 1] - 2 * T[i] + T[i + 1])

            T[i] = T[i] + dTdt[i] * dt

        # # The last and first incremental temperatures.
        dTdt[0] = 0  # taken care of above.
        dTdt[-1] = alph[m] / dz[-2] ** 2 * (T[-2] - 2 * T[-1] + Tcon)

        # Updating the temperature with depth.
        T[0] = T[0] + dTdt[0] * dt
        T[-1] = T[-1] + dTdt[-1] * dt

        if j != (len(t) - 1) and (t[j] - tprev) == 1:
            # -----------Go to the next HOUR INCREMENT------------
            # RadData['data'][k,:] = [Qrad, Qs, Qa, Qe, Qz, Qx, Rhr_eng] # Update the RadData dictionary.
            tprev = t[j]  # Update the previous time step.
            k = k + 1  # Update the time counter.
            Tair = Tair_F[k]
            time_hr += 1
            # Check whether the day is finished and we need to go to the next day.
            if time_hr > 23:
                # -----------Go to the next DAY INCREMENT-----------
                time_hr = 0
                time_day += 1
                # Check whether the month is finished and we need to go to the next month.
                if time_day > NoDays[time_month - 1]:
                    # -----------Go to the next MONTH INCREMENT ------------
                    time_day = 1
                    time_month += 1
                    if time_month > 12:
                        # -----------Go to the next YEAR INCREMENT------------
                        time_month = 1
                        time_year += 1

            # Update the radiation properties.
            # date = str(int(time_year)) + '-' + str(int(time_month)) + '-' + str(int(time_day))
            # date = np.array([time_year, time_month, time_day]).astype(int)
            # tSR, tSS = SunUpDown(date, latitude, longitude, daylight_saving, UTC)
            tSR, tSS = SunUpDown(time_year, time_month, time_day, latitude, longitude, daylight_saving, UTC)

            day_cur = np.sum(NoDays[:time_month - 1]) + time_day  # Current day number
            Rday_eng = SolarRadiation(latitude, day_cur)  # Calculate daily solar radiation.
            #   R_eng = Btu/ft^2-day
            Rhr_eng = SolarRadiation_Daily2Hourly(Rday_eng, tSR, tSS, time_hr)  # Update the solar radiation at
            #   current t

            # Use an interpolation for updating the temperature gradiant.
            # %%%%%%%%%%%%%%%%%% SLLLOOOWWWWW
            # Tp_F_mek = np.interp(zi_PV_in, zi_in, np.squeeze(T))
            # %%%%%%%%%%%%%%%%%% SLLLOOOWWWWW
            Tpave_mek_F[k - 1, :] = np.interp(zi_PV_in, zi_in, T) #Tp_F_mek
            Tsurf_mek_F[k - 1] = T[0]

    return Tpave_mek_F, Tsurf_mek_F