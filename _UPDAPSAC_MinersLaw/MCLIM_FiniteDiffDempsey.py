# ==== Import the required libraties ====
import numpy as np
import os
from time import time
from scipy.stats import norm
from mclimutil import f_read_hcd
from mclimutil_njit import f_iterate_throug_time

# from MClim_Util import f_iterate_throug_time

# Define the main function.
def FiniteDiffDempsey(AirTempFile, AirTempFolder, latitude, longitude, daylight_saving, UTC, openmonth,
    tdesign, gd, K, C, TH_in_cum, zi_PV_in, absorp, ClimDataType, climateModel):
    """
    This function is the main code for Finite Difference method of Dempsey.
    :param Input: This input variable is a dictionary of all required inputs, read from Json.
    :return: Several parameters as seen in the return section.
    """
    alph = K / (gd * C)  # Thermal diffusivity (ft2/hr)
    dt = 1 / 5 # time step in hrs: make sure multiples will yield integer hours
    Tcon = 51  #51  final constant temperature at a depth of 144 inches (Dempsey 1969)

    # Specify the opening month.
    MonthNames  = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                   'September', 'October', 'November', 'December']
    NoDays      = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]).reshape(-1,1)
    Imonth_start= [Indx for Indx in range(len(MonthNames)) if MonthNames[Indx].lower() == openmonth.lower()]
    Imonth_start= Imonth_start[0] + 1           # Stating from one (it was starting from zero).

    # read and process the climate file (.hcd or equivalent file)
    Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum, FreezeIndex, \
    Prec_in_an_avg, yrstart = f_read_hcd(AirTempFolder, AirTempFile, Imonth_start, tdesign, ClimDataType)

    # Post analysis of Tair_F, PSun, and zi_PV_in
    MAAT_F      = np.mean(Tair_F)
    PSun        = PSun / 100                # From percent to decimal.
    num_zi_PV   = len(zi_PV_in)

    # Calculate the critical coordinates
    Y           = 144                       # inches, termination depth <<<Should be Hard code?>>>
    X           = 73                        # inches %sum(TH_in);       <<<Should be Hard code?>>>
    dX          = X / 36.5
    W           = Y - X
    Nsubf       = 3
    dW          = W / Nsubf * np.ones(Nsubf)
    Wi          = np.cumsum(dW) - dW / 2    # Mid points.
    zi_in       = np.append(np.arange(0, X - dX / 2 + 1e-6, dX), list(Wi + X) + [Y + dW[0] / 2])
    zi_ft       = zi_in / 12                # convert in to ft
    n           = len(zi_ft)
    dz          = np.gradient(zi_ft)


    dtmax       = min(gd) * min(C) * min(dz) ** 2 / (2 * max(K))

    if dt > dtmax:
        print(f'dt= {dt} > dtmax = {dtmax}')
        print(f'dt exceeds dtmax for the surface-pavementheat transfer')
        dt = 0.95 * dtmax
        print(f'new dt = {dt}')
        # raise Exception('dt exceeds dtmax')

    # Defining time counters for calculating the solar radiations and sunrise and sunset during time.
    tend        = 24 * 365 * tdesign
    t           = np.arange(0, tend+1e-6, dt)
    TimeKeep    = time()


    # function to perform the main iterations through time
    Tpave_mek_F, Tsurf_mek_F = f_iterate_throug_time(t, PSun, U_mph, Tair_F, gd, C, dz, K, dt, zi_in, TH_in_cum, zi_PV_in, n, climateModel,
                          absorp, alph, Tcon,NoDays, latitude, longitude, daylight_saving, UTC,
                          Yr, Mo, Dy, Imonth_start, yrstart)

    # Display the time passed.
    TotalTime   = time() - TimeKeep
    print('Climatic model has been successfully within %.2f seconds!' % TotalTime)

    # Calculating air quintiles
    print('Calculating temperature quintiles:')
    numMo = tdesign * 12
    NL = len(zi_PV_in)
    Tsubdum_M = np.zeros((int(numMo), NL+1, 8))

    for j in range(num_zi_PV):                                              # Iteration for different depths
        TquintPavezi_mek    = CalculateQuantiles(Yr, Mo, Tpave_mek_F[:,j], 0)
        for i in range(len(TquintPavezi_mek)):
            Tsubdum_M[i, j, 0] = j + 1
            Tsubdum_M[i, j, 1:] = TquintPavezi_mek[i]

    # # These variables do not needed in the code, but still here as they exists in Prof. Kutay's one.
    # tdesignHRS  = tdesign * 365 * 24                                        # total number of hours during design life
    # Tair_FHRS   = len(Tair_F)

    # Returning section.
    return Tsubdum_M, MAAT_F, Yr, Mo, Tpave_mek_F, Tsurf_mek_F, Prec_in, FreezeIndex, Prec_in_an_avg, \
           yrstart, Dy, Hr



# ======================================================================================================================
def CalculateQuantiles(Year, Month, TEMP, PLOT = 0):
    """
    This function calculates five quantiles for the temperatures within each month during the analysis pavement life. In
        this funciton, I seperated the plot function as it required further computations. However, for most cases the
        plotting section is not activated and the function can be more optimized.
        <<<The plot section is not included yet>>>
        The input parameters are:
    :param Yr: An array of years for each hourly data point.
    :param Mo: An array of month for each hourly data point.
    :param Tpave_mek_F: An array of hourly temperature at the given depth for each hour.
    :param PLOT: The boolian variable for plotting or not plotting the results.
    :return: The quantiled data: refer to the return section.
    """

    # Specify the reliability levels.
    ReliabilityLevels   = [0.1, 0.30, 0.50, 0.70, 0.90]                 # In decimal.
    z_quint             = norm.ppf(ReliabilityLevels)                   # Convert to the standard normal deviate values.

    if PLOT == 0:
        # If PLOT variable is NOT activated.

        # Prepare the variables.
        Yu                  = np.unique(Year)
        Tquintiles          = np.array([])
        for i in range(len(Yu)):                                        # Iteration on each year.
            T_cur_Yr        = TEMP[Year ==  Yu[i]]                      # An array of temperatures in current year.
            Mo_cur_Yr       = Month[Year == Yu[i]]                      # An array of monthes in the current year.
            Mo              = np.unique(Mo_cur_Yr)

            for j in range(len(Mo)):                                    # Iteration on each month in the current year.
                T_cur_Mo    = T_cur_Yr[Mo_cur_Yr == Mo[j]]              # An array of temperatures in the current month.

                # Fitting a normal distribution to the data points.
                mu, std     = norm.fit(T_cur_Mo)

                # Calculate the temperature quantiles.
                x_Tqi       = z_quint * std + mu

                # Save the results.
                Tquintiles  = np.append(Tquintiles, [Yu[i], Mo[j]] + list(x_Tqi))

        # Return the quantiles.
        Tquintiles          = Tquintiles.reshape(-1, 7)
        return Tquintiles

    else:
        # If PLOT variable is activated.
        pass
        # NOT included yet.