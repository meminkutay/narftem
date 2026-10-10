# === importing the required libraries.
import gzip
import numpy as np
import pandas as pd


def ReadHCM(FilePath, AirTempFile, Imonth_start, tdesign, ClimaticModel):
    """
    This function is for reading the input using HCD or HCM files or from web.
        The HCM file could have upto 6 headers as:
            1. Date/Hour
            2. Temperature (F)
            3. Wind Speed (mph)
            4. Sunshine (%)
            5. Precipitation (in)
            6. Humidity (%)
    :param FilePath: The file path of the Airtemperatures, also maybe a URL.
    :param AirTempFile: The file name: Not required by now, but if the download method changes, it may be required.
    :param Imonth_start: The index of the starting month, starting from <0>.
    :param tdesign: The design life of the pavement.
    :param ClimaticModel: Could be "NARR" in which temperature in F and percepitation in in. Or it could be "MERRA2" in
        which temperature in C and percipitation in mm.
    :return: Refer to the return section for the returned parameters.
    """
    print('Reading climatic hcm file: %s' % FilePath)

    # === Read the file in an try/except structure ===
    A = ReadClimateData(FilePath)
    Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum = extractDATAfromA(A)

    # Available years in the data set.
    YrsAvail = np.unique(Yr)

    # Specify the first and last years.
    yrstart = Yr[0]
    yrend = np.floor(yrstart + tdesign)

    # pdb.set_trace()
    # Check whether there are enough data for the design period.
    if max(YrsAvail) <= yrend:
        # Repeat the airTemp matrix and re-extract the data from it.
        A = RepeatData(A, yrend)
        # Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum = extractDATAfromA(A)
        # YrsAvail    = np.unique(Yr)                     # Available years in the data set.

    # # Create and filter the "DATA" matrix.
    # DATA        = np.concatenate((Yr.reshape(-1,1),   Mo.reshape(-1,1),      Dy.reshape(-1,1),
    #                               Hr.reshape(-1,1),   Tair_F.reshape(-1,1),  U_mph.reshape(-1,1),
    #                               PSun.reshape(-1,1), Prec_in.reshape(-1,1), PHum.reshape(-1,1)),
    #                              axis = 1)

    # for some reason, the code below does not work at ANL's clusters!!!
    # DATA = delrows(DATA, yrstart, yrend, Imonth_start)
    # print(f' --> Size of DATA = {len(DATA)}')
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    DATA = cropdata(A, yrstart, yrend, Imonth_start)
    print(f' --> Size of DATA2 = {len(DATA)}')

    # pdb.set_trace()
    # Re-extract the data
    Yr = DATA[:, 0]
    Mo = DATA[:, 1]
    Dy = DATA[:, 2]
    Hr = DATA[:, 3]
    Tair_F = DATA[:, 4]
    U_mph = DATA[:, 5]
    PSun = DATA[:, 6]
    Prec_in = DATA[:, 7]
    PHum = DATA[:, 8]

    # Calculating the Freezing index (NOT Prof. Kutay's code).
    if ClimaticModel == 'NARR':
        Tair_Freez = Tair_F - 32
    else:
        Tair_Freez = Tair_F
    DelRows = np.where(Tair_Freez >= 0)[0]
    Tair_Freez = np.delete(Tair_Freez, DelRows, axis=0)
    FreezeIndex = -np.sum(Tair_Freez) / 24 / tdesign  # In degree-days per year
    print('Freeze Index = ' + str(FreezeIndex) + ' Degree F days per year')
    # Calculating the average precipitation.

    Prec_in_anhr = Prec_in.reshape(-1, 24 * 365)
    Prec_in_an = np.sum(Prec_in_anhr, axis=1)
    Prec_in_an_avg = round(np.mean(Prec_in_an), 1)

    # Returning section.
    return Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum, FreezeIndex, Prec_in_an_avg, yrstart


def extractDATAfromA(A):
    # === Analyzing the data in the file ===
    Date = A[:, 0]
    Tair_F = A[:, 1]
    U_mph = A[:, 2]
    PSun = A[:, 3]
    Prec_in = A[:, 4]
    PHum = A[:, 5]

    # extract the exact hour, day, month, and year data.
    Hr = np.mod(Date, 10 ** 2)
    Dy = np.floor(np.mod(Date, 10 ** 4) / 10 ** 2)
    Mo = np.floor(np.mod(Date, 10 ** 6) / 10 ** 4)
    Yr = np.floor(np.mod(Date, 10 ** 10) / 10 ** 6)

    return Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum


def cropdata(A, yrstart, yrend, Imonth_start):
    Date = A[:, 0]
    Date0 = yrstart * 1e6 + Imonth_start * 1e4
    Date1 = yrend * 1e6 + Imonth_start * 1e4

    Ind = np.all([Date >= Date0, Date < Date1], axis=0)
    A2 = A[Ind].copy()

    Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum = extractDATAfromA(A2)

    DATA = np.concatenate((Yr.reshape(-1, 1), Mo.reshape(-1, 1), Dy.reshape(-1, 1),
                           Hr.reshape(-1, 1), Tair_F.reshape(-1, 1), U_mph.reshape(-1, 1),
                           PSun.reshape(-1, 1), Prec_in.reshape(-1, 1), PHum.reshape(-1, 1)),
                          axis=1)

    # # Eliminate the data of Feburary 29th, because it is leap day
    DelRows = np.all([DATA[:, 1] == 2, DATA[:, 2] == 29], axis=0)
    Ikeep = np.where(DelRows == False)[0]
    DATA2 = DATA[Ikeep].copy()

    # for some reason, the code below does not work at ANL's clusters!!!
    # DelRows     = np.all([DATA[:,1] == 2,   DATA[:,2] == 29], axis = 0)
    # DATA        = np.delete(DATA, DelRows, axis = 0)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return DATA2


# for some reason, the def below does not work at ANL's clusters!!!
def delrows(DATA, yrstart, yrend, Imonth_start):
    DATA = DATA[np.all([DATA[:, 0] >= yrstart, DATA[:, 0] <= yrend], axis=0)]
    DelRows = np.all([DATA[:, 0] == yrstart, DATA[:, 1] < Imonth_start], axis=0)
    DATA = np.delete(DATA, DelRows, axis=0)
    DelRows = np.all([DATA[:, 0] == yrend, DATA[:, 1] >= Imonth_start], axis=0)
    DATA = np.delete(DATA, DelRows, axis=0)

    # Eliminate the data of Feburary 29th, because it is leap day
    DelRows = np.all([DATA[:, 1] == 2, DATA[:, 2] == 29], axis=0)
    DATA = np.delete(DATA, DelRows, axis=0)

    return DATA


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# ======================================================================================================================
def ReadHCM_msrdQrad(FilePath, AirTempFile, Imonth_start, tdesign):
    """
    This function is for reading the input using HCD or CSV files or from web.
        The HCM file could have upto 6 headers as:
            1. Date/Hour
            2. Temperature (F)
            3. Wind Speed (mph)
            4. Sunshine (%)
            5. Precipitation (in)
            6. Humidity (%)
            7. Net Incoming Surface Shortwave Radiation (Watt/squared meter)
            8. Top of Atmosphere Solar Radiation (Watt/squared meter)
            9. Longwave Radiation (Watt/squared meter)

    :param FilePath: The file path of the Airtemperatures, also maybe a URL.
    :param AirTempFile: The file name: Not required by now, but if the download method changes, it may be required.
    :param Imonth_start: The index of the starting month, starting from <0>.
    :param tdesign: The design life of the pavement.
    :return: Refer to the return section for the returned parameters.
    """
    print('Reading climatic hcm file: %s' % FilePath)

    # === Read the file in an try/except structure ===
    A = ReadClimateData(FilePath)

    # === Analyzing the readed file ===
    Date = A[:, 0]
    Yr = np.floor(np.mod(Date, 10 ** 10) / 10 ** 6)
    YrsAvail = np.unique(Yr)  # Available years in the data set.

    # Eliminate the first year data, due to incomplete data and relevant complications.
    DelRows = np.where(Yr == YrsAvail[0])[0]
    A = np.delete(A, DelRows, axis=0)

    # continue computations starting the second year data
    Date = A[:, 0]
    Tair_F = A[:, 1]
    U_mph = A[:, 2]
    PSun = A[:, 3]
    Prec_in = A[:, 4]
    PHum = A[:, 5]
    Qshortwave_wm2 = A[:, 6]
    SolarRad_wm2 = A[:, 7]
    Qlongwave_wm2 = A[:, 8]

    # extract the exact hour, day, month, and year data.
    Hr = np.mod(Date, 10 ** 2)
    Dy = np.floor(np.mod(Date, 10 ** 4) / 10 ** 2)
    Mo = np.floor(np.mod(Date, 10 ** 6) / 10 ** 4)
    Yr = np.floor(np.mod(Date, 10 ** 10) / 10 ** 6)
    YrsAvail = np.unique(Yr)  # Available years in the data set.

    # Specify the first and last years.
    yrstart = Yr[0]
    yrend = np.floor(yrstart + tdesign)

    # Check whether there are enough data for the design period.
    if max(YrsAvail) <= yrend:
        # Repeat the airTemp matrix and re-extract the data from it.
        A = RepeatData(A, yrend)
        Date = A[:, 0]
        Tair_F = A[:, 1]
        U_mph = A[:, 2]
        PSun = A[:, 3]
        Prec_in = A[:, 4]
        PHum = A[:, 5]
        Qshortwave_wm2 = A[:, 6]
        SolarRad_wm2 = A[:, 7]
        Qlongwave_wm2 = A[:, 8]
        # extract the exact hour, day, month, and year data.
        Hr = np.mod(Date, 10 ** 2)
        Dy = np.floor(np.mod(Date, 10 ** 4) / 10 ** 2)
        Mo = np.floor(np.mod(Date, 10 ** 6) / 10 ** 4)
        Yr = np.floor(np.mod(Date, 10 ** 10) / 10 ** 6)
        YrsAvail = np.unique(Yr)  # Available years in the data set.

    # Create and filter the "DATA" matrix.
    DATA = np.concatenate((Yr.reshape(-1, 1), Mo.reshape(-1, 1), Dy.reshape(-1, 1),
                           Hr.reshape(-1, 1), Tair_F.reshape(-1, 1), U_mph.reshape(-1, 1),
                           PSun.reshape(-1, 1), Prec_in.reshape(-1, 1), PHum.reshape(-1, 1),
                           Qshortwave_wm2.reshape(-1, 1), SolarRad_wm2.reshape(-1, 1),
                           Qlongwave_wm2.reshape(-1, 1)),
                          axis=1)
    DATA = DATA[np.all([DATA[:, 0] >= yrstart, DATA[:, 0] <= yrend], axis=0)]
    DelRows = np.all([DATA[:, 0] == yrstart, DATA[:, 1] < Imonth_start], axis=0)
    DATA = np.delete(DATA, DelRows, axis=0)
    DelRows = np.all([DATA[:, 0] == yrend, DATA[:, 1] >= Imonth_start], axis=0)
    DATA = np.delete(DATA, DelRows, axis=0)

    # Eliminate the data of Feburary 29th, because it is leap day.
    DelRows = np.all([DATA[:, 1] == 2, DATA[:, 2] == 29], axis=0)
    DATA = np.delete(DATA, DelRows, axis=0)

    # Re-extract the data
    Yr = DATA[:, 0]
    Mo = DATA[:, 1]
    Dy = DATA[:, 2]
    Hr = DATA[:, 3]
    Tair_F = DATA[:, 4]
    U_mph = DATA[:, 5]
    PSun = DATA[:, 6]
    Prec_in = DATA[:, 7]
    PHum = DATA[:, 8]
    Qshortwave_wm2 = DATA[:, 9]
    SolarRad_wm2 = DATA[:, 10]
    Qlongwave_wm2 = DATA[:, 11]

    # Calculating the Freezing index (NOT Prof. Kutay's code).
    Tair_Freez = Tair_F - 32
    DelRows = np.where(Tair_Freez >= 0)[0]
    Tair_Freez = np.delete(Tair_Freez, DelRows, axis=0)
    FreezeIndex = -np.sum(Tair_Freez) / 24 / tdesign  # In degree-days per year

    # Calculating the average precipitation.
    Prec_in_anhr = Prec_in.reshape(-1, 24 * 365)
    Prec_in_an = np.sum(Prec_in_anhr, axis=1)
    Prec_in_an_avg = round(np.mean(Prec_in_an), 1)

    # Returning section.
    return Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum, FreezeIndex, Prec_in_an_avg, yrstart, Qshortwave_wm2, \
        SolarRad_wm2, Qlongwave_wm2


# ======================================================================================================================
def ReadClimateData(FilePath):
    try:
        with gzip.open(FilePath, "rt") as f_in:  # "rt" is for reading text
            df = pd.read_csv(f_in, header=None)
            print(f'Read the {FilePath}')

        A = df.to_numpy()

        # # Generate the CSV file path by removing the '.gz' extension
        # csv_file_path = os.path.splitext(FilePath)[0]
        #
        # # Write the DataFrame to a CSV file
        # df.to_csv(csv_file_path, index=False)
        # print(f'Wrote the data to {csv_file_path}')

    except Exception as e:

        raise Exception(f'Error: {e} while reading {FilePath}')

    # Return section.
    return A


# ======================================================================================================================
def RepeatData(Mat, YearMax):
    """
    This function is for repeating the AirTemperature matrix several times, till the maximum year is
        included in the design.
    :param Mat: The intended matrix.
    :param YearMax: The maximum year.
    :return: The updated Mat matrix is returned.
    """

    # Print the warning that the data set was not covering all the design period.
    print('********************************')
    print('The analysis duration is longer than the climatic data avialable. The LAST YEAR is Copyid to the end ' + \
          'of the data tiil the design period is met (i.e., not mirroring)...')
    print('********************************')

    # Find the corresponding date of the last date in its previous year.
    Date = Mat[:, 0]
    lastDate = str(Date[-1])
    lastYr = int(lastDate[:4]) - 1
    datePrevYr = float(str(lastYr) + lastDate[4:])
    Indx1 = np.where(Date == datePrevYr)[0]
    if len(Indx1) == 0:
        print('Last date is %s' % lastDate + ', but corresponding previous year <%s> is not in dataset!' % str(
            datePrevYr))
        raise Exception('At least 1 year of climatic data is needed!!!')

    # Required number of repeating the last year.
    RepeatNumber = YearMax - lastYr + 1
    Indx1 = int(Indx1)

    # Specify the matrix of last year.
    RepeatedMat = Mat[Indx1 + 1:, :].copy()

    # Add the repeated mat to the input Matrix.
    for ii in range(RepeatNumber):
        RepeatedMat[:, 0] += 1e6  # Add one year to the Repeated matrix.
        Mat = np.concatenate((Mat, RepeatedMat), axis=0)

    # Return the updated Matrix.
    return Mat
