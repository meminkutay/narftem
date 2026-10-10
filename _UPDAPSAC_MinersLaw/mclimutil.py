from MCLIM_01ReadHCD import ReadHCM, ReadHCM_msrdQrad
import os

def f_read_hcd(AirTempFolder, AirTempFile, Imonth_start, tdesign, ClimDataType):
    # Reading the HCD file.
    if ClimDataType.upper() == 'NARR':
        print('Reading the NARR data file:')
        Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum, FreezeIndex, Prec_in_an_avg, yrstart =  \
            ReadHCM(os.path.join(AirTempFolder, AirTempFile), AirTempFile, Imonth_start, tdesign, 'NARR')

    elif ClimDataType.upper() == 'MERRA2':
        print('Reading the MERRA2 data file')
        Yr, Mo, Dy, Hr, Tair_C, U_ms, PSun, Prec_mm, PHum, FreezeIndex, Prec_in_an_avg, yrstart =  \
            ReadHCM(os.path.join(AirTempFolder, AirTempFile), AirTempFile, Imonth_start, tdesign, 'MERRA2')
        # convert MERRA temp in C to F
        Tair_F      = (Tair_C * 1.8) + 32
        Prec_in     = Prec_mm / 25.4
        U_mph       = U_ms    * 2.23964

    elif ClimDataType.upper() == 'MERRA2-MSRDQ':
        print('Reading the MERRA2-mod data file')
        Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum, FreezeIndex, Prec_in_an_avg, yrstart, Qshortwave_wm2, \
        SolarRad_wm2, Qlongwave_wm2 =  \
            ReadHCM_msrdQrad(os.path.join(AirTempFolder, AirTempFile), AirTempFile, Imonth_start, tdesign)

    else:
        raise Exception('Could not read the hcd file')
    return Yr, Mo, Dy, Hr, Tair_F, U_mph, PSun, Prec_in, PHum, FreezeIndex, Prec_in_an_avg, yrstart
