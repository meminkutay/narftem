import pickle
import sys
import time
from os import path
import os
from UPDAPSAC_STEP1 import UPDAPSAC_STEP1
from UPDAPSAC_STEP2 import f_struc_resp_ACGB, f_struc_resp_ACCSM, f_struc_resp_ACEAC
import updapsutil as utl
from f_calc_distress_atrel import f_calc_distress_atrel
from datetime import datetime


def UPCAPSACRunOneFile(inps, jsonfld, jsonfile, fld_out, json_output_file,
                       StatusTxtFile, ErrorTxtFile,
                       local_climatic_file):
    print(f' -->>> INPUT TO UPCAPSACRunOneFile IS {jsonfld}, {jsonfile}')

    try:

        jsonfilenoext = os.path.splitext(jsonfile)[0]

        inps['fldAirTemp'], inps['fnameAirTemp'] = os.path.split(local_climatic_file)

        starttime = time.perf_counter()

        print(f'Output folder: {fld_out}\nJson_output_file: {json_output_file}\n StatusTxtFile: {StatusTxtFile}')

        txttoadd = f' --> Starting Step I for file {jsonfile} @ {jsonfld} on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        print(txttoadd)
        utl.appendtxt(fld_out, StatusTxtFile, txttoadd)

        # ___ STEP I is a preprocessor, common to all AC pavement types,
        # ___ It also includes climatic and thermal cracking models _____
        # Step1filename = fld_out + "/" + jsonfilenoext + "-Step1Output"
        Step1filename = os.path.join(fld_out, jsonfilenoext + "-Step1Output")

        # In windows, use // instead of backslash

        if path.exists(Step1filename + ".pkl"):
            # This is to skip the step I, if it is previously ran already.
            # USE ONLY DURING DEVELOPMENT STAGE, TO SKIP STEP I.
            # THIS IF STATEMENT IS WILL BE DELETED ONCE THE CODING IS COMPLETED!!!
            Step1Output = pickle.load(open(Step1filename + ".pkl", "rb"))
            print(f' !!! --> Step I is loaded from previously run data <-- !!!')
        else:
            # _________________________________________________________________________________________
            Step1Output = UPDAPSAC_STEP1(inps, fld_out, StatusTxtFile)
            # _________________________________________________________________________________________
            # Save outputs - only during development stage...
            # savemat(Step1filename + ".mat", Step1Output)
            # pickle.dump(Step1Output, open(Step1filename + ".pkl", "wb"))

        print(f' --> Step I (preprocessing + climate model + thermal cracking) is done for file {jsonfile} @ {jsonfld}')

        txttoadd = f' --> Starting Step II (struct response) for file {jsonfile} @ {jsonfld} \n'
        print(txttoadd)
        utl.appendtxt(fld_out, StatusTxtFile, txttoadd)

        # ___ STEP II is where the structural and damage responses are computed
        CSMExists = Step1Output['CSMExists']
        EACExists = Step1Output['EACExists']
        Step1Output['fld_out'] = fld_out
        Step1Output['StatusTxtFile'] = StatusTxtFile
        Step1Output['json_output_file'] = json_output_file

        if not CSMExists and not EACExists:
            # _________________________________________________________________________________________
            outputPythi = f_struc_resp_ACGB(Step1Output)
            # outputPythi = {'Debug': True}
            # _________________________________________________________________________________________

        elif CSMExists and not EACExists:
            # ___________________________________________________________________________________________
            outputPythi = f_struc_resp_ACCSM(Step1Output)
            # _________________________________________________________________________________________

        elif EACExists and not CSMExists:
            # _________________________________________________________________________________________
            outputPythi = f_struc_resp_ACEAC(Step1Output)
            # _________________________________________________________________________________________

        else:
            print(f' ***** --> this type of structure cannot be modeled')
            print(f' ***** --> this type of structure cannot be modeled')
            print(f' ***** --> this type of structure cannot be modeled')
            sys.exit()

        print(f' --> Step II (struct response) is done for file {jsonfile} @ {jsonfld}')

        outputPythi['TC_mo'] = Step1Output['TC_mo'].tolist()
        TotalElapsedTime: float = time.perf_counter() - starttime
        print(f' --> Total Elapsed Time for this JSON is {TotalElapsedTime: .6} seconds')
        outputPythi['Runtime_sec'] = int(TotalElapsedTime)
        outputPythi['jsonfilenoext'] = jsonfilenoext

        # Save outputs
        # savemat(fld_out + "/" + jsonfilenoext + "-Python-output.mat", outputPythi)
        # pickle.dump(outputPythi, open(fld_out + "/" + jsonfilenoext + "-Python-output.pkl", "wb"))
        # outputPythi = utl.Numpy2List4JsonSave(outputPythi)

        if 'TRCac_mo' in outputPythi:
            ld = {
                "FCbu": outputPythi['FCbu_mo'],
                "FCtd": outputPythi['FCtd_mo'],
                "rut_TAC": outputPythi['rut_tot_AC_mo'],
                "rut_TBASE": outputPythi['rut_tot_BASE_mo'],
                "rut_TSUBG": outputPythi['rut_tot_SUBG_mo'],
                "rutTOT": outputPythi['rutTOT_mo'],
                "IRI": outputPythi['IRI_mo'],
                "TC_mo": outputPythi['TC_mo'],
                "TRCac": outputPythi['TRCac'],
                "CSMExists": CSMExists,
                "EACExists": EACExists,
            }

        else:

            ld = {
                "FCbu": outputPythi['FCbu_mo'],
                "FCtd": outputPythi['FCtd_mo'],
                "rut_TAC": outputPythi['rut_tot_AC_mo'],
                "rut_TBASE": outputPythi['rut_tot_BASE_mo'],
                "rut_TSUBG": outputPythi['rut_tot_SUBG_mo'],
                "rutTOT": outputPythi['rutTOT_mo'],
                "IRI": outputPythi['IRI_mo'],
                "TC_mo": outputPythi['TC_mo'],
                "CSMExists": CSMExists,
                "EACExists": EACExists,
            }

        stdev_data = inps['stdevData']
        reliability_thresholds = inps['reliabilityThresholds']

        ld = f_calc_distress_atrel(ld, stdev_data, reliability_thresholds, plt_flag=0)
        # Update `outputPythi` with the modified variables from `ld`
        outputPythi.update({
            "totrutAtRel": ld["totrutAtRel"],
            "acrutAtRel": ld["acrutAtRel"],
            "fcbuAtRel": ld["fcbuAtRel"],
            "fctdtAtRel": ld["fctdtAtRel"],
            "tc_ACAtRel": ld["tc_ACAtRel"],
            "iriAtRel": ld["iriAtRel"],
        })

        # If 'CSMExists' or 'EACExists' conditions apply, add reflective cracking variables
        if ld.get('CSMExists') or ld.get('EACExists'):
            outputPythi.update({
                "reflCrackACAtRel": ld["reflCrackACAtRel"],
            })

        utl.save_to_output(outputPythi, fld_out, json_output_file)
        txttoadd = f'Run completed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        print(txttoadd)
        utl.appendtxt(fld_out, StatusTxtFile, txttoadd)

    except Exception as ERR:
        print(ERR)
        txttoadd = f' --> An error has been encountered: {ERR} \n'
        print(txttoadd)
        utl.appendtxt(fld_out, ErrorTxtFile, txttoadd)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        txttoadd = f'Error details: {exc_type}, {fname}, {exc_tb.tb_lineno}'
        print(txttoadd)
        utl.appendtxt(fld_out, StatusTxtFile, txttoadd)
        outputPythi = []

    return outputPythi
