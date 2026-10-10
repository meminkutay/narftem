import datetime
import fnmatch
import json
import math
import os
import pickle
import numpy as np
import boto3
import threading

s3_client = boto3.client('s3')
# Thread-safe lock for shared writes
write_lock = threading.Lock()
status_logs = []

def Search4JsonFiles(path, SubDir, JsonList):
    """
    This function search for any Json input files in the provided path and collect them.
    :param path: The directory of the current folder, provided by user.
    :param SubDir: The sub-directory of the Json in the parent directory.
    :param JsonList: The list of all Json files in the directory.
    :return: The updated JsonList variable.
    """
    for fileName in os.listdir(path):
        if os.path.isdir(os.path.join(path, fileName)):  # The recursive part for searching sub-folders.
            JsonList = Search4JsonFiles(os.path.join(path, fileName),
                                        os.path.join(SubDir, fileName),
                                        JsonList)
            continue
        # Now check the files for Json ones.
        if fnmatch.fnmatch(fileName, '*.json'):
            print(f'File added: {os.path.join(path, fileName)}')
            JsonList.append({'LocalPath': os.path.join(path, fileName),
                             'FileName': fileName,
                             'SubDir': SubDir})
    # Return the updated JsonList.
    return JsonList


def Numpy2List4JsonSave(Data):
    """
    This function converts all the numpy data types into the list for a given dictionary. This is because the "json"
        module can not handle the numpy data types.
    NOTE: This function only supports the inputs of the "Dictionary" and "list" types.
    :param Data: The input dictionary to be treated.
    :return: The updated Data dictionary.
    """
    if type(Data) == dict:
        for key in Data.keys():
            print(f' - {key}')
            if type(Data[key]) == np.ndarray:
                Data[key] = Data[key].tolist()
            else:
                Data[key] = Numpy2List4JsonSave(Data[key])
    elif type(Data) == list:
        for i in range(len(Data)):
            if type(Data[i]) == np.ndarray:
                Data[i] = Data[i].tolist()
            else:
                Data[i] = Numpy2List4JsonSave(Data[i])
    else:

        raise ValueError(f'The variable input format is <{type(Data)}>, which is NOT supported. This function only '
                         f'handle "dict" or "list".')
    return Data


def create_json_clusters_and_job_files(JsonFiles2Run, numclusters, eltimemean, eltimemax, LocalJsonFolder,
                                       servernode, RemoteCodeDir, RemoteCodeName, ppnreq, anl_user):
    # Creating the clusters for running.
    #   Each cluster should take less than "targetruntime" run time. Using a quick run time study, the average time for
    #   running each pavement section (20 years design life) is about eltimemean minutes, and for 40 years design life can take
    #   up to eltimemax minutes. Therefore, here we assumed the minimum run time of 5 minutes and using a linear interpolation
    #   for design lives of more than 20 years.
    cluster_jsons = []
    cluster_key = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '-')
    RunTime = 0
    numjson = len(JsonFiles2Run)

    numjsonpercluster = math.ceil(numjson / numclusters)

    allclusters = {}
    n = 0
    ClusterNo = 0

    fldout_job = os.path.join(RemoteCodeDir, '_Jobs_' + cluster_key)
    if not os.path.isdir(fldout_job):
        os.mkdir(fldout_job)

    for i in range(numjson):
        # Read the design life of the current pavement.
        with open(JsonFiles2Run[i]['LocalPath'], 'r') as file:
            DesignLife = float(json.load(file)['tDesign'])
        EstimatedRunTime = eltimemean + (eltimemax - eltimemean) * (DesignLife - 20.0)
        EstimatedRunTime = eltimemean if EstimatedRunTime < eltimemean else EstimatedRunTime
        JsonFiles2Run[i]['EstimatedRunTime_min'] = np.ceil(EstimatedRunTime)
        JsonFiles2Run[i]['RemotePath'] = JsonFiles2Run[i]['LocalPath']

        if (n >= numjsonpercluster):
            clustercur = {}

            # Save the cluster in the folder
            RunTime_hr = math.ceil(RunTime / 60.0 * 1.2)  # 20% more than estimated runtime
            Twall = f'{RunTime_hr}:00:00'
            cluster_pkl_path, cluster_job_path = save_jsoncluster(cluster_jsons, fldout_job,
                                                                  cluster_key, ClusterNo, servernode,
                                                                  RemoteCodeDir, RemoteCodeName, Twall, ppnreq)

            clustercur['cluster_jsons'] = cluster_jsons
            clustercur['cluster_key'] = cluster_key
            clustercur['cluster_pkl_path'] = cluster_pkl_path
            clustercur['cluster_job_path'] = cluster_job_path
            allclusters[ClusterNo] = clustercur

            # submit the job to anl
            job_path = clustercur['cluster_job_path']
            print(f'Submitting {job_path}')
            os.system(f'qsub {job_path}; '
                      f'qstat -u {anl_user}; ')

            # Reset parameters for next cluster
            cluster_key = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '-')
            cluster_jsons = []
            n = 0
            ClusterNo = ClusterNo + 1
            RunTime = EstimatedRunTime

        else:

            RunTime += EstimatedRunTime

        cluster_jsons.append(JsonFiles2Run[i])
        n = n + 1

    # Save the last cluster in the folder
    RunTime_hr = math.ceil(RunTime / 60.0 * 1.2)  # 20% more than estimated runtime
    Twall = f'{RunTime_hr}:00:00'
    cluster_pkl_path, cluster_job_path = save_jsoncluster(cluster_jsons, fldout_job,
                                                          cluster_key, ClusterNo, servernode,
                                                          RemoteCodeDir, RemoteCodeName, Twall, ppnreq)

    clustercur['cluster_jsons'] = cluster_jsons
    clustercur['cluster_key'] = cluster_key
    clustercur['cluster_pkl_path'] = cluster_pkl_path
    clustercur['cluster_job_path'] = cluster_job_path
    allclusters[ClusterNo] = clustercur

    # submit the job to anl
    job_path = clustercur['cluster_job_path']
    print(f'Submitting {job_path}')
    os.system(f'qsub {job_path}; '
              f'qstat -u {anl_user}; ')

    return allclusters


def save_jsoncluster(JsonList, fldout_job, CurrentKey, ClusterNo, HPCCnode, RemoteCodeDir, RemoteCodeName, Twall,
                     ppnreq):
    RemotePklPath = os.path.join(fldout_job, f'JsonList-{len(JsonList)}-{CurrentKey}.pkl')
    RemoteJobPath = os.path.join(fldout_job, f'Job-{len(JsonList)}-{CurrentKey}.job')

    with open(RemotePklPath, 'wb') as file:
        pickle.dump(JsonList, file)

    create_job_files(RemoteJobPath, HPCCnode, RemoteCodeDir, RemoteCodeName, RemotePklPath, Twall, ppnreq)

    print(f'Cluster {ClusterNo} with {len(JsonList)} jsons has been saved: {RemotePklPath}')
    print(f'    Job file has been saved: {RemoteJobPath}')
    return RemotePklPath, RemoteJobPath


def create_job_files(LocalJobPath, HPCCnode, RemoteCodeDir, RemoteCodeName, RemotePklPath, Twall, ppnreq):
    # Generating the job file for submition to HPCC.
    print('--> Generating the job file:' + LocalJobPath)
    # generate the job file in the local directory first.
    with open(LocalJobPath, 'w') as outfile:
        outfile.write('# \n')
        outfile.write(f'#PBS -q {HPCCnode}              \n')
        outfile.write(f'#PBS -l nodes=1:ppn={ppnreq}         \n')
        outfile.write(f'#PBS -l walltime={Twall}       \n')
        outfile.write('\n')
        outfile.write(f'module load python/3.8.11;\n '
                      f'cd "{RemoteCodeDir}";\n'
                      f'python3 {RemoteCodeName} "{RemotePklPath}" \n')


def create_json_clusters_basedon_runtime(JsonFiles2Run, targetruntime):
    # Creating the clusters for running.
    #   Each cluster should take less than "targetruntime" run time. Using a quick run time study, the average time for
    #   running each pavement section (20 years design life) is about 5 minutes, and for 40 years design life can take
    #   up to 8 minutes. Therefore, here we assumed the minimum run time of 5 minutes and using a linear interpolation
    #   for design lives of more than 20 years.
    Clusters = {}
    CurrentKey = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '-')
    Clusters[CurrentKey] = []
    RunTime = 0

    for i in range(len(JsonFiles2Run)):
        # for i in range(2):
        # Read the design life of the current pavement.
        with open(JsonFiles2Run[i]['LocalPath'], 'r') as file:
            DesignLife = float(json.load(file)['tDesign'])
        EstimatedRunTime = 8.0 + (15.0 - 8.0) * (DesignLife - 20.0)
        EstimatedRunTime = 8.0 if EstimatedRunTime < 8.0 else EstimatedRunTime
        TotalTime = RunTime + EstimatedRunTime
        if TotalTime > targetruntime:  #
            # time.sleep(0.5)  # Half a second pause.
            print(
                f'Cluster number <{len(Clusters)}> with number of jsons = {len(Clusters[CurrentKey])} has been added: {CurrentKey}')

            CurrentKey = str(datetime.datetime.now()).replace(':', '-').replace('.', '-').replace(' ', '-')
            Clusters[CurrentKey] = []
            Clusters[CurrentKey].append(JsonFiles2Run[i])

            RunTime = EstimatedRunTime
        else:
            RunTime += EstimatedRunTime
            Clusters[CurrentKey].append(JsonFiles2Run[i])
    print(
        f'Cluster number <{len(Clusters)}> with number of jsons = {len(Clusters[CurrentKey])} has been added: {CurrentKey}')

    return Clusters


def f_process_traf_N(traf_N, freq_Ncorr_CR, zi_AC_bu_fat, zi_AC_td_fat, zi_sub_inAC, zi_sub_inBASE, zi_sub_inSUBG,
                     zi_CSM_fat, zi_EAC_bu_fat):
    # traf_N = {'ESALtot': ESALtot, 'NAij_SitwDay': NAij_SitwDay, 'NAij_TatwDay': NAij_TatwDay,
    #           'NAij_TrtwDay': NAij_TrtwDay,
    #           'NAij_QdtwDay': NAij_QdtwDay, 'NoDays': NoDays, 'NAij_SitwMoQnt': NAij_SitwMoQnt,
    #           'NAij_TatwMoQnt': NAij_TatwMoQnt,
    #           'NAij_TrtwMoQnt': NAij_TrtwMoQnt, 'NAij_QdtwMoQnt': NAij_QdtwMoQnt}
    # freq_Ncorr_CR = {'ZC_in_sub': ZC_in_sub, 'freq_si': freq_si, 'freq_ta': freq_ta, 'freq_tr': freq_tr,
    #               'freq_qd': freq_qd, 'Ncorr_ta': Ncorr_ta, 'Ncorr_tr': Ncorr_tr, 'Ncorr_qd': Ncorr_qd,
    #               'CR_in_si':CR_in_si, 'CR_in_ta':CR_in_ta, 'CR_in_tr': CR_in_tr,'CR_in_qd': CR_in_qd}
    NAij_SitwMoQnt = traf_N['NAij_SitwMoQnt']
    NAij_TatwMoQnt = traf_N['NAij_TatwMoQnt']
    NAij_TrtwMoQnt = traf_N['NAij_TrtwMoQnt']
    NAij_QdtwMoQnt = traf_N['NAij_QdtwMoQnt']

    Ncorr_ta = freq_Ncorr_CR['Ncorr_ta']
    Ncorr_tr = freq_Ncorr_CR['Ncorr_tr']
    Ncorr_qd = freq_Ncorr_CR['Ncorr_qd']

    numMo = len(NAij_TatwMoQnt)
    numLoadsSi = len(NAij_SitwMoQnt[0])
    numLoadsTa = len(NAij_TatwMoQnt[0])
    numLoadsTr = len(NAij_TrtwMoQnt[0])
    numLoadsQd = len(NAij_QdtwMoQnt[0])

    def ff_apply_Ncorr(zi_sub):

        if isinstance(zi_sub, (list, tuple, np.ndarray)):
            Nzi = len(zi_sub)
            Nc_si = np.zeros((numMo, Nzi, numLoadsSi))
            Nc_ta = np.zeros((numMo, Nzi, numLoadsTa))
            Nc_tr = np.zeros((numMo, Nzi, numLoadsTr))
            Nc_qd = np.zeros((numMo, Nzi, numLoadsQd))
            for j in range(numMo):
                for i in range(Nzi):
                    I1 = np.where(freq_Ncorr_CR['ZC_in_sub'] == zi_sub[i])[0]
                    Nc_si[j, i, :] = NAij_SitwMoQnt[j]
                    Nc_ta[j, i, :] = Ncorr_ta[I1] * NAij_TatwMoQnt[j]
                    Nc_tr[j, i, :] = Ncorr_tr[I1] * NAij_TrtwMoQnt[j]
                    Nc_qd[j, i, :] = Ncorr_qd[I1] * NAij_QdtwMoQnt[j]
        else:
            Nzi = 1
            Nc_si = np.zeros((numMo, Nzi, numLoadsSi))
            Nc_ta = np.zeros((numMo, Nzi, numLoadsTa))
            Nc_tr = np.zeros((numMo, Nzi, numLoadsTr))
            Nc_qd = np.zeros((numMo, Nzi, numLoadsQd))
            for j in range(numMo):
                I1 = np.where(freq_Ncorr_CR['ZC_in_sub'] == zi_sub)[0]
                if not I1:
                    ddiff = abs(freq_Ncorr_CR['ZC_in_sub'] - zi_sub)
                    I1 = np.where(ddiff == min(ddiff))[0][0]
                    print(f' --> For Ncorr, using z= {freq_Ncorr_CR["ZC_in_sub"][I1]}, instead of z = {zi_sub}')

                Nc_si[j, 0, :] = NAij_SitwMoQnt[j]
                Nc_ta[j, 0, :] = Ncorr_ta[I1] * NAij_TatwMoQnt[j]
                Nc_tr[j, 0, :] = Ncorr_tr[I1] * NAij_TrtwMoQnt[j]
                Nc_qd[j, 0, :] = Ncorr_qd[I1] * NAij_QdtwMoQnt[j]

        return Nc_si, Nc_ta, Nc_tr, Nc_qd

    Nc_acrut_si, Nc_acrut_ta, Nc_acrut_tr, Nc_acrut_qd = ff_apply_Ncorr(zi_sub_inAC)
    Nc_bufc_si, Nc_bufc_ta, Nc_bufc_tr, Nc_bufc_qd = ff_apply_Ncorr(zi_AC_bu_fat)
    Nc_tdfc_si, Nc_tdfc_ta, Nc_tdfc_tr, Nc_tdfc_qd = ff_apply_Ncorr(zi_AC_td_fat)
    Nc_baserut_si, Nc_baserut_ta, Nc_baserut_tr, Nc_baserut_qd = ff_apply_Ncorr(zi_sub_inBASE)
    Nc_subgrut_si, Nc_subgrut_ta, Nc_subgrut_tr, Nc_subgrut_qd = ff_apply_Ncorr(zi_sub_inSUBG)

    Nc_csm_si, Nc_csm_ta, Nc_csm_tr, Nc_csm_qd = ff_apply_Ncorr(zi_CSM_fat)
    Nc_eac_si, Nc_eac_ta, Nc_eac_tr, Nc_eac_qd = ff_apply_Ncorr(zi_EAC_bu_fat)

    traf_NCor = {}
    traf_NCor['Nc_acrut_si'] = Nc_acrut_si
    traf_NCor['Nc_acrut_ta'] = Nc_acrut_ta
    traf_NCor['Nc_acrut_tr'] = Nc_acrut_tr
    traf_NCor['Nc_acrut_qd'] = Nc_acrut_qd

    traf_NCor['Nc_bufc_si'] = Nc_bufc_si
    traf_NCor['Nc_bufc_ta'] = Nc_bufc_ta
    traf_NCor['Nc_bufc_tr'] = Nc_bufc_tr
    traf_NCor['Nc_bufc_qd'] = Nc_bufc_qd

    traf_NCor['Nc_tdfc_si'] = Nc_tdfc_si
    traf_NCor['Nc_tdfc_ta'] = Nc_tdfc_ta
    traf_NCor['Nc_tdfc_tr'] = Nc_tdfc_tr
    traf_NCor['Nc_tdfc_qd'] = Nc_tdfc_qd

    traf_NCor['Nc_baserut_si'] = Nc_baserut_si
    traf_NCor['Nc_baserut_ta'] = Nc_baserut_ta
    traf_NCor['Nc_baserut_tr'] = Nc_baserut_tr
    traf_NCor['Nc_baserut_qd'] = Nc_baserut_qd

    traf_NCor['Nc_subgrut_si'] = Nc_subgrut_si
    traf_NCor['Nc_subgrut_ta'] = Nc_subgrut_ta
    traf_NCor['Nc_subgrut_tr'] = Nc_subgrut_tr
    traf_NCor['Nc_subgrut_qd'] = Nc_subgrut_qd

    traf_NCor['Nc_csm_si'] = Nc_csm_si
    traf_NCor['Nc_csm_ta'] = Nc_csm_ta
    traf_NCor['Nc_csm_tr'] = Nc_csm_tr
    traf_NCor['Nc_csm_qd'] = Nc_csm_qd

    traf_NCor['Nc_eac_si'] = Nc_eac_si
    traf_NCor['Nc_eac_ta'] = Nc_eac_ta
    traf_NCor['Nc_eac_tr'] = Nc_eac_tr
    traf_NCor['Nc_eac_qd'] = Nc_eac_qd

    return traf_NCor


def f_process_calib_coeffs(inps, Va_PV_inAC, Vbeff_PV_inAC, TH_subAC, TH_subBASE, zi_sub_inAC):
    calibCoeffs = {}
    # ******* BOTTOM-UP CRACKING CALIBRATION COEFFICIENTS *******
    # For the bottom-up cracking Nf equation
    calibCoeffs['Bi_bu'] = inps['biBu']
    calibCoeffs['ki_bu'] = inps['kiBu']
    Vabu = Va_PV_inAC[-1]
    Vbebu = Vbeff_PV_inAC[-1]
    C_bu = 10 ** (4.84 * ((Vbebu / (Vbebu + Vabu)) - 0.69))
    h_AC = np.sum(TH_subAC)

    bui = [0.000398, 0.003602, 11.02, 3.49]
    CH_bu = (bui[0] + bui[1] / (1 + np.exp(bui[2] - bui[3] * h_AC))) ** -1

    calibCoeffs['C_bu'] = C_bu
    calibCoeffs['CH_bu'] = CH_bu

    # For Existing AC the bottom-up cracking Nf equation --- placeholder. Can be input in the future.
    calibCoeffs['Bi_bueac'] = inps['biBu']
    calibCoeffs['ki_bueac'] = inps['kiBu']
    calibCoeffs['C_bueac'] = C_bu
    calibCoeffs['CH_bueac'] = CH_bu

    # For the bottom-up cracking transfer function
    calibCoeffs['C_TF_bu'] = inps['cTfBu']
    Cprime_2 = -2.40874 - 39.748 * (1 + h_AC) ** -2.856
    Cprime_1 = -2 * Cprime_2
    calibCoeffs['Cprime_1'] = Cprime_1
    calibCoeffs['Cprime_2'] = Cprime_2

    # ******* TOP-DOWN CRACKING CALIBRATION COEFFICIENTS *******
    # For the top-down cracking Nf equation
    calibCoeffs['Bi_td'] = inps['biTd']
    calibCoeffs['ki_td'] = inps['kiTd']
    Vatd = Va_PV_inAC[0]
    Vbetd = Vbeff_PV_inAC[0]
    C_td = 10 ** (4.84 * ((Vbetd / (Vbetd + Vatd)) - 0.69))
    # bti = [0.01, 12, 15.676, 2.8186]
    # CH_td = (bti[0] + bti[1] / (1 + np.exp(bti[2] - bti[3] * h_AC))) ** -1
    CH_td = 5  # This works better (5 / 27 / 2019). Since our model is different (based on principal strains)
    calibCoeffs['C_td'] = C_td
    calibCoeffs['CH_td'] = CH_td

    # For the top-down cracking transfer function
    calibCoeffs['C_TF_td'] = inps['cTfTd']

    # ******* IRI CALIBRATION COEFFICIENTS *******
    calibCoeffs['C_IRI'] = inps['cIRI']
    calibCoeffs['IRIo'] = inps['iriO']

    # ******* AC RUTTING CALIBRATION COEFFICIENTS *******
    C1rut = -0.1039 * h_AC ** 2 + 2.4868 * h_AC - 17.342
    C2rut = 0.0172 * h_AC ** 2 - 1.7331 * h_AC + 27.428
    kzrut = (C1rut + C2rut * zi_sub_inAC) * 0.328196 ** zi_sub_inAC
    kzrut[kzrut < 0] = 0
    calibCoeffs['kzrut'] = kzrut
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Make sure the beta and k values of each AC sublayer is entered as an input in the future...
    Bri = inps['briAC']
    kri = inps['kriAC']
    kriz = np.zeros((len(TH_subAC), 3))
    Briz = np.zeros((len(TH_subAC), 3))
    for i in range(len(TH_subAC)):
        kriz[i] = kri
        Briz[i] = Bri
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    calibCoeffs['kriz'] = kriz
    calibCoeffs['Briz'] = Briz

    # ******* BASE/SUBBASE (BOTH CALLED BASE HERE) RUTTING CALIBRATION COEFFICIENTS *******
    BriB = inps['briB']  # * inps['kri_base']
    # Make sure the rutting beta values of each BASE sublayer is entered as an input in the future...
    BriBsub = np.zeros(len(TH_subBASE))
    for i in range(len(TH_subBASE)):
        BriBsub[i] = BriB
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    calibCoeffs['BriBsub'] = BriBsub  # BriBsub = BriB values for each sublayer.

    # ******* SUBGRADE RUTTING CALIBRATION COEFFICIENTS *******
    calibCoeffs['BriS'] = inps['briS']  # * inps['kri_subgrade']

    # ******* FOR REFLECTIVE CRACKING, IF CSM OR EAC EXISTS *******
    RFCa_rc = 3.5 + 0.75 * h_AC
    RFCb_rc = -0.688584 - 3.37302 * h_AC ** (-0.915469)
    calibCoeffs['RFCa_rc'] = RFCa_rc
    calibCoeffs['RFCb_rc'] = RFCb_rc

    return calibCoeffs


def f_get_Ncorr_ax(Ncorr_ta, Ncorr_tr, Ncorr_qd, taLoadsLb, trLoadsLb, qdLoadsLb, zi_sub_in, zi_AC_bu_fat):
    P_ta = taLoadsLb / 8
    P_tr = trLoadsLb / 12
    P_qd = qdLoadsLb / 16

    Ncorr_ta_z1 = np.squeeze(Ncorr_ta[:, P_ta == 4500])
    Ncorr_tr_z1 = np.squeeze(Ncorr_tr[:, P_tr == 4500])
    Ncorr_qd_z1 = np.squeeze(Ncorr_qd[:, P_qd == 4500])

    # # this is for the surface layer, i.e., top down cracking
    # Ncorr_ta_z0 = Ncorr_ta_z1[0]
    # Ncorr_tr_z0 = Ncorr_tr_z1[0]
    # Ncorr_qd_z0 = Ncorr_qd_z1[0]
    #

    # Interp example: yp = interp(xp, x, y)

    Ncorr_ta_zbu = np.interp(zi_AC_bu_fat, zi_sub_in,
                             Ncorr_ta_z1)  # correction factor for the tandem axle n (for moving load simulation)
    Ncorr_tr_zbu = np.interp(zi_AC_bu_fat, zi_sub_in,
                             Ncorr_tr_z1)  # correction factor for the tandem axle n (for moving load simulation)
    Ncorr_qd_zbu = np.interp(zi_AC_bu_fat, zi_sub_in,
                             Ncorr_qd_z1)  # correction factor for the tandem axle n (for moving load simulation)

    return Ncorr_ta_zbu, Ncorr_tr_zbu, Ncorr_qd_zbu


def f_get_Ncorr_axEAC(Ncorr_ta, Ncorr_tr, Ncorr_qd, taLoadsLb, trLoadsLb, qdLoadsLb, zi_sub_in, zi_EAC_bu_fat):
    P_ta = taLoadsLb / 8
    P_tr = trLoadsLb / 12
    P_qd = qdLoadsLb / 16

    Ncorr_ta_z1 = Ncorr_ta[:, P_ta == 4500]
    Ncorr_tr_z1 = Ncorr_tr[:, P_tr == 4500]
    Ncorr_qd_z1 = Ncorr_qd[:, P_qd == 4500]

    Ncorr_ta_zbuEAC = np.interp(zi_sub_in, Ncorr_ta_z1,
                                zi_EAC_bu_fat)  # correction factor for the tandem axle n (for moving load simulation)
    Ncorr_tr_zbuEAC = np.interp(zi_sub_in, Ncorr_tr_z1,
                                zi_EAC_bu_fat)  # correction factor for the tandem axle n (for moving load simulation)
    Ncorr_qd_zbuEAC = np.interp(zi_sub_in, Ncorr_qd_z1,
                                zi_EAC_bu_fat)  # correction factor for the tandem axle n (for moving load simulation)

    return Ncorr_ta_zbuEAC, Ncorr_tr_zbuEAC, Ncorr_qd_zbuEAC


def f_analysis_locations(Sx, Sy_Ta, Sy_Tr, Sy_Qd, CR_in, ZC_in):
    # Analysis locations in X-direction (perpendicular to the traffic direction)
    X = np.hstack((0, (Sx / 2 - CR_in) / 2, (Sx / 2 - CR_in), Sx / 2, (Sx / 2 + CR_in), (Sx / 2 + CR_in + 4),
                   (Sx / 2 + CR_in + 8),
                   (Sx / 2 + CR_in + 16), (Sx / 2 + CR_in + 24), (Sx / 2 + CR_in + 32)))

    # Analysis locations in Y direction (along traffic direction)
    YTa = np.arange(0, 1.5, 0.5) * Sy_Ta
    YTr = np.arange(0, 2.5, 0.5) * Sy_Tr
    YQd = np.arange(0, 3.5, 0.5) * Sy_Qd

    xy_si = np.zeros((len(X), 2))
    xy_si[:, 0] = X
    xy_ta = mcombvec(X, YTa)
    xy_tr = mcombvec(X, YTr)
    xy_qd = mcombvec(X, YQd)

    # single axle
    YT_si = np.zeros(2)
    XT_si = np.zeros(2)
    YT_si[0] = 0
    YT_si[1] = 0
    XT_si[0] = -Sx / 2
    XT_si[1] = Sx / 2

    # tandem axle
    YT_ta = np.zeros(4)
    XT_ta = np.zeros(4)
    YT_ta[0:2] = 0
    YT_ta[2:4] = Sy_Ta
    XT_ta[0] = -Sx / 2
    XT_ta[2] = -Sx / 2
    XT_ta[1] = Sx / 2
    XT_ta[3] = Sx / 2

    # tridem axle
    YT_tr = np.zeros(6)
    XT_tr = np.zeros(6)
    YT_tr[0:2] = 0
    YT_tr[2:4] = Sy_Tr
    YT_tr[4:6] = 2 * Sy_Tr

    XT_tr[0] = -Sx / 2
    XT_tr[2] = -Sx / 2
    XT_tr[4] = -Sx / 2
    XT_tr[1] = Sx / 2
    XT_tr[3] = Sx / 2
    XT_tr[5] = Sx / 2

    # quad axle
    YT_qd = np.zeros(8)
    XT_qd = np.zeros(8)
    YT_qd[0:2] = 0
    YT_qd[2:4] = Sy_Qd
    YT_qd[4:6] = 2 * Sy_Qd
    YT_qd[6:8] = 3 * Sy_Qd

    XT_qd[0] = -Sx / 2
    XT_qd[2] = -Sx / 2
    XT_qd[4] = -Sx / 2
    XT_qd[6] = -Sx / 2
    XT_qd[1] = Sx / 2
    XT_qd[3] = Sx / 2
    XT_qd[5] = Sx / 2
    XT_qd[7] = Sx / 2

    [RM_si, theta_si] = f_calcl_RM_theta(xy_si, XT_si, YT_si, 2)
    [RM_ta, theta_ta] = f_calcl_RM_theta(xy_ta, XT_ta, YT_ta, 4)
    [RM_tr, theta_tr] = f_calcl_RM_theta(xy_tr, XT_tr, YT_tr, 6)
    [RM_qd, theta_qd] = f_calcl_RM_theta(xy_qd, XT_qd, YT_qd, 8)

    xyz_si = mcombvec2(xy_si, ZC_in[0])
    xyz_ta = mcombvec2(xy_ta, ZC_in[0])
    xyz_tr = mcombvec2(xy_tr, ZC_in[0])
    xyz_qd = mcombvec2(xy_qd, ZC_in[0])
    SI_X = len(np.unique(xyz_si[:, 0]))  # Number of unique X coordinates in the tire/axle coordinate matrix
    SJ2_Y = len(np.unique(xyz_ta[:, 1]))  # Number of unique Y coordinates in the tire/axle coordinate matrix
    SJ3_Y = len(np.unique(xyz_tr[:, 1]))  # Number of unique Y coordinates in the tire/axle coordinate matrix
    SJ4_Y = len(np.unique(xyz_qd[:, 1]))  # Number of unique Y coordinates in the tire/axle coordinate matrix

    analysisLoc = {'RM_si': RM_si, 'theta_si': theta_si,
                   'RM_ta': RM_ta, 'theta_ta': theta_ta,
                   'RM_tr': RM_tr, 'theta_tr': theta_tr,
                   'RM_qd': RM_qd, 'theta_qd': theta_qd,
                   'xyz_si': xyz_si, 'xyz_ta': xyz_ta,
                   'xyz_tr': xyz_tr, 'xyz_qd': xyz_qd,
                   'SI_X': SI_X, 'SJ2_Y': SJ2_Y, 'SJ3_Y': SJ3_Y, 'SJ4_Y': SJ4_Y}

    # savemat("outputs_Python_f_calcl_RM_theta.mat", analysisLoc)
    # RM_si, RM_ta, RM_tr, RM_qd, theta_si, theta_ta, theta_tr, theta_qd, xyz_si, xyz_ta, xyz_tr, xyz_qd
    return analysisLoc


def f_calcl_RM_theta(xy, XT, YT, Ncirc):
    RMt = np.zeros((len(xy), Ncirc))
    thetat = np.zeros((len(xy), Ncirc))

    for i in range(Ncirc):
        dx_eval = xy[:, 0] - XT[i]
        dy_eval = xy[:, 1] - YT[i]
        RMt[:, i] = (dx_eval ** 2 + dy_eval ** 2) ** 0.5
        thetat[:, i] = np.arctan2(dy_eval, dx_eval) * 180 / np.pi

    return RMt, thetat


def mcombvec2(X, Y):
    # THis code is to combine one matrix of lenX*SJ and a vector of 1*lenY
    lenX = len(X)
    lenY = len(Y)
    SJ = len(X[0])  # number of columns

    X: float = X
    Y: float = Y
    Z = np.zeros((lenX * lenY, SJ + 1))
    n = 0
    for j in range(lenY):
        for i in range(lenX):
            Z[n, 0:SJ] = X[i]
            Z[n, SJ:] = Y[j]
            n = n + 1

    return Z


def mcombvec(X, Y):
    # THis code is to combine two vectors of 1*lenX an 1*lenY
    lenX = len(X)
    lenY = len(Y)

    X: float = X
    Y: float = Y
    Z = np.zeros((lenX * lenY, 2))
    n = 0
    for j in range(lenY):
        for i in range(lenX):
            Z[n, 0] = X[i]
            Z[n, 1] = Y[j]
            n = n + 1

    return Z


def mlinspace(minVal, maxVal, numvals):
    minV: float = minVal
    maxV: float = maxVal
    stepp = (maxV - minV) / (numvals - 1)
    outArray: float = np.arange(minV, maxV + stepp, stepp)
    return outArray


def mlogspace(logminVal, logmaxVal, numvals):
    logmin: float = logminVal
    logmax: float = logmaxVal
    stepp = (logmax - logmin) / (numvals - 1)
    logOutArray: float = np.arange(logmin, logmax + stepp, stepp)
    outArray = 10 ** logOutArray
    return outArray


def f_Et_from_Estar(c, p, Nmxwll):
    # fR = np.logspace(-16, 16, 500).T
    # logfR = np.log10(fR)
    print('  ')
    print('**********************')
    print(' Converting |E*| to E(t) curve ')
    logfR = np.arange(-16, 16, 32 / 500)
    fR = 10 ** logfR
    logtR = -logfR

    Estarfit = 10 ** (c[0] + c[1] / (1 + np.exp(c[2] + c[3] * logtR)))
    deltafit = p[0] * np.exp(-(p[1] - logfR) ** 2 / (2 * p[2] ** 2))
    # -- calculate storage modulus
    Ep = Estarfit * (np.cos(np.deg2rad(deltafit)))
    # wR = 2 * np.pi * fR

    # -assume a set of taui values and calculate Ei s
    logminfR: float = np.log10(np.min(fR))
    logmaxfR: float = np.log10(np.max(fR))
    # taui = 1.2 * np.logspace(np.log10(np.min(fR)), np.log10(np.max(fR)), Nmxwll).T
    taui = mlogspace(logminfR, logmaxfR, Nmxwll)

    fRp = fR
    wRp = 2 * np.pi * fRp
    Einf = np.min(Ep)

    A = np.empty((len(wRp), len(taui)))
    for i in range(len(wRp)):
        for j in range(len(taui)):
            A[i, j] = wRp[i] ** 2 * taui[j] ** 2 / (wRp[i] ** 2 * taui[j] ** 2 + 1)

    C = Ep - Einf
    # --- solve AX = C - least-squares solution to a linear matrix equation
    Eid = np.linalg.lstsq(A, C, rcond=None)  # this does not work with Numba
    # Eid = np.linalg.lstsq(A, C) # this works with Numba but gives a warning
    Ei = Eid[0]

    # --Calculate fit to Ep
    Epprony = np.empty((len(wRp)))
    for i in range(len(wRp)):
        Epprony[i] = Einf + np.sum(wRp[i] ** 2 * taui ** 2 * Ei / (wRp[i] ** 2 * taui ** 2 + 1))

    # Calculating Alpha
    print('  ')
    print('**********************')
    print(' Calculating slope (m) of the Et curve ')
    # t = np.logspace(-16, 16, 76)
    t = mlogspace(-16, 16, 76)

    Et = np.zeros(len(t))

    for i in range(len(t)):
        Et[i] = Einf + np.sum(Ei * np.exp(-t[i] / taui))

    # computing the alpha
    logt = np.log10(t)
    logE = np.log10(Et)

    dlogE1 = (logE[0] - logE[1]) / (logt[0] - logt[1])

    dlogE = np.gradient(logE, logt)
    slopem = np.min(dlogE)
    Ind = np.argmin(dlogE)
    interc = logE[Ind] - slopem * logt[Ind]

    Nt = 5
    xl = logt[Ind - Nt: Ind + Nt]
    yl = slopem * xl + interc

    xl = 10 ** xl
    yl = 10 ** yl

    m = np.abs(slopem)

    # print(f'm = {m}')
    #
    # fig, axs = plt.subplots(2, 1)
    # axs[0].loglog(fRp, Epprony, marker = '+', markerfacecolor = 'red', markersize = 0, color = 'red', linewidth = 1, label = 'Epprony')
    # axs[0].loglog(fRp, Ep, marker = 'o', markerfacecolor = None, markersize = 0, color = 'blue', linewidth = 1, label = 'Ep')
    # axs[0].set(ylabel = "Eprime")
    # axs[0].set(xlabel = "fR (Hz)")
    #
    # axs[1].loglog(t, Et, marker = '+', markerfacecolor = 'red', markersize = 2, color = 'red', linewidth = 1, label = 'E(t)')
    # axs[1].set(ylabel = "E(t)")
    # axs[1].set(xlabel = "t (s)")
    # plt.legend()
    # plt.show()
    pronyCoeffsAC = {'Einf': Einf, 'Ei': Ei, 'taui': taui, 'm': m, 't': t, 'Et': Et}

    return pronyCoeffsAC


# def f_construct_Esubdum_M(E_sub, numMo, SublayerTQ, numzi_inAC, numQuint):
#     NL = len(E_sub)
#     ncolumns = numQuint+2
#     nrowspermonth = NL
#     EsubdumM = np.zeros(numMo * ncolumns * nrowspermonth).reshape(numMo, nrowspermonth, ncolumns)
#     for i in range(numMo):
#         for k in range(numzi_inAC):
#             dummy = SublayerTQ['EQaged'][0]
#             EsubdumM[i, k, :] = dummy[k][i]
#         for kk in range(numzi_inAC,NL):
#             EsubdumM[i, kk, :2] = EsubdumM[i, kk-1, :2]
#             EsubdumM[i, kk, 2:] = E_sub[kk]
#
#     return EsubdumM, NL

def loadjson(localjsonfld, jsonfile, bucket_name="meapa2-io", username="meminkutay"):
    """
    Load a JSON file, either locally or from an S3 bucket.

    :param localjsonfld: Folder path or S3 prefix (e.g., "s3:").
    :param jsonfile: File name.
    :param bucket_name: S3 bucket name (default: "meapa2-io").
    :param username: Optional username to include in the S3 key.
    :return: Parsed JSON data.
    """
    if localjsonfld.lower().startswith("s3:"):
        # Reading from S3
        try:
            # Construct the S3 key
            s3_client = boto3.client('s3')
            s3_key = f"{username}/{jsonfile}" if username else jsonfile
            print(f"Downloading input JSON from S3: s3://{bucket_name}/{s3_key}")

            # Get the JSON content from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            data = json.load(response['Body'])
            print(f"Successfully loaded JSON from S3: s3://{bucket_name}/{s3_key}")
            return data
        except Exception as e:
            print(f"Error loading JSON from S3: {e}")
            raise
    else:
        # Reading locally
        try:
            with open(os.path.join(localjsonfld, jsonfile), 'r') as f:
                data = json.load(f)
            print(f"Successfully loaded JSON from local file: {os.path.join(localjsonfld, jsonfile)}")
            return data
        except Exception as e:
            print(f"Error loading JSON from local file: {e}")
            raise


def save_to_output(outputPythi, fld_out, json_output_file):
    if fld_out.lower().startswith("s3:"):
        # Extract the bucket name and key from fld_out
        s3 = boto3.client('s3')
        bucket_name = fld_out.split('/')[2]
        s3_key = '/'.join(fld_out.split('/')[3:] + [f"{json_output_file}"])

        # Save JSON to S3
        print(f"--> Saving the JSON output to S3: s3://{bucket_name}/{s3_key}")
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json.dumps(outputPythi),  # Serialize JSON data
            ContentType='application/json'
        )
        print(f"--> Saved: s3://{bucket_name}/{s3_key}")


    else:
        # Save locally
        outjson = os.path.join(fld_out, f"{json_output_file}")
        print(f" --> Saving the JSON output locally: {outjson}")
        with open(outjson, 'w') as outfile:
            json.dump(outputPythi, outfile)


def delete_file(jsonfld, file_to_delete):
    """
    Delete a single file, either from S3 or locally.

    :param jsonfld: Folder path or S3 prefix (e.g., "s3:").
    :param file_to_delete: Name of the file to delete.
    """
    if jsonfld.lower().startswith("s3:"):
        # Extract the bucket name and prefix (key folder)
        s3 = boto3.client('s3')
        bucket_name = jsonfld.split('/')[2]
        prefix = "/".join(jsonfld.split('/')[3:])  # Extract the S3 folder/key prefix if any

        # Construct full key for the file
        file_key = f"{prefix}/{file_to_delete}" if prefix else file_to_delete

        # Delete file from S3
        try:
            print(f"Deleting file: s3://{bucket_name}/{file_key}")
            s3.delete_object(Bucket=bucket_name, Key=file_key)
            print(f"Successfully deleted file: {file_to_delete}")
        except Exception as e:
            print(f"Error deleting file from S3: {e}")
    else:
        # Handle local file deletion
        file_path = os.path.join(jsonfld, file_to_delete)

        try:
            print(f"Deleting local file: {file_path}")
            os.remove(file_path)
            print(f"Successfully deleted local file: {file_to_delete}")
        except FileNotFoundError:
            print(f"Local file not found: {file_path}")
        except Exception as e:
            print(f"Error deleting local file: {e}")


def savejson(localjsonfld, jsonfile, data):
    with open(localjsonfld + jsonfile, 'w') as outfile:
        json.dump(data, outfile)


def appendtxtold(txtfld, txtfile, txttoadd):
    with open(txtfld + '/' + txtfile, 'a') as outfile:
        outfile.write(txttoadd)


def appendtxt(txtfld, txtfile, txttoadd, bucket_name="meapa2-io", username="meminkutay"):
    """
    Append text to a file, either locally or in an S3 bucket, with thread-safe aggregation.

    :param txtfld: Folder path or S3 prefix (e.g., "s3:").
    :param txtfile: File name.
    :param txttoadd: Text to append.
    :param bucket_name: S3 bucket name (default: "meapa2-io").
    :param username: Optional username to include in the S3 key.
    """
    global status_logs

    if txtfld.lower().startswith("s3:"):
        # Writing to S3
        try:
            # Construct the S3 key
            s3_key = f"{username}/{txtfile}" if username else txtfile
            print(f"Uploading s3_key: {s3_key} to bucket_name: {bucket_name}")

            with write_lock:
                # Append the new text to the shared log
                status_logs.append(txttoadd)

                # Aggregate the log and upload to S3
                updated_content = "\n".join(status_logs)
                boto3.client('s3').put_object(Bucket=bucket_name, Key=s3_key, Body=updated_content.encode('utf-8'))

            print(f"Successfully appended text to S3 object: s3://{bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Error appending text to S3: {e}")
    else:
        # Writing locally
        try:
            os.makedirs(txtfld, exist_ok=True)  # Ensure local directory exists
            with write_lock:
                # Append the new text to the shared log
                status_logs.append(txttoadd)

                # Write the aggregated log to the local file
                updated_content = "\n".join(status_logs)
                with open(os.path.join(txtfld, txtfile), 'w') as outfile:
                    outfile.write(updated_content)

            print(f"Successfully appended text to local file: {os.path.join(txtfld, txtfile)}")
        except Exception as e:
            print(f"Error appending text to local file: {e}")



def strcmp(l1, s):
    matched_indexes = []
    i = 0
    length = len(l1)

    while i < length:
        if s == l1[i]:
            matched_indexes.append(i)
        i += 1
    return matched_indexes


def makedir(outfld):
    if not os.path.exists(outfld):
        os.makedirs(outfld)


def sigdigit(num, sigd):
    return round(num * 10 ** sigd) / (10 ** sigd)


# function to revise the json file for certain settings.
def editjson(localjsonfld, jsonfile):
    data = loadjson(localjsonfld, jsonfile)
    print('Before <--' + data['fldAirTemp'] + ',' + data['climateModel'] + ',' + data['climateType'], ',',
          data['savePeriod'][1])
    data['fldAirTemp'] = '/Users/muhammedkutay/matlab/data/weather'
    # data['climateType'] = 'MERRA2'
    data['savePeriod'][0] = 5
    data['savePeriod'][1] = 22000
    # data['climateModel'] = 'Original'
    print('After --> ' + data['fldAirTemp'] + ',' + data['climateModel'] + ',' + data['climateType'], ',',
          data['savePeriod'][1])

    savejson(localjsonfld, jsonfile, data)
