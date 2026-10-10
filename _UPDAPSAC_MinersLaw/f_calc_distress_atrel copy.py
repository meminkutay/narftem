import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv
from scipy.io import loadmat
import updapsutil as utl


def f_calc_distress_atrel(ld, stdev_data, reliability_thresholds, plt_flag=1):
    # Initialize reliability and threshold data
    ld['rel_fctd'] = reliability_thresholds[1]['distressReliability']
    ld['rel_fcbu'] = reliability_thresholds[2]['distressReliability']
    ld['rel_tc_AC'] = reliability_thresholds[3]['distressReliability']
    ld['rel_rutTOT'] = reliability_thresholds[4]['distressReliability']
    ld['rel_rut_TAC'] = reliability_thresholds[5]['distressReliability']
    ld['rel_reflCrackAC'] = reliability_thresholds[6]['distressReliability']
    ld['rel_IRI'] = max([r['distressReliability'] for r in reliability_thresholds[1:]])

    ld['IRIThresh'] = reliability_thresholds[0]['distressThreshold']
    ld['FCtdThresh'] = reliability_thresholds[1]['distressThreshold']
    ld['FCbuThresh'] = reliability_thresholds[2]['distressThreshold']
    ld['TCACThresh'] = reliability_thresholds[3]['distressThreshold']
    ld['RutThresh'] = reliability_thresholds[4]['distressThreshold']
    ld['RutACThresh'] = reliability_thresholds[5]['distressThreshold']
    ld['reflCrackACThresh'] = reliability_thresholds[6]['distressThreshold']

    # Initialize standard deviation equations
    ld['eqnAcbufcstdev'] = ff_process_equation(stdev_data['acbufcstdev'])
    ld['eqnAcrefcrstdev'] = ff_process_equation(stdev_data['acrefcrstdev'])
    ld['eqnActcstdev'] = ff_process_equation(stdev_data['actcstdev'])
    ld['eqnActdfcstdev'] = ff_process_equation(stdev_data['actdfcstdev'])
    ld['eqnBaserutstdev'] = ff_process_equation(stdev_data['baserutstdev'])
    ld['eqnSubgrutstdev'] = ff_process_equation(stdev_data['subgrutstdev'])
    ld['eqnAcrutstdev'] = ff_process_equation(stdev_data['acrutstdev'])

    # Prepare arrays for calculation
    Nmo = len(ld['FCbu'])
    distress_vars = ['stdev_acrut', 'stdev_baserut', 'stdev_subgrut', 'stdev_fcbu',
                     'stdev_fctd', 'stdev_tc_AC', 'stdev_totrut', 'iriAtRel']
    for var in distress_vars:
        ld[var] = np.zeros(Nmo)

    # Initialize keys as NumPy arrays with the appropriate size (Nmo)
    Nmo = len(ld['FCbu'])  # Assuming 'FCbu' is already initialized and its length determines the size

    # Initialize result arrays for distress at reliability
    ld['totrutAtRel'] = np.zeros(Nmo)
    ld['acrutAtRel'] = np.zeros(Nmo)
    ld['fcbuAtRel'] = np.zeros(Nmo)
    ld['fctdtAtRel'] = np.zeros(Nmo)
    ld['tc_ACAtRel'] = np.zeros(Nmo)
    ld['iriAtRel'] = np.zeros(Nmo)

    # Conditional initialization if 'CSMExists' or 'EACExists'
    if ld.get('CSMExists') or ld.get('EACExists'):
        ld['stdev_reflCrackAC'] = np.zeros(Nmo)
        ld['reflCrackACAtRel'] = np.zeros(Nmo)

    for i in range(Nmo):  # Python uses 0-based indexing
        # Calculate the standard deviation values for each distress
        ld['stdev_acrut'][i] = compute_stdev(ld['rut_TAC'][i], ld['eqnAcrutstdev'], "acrut")
        ld['stdev_baserut'][i] = compute_stdev(ld['rut_TBASE'][i], ld['eqnBaserutstdev'], "baserut")
        ld['stdev_subgrut'][i] = compute_stdev(ld['rut_TSUBG'][i], ld['eqnSubgrutstdev'], "subrut")
        ld['stdev_fcbu'][i] = compute_stdev(ld['FCbu'][i] / 100, ld['eqnAcbufcstdev'], "bottom")
        ld['stdev_fctd'][i] = compute_stdev(ld['FCtd'][i], ld['eqnActdfcstdev'], "top")
        ld['stdev_tc_AC'][i] = compute_stdev(ld['TC_mo'][i], ld['eqnActcstdev'], "thermal")

        # Combine the standard deviations to compute total rutting standard deviation
        ld['stdev_totrut'][i] = np.sqrt(
            ld['stdev_acrut'][i] ** 2 +
            ld['stdev_baserut'][i] ** 2 +
            ld['stdev_subgrut'][i] ** 2
        )

        # Compute IRI standard deviation
        std_iri = compute_stdev_iri(ld['IRI'][i], i)



        # Calculate the distress values at reliability
        ld['totrutAtRel'][i] = compute_distress_at_reliability(ld['rutTOT'][i], ld['stdev_totrut'][i], ld['rel_rutTOT'],
                                                               4.0)
        ld['acrutAtRel'][i] = compute_distress_at_reliability(ld['rut_TAC'][i], ld['stdev_acrut'][i], ld['rel_rut_TAC'],
                                                              3.0)
        ld['fcbuAtRel'][i] = compute_distress_at_reliability(ld['FCbu'][i], ld['stdev_fcbu'][i], ld['rel_fcbu'], 100.0)
        ld['fctdtAtRel'][i] = compute_distress_at_reliability(ld['FCtd'][i], ld['stdev_fctd'][i], ld['rel_fctd'],
                                                              15000.0)
        ld['tc_ACAtRel'][i] = compute_distress_at_reliability(ld['TC_mo'][i], ld['stdev_tc_AC'][i], ld['rel_tc_AC'],
                                                              15000.0)

        # Additional calculations if conditions are met
        if ld.get('CSMExists') or ld.get('EACExists'):
            ld['stdev_reflCrackAC'][i] = compute_stdev(ld['TRCac'][i], ld['eqnAcrefcrstdev'], "transverse")
            ld['reflCrackACAtRel'][i] = compute_distress_at_reliability(
                ld['TRCac'][i], ld['stdev_reflCrackAC'][i], ld['rel_reflCrackAC'], 100.0
            )

        # Compute distress at reliability for IRI
        ld['iriAtRel'][i] = compute_distress_at_reliability(ld['IRI'][i], std_iri, ld['rel_IRI'], 400)

    # Plot if required
    if plt_flag == 1:
        ff_plot(ld)

    return ld


def compute_distress_at_reliability(distress_val, stdev_distress, reliability, dist_maxx):
    p = 1 - reliability / 100
    zval = np.sqrt(2) * erfcinv(2 * p)
    distress_val_rel = distress_val + zval * stdev_distress
    distress_val_rel = max(0.0, min(distress_val_rel, dist_maxx))
    return distress_val_rel


def compute_stdev(x, eqn, var_name):
    try:
        eqn = eqn.replace(var_name, "x")
        return eval(eqn)
    except Exception as err:
        print(f'Error in {var_name}, {eqn}: {err}')
        return None



def compute_stdev_iri(iri, i):
    if i == 0:
        return iri / 10
    else:
        return 25.1148 * np.log(iri) - 87.95062


def ff_process_equation(eqn):
    eqn = eqn.lower().replace("pow", "np.power")
    eqn = eqn.lower().replace("exp", "np.exp")
    eqn = eqn.lower().replace("log10", "np.log10")

    return eqn


def ff_plot(ld):
    Nmo = len(ld['FCbu'])
    tmoR = np.arange(1, Nmo + 1)
    plt.figure(figsize=(12, 10))

    plt.subplot(321)
    plt.plot(tmoR, ld['IRI'], '.-', label='IRI (50%)')
    plt.plot(tmoR, ld['iriAtRel'], 'x--', label=f"IRI ({ld['rel_IRI']}%)")
    plt.axhline(y=ld['IRIThresh'], color='r', linewidth=2, label='IRI threshold')
    plt.xlabel('Months')
    plt.ylabel('IRI (in/mile)')
    plt.legend()

    # Additional subplots for other distress variables can be added similarly
    plt.tight_layout()
    plt.show()


def mat_to_dict(mat_obj):
    """
    Recursively convert MATLAB structures to Python dictionaries.
    """
    if isinstance(mat_obj, np.ndarray) and mat_obj.dtype.names:  # MATLAB struct
        return {name: mat_to_dict(mat_obj[name][0]) for name in mat_obj.dtype.names}
    elif isinstance(mat_obj, np.ndarray) and mat_obj.size == 1:  # MATLAB scalar
        return mat_to_dict(mat_obj[0])
    elif isinstance(mat_obj, np.ndarray):  # MATLAB array
        return mat_obj.flatten()
    else:  # Base case: leave other types as-is
        return mat_obj


if __name__ == "__main__":
    # Load the .mat file
    mat_contents = loadmat('f_calc_distress_atrel_inp.mat')

    # # Access the 'ld' variable from the loaded data
    # ld = mat_contents['ld'][0][0]
    # Convert 'ld' MATLAB struct to Python dictionary
    ld = mat_to_dict(mat_contents['ld'])

    jsonfld = "/Users/kutay/Downloads"
    jsonfile = "CE431.json"
    inps = utl.loadjson(jsonfld, jsonfile)

    stdev_data = inps['stdevData']
    reliability_thresholds = inps['reliabilityThresholds']
    f_calc_distress_atrel(ld, stdev_data, reliability_thresholds, plt_flag=1)
