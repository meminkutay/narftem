import numpy as np
# import matplotlib.pyplot as plt
from scipy.special import erfcinv
from scipy.io import loadmat
import updapsutil as utl


def f_calc_distress_atrel(ld, stdev_data, reliability_thresholds, plt_flag=1):
    """
    Main function to calculate distress at reliability thresholds.
    """

    # Set reliability and threshold values in `ld`
    reliability_keys = ['rel_fctd', 'rel_fcbu', 'rel_tc_AC', 'rel_rutTOT', 'rel_rut_TAC', 'rel_reflCrackAC']
    threshold_keys = ['IRIThresh', 'FCtdThresh', 'FCbuThresh', 'TCACThresh', 'RutThresh', 'RutACThresh',
                      'reflCrackACThresh']
    reliability_values = [r['distressReliability'] for r in reliability_thresholds[1:]]
    threshold_values = [t['distressThreshold'] for t in reliability_thresholds]

    for key, val in zip(reliability_keys, reliability_values):
        ld[key] = val
    for key, val in zip(threshold_keys, threshold_values):
        ld[key] = val
    ld['rel_IRI'] = max(reliability_values)

    # Process standard deviation equations
    stdev_keys = ['acbufcstdev', 'acrefcrstdev', 'actcstdev', 'actdfcstdev', 'baserutstdev', 'subgrutstdev',
                  'acrutstdev']
    for key in stdev_keys:
        ld[f"eqn{key}"] = ff_process_equation(stdev_data[key])

    # Initialize arrays
    Nmo = len(ld['FCbu'])
    distress_vars = ['stdev_acrut', 'stdev_baserut', 'stdev_subgrut', 'stdev_fcbu',
                     'stdev_fctd', 'stdev_tc_AC', 'stdev_totrut', 'iriAtRel']
    result_vars = ['totrutAtRel', 'acrutAtRel', 'fcbuAtRel', 'fctdtAtRel', 'tc_ACAtRel', 'iriAtRel']

    for var in distress_vars + result_vars:
        ld[var] = np.zeros(Nmo).tolist()

    if ld.get('CSMExists') or ld.get('EACExists'):
        ld['stdev_reflCrackAC'] = np.zeros(Nmo).tolist()
        ld['reflCrackACAtRel'] = np.zeros(Nmo).tolist()

    # Compute values for each month
    for i in range(Nmo):
        ld['stdev_acrut'][i] = compute_stdev(ld['rut_TAC'][i], ld['eqnacrutstdev'], "acrut")
        ld['stdev_baserut'][i] = compute_stdev(ld['rut_TBASE'][i], ld['eqnbaserutstdev'], "baserut")
        ld['stdev_subgrut'][i] = compute_stdev(ld['rut_TSUBG'][i], ld['eqnsubgrutstdev'], "subrut")
        ld['stdev_fcbu'][i] = compute_stdev(ld['FCbu'][i] / 100, ld['eqnacbufcstdev'], "bottom")
        ld['stdev_fctd'][i] = compute_stdev(ld['FCtd'][i], ld['eqnactdfcstdev'], "top")
        ld['stdev_tc_AC'][i] = compute_stdev(ld['TC_mo'][i], ld['eqnactcstdev'], "thermal")

        # Combine standard deviations
        ld['stdev_totrut'][i] = np.sqrt(
            ld['stdev_acrut'][i] ** 2 + ld['stdev_baserut'][i] ** 2 + ld['stdev_subgrut'][i] ** 2)

        # Compute IRI standard deviation
        std_iri = compute_stdev_iri(ld['IRI'][i], i)

        # Compute distress at reliability
        ld['totrutAtRel'][i] = compute_distress_at_reliability(ld['rutTOT'][i], ld['stdev_totrut'][i], ld['rel_rutTOT'],
                                                               4.0)
        ld['acrutAtRel'][i] = compute_distress_at_reliability(ld['rut_TAC'][i], ld['stdev_acrut'][i], ld['rel_rut_TAC'],
                                                              3.0)
        ld['fcbuAtRel'][i] = compute_distress_at_reliability(ld['FCbu'][i], ld['stdev_fcbu'][i], ld['rel_fcbu'], 100.0)
        ld['fctdtAtRel'][i] = compute_distress_at_reliability(ld['FCtd'][i], ld['stdev_fctd'][i], ld['rel_fctd'],
                                                              15000.0)
        ld['tc_ACAtRel'][i] = compute_distress_at_reliability(ld['TC_mo'][i], ld['stdev_tc_AC'][i], ld['rel_tc_AC'],
                                                              15000.0)
        ld['iriAtRel'][i] = compute_distress_at_reliability(ld['IRI'][i], std_iri, ld['rel_IRI'], 400)

        if ld.get('CSMExists') or ld.get('EACExists'):
            ld['stdev_reflCrackAC'][i] = compute_stdev(ld['TRCac'][i], ld['eqnacrefcrstdev'], "transverse")
            ld['reflCrackACAtRel'][i] = compute_distress_at_reliability(ld['TRCac'][i], ld['stdev_reflCrackAC'][i],
                                                                        ld['rel_reflCrackAC'], 100.0)

    # # Plot results if flag is set
    # if plt_flag:
    #     ff_plot(ld)

    return ld


def compute_distress_at_reliability(distress_val, stdev_distress, reliability, dist_maxx):
    """
    Compute distress at a given reliability.
    """
    p = 1 - reliability / 100
    zval = np.sqrt(2) * erfcinv(2 * p)
    distress_val_rel = distress_val + zval * stdev_distress
    return np.clip(distress_val_rel, 0.0, dist_maxx)


def compute_stdev(x, eqn, var_name):
    """
    Compute standard deviation using the provided equation.
    """
    try:
        eqn = eqn.replace(var_name, "x")
        return eval(eqn)
    except Exception as err:
        print(f"Error in {var_name}, {eqn}: {err}")
        return None


def compute_stdev_iri(iri, i):
    """
    Compute standard deviation for IRI.
    """
    return iri / 10 if i == 0 else 25.1148 * np.log(iri) - 87.95062


def ff_process_equation(eqn):
    """
    Process equations to be Python-compatible.
    """
    return eqn.lower().replace("pow", "np.power").replace("exp", "np.exp").replace("log10", "np.log10")

#
# def ff_plot(ld):
#     """
#     Generate plots for distress variables using the Python equivalent of MATLAB plotting.
#     """
#     Nmo = len(ld['FCbu'])  # Number of months
#     tmoR = np.arange(1, Nmo + 1)  # Months (1-based indexing for consistency with MATLAB)
#
#     plt.figure(figsize=(15, 12), facecolor='white')
#
#     # Subplot 1: IRI
#     plt.subplot(321)
#     plt.plot(tmoR, ld['IRI'], '.-', label='IRI (50%)')
#     plt.plot(tmoR, ld['iriAtRel'], 'x--', label=f"IRI ({ld['rel_IRI']}%)")
#     plt.plot([1, Nmo], [ld['IRIThresh'], ld['IRIThresh']], 'r-', linewidth=4, label='IRI threshold')
#     plt.xlabel('Months')
#     plt.ylabel('IRI (in/mile)')
#     plt.legend(loc='upper left', fontsize=10)
#     plt.grid(True)
#
#     # Subplot 2: Rutting
#     plt.subplot(322)
#     plt.plot(tmoR, ld['rut_TAC'], '.-', label='AC rut (50%)')
#     plt.plot(tmoR, ld['acrutAtRel'], 'x--', label=f"AC rut ({ld['rel_rut_TAC']}%)")
#     plt.plot(tmoR, ld['rutTOT'], 'o-', label='Total rut (50%)')
#     plt.plot(tmoR, ld['totrutAtRel'], '+--', label=f"Total rut ({ld['rel_rutTOT']}%)")
#     plt.plot([1, Nmo], [ld['RutACThresh'], ld['RutACThresh']], 'r--', linewidth=2, label='AC rut threshold')
#     plt.plot([1, Nmo], [ld['RutThresh'], ld['RutThresh']], 'r-', linewidth=4, label='Total rut threshold')
#     plt.xlabel('Months')
#     plt.ylabel('Rutting (in)')
#     plt.legend(loc='upper left', fontsize=10)
#     plt.grid(True)
#
#     # Subplot 3: Bottom-up Fatigue Cracking
#     plt.subplot(323)
#     plt.plot(tmoR, ld['FCbu'], '.-', label='FCbu (50%)')
#     plt.plot(tmoR, ld['fcbuAtRel'], 'x--', label=f"FCbu ({ld['rel_fcbu']}%)")
#     plt.plot([1, Nmo], [ld['FCbuThresh'], ld['FCbuThresh']], 'r-', linewidth=4, label='FCbu threshold')
#     plt.xlabel('Months')
#     plt.ylabel('Bottom-up Fatigue Cracking (%)')
#     plt.legend(loc='upper left', fontsize=10)
#     plt.grid(True)
#
#     # Subplot 4: Top-down Fatigue Cracking
#     plt.subplot(324)
#     plt.plot(tmoR, ld['FCtd'], '.-', label='FCtd (50%)')
#     plt.plot(tmoR, ld['fctdtAtRel'], 'x--', label=f"FCtd ({ld['rel_fctd']}%)")
#     plt.plot([1, Nmo], [ld['FCtdThresh'], ld['FCtdThresh']], 'r-', linewidth=4, label='FCtd threshold')
#     plt.xlabel('Months')
#     plt.ylabel('Top-down Fatigue Cracking (ft/mile)')
#     plt.legend(loc='upper left', fontsize=10)
#     plt.grid(True)
#
#     # Subplot 5: Thermal Cracking
#     plt.subplot(325)
#     plt.plot(tmoR, ld['TC_mo'], '.-', label='TC (50%)')
#     plt.plot(tmoR, ld['tc_ACAtRel'], 'x--', label=f"TC ({ld['rel_tc_AC']}%)")
#     plt.plot([1, Nmo], [ld['TCACThresh'], ld['TCACThresh']], 'r-', linewidth=4, label='TC threshold')
#     plt.xlabel('Months')
#     plt.ylabel('Thermal Cracking (ft/mile)')
#     plt.legend(loc='upper left', fontsize=10)
#     plt.grid(True)
#
#     # Subplot 6: Reflective Cracking (if applicable)
#     if ld.get('CSMExists') or ld.get('EACExists'):
#         plt.subplot(326)
#         plt.plot(tmoR, ld['TRCac'], 'x-', label='Ref. Crack (50%)')
#         plt.plot(tmoR, ld['reflCrackACAtRel'], 'x--', label=f"Ref. Crack ({ld['rel_reflCrackAC']}%)")
#         plt.plot([1, Nmo], [ld['reflCrackACThresh'], ld['reflCrackACThresh']], 'r-', linewidth=4,
#                  label='Ref. Crack threshold')
#         plt.xlabel('Months')
#         plt.ylabel('Reflective Cracking (%)')
#         plt.legend(loc='upper left', fontsize=10)
#         plt.grid(True)
#
#     # Adjust layout and show plot
#     # plt.tight_layout()
#     plt.show()
#

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
    else:
        return mat_obj


def compare_matlab_python(ld_out_Python, ld_out_Matlab):
    print("\nComparison of Python and MATLAB Results:\n")
    keys_in_both = set(ld_out_Python.keys()).intersection(ld_out_Matlab.keys())
    for key in sorted(keys_in_both):
        python_value = ld_out_Python[key]
        matlab_value = ld_out_Matlab[key]

        # Check if the values are arrays
        if isinstance(python_value, np.ndarray) and isinstance(matlab_value, np.ndarray):
            if python_value.shape != matlab_value.shape:
                print(f"Key: {key} - Shape Mismatch (Python: {python_value.shape}, MATLAB: {matlab_value.shape})")
                continue

            # Compute absolute and relative differences
            abs_diff = np.abs(python_value - matlab_value)
            rel_diff = np.abs(abs_diff / (np.abs(matlab_value) + 1e-12))  # Avoid division by zero
            max_abs_diff = np.max(abs_diff)
            max_rel_diff = np.max(rel_diff)

            if max_abs_diff > 1e-6:  # Tolerance for absolute differences
                print(f"--> BAD: Key: {key} - Max Abs Diff: {max_abs_diff:.6f}, Max Rel Diff: {max_rel_diff:.6f}")
            else:
                print(f"-> Good: Key: {key} - Max Abs Diff: {max_abs_diff:.6f}, Max Rel Diff: {max_rel_diff:.6f}")

        elif isinstance(python_value, (int, float)) and isinstance(matlab_value, (int, float)):
            # Compare scalar values
            abs_diff = abs(python_value - matlab_value)
            rel_diff = abs_diff / (abs(matlab_value) + 1e-12)
            if abs_diff > 1e-6:
                print(f"Key: {key} - Abs Diff: {abs_diff:.6f}, Rel Diff: {rel_diff:.6f}")

        else:
            # Handle other types (e.g., strings, dictionaries)
            if python_value != matlab_value:
                print(f"Key: {key} - Value Mismatch (Python: {python_value}, MATLAB: {matlab_value})")

    print("\nComparison Complete.")


if __name__ == "__main__":
    # Load input data
    mat_contents = loadmat('f_calc_distress_atrel_inp.mat')
    ld = mat_to_dict(mat_contents['ld'])

    # Load JSON configuration
    jsonfld = "/Users/kutay/Downloads"
    jsonfile = "CE431.json"
    inps = utl.loadjson(jsonfld, jsonfile)

    # Input parameters
    stdev_data = inps['stdevData']
    reliability_thresholds = inps['reliabilityThresholds']

    # Compute Python results
    ld_out_Python = f_calc_distress_atrel(ld, stdev_data, reliability_thresholds, plt_flag=1)

    # Load MATLAB results
    mat_contents = loadmat('f_calc_distress_atrel_out.mat')
    ld_out_Matlab = mat_to_dict(mat_contents['ld'])

    # Compare results
    compare_matlab_python(ld_out_Python, ld_out_Matlab)
