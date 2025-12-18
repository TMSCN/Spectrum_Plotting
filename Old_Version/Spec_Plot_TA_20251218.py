# Latest update: 2025-12-17

import os
import re
import math
import numpy as np
import json
import scipy
from matplotlib import pyplot as pp
from scipy.optimize import curve_fit

INPUT_JSON_PATH = 'ZZL-283/config_E00_ddFppy_2.json'

#========== Common Notes ===========
# Supported TYPE:
# "UV": UV-Vis Spectrum
# "TA": Transient Absorption Spectrum
# "KinAbs": Kinetic Absorption Data (t/ns, ∆OD/a.u.)

#========== UV/TA/Fl normalization ===========
# NORMALIZATION_METHOD options:
# 0: Use Y_CORR as normalization (default is 1)
# 1: Normalize to epsilon (L·mol^-1·cm^-1)
# 2: Normalize by maximum value

#========== Notes on fitting ===========
# NORMALIZATION_METHOD in kinetic absorption fitting:
# 0: No normalization
# 1: Baseline correction only
# 2: Baseline correction + Normalize maximum to 1
# 3: Normalize by intensity at time-zero point

# SPECIAL NOTES ON THE FITTED k_VALUES: 
# The unit of secondary reaction rate constant k is 1/(μs·a.u.).
# 1/(μs·a.u.) = 1/[ eps * (μs·mol·L^-1)] = 1e6/(eps * (s·mol·L^-1)).
# If the intensity is 0.05 a.u., and c = 0.05 mM, then eps = 0.05 / 0.00005 = 1000.
# For secondary reaction rate constants, if k is for concentration, k' is for intensity:
# Then, we have k' = k / eps. (Secondary). k' = k (Primary).

# Option of FIT_FUNCTION:
# "exp_decay": t, y0, A, tau; y = y0 + A * exp(-t / tau)
# "exp_2decay": t, y0, A1, A2, tau1, tau2; y = y0 + A1 * exp(-t / tau1) + A2 * exp(-t / tau2) 
# "secondary_decay" t, y0, k, r0; y = y0 + 1 / (2 * k * t + r0)
# "primary_secondary_decay" t, y0, k1, k2, const, eps; y = y0 + eps * k1 / (exp(k1 * (t - const)) - 2 * k2)
# "primary_secondary_decay_eps1" t, y0, k1, k2, const; y = y0 + k1 / (exp(k1 * (t - const)) - 2 * k2)
# "primary_secondary_decay_efix" t, y0, k1, k2, const; eps = efix

# Note: In initial guess of primary_secondary_decay, const = - ln(eps* k1 / r0 + 2*k2) / k1, 
# The input const is r0 (~0.05, the maximum of KA spectrum) actually.

#region --- Load configuration from JSON ---
with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

RUN_LIST = config.get("RUN_LIST", [])
RUN_DIR = config.get("RUN_DIR", INPUT_JSON_PATH.split('/')[0])
TYPE = config.get("TYPE")
CONCENTRATION_LIST = config.get("CONCENTRATION_LIST", [])
AUTOLABEL = config.get("AUTOLABEL", False)
LABEL_LIST = config.get("LABEL_LIST", [])
BLANK = config.get("BLANK", [])
COLOR_LIST = config.get("COLOR_LIST", [])
TITLE0 = config.get("TITLE0", "")
WAVELENGTH_DETECTED = config.get("WAVELENGTH_DETECTED", 0)
LENGTH = config.get("LENGTH", 1.0)
FIGSIZE = tuple(config.get("FIGSIZE", [10, 6])) # Convert list to tuple for matplotlib
ENABLE_LEGEND = config.get("ENABLE_LEGEND", True)
SHOW_RUN_NUMBER = config.get("SHOW_RUN_NUMBER", False)
XLABEL = config.get("XLABEL", "Wavelength (nm)")
xmin = config.get("XMIN", 0.0) # Allows for None if not in JSON
xmax = config.get("XMAX", 0.0)
ymin = config.get("ymin", 0.0)
ymax = config.get("ymax", 0.0)
X_CORR = config.get("X_CORR", 0.0) # X-axis correction value.
# When solving kinetic data, this value can reset time-zero point. Resolve after x_scale.
Y_CORR = config.get("Y_CORR", 0.0) # Y-axis correction value.
x_scale = config.get("x_scale", 1.0)
y_scale = config.get("y_scale", 1.0)
Y_SCALE_FACTOR_IN_UNIT = config.get("Y_SCALE_FACTOR_IN_UNIT", 1.0)
SCALE_FACTOR_FLUORE = config.get("SCALE_FACTOR_FLUORE", 1e-4)
FIND_INTERSECTIONS = config.get("FIND_INTERSECTIONS", False)
SHOW_INTERSECTIONS = config.get("SHOW_INTERSECTIONS", False)
INTERSECTIONS_ABS_TOL = config.get("INTERSECTION_ABS_TOL", 5e-2)

DO_FITTING = config.get("DO_FITTING", False)
FIT_FUNCTION = config.get("FIT_FUNCTION", None) # LOAD THE FUNCTION NAME
SHOW_FITTING_INFO = config.get("SHOW_FITTING_INFO", True)
NORMALIZATION_METHOD = config.get("NORMALIZATION_METHOD", 0)
ENABLE_CUSTOM_INITIAL_GUESS = config.get("ENABLE_CUSTOM_INITIAL_GUESS", False)
CUSTOM_INITIAL_GUESS = config.get("CUSTOM_INITIAL_GUESS", [])
POSITION_OF_FITTING_INFO = config.get("POSITION_OF_FITTING_INFO", "top right")
RETRY_FIT_IF_FAIL = config.get("RETRY_FIT_IF_FAIL", True)
WEIGHTING_FOR_FIT = config.get("WEIGHTING_FOR_FIT", False)
WEIGHTING_AFTER = config.get("WEIGHTING_AFTER", 0.0) # us
WEIGHT = config.get("WEIGHT", 0.2)
CUSTOM_EFIX = config.get("CUSTOM_EFIX", 0.0)

#endregion --- End of Configuration Loading ---

#region --- Redefining ---------------

if CUSTOM_EFIX:
    efix = CUSTOM_EFIX
    found_efix = True
else:
    efix = 1.0
    found_efix = False

name_reserved = ''
TO_FIND_TIME_ZERO = False # Whether to find time-zero point in kinetic absorption data

if xmin and xmax and (not TO_FIND_TIME_ZERO):
    xmin += X_CORR
    xmax += X_CORR

#endregion -- End of Redefining ---------------

#region ---- Mode recognition ---------------------
# Determine plot mode and choose type_title / ylabel
# For normal UV-Vis/Fluorescence/TA plotting, plot_mode = 0
# For kinetic absorption fitting, plot_mode = 1
#
plot_mode = 0
type_title = '' # Needless to set, it will be set automatically based on TYPE
ylabel = 'Arbitrary unit'

if TYPE == "TA":
    type_title = 'TA'
    ylabel = '∆OD (a. u.)'
if TYPE == "UV" or TYPE == "UV-Vis":
    type_title = 'UV-Vis'
    ylabel = 'Absorbance (A)'
    if NORMALIZATION_METHOD == 1:
        ylabel = 'ε (L·mol$^{-1}$·cm$^{-1}$)'
if TYPE == "FL" or TYPE == "E00":
    type_title = 'Fluorescence'
    ylabel = 'Counts'
if TYPE == "EX":
    type_title = 'Fluorescence excitation'
    ylabel = 'Counts'
if TYPE == "EM":
    type_title = 'Fluorescence emission'
    ylabel = 'Counts'
if TYPE == "KinAbs" and not DO_FITTING:
    type_title = 'Absorption kinetics'
    ylabel = '∆OD (a. u.)'

if NORMALIZATION_METHOD == 2:
    if TYPE == 'E00' or TYPE == 'FL' or TYPE == 'EX' or TYPE == 'EM':
        ylabel = 'Normalized fluorescence intensity'
    elif TYPE == 'UV' or TYPE == 'UV-Vis':
        ylabel = 'Normalized absorbance'
    else:
        ylabel = 'Normalized unit'

if TYPE == "KinAbs" and DO_FITTING:
    type_title = 'Absorption kinetics'
    ylabel = '∆OD (a. u.)'
    plot_mode = 1

#endregion ---- End of mode recognition ---------------------

#region --- Plotting preprocessing ---------------

if plot_mode == 0:
    pp.figure(figsize=FIGSIZE)
elif plot_mode == 1:
    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = pp.subplots(
        2, 1, 
        figsize=FIGSIZE, 
        sharex=True,  # Both subplots will share the same x-axis
        gridspec_kw={'height_ratios': [3, 1]} # Main plot is 3x taller than residual plot
    )
    fig.subplots_adjust(hspace=0.05) # Remove space between plots

pp.rcParams['font.family'] = 'sans-serif'
pp.rcParams['font.sans-serif'] = ['Arial']
pp.rcParams['font.size'] = 16

label_list = []
if not AUTOLABEL:
    label_list = LABEL_LIST # Get labels
else:
    label_list = [f'Run{r:02d}' for r in RUN_LIST]

if SHOW_RUN_NUMBER:
    for i in range(len(label_list)):
        label_list[i] += f' (Run{RUN_LIST[i]:02d})'

#endregion --- Plotting preprocessing ---------------

#region --- Defining parsing functions ---------------

# Find RunXX_<TYPE>_YY_ZZ.txt file in the directory
# If work_type is '', find RunXX*.txt
def find_run_file(num, directory = RUN_DIR, work_type = ''):
    cwd = rf'{os.getcwd()}' + '\\'
    if work_type == '':
        pattern = re.compile(rf'Run{num:02d}.*\.txt', re.IGNORECASE)
    else:
        pattern = re.compile(rf'Run{num:02d}_{work_type}.*\.txt', re.IGNORECASE)

    for fname in os.listdir(cwd + directory):
        match = pattern.match(fname)
        if match:
            return cwd + directory + '\\' + fname
        
    print(cwd + directory)
    return None  

# This function extracts UV/Fluorescence/TA data from the specified file, 
# also suitable to kinetic absorption data (t/ns, ∆OD/a.u.)
def extract_data(num, directory = RUN_DIR, return_meta = False, type_adjust = ''):
    file_path = find_run_file(num, directory, type_adjust)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Read the data lines
    data = []
    meta_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                x = float(parts[0])
                y = float(parts[1].replace('E+', 'E').replace('E-', 'E-'))  
                # Handles scientific notation
                data.append([x, y])
                continue
            except ValueError:
                pass  # Skip lines that can't be parsed
        meta_lines.append(line.rstrip('\n'))

    if return_meta:
        return np.array(data), meta_lines
    else:
        return np.array(data)

#endregion --- End of Parsing functions ---------------

#region --- Functions to fit ---------------

def exp_decay(t, y0, A, tau):
    """
    Exponential decay function.
    y(t) = y0 + A * exp(-t / tau)
    y0: offset
    A: amplitude
    tau: lifetime
    """
    return y0 + A * np.exp(-t / tau)

def exp_2decay(t, y0, A1, A2, tau1, tau2):
    """
    Double exponential decay function.
    """
    return y0 + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

def secondary_decay(t, y0, k, r0):
    """
    Secondary reaction decay function.
    r0 = 1/c0
    k - rate constants
    y0 - offset
    """
    return y0 + 1 / (k * t + r0)

def primary_secondary_decay(t, y0, k1, k2, const, eps):
    return y0 + k1 / (np.exp(k1 * (t - const)) - 2 * k2) * eps

def primary_secondary_decay_eps1(t, y0, k1, k2, const):
    return y0 + k1 / (np.exp(k1 * (t - const)) - 2 * k2)

def primary_secondary_decay_efix(t, y0, k1, k2, const):
    return y0 + k1 / (np.exp(k1 * (t - const)) - 2 * k2) * efix

def calculate_r_squared(y_true, y_predicted):
    """Calculates the R-squared value."""
    residuals = y_true - y_predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

FIT_MODELS = {
    "exp_decay": exp_decay,
    "exp_2decay": exp_2decay,
    "secondary_decay": secondary_decay,
    "primary_secondary_decay": primary_secondary_decay,
    "primary_secondary_decay_eps1": primary_secondary_decay_eps1,
    "primary_secondary_decay_efix": primary_secondary_decay_efix
}

#endregion --- End of Functions to fit ---------------

#region --- Intersection function ---------------
def find_curve_intersections(x1, y1, x2, y2, xtol=1e-8, abs_tol=4e-2):
    """
    Return a list of (x, y) intersection points between two 1D curves.

    The function interpolates both curves onto the union of x-values, looks
    for sign changes in the difference, and uses a root finder to refine the
    intersection positions.
    """
    x1 = np.asarray(x1); y1 = np.asarray(y1)
    x2 = np.asarray(x2); y2 = np.asarray(y2)

    # Sort by x to be safe
    s1 = np.argsort(x1); x1, y1 = x1[s1], y1[s1]
    s2 = np.argsort(x2); x2, y2 = x2[s2], y2[s2]

    # Build a common grid from unique x-values
    x_common = np.unique(np.concatenate([x1, x2]))
    if x_common.size == 0:
        return []

    y1i = np.interp(x_common, x1, y1)
    y2i = np.interp(x_common, x2, y2)
    diff = y1i - y2i
    y_max = max(np.max(np.abs(y1i)), np.max(np.abs(y2i)))

    intersections = []

    # Exact zeros (within tolerance)
    zero_idx = np.where(np.isclose(diff, 0.0, atol=1e-12, rtol=0))[0]
    for idx in zero_idx:
        xi = float(x_common[idx])
        yi = float(y1i[idx])
        # The point is considered an intersection only if the y-value is significant (yi/ymax is greater than abs_tol)
        if y_max != 0 and (abs(yi) / y_max) >= abs_tol:
            # Then add the intersection
            intersections.append((xi, yi))

    # Sign changes indicate a root in the interval
    signs = np.sign(diff)
    sign_change_idx = np.where(signs[:-1] * signs[1:] < 0)[0]

    for idx in sign_change_idx:
        a = float(x_common[idx]); b = float(x_common[idx+1])
        f = lambda x: np.interp(x, x1, y1) - np.interp(x, x2, y2)
        try:
            root = float(scipy.optimize.brentq(f, a, b, xtol=xtol))
            yroot = float(np.interp(root, x1, y1))
            if y_max != 0 and (abs(yroot) / y_max) >= abs_tol:
                intersections.append((root, yroot))
        except Exception:
            # If brentq fails for any reason, skip this interval
            continue

    # Remove duplicates (within tolerance) and sort by x
    if not intersections:
        return []
    # Sort and merge near-duplicates
    intersections = sorted(intersections, key=lambda t: t[0])
    merged = [intersections[0]]
    for x,y in intersections[1:]:
        if abs(x - merged[-1][0]) > 1e-8:
            merged.append((x,y))
    return merged

#endregion --- End of intersection function ---------------

#region ---- TA spectrum plotting ---------------------

if plot_mode == 0:
    data_arrays = [extract_data(num) for num in RUN_LIST]
    for i, data in enumerate(data_arrays):
        # Data normalization ============================
        if NORMALIZATION_METHOD == 1:
            y_scale =  1000.0 / (CONCENTRATION_LIST[i] * LENGTH * Y_SCALE_FACTOR_IN_UNIT)
        if NORMALIZATION_METHOD == 2:
            y_max_value = np.max(data[:,1])
            if y_max_value != 0:
                y_scale = 1.0 / y_max_value
            else:
                print("!Warning: Maximum value is zero, cannot normalize.")
                y_scale = 1.0
        # Data pre-processing ============================
        data[:,0] = data[:,0] * x_scale + X_CORR
        data[:,1] = data[:,1] * y_scale + Y_CORR
        if not True: # If you want to use a blank data set, set this to True
            pp.plot(data[:,0], data[:,1], label = label_list[i], color = '#b0b0b0')
        else:
            if COLOR_LIST and label_list:
                pp.plot(data[:,0], data[:,1], label = label_list[i], color = COLOR_LIST[i % len(COLOR_LIST)])
            elif label_list:
                pp.plot(data[:,0], data[:,1], label = label_list[i])
            else:
                pp.plot(data[:,0], data[:,1])

    # If there are at least two curves, compute their intersections and mark them
    if FIND_INTERSECTIONS and len(data_arrays) >= 2:
        x1, y1 = data_arrays[0][:,0], data_arrays[0][:,1]
        x2, y2 = data_arrays[1][:,0], data_arrays[1][:,1]
        intersections = find_curve_intersections(x1, y1, x2, y2, abs_tol=INTERSECTIONS_ABS_TOL)
        num_intersections = len(intersections)
        if intersections:
            xs = [p[0] for p in intersections]
            ys = [p[1] for p in intersections]
            if SHOW_INTERSECTIONS:
                # Hollow circle markers: unfilled face, visible edge
                pp.scatter(xs, ys, marker='o', facecolors='none', edgecolors='k', s=80, linewidths=1.5, zorder=10, label='_nolegend_')
            print(f'{num_intersections} intersection(s) between first two curves:')
            for xi, yi in intersections:
                print(f'  x = {xi:.6f}, y = {yi:.6e}')
        else:
            print('No intersections found between first two curves.')

#endregion ---- End of TA spectrum plotting ---------------------

#region --- Absorption kinetics plotting and fitting ---------------------
elif plot_mode == 1:
    # You can loop this if you have multiple curves to fit.
    if RUN_LIST:
        #region --- Data extraction and pre-processing for fitting ---
        data = extract_data(RUN_LIST[0])
        time_data = data[:,0] * x_scale + X_CORR
        intensity_data = data[:,1] * y_scale

        # This finds the index of the time value closest to zero
        zero_index = np.argmin(np.abs(time_data))
        #endregion

        #region Normalization if needed
        scale = 1.0
        baseline = 0.0
        if NORMALIZATION_METHOD == 0:
            intensity_fit = intensity_data.copy()
        else:
            # baseline = mean of points before zero_index (if any), else first point
            if zero_index - 100 > 0:
                baseline = np.mean(intensity_data[:(zero_index - 100)])
            else:
                baseline = intensity_data[0]
            tmp = intensity_data - baseline

            if NORMALIZATION_METHOD == 1:
                intensity_fit = tmp
            elif NORMALIZATION_METHOD == 2:
                scale = np.max(np.abs(tmp))
                if scale == 0 or np.isnan(scale):
                    scale = 1.0
                intensity_fit = tmp / scale
            elif NORMALIZATION_METHOD == 3:
                scale = intensity_data[zero_index] if intensity_data[zero_index] != 0 else 1.0
                intensity_fit = intensity_data / scale
            else:
                # fallback to no normalization
                intensity_fit = intensity_data.copy()
                baseline = 0.0
                scale = 1.0

        # Keep originals for plotting/metrics
        intensity_orig = intensity_data.copy()

        #endregion

        #region Pick the fitting function
        try:
            fit_function = FIT_MODELS[FIT_FUNCTION]
        except KeyError:
            print(f"Error: Fit function '{FIT_FUNCTION}' not recognized. Defaulting to 'exp_decay'.")
            fit_function = exp_decay
        #endregion

        # --- Perform the Fit --------------------------------------------------------
        while RETRY_FIT_IF_FAIL:
            try:
                #region --- Provide initial guesses for the parameters ---
                if fit_function == exp_decay:
                    initial_guess = [
                        intensity_data[-1], # y0: guess the offset is the last data point
                        intensity_data[zero_index] - intensity_data[-1], # A: guess amplitude is the total drop
                        (time_data[-1] - time_data[zero_index]) / 2 # tau: guess lifetime is half the time range
                    ]
                elif fit_function == exp_2decay:
                    initial_guess = [
                        intensity_data[-1], # y0
                        (intensity_data[zero_index] - intensity_data[-1]) * 0.6, # A1
                        (intensity_data[zero_index] - intensity_data[-1]) * 0.3, # A2
                        (time_data[-1] - time_data[zero_index]) / 3, # tau1
                        (time_data[-1] - time_data[zero_index]) / 1.5 # tau2
                    ]
                elif fit_function == secondary_decay:
                    initial_guess = [
                        intensity_data[-1], # y0
                        0.01,               # k: small rate constant
                        0.01 # r0: initial reciprocal concentration
                    ]
                elif fit_function == primary_secondary_decay:
                    initial_guess = [
                        intensity_data[-1], # y0
                        (intensity_data[zero_index] - intensity_data[-1]) * 0.5,               # k1
                        1.0,              # k2
                        0.05,                # const
                        1000.0                 # eps
                    ]
                elif fit_function == primary_secondary_decay_eps1 or fit_function == primary_secondary_decay_efix:
                    initial_guess = [
                        intensity_data[-1], # y0
                        (intensity_data[zero_index] - intensity_data[-1]) * 0.5,               # k1
                        1.0,              # k2
                        0.05                # const
                    ]
                else:
                    print("Error: Unknown fitting function. Cannot perform fitting.")
                    exit()
                #endregion

                #region --- Override initial guess if custom guess is enabled ---
                if ENABLE_CUSTOM_INITIAL_GUESS and CUSTOM_INITIAL_GUESS:
                    initial_guess = CUSTOM_INITIAL_GUESS
                    # 以下部分专为一二级混合衰减设计，促进收敛
                    if fit_function == primary_secondary_decay:
                        initial_guess[3] = - np.log(initial_guess[4] * initial_guess[1] / initial_guess[3] + 2*initial_guess[2]) / initial_guess[1]
                    elif fit_function == primary_secondary_decay_eps1 or fit_function == primary_secondary_decay_efix:
                        initial_guess[3] = - np.log(efix * initial_guess[1] / initial_guess[3] + 2*initial_guess[2]) / initial_guess[1]
                
                initial_guess[0] = (initial_guess[0] - baseline) / scale # Convert y0 to normalized units 
                #endregion

                #region --- Apply weighting if enabled ---
                if WEIGHTING_AFTER <= 0.0:
                    WEIGHTING_FOR_FIT = False
                if WEIGHTING_FOR_FIT:
                    weights = np.ones_like(intensity_data[zero_index:])
                    weights[time_data[zero_index:] > WEIGHTING_AFTER] = WEIGHT  
                    # Reduce weight for data points after WEIGHTING_AFTER time     
                #endregion           

                #region --- Use curve_fit and generate a smooth curve for the fit ---
                popt, pcov = curve_fit(fit_function, 
                                    time_data[zero_index:], intensity_fit[zero_index:], 
                                    p0=initial_guess,
                                    sigma=weights if WEIGHTING_FOR_FIT else None,
                                    method='trf',
                                    maxfev=5000
                                    )
                
                tnum = len(time_data[zero_index:])
                t_fit = np.linspace(time_data[zero_index], time_data.max(), tnum)
                y_fit = fit_function(t_fit, *popt)
                
                # y_fit currently is in normalized units -> convert to original units
                y_fit_plot = y_fit * scale + baseline
                # Calculate residuals: Residual = Original Y - Fitted Y
                residuals = intensity_orig[zero_index:] - (fit_function(time_data[zero_index:], *popt) * scale + baseline)

                R2 = calculate_r_squared(intensity_orig[zero_index:], y_fit_plot)
                #endregion
                
                #region --- Print the fitted results ---
                print(f"Fit successful for Run {RUN_LIST[0]:02d}:")
                info_lines = [f'Func: \"{FIT_FUNCTION}\"', 
                            f'$R^2$ = {R2:.4f}',]
                if fit_function == exp_decay:
                    info_lines.append(f"$τ$ = {popt[2]:.4f} μs")
                    info_lines.append(f"$A$ = {popt[1]:.4f}")
                    info_lines.append(f"$y_0$ = {popt[0]:.4f}")
                elif fit_function == exp_2decay:
                    info_lines.append(f"$τ_1$ = {popt[3]:.4f} μs ({abs(popt[1])*100/(abs(popt[1])+abs(popt[2])):.1f}%)")
                    info_lines.append(f"$τ_2$ = {popt[4]:.4f} μs ({abs(popt[2])*100/(abs(popt[1])+abs(popt[2])):.1f}%)")
                    info_lines.append(f"$A_1$ = {popt[1]:.4f}")
                    info_lines.append(f"$A_2$ = {popt[2]:.4f}")
                    info_lines.append(f"$y_0$ = {popt[0]:.4f}")
                elif fit_function == secondary_decay:
                    info_lines.append(f"$k^'$ = {popt[1]:.4f} /(μs·a.u.)")
                    info_lines.append(f"1/$A_0$ = {popt[2]:.4f}")
                    info_lines.append(f"$y_0$ = {popt[0]:.4f}")
                elif fit_function == primary_secondary_decay:
                    info_lines.append(f"$k_1$ = {popt[1]:.4f} /μs")
                    info_lines.append(f"$k_2^'$ = {popt[2]:.4f} /(μs·a.u.)")
                    info_lines.append(f"const = {popt[3]:.4f} μs")
                    info_lines.append(f"eps = {popt[4]:.4f}")
                    info_lines.append(f"$y_0$ = {popt[0]:.4f}")
                elif fit_function == primary_secondary_decay_eps1:
                    info_lines.append(f"$k_1$ = {popt[1]:.4f} /μs")
                    info_lines.append(f"$k_2^'$ = {popt[2]:.4f} /(μs·a.u.)")
                    info_lines.append(f"const = {popt[3]:.4f} μs")
                    info_lines.append(f"$y_0$ = {popt[0]:.4f}")
                elif fit_function == primary_secondary_decay_efix and found_efix:
                    info_lines.append(f"$k_1$ = {popt[1]:.4f} /μs")
                    info_lines.append(f"$k_2^'$ = {popt[2]:.4f} /(μs·a.u.)")
                    info_lines.append(f"const = {popt[3]:.4f} μs")
                    info_lines.append(f"eps = {efix:.4f} (fixed)")
                    info_lines.append(f"$y_0$ = {popt[0]:.4f}")
                if WEIGHTING_FOR_FIT:
                    info_lines.append(f"Weight({WEIGHT}) when t > {WEIGHTING_AFTER} μs")

                info_text = "\n".join(info_lines)
                print(info_text)
                #endregion

                #region --- Calculate and print residual statistics---
                residual_mean = np.mean(np.abs(residuals))
                print(f"Mean of residuals = {residual_mean}")
                residual_max = np.max(np.abs(residuals))
                print(f"Largest residual = {residual_max}")
                #endregion

                #region --- Check R² value for fit quality ---
                if np.abs(R2) < 0.2:
                    print(f"!Warning: Low R² value ({R2:.4f}) indicates poor fit quality.")
                    if RETRY_FIT_IF_FAIL:
                        retry = input("Retry with adjusted initial guesses? y/n: ").lower()
                        if retry != 'n':
                            # Adjust initial guesses slightly for retry
                            print("Enter new initial guesses. eg: xx yy zz")
                            print(f"Current initial guess: {initial_guess}")
                            input_initial_guess = input().strip()
                            CUSTOM_INITIAL_GUESS = [float(x) for x in input_initial_guess.split()]
                            ENABLE_CUSTOM_INITIAL_GUESS = True
                            continue
                #endregion

                #region --- Refit if fit_function == primary_secondary_decay_efix ---
                if fit_function == primary_secondary_decay_efix and (not found_efix):
                    efix = fit_function(time_data[zero_index], *popt) / CONCENTRATION_LIST[0] * 1000
                    print(f"Calculated efix = {efix:.4f}")
                    # Then refit with fixed efix
                    CUSTOM_INITIAL_GUESS = list(popt)
                    CUSTOM_INITIAL_GUESS[3] = intensity_data[zero_index]
                    ENABLE_CUSTOM_INITIAL_GUESS = True
                    found_efix = True
                    continue
                #endregion

                #region --- Plot Original Data and Fit ---

                # Plot the raw data points on the top subplot (ax1)
                ax1.plot(time_data, intensity_data, 
                        label=f'Data',
                        color="#6babd8")
                
                # Plot the fitted curve on ax1
                ax1.plot(t_fit, y_fit_plot, color='#0000aa', linestyle='-', 
                        label=f'Fit')
                
                # Add fitting results into a text box on the diagram ---
                if SHOW_FITTING_INFO:
                    v = 'bottom'
                    y_off = 0.07
                    if 'top' in POSITION_OF_FITTING_INFO:
                        v = 'top'
                        y_off = 0.95
                    h = 'left' if 'left' in POSITION_OF_FITTING_INFO else 'right'
                    ax1.text(0.97, y_off, info_text, 
                            transform=ax1.transAxes, 
                            fontsize=12,
                            verticalalignment=v, 
                            horizontalalignment=h,
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
                    
                # Plot residuals on the bottom subplot (ax2)
                ax2.scatter(time_data[zero_index:], residuals, facecolors='none', edgecolors='gray', s=20)
                ax2.axhline(0, color='black', linestyle='--', linewidth=1) # Add a zero line
                
                #endregion

                if RETRY_FIT_IF_FAIL:
                    break

            except RuntimeError:
                print(f"Error: Could not find a fit for Run {RUN_LIST[0]:02d}.")
                if RETRY_FIT_IF_FAIL:
                    retry = input("Retry with adjusted initial guesses? y/n: ").lower()
                    if retry != 'y':
                        exit()
                    else:
                        # Adjust initial guesses slightly for retry
                        print("Enter new initial guesses. eg: xx yy zz")
                        print(f"Current initial guess: {initial_guess}")
                        input_initial_guess = input().strip()
                        CUSTOM_INITIAL_GUESS = [float(x) for x in input_initial_guess.split()]
                        ENABLE_CUSTOM_INITIAL_GUESS = True
   
#endregion --- End of Absorption kinetics plotting and fitting ---------------------

#region --- Finalizing the plot ---------------

if plot_mode == 0:
    pp.ylabel(ylabel)
    pp.xlabel(XLABEL)

    pp.autoscale(enable = True, axis = 'x', tight = True)
    pp.autoscale(enable = True, axis = 'y')
    if ENABLE_LEGEND and label_list:
        pp.legend(loc = "upper right")

    TITLE = rf'{type_title} spectrum of {TITLE0}' # EXAMINE it carefully!!!

    pp.title(TITLE, fontsize = 18, fontweight = 'bold')

    if xmin and xmax:
        pp.xlim(xmin, xmax)
    if ymin and ymax:
        pp.ylim(ymin, ymax)

elif plot_mode == 1:
    # Finalize the top subplot (ax1)
    final_title = rf'{type_title} of {TITLE0}'
    if WAVELENGTH_DETECTED:
        final_title += rf' at {WAVELENGTH_DETECTED} nm'
    else:
        print("!Warning: WAVELENGTH_DETECTED is not set properly.")
    
    ax1.set_ylabel(ylabel)
    ax1.set_title(final_title, 
                  fontsize=18, fontweight='bold')
    if xmin is not None and xmax is not None:
        ax1.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax1.set_ylim(ymin, ymax)
    if ENABLE_LEGEND:
        ax1.legend(loc="best")
    
    # Finalize the bottom subplot (ax2)
    ax2.set_xlabel(XLABEL)
    ax2.set_ylabel('Residuals')
    if xmin is not None and xmax is not None:
        ax2.set_xlim(xmin, xmax)

#endregion --- End of Finalizing the plot ---------------

#region --- Save and show the plot --------------

if type_title == '':
    print("!Warning: The input TYPE is not recognized.")

yorn = input('Show Figures? y/n/x: ').lower()
if yorn == 'y':
    filename = RUN_DIR + '_' + TYPE + '_' + '_'.join(label_list) + '_' + TITLE0.replace(' ', '_')
    if TYPE == "KinAbs":
        filename += f'_{WAVELENGTH_DETECTED}nm'
        if DO_FITTING:
            filename += f'_{FIT_FUNCTION}'
        if WEIGHTING_FOR_FIT:
            filename += f'_WAfter{WEIGHTING_AFTER}_W{WEIGHT}'
    filename += name_reserved

    pp.savefig(f'{os.getcwd()}\\{filename}.png')
    print(f'{os.getcwd()}\\{filename}.png is saved.')

elif yorn == 'x':
    print('Exit without showing or saving figures.')
    pp.close()
    exit()

pp.show()
pp.close()

#endregion --- End of Save and show the plot --------------
