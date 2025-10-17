import os
import re
import math
import numpy as np
import json
import scipy
from matplotlib import pyplot as pp
from scipy.optimize import curve_fit

INPUT_JSON_PATH = 'ZZL-224/config_kin.json'

# --- Load Configuration from JSON ---
with open(INPUT_JSON_PATH, 'r') as f:
    config = json.load(f)

RUN_LIST = config.get("RUN_LIST", [])
RUN_DIR = config.get("RUN_DIR", INPUT_JSON_PATH.split('/')[0])
TYPE = config.get("TYPE", "TA")
CONCENTRATION_LIST = config.get("CONCENTRATION_LIST", [])
AUTOLABEL = config.get("AUTOLABEL", False)
LABEL_LIST = config.get("LABEL_LIST", [])
BLANK = config.get("BLANK", [])
COLOR_LIST = config.get("COLOR_LIST", [])
TITLE0 = config.get("TITLE0", "")
WAVELENGTH_DETECTED = config.get("WAVELENGTH_DETECTED", 0)
LENGTH = config.get("LENGTH", 1.0)
FIGSIZE = tuple(config.get("FIGSIZE", [10, 6])) # Convert list to tuple for matplotlib
XLABEL = config.get("XLABEL", "Wavelength (nm)")
xmin = config.get("XMIN") # Allows for None if not in JSON
xmax = config.get("XMAX")
YMIN = config.get("YMIN")
YMAX = config.get("YMAX")
X_CORR = config.get("X_CORR", 0.0) # X-axis correction value, 
# When solving kinetic data, this value can reset time-zero point. Resolve after X_SCALE.
NORMALIZATION_METHOD = config.get("NORMALIZATION_METHOD", 0)
X_SCALE = config.get("X_SCALE", 1.0)
Y_SCALE = config.get("Y_SCALE", 1.0)
SCALE_FACTOR = config.get("SCALE_FACTOR", 1.0)
SCALE_FACTOR_FLUORE = config.get("SCALE_FACTOR_FLUORE", 1e-4)
DO_FITTING = config.get("DO_FITTING", False)
FIT_FUNCTION = config.get("FIT_FUNCTION", None) # LOAD THE FUNCTION NAME
SHOW_FITTING_INFO = config.get("SHOW_FITTING_INFO", True)
# Option of FIT_FUNCTION:
# "exp_decay": t, y0, A, tau; y = y0 + A * exp(-t / tau)
# "exp_2decay": t, y0, A1, A2, tau1, tau2; y = y0 + A1 * exp(-t / tau1) + A2 * exp(-t / tau2) 
# "secondary_decay" t, y0, k, r0; y = y0 + 1 / (k * t + r0)

# --- End of Configuration Loading ---

# -- Redefining ---------------

LABEL_LIST = []
XLABEL = "Time (μs)"
name_reserved = ''
TO_FIND_TIME_ZERO = False # Whether to find time-zero point in kinetic absorption data

if xmin is not None and xmax is not None and (not TO_FIND_TIME_ZERO):
    xmin += X_CORR
    xmax += X_CORR

# --- Plotting Setup ---------------

if TYPE == "TA" or (TYPE == "KinAbs" and not DO_FITTING):
    pp.figure(figsize=FIGSIZE)
elif TYPE == "KinAbs" and DO_FITTING:
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
type_title = '' # Needless to set, it will be set automatically based on TYPE

# --- Parsing functions ---------------

# Find RunXX_TA_YY_ZZ.txt file in the directory
def find_run_file(num, directory = RUN_DIR, work_type = 'TA'):
    cwd = rf'{os.getcwd()}' + '\\'
    pattern = re.compile(rf'Run{num:02d}_{work_type}.*\.txt', re.IGNORECASE)
    for fname in os.listdir(cwd + directory):
        match = pattern.match(fname)
        if match:
            return cwd + directory + '\\' + fname
    print(cwd + directory)
    return None  

# This function extracts TA data from the specified file, 
# also suitable to kinetic absorption data (t/ns, ∆OD/a.u.)
def extract_TA_data(num, directory = RUN_DIR):
    file_path = find_run_file(num, directory, TYPE)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Read the data lines
    data = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                wavelength = float(parts[0])
                absorbance = float(parts[1])
                data.append([wavelength, absorbance])
            except ValueError:
                continue  # Skip lines that can't be parsed
    return np.array(data)

def extract_fluore_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        # Skip header or label lines
        if len(parts) == 2:
            try:
                x = float(parts[0])
                y = float(parts[1].replace('E+', 'E').replace('E-', 'E-'))  # Handles scientific notation
                data.append([x, y])
            except ValueError:
                continue  # Skip lines that can't be parsed
    return np.array(data)

# --- Functions to fit ---------------

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
    "secondary_decay": secondary_decay
}

# Plotting preprocessing ============================

ylabel = ""
label_list = []
if not AUTOLABEL:
    label_list = LABEL_LIST # Get labels
else:
    label_list = [f'Run{r:02d}' for r in RUN_LIST]

# Pick the fitting function
try:
    fit_function = FIT_MODELS[FIT_FUNCTION]
except KeyError:
    print(f"Error: Fit function '{FIT_FUNCTION}' not recognized. Defaulting to 'exp_decay'.")
    fit_function = exp_decay

#TA spectrum plotting ============================

if TYPE == "TA":
    type_title = 'TA'
    ylabel = '∆OD (a. u.)'
    data_arrays = [extract_TA_data(num) for num in RUN_LIST]
    for i, data in enumerate(data_arrays):
        # Data pre-processing ============================
        data[:,0] = data[:,0] * X_SCALE + X_CORR
        data[:,1] = data[:,1] * Y_SCALE
        if not True: # If you want to use a blank data set, set this to True
            pp.plot(data[:,0], data[:,1], label = label_list[i], color = '#b0b0b0')
        else:
            pp.plot(data[:,0], data[:,1], label = label_list[i], color = COLOR_LIST[i % len(COLOR_LIST)])

elif TYPE == "KinAbs":
    type_title = 'Absorption kinetics'
    ylabel = '∆OD (a. u.)'
    
    # This example fits the first dataset from RUN_LIST. 
    # You can loop this if you have multiple curves to fit.
    if RUN_LIST:
        data = extract_TA_data(RUN_LIST[0])
        # Data pre-processing
        time_data = data[:,0] * X_SCALE + X_CORR
        intensity_data = data[:,1] * Y_SCALE

        # This finds the index of the time value closest to zero
        zero_index = np.argmin(np.abs(time_data))

        # --- Perform the Fit --------------------------------------------------------
        try:
            # Provide initial guesses for the parameters
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
                    (intensity_data[zero_index] - intensity_data[-1]) * 0.4, # A2
                    (time_data[-1] - time_data[zero_index]) / 3, # tau1
                    (time_data[-1] - time_data[zero_index]) / 1.5 # tau2
                ]
            elif fit_function == secondary_decay:
                initial_guess = [
                    intensity_data[-1], # y0
                    0.01,               # k: small rate constant
                    0.01 # r0: initial reciprocal concentration
                ]
            else:
                print("Error: Unknown fitting function. Cannot perform fitting.")
                raise RuntimeError
            
            # Use curve_fit to find the best parameters
            popt, pcov = curve_fit(fit_function, 
                                   time_data[zero_index:], intensity_data[zero_index:], 
                                   p0=initial_guess)
            
            # --- Plot Original Data and Fit ---
            # Plot the raw data points on the top subplot (ax1)
            ax1.plot(time_data, intensity_data, 
                     label=f'Data',
                     color="#6babd8")
            
            # Generate a smooth curve for the fit
            tnum = len(time_data[zero_index:])
            t_fit = np.linspace(time_data[zero_index], time_data.max(), tnum)
            y_fit = fit_function(t_fit, *popt)
            R2 = calculate_r_squared(intensity_data[zero_index:], y_fit)
            
            # Plot the fitted curve on ax1
            ax1.plot(t_fit, y_fit, color='#0000aa', linestyle='-', 
                     label=f'Fit')
            
            # Print the fitted results
            print(f"Fit successful for Run {RUN_LIST[0]:02d}:")
            info_lines = [f'Func: \"{FIT_FUNCTION}\"', f'$R^2$ = {R2:.4f}']
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
                info_lines.append(f"$k$ = {popt[1]:.4f} μs⁻¹")
                info_lines.append(f"$r_0$ = {popt[2]:.4f}")
                info_lines.append(f"$y_0$ = {popt[0]:.4f}")

            info_text = "\n".join(info_lines)
            print(info_text)

            # --- Add fitting results into a text box on the diagram ---
            if SHOW_FITTING_INFO:
                ax1.text(0.97, 0.95, info_text, 
                         transform=ax1.transAxes, 
                         fontsize=12,
                         verticalalignment='top', 
                         horizontalalignment='right',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

            # --- Calculate and Plot Residuals ---
            # Residual = Original Y - Fitted Y
            residuals = intensity_data[zero_index:] - fit_function(time_data[zero_index:], *popt)
            
            # Plot residuals on the bottom subplot (ax2)
            ax2.scatter(time_data[zero_index:], residuals, facecolors='none', edgecolors='gray', s=20)
            ax2.axhline(0, color='black', linestyle='--', linewidth=1) # Add a zero line
            
        except RuntimeError:
            print(f"Error: Could not find a fit for Run {RUN_LIST[0]:02d}. Plotting raw data only.")
            ax1.plot(time_data, intensity_data, 
                     label='Data',
                     color="#6babd8")
            
# Finalizing the plot ============================

if TYPE == "TA" or (TYPE == "KinAbs" and not DO_FITTING):
    pp.ylabel(ylabel)
    pp.xlabel(XLABEL)

    pp.autoscale(enable = True, axis = 'x', tight = True)
    pp.autoscale(enable = True, axis = 'y')
    if label_list:
        pp.legend(loc = "best")

    TITLE = rf'{type_title} spectrum of {TITLE0}' # EXAMINE it carefully!!!

    pp.title(TITLE, fontsize = 18, fontweight = 'bold')

    if xmin is not None and xmax is not None:
        pp.xlim(xmin, xmax)
    if YMIN is not None and YMAX is not None:
        pp.ylim(YMIN, YMAX)

elif TYPE == "KinAbs" and DO_FITTING:
    # Finalize the top subplot (ax1)
    final_title = rf'{type_title} of {TITLE0}'
    if WAVELENGTH_DETECTED:
        final_title += rf' at {WAVELENGTH_DETECTED} nm'
    else:
        print("Warning: WAVELENGTH_DETECTED is not set properly.")
    
    ax1.set_ylabel(ylabel)
    ax1.set_title(final_title, 
                  fontsize=18, fontweight='bold')
    if xmin is not None and xmax is not None:
        ax1.set_xlim(xmin, xmax)
    if YMIN is not None and YMAX is not None:
        ax1.set_ylim(YMIN, YMAX)
    ax1.legend(loc="best")
    
    # Finalize the bottom subplot (ax2)
    ax2.set_xlabel(XLABEL)
    ax2.set_ylabel('Residuals')
    if xmin is not None and xmax is not None:
        ax2.set_xlim(xmin, xmax)

# --- Save and show the plot --------------

if input('Save Figures? y/n: ') == 'y':
    filename = RUN_DIR + '_' + TYPE + '_' + '_'.join(label_list) + name_reserved
    pp.savefig(f'{os.getcwd()}\\{filename}.png')
    print(f'{os.getcwd()}\\{filename}.png is saved.')

pp.show()
pp.close()
