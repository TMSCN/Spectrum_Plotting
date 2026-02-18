"""SpecPlot - Powerful tool for spectrum plotting and data processing."""
import os
import sys
import re
import json
import math
import tkinter as tk
from tkinter import filedialog

import numpy as np
import scipy
from matplotlib import pyplot as pp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
#from scipy.signal import convolve

LATEST_UPDATE = "2026-02-17"
VERSION = "0.2"

#========== Common Notes ===========
# Supported TYPE:
# "UV": UV-Vis Spectrum
# "TA": Transient Absorption Spectrum
# "KinAbs"/"KA": Kinetic Absorption Data (t/ns, ∆OD/a.u.)

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
# "linear": t, y0, k; y = y0 + k * t
# "exp_decay": t, y0, A, tau; y = y0 + A * exp(-t / tau)
# "exp_2decay": t, y0, A1, A2, tau1, tau2; y = y0 + A1 * exp(-t / tau1) + A2 * exp(-t / tau2)
# "exp_3decay": t, y0, A1, A2, A3, tau1, tau2, tau3;
# y = y0 + A1 * exp(-t / tau1) + A2 * exp(-t / tau2) + A3 * exp(-t / tau3)
# "secondary_decay" t, y0, k, r0; y = y0 + 1 / (2 * k * t + r0)
# "primary_secondary_decay" t, y0, k1, k2, const, eps;
# y = y0 + k1 / (exp(k1 * (t - const)) - 2 * k2)
# "primary_secondary_seperate" t, y0, A, k1, k2, r0;
# y = y0 + A * exp( - k1 * t) + 1 / ( k2 * t + r0)
# "primary_rise": t, y0, A, k; y = y0 + A * (1 - exp(-k * t))
# "primary_rise_linear_offset": t, y0, A, k, B; y = y0 + A * (1 - exp(-k * t)) + B * t
# "double_primary_rise": t, y0, A1, k1, A2, k2;
# y = y0 + A1 * (1 - exp(-k1 * t)) + A2 * (1 - exp(-k2 * t))
# "double_primary_rise_linear_offset":
# t, y0, A1, k1, A2, k2, B; y = y0 + A1 * (1 - exp(-k1 * t)) + A2 * (1 - exp(-k2 * t)) + B * t
# "primary_cascade": t, y0, A, k1, k2; y = y0 + A  * (exp(-k1 * t) - exp(-k2 * t))

# Note: In initial guess of primary_secondary_decay, const = - ln(k1 / r0 + 2*k2) / k1,
# The input const is r0 (~0.05, the maximum of KA spectrum) actually.

#region --- Basic functions ---

# Colors for terminal output
class Colors:
    """Define ANSI escape codes for colored terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def infer_time_unit(time_scale):
    """Infer time unit string based on the given scale (in seconds)."""
    mapping = {1.0: 'ns', 1e-3: 'μs', 1e-6: 'ms', 1e-9: 's'}
    # exact match (within tol)
    for scale_key, unit_val in mapping.items():
        if math.isclose(time_scale, scale_key, rel_tol=1e-9, abs_tol=1e-12):
            return unit_val
    # fallback: choose the nearest by log10 distance
    try:
        closest = min(mapping.keys(), key=lambda k: abs(math.log10(time_scale) - math.log10(k)))
        return mapping[closest]
    except (ValueError, ZeroDivisionError):
        return 'ns'

def detect_time_zero(time, intensity, search_fraction=0.25, smooth_window=5, mode=-1):
    """
    Mode:
    0 - Return the maximum intensity point as time-zero.
    1 - Heuristically detect the time-zero point in a kinetic trace.
    Strategy:
    - Compute the numerical derivative dy/dt.
    - Smooth the derivative with a moving average.
    - Search for the largest absolute slope within the early-time window
      (controlled by `search_fraction`).
    Returns the detected time value (float). If detection fails, returns
    the time corresponding to the minimum absolute time (closest to zero).
    Else - No adjusting.
    """
    if mode == 0:
        return float(time[np.argmax(np.abs(intensity))])
    if mode == 1:
        t = np.asarray(time)
        y = np.asarray(intensity)
        if t.size < 3:
            return float(t[0])

        # Define search window in time domain
        tmin, tmax = t.min(), t.max()
        cutoff = tmin + search_fraction * (tmax - tmin)
        mask = t <= cutoff
        if np.count_nonzero(mask) < 3:
            mask = np.ones_like(t, dtype=bool)

        # Numerical derivative
        dy = np.gradient(y, t)

        # Smooth derivative with moving average
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / float(smooth_window)
            dy_s = np.convolve(dy, kernel, mode='same')
        else:
            dy_s = dy

        # Choose index of maximum absolute (smoothed) slope within mask
        try:
            idxs = np.where(mask)[0]
            rel_idx = int(np.argmax(np.abs(dy_s[idxs])))
            detected_tz = float(t[idxs][rel_idx])
            return detected_tz
        except (IndexError, ValueError):
            return float(t[np.argmin(np.abs(t))])
    return 0.0

#endregion --- Basic functions ---

#region --- Basic fitting functions ---

def linear(t, y0, k):
    """y0 + k*t"""
    return y0 + k * t

def exp_decay(t, y0, A, tau):
    """y0 + A*exp(-t/tau)"""
    return y0 + A * np.exp(-t / tau)

def exp_2decay(t, y0, A1, A2, tau1, tau2):
    """y0 + A1*exp(-t/tau1) + A2*exp(-t/tau2)"""
    return y0 + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

def exp_3decay(t, y0, A1, A2, A3, tau1, tau2, tau3):
    """y0 + A1*exp(-t/tau1) + A2*exp(-t/tau2) + A3*exp(-t/tau3)"""
    return y0 + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + A3 * np.exp(-t / tau3)

def secondary_decay(t, y0, k, r0):
    """y0 + 1/(k*t + r0)"""
    return y0 + 1 / (k * t + r0)

def primary_secondary_decay(t, y0, k1, k2, const):
    """y0 + k1/(exp(k1*(t-const)) - 2*k2)"""
    return y0 + k1 / (np.exp(k1 * (t - const)) - 2 * k2)

def primary_secondary_seperate(t, y0, A, k1, k2, r0):
    """y0 + A*exp(-k1*t) + 1/(k2*t + r0)"""
    return y0 + A * np.exp( - k1 * t) + 1 / ( k2 * t + r0)

def primary_rise(t, y0, A, k):
    """y0 + A*(1 - exp(-k*t))"""
    return y0 + A * (1 - np.exp(-k * t))

def primary_rise_linear_offset(t, y0, A, k, B):
    """y0 + A*(1 - exp(-k*t)) + B*t"""
    return y0 + A * (1 - np.exp(-k * t)) + B * t

def primary_rise_double_offset(t, y0, A, k, B, t0):
    """y0 + A*(1 - exp(-k*(t-t0))) + B*t"""
    return y0 + A * (1 - np.exp(-k * (t - t0))) + B * t

def primary_rise_triple_offset(t, y0, A, k, B, C, t0):
    """y0 + A*(C - exp(-k*(t-t0))) + B*t"""
    return y0 + A * (C - np.exp(-k * (t - t0))) + B * t

def double_primary_rise(t, y0, A1, k1, A2, k2):
    """y0 + A1*(1-exp(-k1*t)) + A2*(1-exp(-k2*t))"""
    return y0 + A1 * (1 - np.exp(-k1 * t)) + A2 * (1 - np.exp(-k2 * t))

def double_primary_rise_linear_offset(t, y0, A1, k1, A2, k2, B):
    """y0 + A1*(1-exp(-k1*t)) + A2*(1-exp(-k2*t)) + B*t"""
    return y0 + A1 * (1 - np.exp(-k1 * t)) + A2 * (1 - np.exp(-k2 * t)) + B * t

def primary_cascade(t, y0, A, k1, k2):
    """y0 + A*(exp(-k1*t) - exp(-k2*t))"""
    return y0 + A  * (np.exp(-k1 * t) - np.exp(-k2 * t))

def gaussian(t, y0, A, t0, sigma):
    """y0 + A*exp((t-t0)^2/(2*sigma^2))"""
    return y0 + A * np.exp((t - t0)**2 / (2 * sigma ** 2))

def primary_rise_with_gaussian(t, y0, A, k, B, t0, sigma, C):
    """y0 + A*(1-exp(-k*t)) + B*exp(-0.5*(t-t0/sigma)^2) + C*t"""
    return y0 + A * (1 - np.exp(-k * t)) + B * np.exp( -0.5 * (t - t0 / sigma) ** 2) + C * t

FIT_MODELS = {
    "linear": linear,
    "linear_decay": linear,
    "zero": linear,
    "zero_decay": linear,
    "l": linear,
    "x": linear,
    "0": linear,

    "exp": exp_decay,
    "decay": exp_decay,
    "exp_decay": exp_decay,
    "primary": exp_decay,
    "primary_decay": exp_decay,
    "ed": exp_decay,
    "pd": exp_decay,
    "e^x": exp_decay,
    "e^-x": exp_decay,
    "1": exp_decay,

    "exp2": exp_2decay,
    "2decay": exp_2decay,
    "exp_2decay": exp_2decay,
    "primary_2decay": exp_2decay,
    "double_decay": exp_2decay,
    "double_primary": exp_2decay,
    "ded": exp_2decay,
    "dpd": exp_2decay,
    "11": exp_2decay,

    "exp3": exp_3decay,
    "3decay": exp_3decay,
    "exp_3decay": exp_3decay,
    "primary_3decay": exp_3decay,
    "triple_decay": exp_3decay,
    "triple_primary": exp_3decay,
    "ted": exp_3decay,
    "tpd": exp_3decay,
    "111": exp_3decay,

    "secondary": secondary_decay,
    "secondary_decay": secondary_decay,
    "reciprocal": secondary_decay,
    "reciprocal_decay": secondary_decay,
    "sd": secondary_decay,
    "rd": secondary_decay,
    "1/x": secondary_decay,
    "2": secondary_decay,

    "primary_secondary": primary_secondary_decay,
    "primary_secondary_decay": primary_secondary_decay,
    "psd": primary_secondary_decay,
    "spd": primary_secondary_decay,
    "12": primary_secondary_decay,
    "21": primary_secondary_decay,

    "primary_secondary_seperate": primary_secondary_seperate,
    "pss": primary_secondary_seperate,
    "1-2": primary_secondary_seperate,
    "2-1": primary_secondary_seperate,

    "rise": primary_rise,
    "primary_rise": primary_rise,
    "pr": primary_rise,
    "er": primary_rise,
    "1-e^x": primary_rise,
    "1+": primary_rise,

    "rise0": primary_rise_linear_offset,
    "risel": primary_rise_linear_offset,
    "primary_rise_linear_offset": primary_rise_linear_offset,
    "prl": primary_rise_linear_offset,
    "erl": primary_rise_linear_offset,
    "1+0": primary_rise_linear_offset,
    "10+": primary_rise_linear_offset,
    "01+": primary_rise_linear_offset,

    "rise2": primary_rise_double_offset,
    "primary_rise_double_offset": primary_rise_double_offset,
    "prd": primary_rise_double_offset,
    "erd": primary_rise_double_offset,
    "1+00": primary_rise_double_offset,
    "001+": primary_rise_double_offset,

    "rise3": primary_rise_triple_offset,
    "primary_rise_triple_offset": primary_rise_triple_offset,
    "prt": primary_rise_triple_offset,
    "ert": primary_rise_triple_offset,
    "1+000": primary_rise_triple_offset,
    "0001+": primary_rise_triple_offset,

    "double_rise": double_primary_rise,
    "double_primary_rise": double_primary_rise,
    "dpr": double_primary_rise,
    "11+": double_primary_rise,

    "double_rise0": double_primary_rise_linear_offset,
    "double_primary_rise_linear_offset": double_primary_rise_linear_offset,
    "dprl": double_primary_rise_linear_offset,
    "derl": double_primary_rise_linear_offset,
    "110+": double_primary_rise_linear_offset,
    "011+": double_primary_rise_linear_offset,
    "11+0": double_primary_rise_linear_offset,

    "cascade": primary_cascade,
    "primary_cascade": primary_cascade,
    "pc": primary_cascade,
    "1-1": primary_cascade,

    "gaussian": gaussian,
    "g": gaussian,

    "rise_gaussian": primary_rise_with_gaussian,
    "primary_rise_with_gaussian": primary_rise_with_gaussian,
    "prg": primary_rise_with_gaussian,
    "1+g": primary_rise_with_gaussian,
    "g1+": primary_rise_with_gaussian,

}

#endregion --- End of basic fitting functions ---

#region --- Initialize ---

# Invoke a file dialog to select the input JSON configuration file
def select_file():
    """Call the filedialog and select an input file. Return str: route/filename."""
    root = tk.Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    fn = filedialog.askopenfilename()
    root.destroy()
    return fn

def initialize(input_json_path=None):
    """
    Load configuration from JSON and set module-level globals.
    If input_json_path is None, show file dialog.
    Then, set up global variables, decide mode of plotting / plotting settings.
    Finally return the loaded config dict.
    """
    global INPUT_JSON_PATH, config, RUN_LIST, RUN_DIR, TYPE, CONCENTRATION_LIST
    global FILENAME, FILENAME_MODE
    global AUTOLABEL, LABEL_LIST, BLANK, COLOR_LIST, TITLE0, TITLE_FONTSIZE
    global LINEWIDTH, WAVELENGTH_DETECTED, LENGTH, FIGSIZE, ENABLE_LEGEND, FONTSIZE
    global SHOW_RUN_NUMBER, XLABEL, YLABEL, XMIN, XMAX, YMIN, YMAX, YMAX_OF_RESIDUAL
    global X_CORR, Y_CORR, x_scale, y_scale, Y_SCALE_FACTOR_IN_UNIT, SHOW_STATISTICS
    global SCALE_FACTOR_FLUORE, X_CUT_AFTER, FIND_INTERSECTIONS, SHOW_INTERSECTIONS
    global INTERSECTION_ABS_TOL, DO_FITTING, FIT_FUNCTION, SHOW_FITTING_INFO
    global NORMALIZATION_METHOD, ENABLE_CUSTOM_INITIAL_GUESS, CUSTOM_INITIAL_GUESS
    global UPPER_BOUND, LOWER_BOUND, FIX_VALUE, ERROR_ANALYSIS, BASELINE_ID
    global POSITION_OF_FITTING_INFO, RETRY_FIT_IF_FAIL, WEIGHTING_FOR_FIT
    global IRF_ID, IRF_WIDTH, DISCARD_POINTS_WHEN_RECONV, RECONV_TZ
    global WEIGHTING_AFTER, WEIGHT, TO_FIND_TIME_ZERO, AUTO_FIND_TIME_ZERO
    global LOOP, COPY_FILENAME_TO_CLIPBOARD
    global type_title, plot_mode, label_list

    if input_json_path == "NULL":
        print('No configuration file is loaded')
        plot_mode = -1
        return {}

    if input_json_path is None:
        INPUT_JSON_PATH = select_file()
    else:
        INPUT_JSON_PATH = input_json_path

    if INPUT_JSON_PATH:
        print('Reading configuration from:', INPUT_JSON_PATH)
    else:
        print('No configuration file selected. Exiting.')
        sys.exit()

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Assign simple items (same keys and defaults as before)
    LOOP = config.get("LOOP", True)
    RUN_LIST = config.get("RUN_LIST", [])
    if not RUN_LIST:
        print("RUN_LIST is empty. Exiting.")
        sys.exit()

    CONCENTRATION_LIST = config.get("CONCENTRATION", [])
    if not CONCENTRATION_LIST:
        CONCENTRATION_LIST = config.get("CONCENTRATION_LIST", [1.0])
    RUN_DIR = config.get("RUN_DIR", None)
    FILENAME = config.get("FILENAME", "")
    TYPE = config.get("TYPE","")
    AUTOLABEL = config.get("AUTOLABEL", False)
    LABEL_LIST = config.get("LABEL_LIST", [])
    BLANK = config.get("BLANK", [])
    COLOR_LIST = config.get("COLOR_LIST", [])
    TITLE0 = config.get("TITLE0", "")
    TITLE_FONTSIZE = config.get("TITLE_FONTSIZE", 18)
    LINEWIDTH = config.get("LINEWIDTH", 1.5)
    WAVELENGTH_DETECTED = config.get("WAVELENGTH_DETECTED", 0)
    LENGTH = config.get("LENGTH", 1.0)
    FIGSIZE = tuple(config.get("FIGSIZE", [10, 6]))
    FONTSIZE = config.get("FONTSIZE", 16)
    ENABLE_LEGEND = config.get("ENABLE_LEGEND", True)
    SHOW_RUN_NUMBER = config.get("SHOW_RUN_NUMBER", False)
    XLABEL = config.get("XLABEL", "")
    YLABEL = config.get("YLABEL", "")

    XMIN = config.get("XMIN", 0.0)
    XMAX = config.get("XMAX", 0.0)
    YMIN = config.get("YMIN", 0.0)
    YMAX = config.get("YMAX", 0.0)
    YMAX_OF_RESIDUAL = config.get("YMAX_OF_RESIDUAL", 0.0)
    X_CORR = config.get("X_CORR", 0.0)
    Y_CORR = config.get("Y_CORR", 0.0)
    x_scale = config.get("X_SCALE", 1.0)
    y_scale = config.get("Y_SCALE", 1.0)
    Y_SCALE_FACTOR_IN_UNIT = config.get("Y_SCALE_FACTOR_IN_UNIT", 1.0)
    SCALE_FACTOR_FLUORE = config.get("SCALE_FACTOR_FLUORE", 1e-4)
    X_CUT_AFTER = config.get("X_CUT_AFTER", None)
    SHOW_STATISTICS = config.get("SHOW_STATISTICS", True)

    FIND_INTERSECTIONS = config.get("FIND_INTERSECTIONS", False)
    SHOW_INTERSECTIONS = config.get("SHOW_INTERSECTIONS", False)
    INTERSECTION_ABS_TOL = config.get("INTERSECTION_ABS_TOL", 1e-2)

    DO_FITTING = config.get("DO_FITTING", False)
    FIT_FUNCTION = config.get("FIT_FUNCTION", None)
    SHOW_FITTING_INFO = config.get("SHOW_FITTING_INFO", 1)
    NORMALIZATION_METHOD = config.get("NORMALIZATION_METHOD", 0)
    ENABLE_CUSTOM_INITIAL_GUESS = config.get("ENABLE_CUSTOM_INITIAL_GUESS", False)
    CUSTOM_INITIAL_GUESS = config.get("CUSTOM_INITIAL_GUESS", [])
    POSITION_OF_FITTING_INFO = config.get("POSITION_OF_FITTING_INFO", "top right")
    RETRY_FIT_IF_FAIL = config.get("RETRY_FIT_IF_FAIL", True)
    WEIGHTING_FOR_FIT = config.get("WEIGHTING_FOR_FIT", False)
    WEIGHTING_AFTER = config.get("WEIGHTING_AFTER", 0.0)
    WEIGHT = config.get("WEIGHT", 0.2)
    BASELINE_ID = config.get("BASELINE", 0)
    IRF_ID = config.get("IRF", 0)
    IRF_WIDTH = config.get("IRF_WIDTH", 80.0)
    DISCARD_POINTS_WHEN_RECONV = config.get("DISCARD_POINTS_WHEN_RECONV", 50)
    RECONV_TZ = config.get("RECONV_TZ", False)
    ERROR_ANALYSIS = config.get("ERROR_ANALYSIS", False)

    FILENAME_MODE = config.get("FILENAME_MODE", 0)
    COPY_FILENAME_TO_CLIPBOARD = config.get("COPY_FILENAME_TO_CLIPBOARD", True)

    TO_FIND_TIME_ZERO = config.get("TO_FIND_TIME_ZERO", False)
    AUTO_FIND_TIME_ZERO = config.get("AUTO_FIND_TIME_ZERO", False)
    # Optional bounds provided in the json config (lists of values for parameters)
    UPPER_BOUND = config.get("UPPER_BOUND", [])
    LOWER_BOUND = config.get("LOWER_BOUND", [])
    FIX_VALUE = config.get("FIX_VALUE", [])

    # --- Global variables override logic ---

    # RUN_DIR default handling
    if RUN_DIR is None:
        json_dir = os.path.dirname(INPUT_JSON_PATH)
        try:
            RUN_DIR = os.path.relpath(json_dir, os.getcwd())
        except (ValueError, TypeError):
            RUN_DIR = json_dir
        if RUN_DIR in ('.', './'):
            RUN_DIR = ''
        RUN_DIR = RUN_DIR.replace('\\', '/').lstrip('./\\')

    # Convert scalars to per-run lists
    if isinstance(X_CORR, (int, float)):
        X_CORR = [float(X_CORR)] * len(RUN_LIST)
    if isinstance(Y_CORR, (int, float)):
        Y_CORR = [float(Y_CORR)] * len(RUN_LIST)
    if isinstance(x_scale, (int, float)):
        x_scale = [float(x_scale)] * len(RUN_LIST)
    if isinstance(y_scale, (int, float)):
        y_scale = [float(y_scale)] * len(RUN_LIST)

    # Extend lists if necessary
    while len(X_CORR) < len(RUN_LIST):
        X_CORR.append(X_CORR[-1])
    while len(Y_CORR) < len(RUN_LIST):
        Y_CORR.append(Y_CORR[-1])
    while len(x_scale) < len(RUN_LIST):
        x_scale.append(x_scale[-1])
    while len(y_scale) < len(RUN_LIST):
        y_scale.append(y_scale[-1])

    if isinstance(CONCENTRATION_LIST, (int, float)):
        CONCENTRATION_LIST = [CONCENTRATION_LIST]

    # Traverse the CONCENTRATION_LIST to check for default or negative values
    for i, c in enumerate(CONCENTRATION_LIST):
        if c == 1.0:
            print(f"{Colors.WARNING}!Warning: Input concentration {i+1} is the default value (1.0). Please double-check if this is intended.{Colors.END}")
        if c < 0:
            print(f"{Colors.WARNING}!Warning: Input concentration {i+1} is non-positive. Replacing with 1.0.{Colors.END}")
            CONCENTRATION_LIST[i] = 1.0

    # Fill CONCENTRATION_LIST if too short or empty
    if len(CONCENTRATION_LIST) < len(RUN_LIST):
        while len(CONCENTRATION_LIST) < len(RUN_LIST):
            if CONCENTRATION_LIST:
                CONCENTRATION_LIST.append(CONCENTRATION_LIST[-1])
            else:
                CONCENTRATION_LIST.append(1.0)
        print(f'{Colors.WARNING}!Warning: CONCENTRATION_LIST length is less than RUN_LIST length. Extended by repeating last value.{Colors.END}')

    # Adjust BASELINE_ID to list
    if isinstance(BASELINE_ID, int):
        BASELINE_ID = [BASELINE_ID] * len(RUN_LIST)
    while len(BASELINE_ID) < len(RUN_LIST):
        BASELINE_ID.append(0)

    # Adjust IRF_ID to list
    if isinstance(IRF_ID, int):
        IRF_ID = [IRF_ID] * len(RUN_LIST)
    while len(IRF_ID) < len(RUN_LIST):
        IRF_ID.append(0)

    # Adjust XMIN/XMAX when X_CORR != 0
    if (XMIN or XMAX) and X_CORR and (not TO_FIND_TIME_ZERO):
        XMIN += X_CORR[0]
        XMAX += X_CORR[0]

    if config.get("SHOW_ALL_STATISTICS", False):
        SHOW_STATISTICS = True

    # Convert boolean SHOW_FITTING_INFO into int
    if isinstance(SHOW_FITTING_INFO, bool) and SHOW_FITTING_INFO:
        SHOW_FITTING_INFO = 1
    if isinstance(SHOW_FITTING_INFO, bool) and (not SHOW_FITTING_INFO):
        SHOW_FITTING_INFO = 0

    # Decide the plot mode:
    # 0 - Most spectrum / 1 - Kinetics with fitting
    plot_mode = 0
    type_title = ''
    _xlabel = XLABEL
    _ylabel = YLABEL

    # plot_mode = 0
    if TYPE == "TA":
        type_title = 'TA'
        YLABEL = '∆OD (a. u.)'
    if TYPE in ('UV', 'UV-Vis'):
        type_title = 'UV-Vis'
        YLABEL = 'Absorbance (A)'
        if NORMALIZATION_METHOD == 1:
            YLABEL = 'ε (L·mol$^{-1}$·cm$^{-1}$)'
    if TYPE in ("FL", "E00"):
        type_title = 'Fluorescence'
        YLABEL = 'Counts'
    if TYPE == "EX":
        type_title = 'Fluorescence excitation'
        YLABEL = 'Counts'
    if TYPE == "EM":
        type_title = 'Fluorescence emission'
        YLABEL = 'Counts'
    if TYPE in ('KinAbs', 'KA') and not DO_FITTING:
        type_title = 'Absorption kinetics'
        YLABEL = '∆OD (a. u.)'
    if TYPE == "TC-SPC" and not DO_FITTING:
        type_title = 'TC-SPC'
        YLABEL = 'Counts'

    # Normalize with the max of y
    if NORMALIZATION_METHOD == 2:
        if TYPE in ('E00', 'FL', 'EX', 'EM'):
            YLABEL = 'Normalized fluorescence intensity'
        elif TYPE in ('UV', 'UV-Vis'):
            YLABEL = 'Normalized absorbance'
        else:
            YLABEL = 'Normalized unit'

    # plot_mode = 1
    if TYPE in ('KinAbs', 'KA') and DO_FITTING:
        type_title = 'Absorption kinetics'
        XLABEL = f'Time ({infer_time_unit(x_scale[0])})'
        YLABEL = '∆OD (a. u.)'
        plot_mode = 1
    elif TYPE == "TC-SPC" and DO_FITTING:
        type_title = 'TC-SPC'
        XLABEL = f'Time ({infer_time_unit(x_scale[0])})'
        YLABEL = 'Counts'
        plot_mode = 1

    # Warning on type_title
    if type_title == '':
        print(f"{Colors.WARNING}!Warning: The input TYPE is not recognized.{Colors.END}")

    # Override x/ylabel with the custom values
    if _xlabel:
        XLABEL = _xlabel
    if _ylabel:
        YLABEL = _ylabel

    # Determine labels in the legend
    label_list = []
    if not AUTOLABEL:
        label_list = LABEL_LIST # Get labels
    else:
        for r in RUN_LIST:
            if isinstance(r, int):
                label_list.append(f'Run{r:02d}')
            else:
                label_list.append(str(r))

    # Fill the label_list with ""
    if len(label_list) < len(RUN_LIST):
        while len(label_list) < len(RUN_LIST):
            label_list.append("")

    # Add run_number
    if SHOW_RUN_NUMBER:
        for i, _ in enumerate(label_list):
            if isinstance(RUN_LIST[i], int):
                label_list[i] += f' (Run{RUN_LIST[i]:02d})'
            else:
                label_list[i] += f' (Run{RUN_LIST[i]})'

    # Global settings of figures
    pp.rcParams['font.family'] = 'sans-serif'
    pp.rcParams['font.sans-serif'] = ['Arial']
    pp.rcParams['font.size'] = FONTSIZE

    # Set up color list if input COLOR_LIST is a string (colormap name)
    if isinstance(COLOR_LIST, str):
        ncolors = max(1, len(RUN_LIST))
        cmap_name = config.get("COLOR_LIST", "viridis")
        try:
            cmap = pp.get_cmap(cmap_name)
        except ValueError:
            print(f'{Colors.WARNING}!Warning: Colormap name "{cmap_name}" is invalid. Using "viridis" instead.{Colors.END}')
            cmap = pp.get_cmap("viridis")
        if ncolors == 1:
            COLOR_LIST = [cmap(0.5)]
        else:
            COLOR_LIST = [cmap(i / (ncolors - 1)) for i in range(ncolors)]
        if config.get("COLORMAP_REVERSED", False):
            COLOR_LIST = COLOR_LIST[::-1]

    return config

print(f"{Colors.BLUE}{Colors.BOLD}SpecPlot v{VERSION} (Last update:{LATEST_UPDATE}){Colors.END}")

# Resolve the input from console, then initialize.
CMD_INPUT = None
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            CMD_INPUT = arg
            break
config = initialize(CMD_INPUT) # Load configuration and initialize

#endregion --- Initialize ---

#region --- Parsing functions ---------------

def find_run_file(num, directory = None, work_type = ''):
    '''Find RunXX_<TYPE>_YY_ZZ.txt file in the directory, 
    if work_type is '', find RunXX*.txt.'''
    if directory is None:
        directory = RUN_DIR
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

def extract_data(num, directory = None, return_meta = False, type_adjust = ''):
    '''This function extracts UV/Fluorescence/TA data from the specified file,
    also suitable to kinetic absorption data (t/ns, ∆OD/a.u.)'''
    if directory is None:
        directory = RUN_DIR
    file_path = find_run_file(num, directory, type_adjust)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Read the data lines
    data_pts = []
    meta_lines = []

    for line in lines:
        parts = line.strip().split()
        # Accept data lines with more than two columns, provided every part is numeric
        if len(parts) >= 2:
            try:
                # Parse possible FORTRAN-style 'D' or 'E+' exponent to 'E'
                cleaned = [p.upper().replace('D', 'E').replace('E+', 'E') for p in parts]
                nums = [float(p) for p in cleaned]
                x = nums[0]
                y = nums[1]
                data_pts.append([x, y])
                continue
            except ValueError:
                pass  # Not a numeric data line
        meta_lines.append(line.rstrip('\n'))

    if return_meta:
        return np.array(data_pts), meta_lines
    return np.array(data_pts)

def parse_order_to_unit(order, time_scale = None, levelup = 0):
    '''Parse order(int) to unit string
        Order: 0 - No unit; 1 - time unit; -1 - time unit inverse;
        -2 - secondary (μs^-1·a.u.^-1); 10 a.u.'''
    if time_scale is None:
        time_scale = x_scale[0]
    time_unit = infer_time_unit(time_scale * (10 ** ( - levelup * 3)))
    if order == 1:
        return f"{time_unit}"
    if order == 2:
        return f"{time_unit}²"
    if order == -1:
        return f"/{time_unit}"
    if order == -2:
        return f"/({time_unit}·a.u.)"
    if order == 10:
        return "a.u."
    return ""

#endregion --- Parsing functions ---------------

#region --- Processing functions ---------------

def reconvolve(func, params, t, irf_t, irf_y, tzero=0.0, offset=0.0, width=None):
    """Numerically convolve function 'func' with a specified experimental IRF 
    (irf_t, irf_y). Return a convolved discrete 1D-ndarray."""
    if width is None or width <= 0:
        width = IRF_WIDTH
    # 10% of width in number of points
    ##extension = 0.1 * width / (t[1] - t[0])
    # Make time grids
    t_grid = np.arange(t.min(), t.max(), t[1] - t[0])
    # Interpolation
    f_irf = interp1d(irf_t, irf_y, kind='cubic', fill_value="extrapolate")
    irf_resampled = f_irf(t_grid)
    t_irf_centered = t_grid - t_grid[np.argmax(irf_resampled)] + tzero
    irf_final = irf_resampled[(t_irf_centered >= -width) & (t_irf_centered <= width)]
    # Normalize to area = 1
    irf_final /= irf_final.sum()

    f_model = func(t_grid - tzero, *params)
    f_model[t_grid < tzero] = 0.0 # Set negative time to zero
    f_conv = np.convolve(f_model, irf_final, mode='same')
    f_conv = f_conv[(t_grid >= t.min()) & (t_grid <= t.max())]
    # f_conv = convolve(f_model, irf_resampled, mode='same')
    # f_conv = np.asarray(f_conv)
    return f_conv + offset

def find_curve_intersections(x1, y1, x2, y2, xtol=1e-8, abs_tol=1e-2):
    """
    Return a list of (x, y) intersection points between two 1D curves.

    The function interpolates both curves onto the union of x-values, looks
    for sign changes in the difference, and uses a root finder to refine the
    intersection positions.
    """
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)
    # Sort by x to be safe
    s1 = np.argsort(x1)
    x1, y1 = x1[s1], y1[s1]
    s2 = np.argsort(x2)
    x2, y2 = x2[s2], y2[s2]
    # Build a common grid (intersection of x1 and x2) from unique x-values
    x_common = np.unique(x1[np.any(np.isclose(x1[:, None], x2[None, :], atol=1e-2), axis=1)])
    if x_common.size == 0:
        return []

    y1i = np.interp(x_common, x1, y1)
    y2i = np.interp(x_common, x2, y2)
    diff = y1i - y2i
    y_max = max(np.max(np.abs(y1i)), np.max(np.abs(y2i)))

    _intersections = []

    # Exact zeros (within tolerance)
    zero_idx = np.where(np.isclose(diff, 0.0, atol=1e-12, rtol=0))[0]
    for idx in zero_idx:
        _xi = float(x_common[idx])
        _yi = float(y1i[idx])
        # The point is considered an intersection only if
        # the y-value is significant (yi/ymax is greater than abs_tol)
        if y_max != 0 and (abs(_yi) / y_max) >= abs_tol:
            # Then add the intersection
            _intersections.append((_xi, _yi))

    # Sign changes indicate a root in the interval
    signs = np.sign(diff)
    sign_change_idx = np.where(signs[:-1] * signs[1:] < 0)[0]

    for idx in sign_change_idx:
        a = float(x_common[idx])
        b = float(x_common[idx+1])
        f = lambda x: np.interp(x, x1, y1) - np.interp(x, x2, y2)
        try:
            root = float(scipy.optimize.brentq(f, a, b, xtol=xtol))
            yroot = float(np.interp(root, x1, y1))
            if y_max != 0 and (abs(yroot) / y_max) >= abs_tol:
                _intersections.append((root, yroot))
        except (ValueError, RuntimeError): # If brentq fails for any reason, skip this interval
            continue
    # Remove duplicates (within tolerance) and sort by x
    if not _intersections:
        return []
    # Sort and merge near-duplicates
    _intersections = sorted(_intersections, key=lambda t: t[0])
    merged = [_intersections[0]]
    for x,y in _intersections[1:]:
        if abs(x - merged[-1][0]) > 1e-8:
            merged.append((x,y))
    return merged

def preprocess_data(i, entry, pmode = None, norm = None, c_list = None):
    '''This function centralizes the preprocessing logic for a single RUN_LIST entry.
    Returns a numpy array with two columns (x, y) or an empty (0,2) array on error.'''
    global y_scale, X_CORR, XMAX, XMIN
    # Resolve defaults at call time from current globals (avoid binding at definition time)
    if pmode is None:
        pmode = plot_mode
    if norm is None:
        norm = NORMALIZATION_METHOD
    if c_list is None:
        c_list = CONCENTRATION_LIST
    # Convert string entry to int if possible
    if isinstance(entry, str):
        try:
            entry = int(entry)
        except ValueError:
            pass  # Keep as string for formula processing

    # integer run
    if isinstance(entry, int):
        data = extract_data(entry)
        # Data normalization (if UV-Vis/Fl/TA is plotted, namely pmode == 0)
        if pmode == 0:
            if norm == 1:
                # Try to get concentration for this run from CONCENTRATION_LIST
                y_scale[i] *= 1000.0 / (c_list[i] * LENGTH * Y_SCALE_FACTOR_IN_UNIT)
            if norm == 2:
                if data.size:
                    y_max_value = np.max(data[:,1])
                else:
                    y_max_value = 0
                if y_max_value != 0:
                    y_scale[i] = 1.0 / y_max_value
                else:
                    print(f"{Colors.WARNING}!Warning: Maximum value is zero, cannot normalize.{Colors.END}")
                    y_scale[i] = 1.0

        # Data pre-processing
        if data.size:
            # For kinetics plot, automatically find the time zero:
            if pmode == 1 and AUTO_FIND_TIME_ZERO:
                _tz = detect_time_zero(data[:,0], data[:,1], mode=0)
                print(f'Autofound time zero at {_tz}')
                # Shift time axis so detected tz becomes zero
                data[:,0] = data[:,0] - _tz
                # Shift XMIN and XMAX
                if (XMIN or XMAX):
                    XMIN -= _tz
                    XMAX -= _tz

            # Then apply scale, corrections, and x_cut
            data[:,0] = data[:,0] * x_scale[i] + X_CORR[i]
            data[:,1] = data[:,1] * y_scale[i] + Y_CORR[i]
            if X_CUT_AFTER is not None:
                data = data[data[:,0] <= X_CUT_AFTER]

        return data

    # subtraction entry: 'A-B' -> compute RunA - RunB on a common x-grid
    elif isinstance(entry, str) and re.match(r"^\s*\d+\s*-\s*\d+\s*$", entry):
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", entry)
        a = int(m.group(1))
        b = int(m.group(2))
        data_a = extract_data(a)
        data_b = extract_data(b)

        # Apply per-run normalization where possible
        local_y_scale_a = y_scale[i]
        local_y_scale_b = y_scale[i]
        if pmode == 0:
            if norm == 1:
                local_y_scale_a *= 1000.0 / (c_list[i] * LENGTH * Y_SCALE_FACTOR_IN_UNIT)
                local_y_scale_b *= 1000.0 / (c_list[i] * LENGTH * Y_SCALE_FACTOR_IN_UNIT)
            if norm == 2:
                ya_max = np.max(data_a[:,1]) if data_a.size else 0
                yb_max = np.max(data_b[:,1]) if data_b.size else 0
                if ya_max != 0:
                    local_y_scale_a = 1.0 / ya_max
                else:
                    print(f"{Colors.WARNING}!Warning: Maximum value is zero for Run{a:02d}; cannot normalize.{Colors.END}")
                if yb_max != 0:
                    local_y_scale_b = 1.0 / yb_max
                else:
                    print(f"{Colors.WARNING}!Warning: Maximum value is zero for Run{b:02d}; cannot normalize.{Colors.END}")

        # Apply scaling and offsets
        tza = 0
        tzb = 0
        if data_a.size:
            # For kinetics plot, automatically find the time zero:
            if pmode == 1 and AUTO_FIND_TIME_ZERO:
                tza = detect_time_zero(data_a[:,0], data_a[:,1], mode=0)
                print(f'Autofound time zero at {tza} for data_a')
                # Shift time axis so detected tz becomes zero
                data_a[:,0] = data_a[:,0] - tza

            data_a[:,0] = data_a[:,0] * x_scale[i] + X_CORR[i]
            data_a[:,1] = data_a[:,1] * local_y_scale_a + Y_CORR[i]

        if data_b.size:
            # For kinetics plot, automatically find the time zero:
            if pmode == 1 and AUTO_FIND_TIME_ZERO:
                tzb = detect_time_zero(data_b[:,0], data_b[:,1], mode=0)
                print(f'Autofound time zero at {tzb} for data_b')
                # Shift time axis so detected tz becomes zero
                data_b[:,0] = data_b[:,0] - tzb

            data_b[:,0] = data_b[:,0] * x_scale[i] + X_CORR[i]
            data_b[:,1] = data_b[:,1] * local_y_scale_b + Y_CORR[i]

        # Shift XMIN and XMAX
        if (XMIN or XMAX) and pmode == 1 and AUTO_FIND_TIME_ZERO:
            XMIN -= max([tza, tzb])
            XMAX -= max([tza, tzb])

        if X_CUT_AFTER is not None:
            if data_a.size:
                data_a = data_a[data_a[:,0] <= X_CUT_AFTER]
            if data_b.size:
                data_b = data_b[data_b[:,0] <= X_CUT_AFTER]

        # Interpolate both onto a common x grid (union of x-values)
        x_common = np.unique(np.concatenate([data_a[:,0] if data_a.size else np.array([]),
                                             data_b[:,0] if data_b.size else np.array([])]))
        if x_common.size == 0:
            print(f"{Colors.WARNING}!Warning: No data points found for '{entry}'.{Colors.END}")
            return np.empty((0,2))

        _zeros = np.zeros_like(x_common)
        y_a = np.interp(x_common, data_a[:,0], data_a[:,1]) if data_a.size else _zeros
        y_b = np.interp(x_common, data_b[:,0], data_b[:,1]) if data_b.size else _zeros
        y_diff = y_a - y_b
        return np.column_stack([x_common, y_diff])

    # addition / averaging entry: 'A+B(+C+...)' -> compute average of specified Runs
    elif isinstance(entry, str) and re.match(r"^\s*\d+(\s*\+\s*\d+)+\s*$", entry):
        nums = [int(n) for n in re.findall(r"\d+", entry)]
        data_list = []
        for _i in nums:
            d = extract_data(_i)
            data_list.append(d)

        # Determine per-run local y scales
        local_y_scales = []
        if pmode == 0 and norm == 1:
            _scale = 1000.0 / (c_list[i] * LENGTH * Y_SCALE_FACTOR_IN_UNIT)
            local_y_scales = [_scale] * len(nums)
        elif pmode == 0 and norm == 2:
            for d in data_list:
                maxv = np.max(d[:,1]) if d.size else 0
                if maxv != 0:
                    local_y_scales.append(1.0 / maxv)
                else:
                    print(f"{Colors.WARNING}!Warning: Maximum value is zero for one of Runs in '{entry}'; using 1.0 for normalization.{Colors.END}")
                    local_y_scales.append(1.0)
        else:
            local_y_scales = [y_scale[i]] * len(nums)

        # Apply scaling, offset and x-correction, and optionally cut
        xtz = []
        for k, d in enumerate(data_list):
            if d.size == 0:
                continue
            # For kinetics plot, automatically find the time zero:
            if pmode == 1 and AUTO_FIND_TIME_ZERO:
                _tz = detect_time_zero(d[:,0], d[:,1])
                print(f'Autofound time zero at {_tz} for data {k}')
                # Shift time axis so detected tz becomes zero
                d[:,0] = d[:,0] - _tz
                xtz.append(_tz)

            # Apply time_zero
            d[:,0] = d[:,0] * x_scale[i] + X_CORR[i]
            d[:,1] = d[:,1] * local_y_scales[k] + Y_CORR[i]
            if X_CUT_AFTER is not None:
                data_list[k] = d[d[:,0] <= X_CUT_AFTER]
            else:
                data_list[k] = d

        # Shift XMIN and XMAX
        if (XMIN or XMAX) and pmode == 1 and AUTO_FIND_TIME_ZERO:
            XMIN -= max(xtz)
            XMAX -= max(xtz)

        # Build common x grid (union) and compute average y
        valid_ds = [d for d in data_list if d.size]
        if not valid_ds:
            print(f"{Colors.WARNING}!Warning: No data points found for '{entry}'.{Colors.END}")
            return np.empty((0,2))

        x_common = np.unique(np.concatenate([d[:,0] for d in valid_ds]))
        if x_common.size == 0:
            print(f"{Colors.WARNING}!Warning: No data points found for '{entry}'.{Colors.END}")
            return np.empty((0,2))

        _ys = [np.interp(x_common, d[:,0], d[:,1]) for d in valid_ds]
        y_avg = np.mean(np.vstack(_ys), axis=0)
        return np.column_stack([x_common, y_avg])

    else:
        print(f"{Colors.ERROR}!Error: Unrecognized RUN_LIST entry '{entry}'.{Colors.END}")
        return np.empty((0,2))

def normalize_before_fitting(y, tzero_index, norm = None, tdata = None,
                             print_statistics = False, pts = 300, main_data = True, conc = 0.0):
    """Normalize data before fitting. Returns a tuple: (intensity_fit, baseline, scale, SNR).
    This function also calculates the statictics: baseline, SNR, etc.
    Returning SNR is a list: [Amplitude, Noise, SNR_value, amp/conc]."""
    if norm is None:
        norm = NORMALIZATION_METHOD
    _scale = 1.0
    _baseline = 0.0
    ymax = np.max(y)
    # SNR initialized to -1.0 to indicate not calculated
    _snr = [0.0, 0.0, -1.0, 0.0]

    # Calculate statistics:
    # baseline = mean of points before tzero_index (if any), else first point
    if tzero_index - pts > 0:
        _baseline = np.mean(y[:(tzero_index - 300)])
        _snr[1] = np.std(y[:(tzero_index - 300)])
        _snr[0] = ymax - _baseline
        if _snr[1] != 0:
            _snr[2] = _snr[0] / _snr[1]
        if conc != 0:
            _snr[3] = _snr[0] / conc * 1e3
    else:
        _baseline = y[0]

    # Print statistics
    if print_statistics:
        if main_data:
            print("Statistics of main data:")
        else:
            print("--------------")
            print("Statistics of IRF/baseline:")
        print(f"Maxvalue = {ymax:.5e}")
        print(f"Mean of baseline = {_baseline:.5e}")
        print(f"Signal amplitude = {_snr[0]:.5e}")
        print(f"Noise (std dev) = {_snr[1]:.5e}")
        if _snr[2] == -1.0:
            print(f"{Colors.WARNING}!Warning: Not enough points before time zero to calculate SNR.{Colors.END}")
        else:
            print(f"SNR = {_snr[2]:.4f}")
        if _snr[3] != 0.0:
            print(f"Amplitude/Concentration = {_snr[3]:.5e} a.u./M")

    tmp = y - _baseline

    if norm == 0:
        _intensity_fit = y.copy()
        return _intensity_fit, _baseline, _scale, _snr
    if norm == 1:
        _intensity_fit = tmp
    elif norm == 2:
        _scale = np.max(np.abs(tmp))
        if _scale == 0 or np.isnan(_scale):
            _scale = 1.0
        _intensity_fit = tmp / _scale
    elif norm == 3:
        _scale = y[tzero_index] if y[tzero_index] != 0 else 1.0
        _intensity_fit = y / _scale
    else:
        _intensity_fit = y.copy()
        _baseline = 0.0
        _scale = 1.0

    return _intensity_fit, _baseline, _scale, _snr

def generate_weight(tdata, start_index, weight_value=None, after=None):
    """Read weighting setting"""
    global WEIGHTING_FOR_FIT
    if weight_value is None:
        weight_value = WEIGHT
    if after is None:
        after = WEIGHTING_AFTER
    if after <= 0.0:
        WEIGHTING_FOR_FIT = False
    if not WEIGHTING_FOR_FIT:
        return None
    w = np.ones_like(tdata[start_index:])
    w[tdata[start_index:] > after] = weight_value
    return w

def generate_initial_guess(t, y, fit_func, zero_index=0,
                           _baseline=0.0, _scale=1.0, baseline_var=False, reconv=False):
    """Generate initial guess, bounds, and value fixing of fitting parameters"""
    global ENABLE_CUSTOM_INITIAL_GUESS, CUSTOM_INITIAL_GUESS
    global UPPER_BOUND, LOWER_BOUND, FIX_VALUE

    # Calculate basic statistics
    intensity_max = np.max(np.abs(y))
    intensity_max_idx = np.argmax(y)

    # Generate initial guess based on fitting function
    if fit_func == linear:
        _initial_guess = [
            y[zero_index], # y0: guess the offset is the first data point
            (y[-1] - y[zero_index]) / (t[-1] - t[zero_index]) # m: slope
        ]
    elif fit_func == exp_decay:
        _initial_guess = [
            y[-1], # y0: guess the offset is the last data point
            y[zero_index] - y[-1], # A: guess amplitude is the total drop
            (t[-1] - t[zero_index]) / 2 # tau: guess lifetime is half the time range
        ]
    elif fit_func == exp_2decay:
        _initial_guess = [
            y[-1], # y0
            (y[zero_index] - y[-1]) * 0.6, # A1
            (y[zero_index] - y[-1]) * 0.3, # A2
            (t[-1] - t[zero_index]) / 3, # tau1
            (t[-1] - t[zero_index]) / 1.5 # tau2
        ]
    elif fit_func == exp_3decay:
        _initial_guess = [
            y[-1], # y0
            (y[zero_index] - y[-1]) * 0.5, # A1
            (y[zero_index] - y[-1]) * 0.3, # A2
            (y[zero_index] - y[-1]) * 0.2, # A3
            (t[-1] - t[zero_index]) / 4, # tau1
            (t[-1] - t[zero_index]) / 2, # tau2
            (t[-1] - t[zero_index]) / 1 # tau3
        ]
    elif fit_func == secondary_decay:
        _initial_guess = [
            y[-1], # y0
            y[zero_index] - y[-1],  # k: small rate constant
            1 / intensity_max # r0: initial reciprocal concentration
        ]
    elif fit_func == primary_secondary_decay:
        _initial_guess = [
            y[-1], # y0
            (y[zero_index] - y[-1]) * 0.5,               # k1
            1.0,              # k2
            intensity_max                # const
        ]
    elif fit_func == primary_secondary_seperate:
        _initial_guess = [
            y[-1], # y0: guess the offset is the last data point
            0.5 * (y[zero_index] - y[-1]), # A: guess amplitude is the total drop
            2 / (t[-1] - t[zero_index]), # tau: guess lifetime is half the time range
            y[zero_index] - y[-1],
            1 / intensity_max
        ]
    elif fit_func == primary_rise:
        _initial_guess = [
            y[zero_index], # y0
            y[-1] - y[zero_index], # A
            100 / (t[-1] - t[zero_index]) # k
        ]
    elif fit_func == primary_rise_linear_offset:
        _initial_guess = [
            y[zero_index], # y0
            y[-1] - y[zero_index], # A
            100 / (t[-1] - t[zero_index]), # k
            0.0 # B
        ]
    elif fit_func == primary_rise_double_offset:
        _initial_guess = [
            y[zero_index], # y0
            y[-1] - y[zero_index], # A
            100 / (t[-1] - t[zero_index]), # k
            0.0, # B
            0.0  # t0
        ]
    elif fit_func == primary_rise_triple_offset:
        _initial_guess = [
            y[zero_index], # y0
            y[-1] - y[zero_index], # A
            100 / (t[-1] - t[zero_index]), # k
            0.0, # B
            1.0, # C
            0.0  # t0
        ]
    elif fit_func == double_primary_rise:
        _initial_guess = [
            y[zero_index], # y0
            y[-1] - y[zero_index], # A1
            100 / (t[-1] - t[zero_index]), # k1
            (y[-1] - y[zero_index]) * 0.5, # A2
            10 / (t[-1] - t[zero_index]) # k2
        ]
    elif fit_func == double_primary_rise_linear_offset:
        _initial_guess = [
            y[zero_index], # y0
            y[-1] - y[zero_index], # A1
            100 / (t[-1] - t[zero_index]), # k1
            (y[-1] - y[zero_index]) * 0.5, # A2
            10 / (t[-1] - t[zero_index]), # k2
            0.0 # B
        ]
    elif fit_func == primary_cascade:
        _initial_guess = [
            y[zero_index], # y0
            y[-1] - y[zero_index], # A
            100 / (t[-1] - t[zero_index]), # k1
            1 / (t[-1] - t[zero_index]) # k2
        ]
    elif fit_func == gaussian:
        _initial_guess = [
            0.0, # y0
            intensity_max, # A
            t[intensity_max_idx], # t0
            1.0 # sigma
        ]
    elif fit_func == primary_rise_with_gaussian:
        _initial_guess = [
            0.0, # y0
            intensity_max, # A
            100 / (t[-1] - t[zero_index]), # k
            intensity_max * 0.2, # B
            -0.001, # t0
            0.001, # sigma
            0.0, # C
        ]
    else:
        _initial_guess = []
        print(f"{Colors.WARNING}!Warning: Unknown fitting function. Cannot perform fitting.{Colors.END}")

    if BVAR: # Add C parameter for _baseline variation
        _initial_guess.append(0.0) # C
    if reconv: # Add tz parameter for reconvolution fitting
        _initial_guess.append(0.0) # tz

    # Apply custom initial guess if enabled
    if ENABLE_CUSTOM_INITIAL_GUESS and CUSTOM_INITIAL_GUESS:
        for i in range(min(len(CUSTOM_INITIAL_GUESS), len(_initial_guess))):
            if CUSTOM_INITIAL_GUESS[i] is not None:
                _initial_guess[i] = CUSTOM_INITIAL_GUESS[i]
        print("Using custom initial guess:", _initial_guess)
    else:
        print("Generated initial guess:", _initial_guess)

    # Special handling for primary_secondary_decay to promote convergence
    if fit_func == primary_secondary_decay:
        _initial_guess[3] = - np.log(_initial_guess[1] / _initial_guess[3] + 2*_initial_guess[2]) / _initial_guess[1]

    # Convert y0 to normalized units
        _initial_guess[0] = (_initial_guess[0] - _baseline) / _scale
    # --- Build default bounds ---
    # Use large finite bounds by default
    huge = 1e14
    tiny = 1e-12
    lower = np.full(len(_initial_guess), -huge, dtype=float)
    upper = np.full(len(_initial_guess), huge, dtype=float)

    # Set some reasonable per-model bounds (e.g. lifetimes/rates should be positive)
    if fit_func == exp_decay:
        # y0, A, tau
        lower = np.array([-huge, -huge, tiny])
        upper = np.array([huge, huge, huge])
    elif fit_func == exp_2decay:
        # y0, A1, A2, tau1, tau2
        lower = np.array([-huge, -huge, -huge, tiny, tiny])
        upper = np.array([huge, huge, huge, huge, huge])
    elif fit_func == exp_3decay:
        # y0, A1, A2, A3, tau1, tau2, tau3
        lower = np.array([-huge, -huge, -huge, -huge, tiny, tiny, tiny])
        upper = np.array([huge, huge, huge, huge, huge, huge, huge])
    elif fit_func == secondary_decay:
        # y0, k, r0  (k >= 0, r0 >= 0)
        lower = np.array([-huge, 0.0, 0.0])
        upper = np.array([huge, huge, huge])
    elif fit_func == primary_secondary_decay:
        # y0, k1, k2, const  (k1>=0, k2>=0)
        lower = np.array([-huge, 0.0, 0.0, -huge])
        upper = np.array([huge, huge, huge, huge])
    elif fit_func == primary_secondary_seperate:
        # y0, A, k1, k2, r0  (k1>=0, k2>=0, r0>=0)
        lower = np.array([-huge, -huge, 0.0, 0.0, 0.0])
        upper = np.array([huge, huge, huge, huge, huge])
    elif fit_func == primary_rise:
        # y0, A, k (k>=0)
        lower = np.array([-huge, -huge, 0.0])
        upper = np.array([huge, huge, huge])
    elif fit_func == primary_rise_linear_offset:
        # y0, A, k, B
        lower = np.array([-huge, -huge, 0.0, -huge])
        upper = np.array([huge, huge, huge, huge])
    elif fit_func == primary_rise_double_offset:
        # y0, A, k, B, t0
        lower = np.array([-huge, -huge, 0.0, -huge, -huge])
        upper = np.array([huge, huge, huge, huge, huge])
    elif fit_func == primary_rise_triple_offset:
        # y0, A, k, B, C, t0
        lower = np.array([-huge, -huge, 0.0, -huge, 0.0, -huge])
        upper = np.array([huge, huge, huge, huge, huge, huge])
    elif fit_func == double_primary_rise:
        # y0, A1, k1, A2, k2
        lower = np.array([-huge, -huge, 0.0, -huge, 0.0])
        upper = np.array([huge, huge, huge, huge, huge])
    elif fit_func == double_primary_rise_linear_offset:
        lower = np.array([-huge, -huge, 0.0, -huge, 0.0, -huge])
        upper = np.array([huge, huge, huge, huge, huge, huge])
    elif fit_func == primary_cascade:
        # y0, A, k1, k2
        lower = np.array([-huge, -huge, 0.0, 0.0])
        upper = np.array([huge, huge, huge, huge])
    elif fit_func == gaussian:
        # y0, A, t0, sigma (sigma > 0)
        lower = np.array([-huge, -huge, -huge, tiny])
        upper = np.array([huge, huge, huge, huge])
    elif fit_func == primary_rise_with_gaussian:
        # y0, A, k, B, t0, sigma, C
        lower = np.array([-huge, -huge, 0.0, -huge, -huge, tiny, -huge])
        upper = np.array([huge, huge, huge, huge, huge, huge, huge])

    if baseline_var:
        lower = np.append(lower, -huge) # C
        upper = np.append(upper, huge)  # C
    if reconv:
        lower = np.append(lower, -huge) # tz
        upper = np.append(upper, huge)  # tz

    # If JSON provided LOWER_BOUND/UPPER_BOUND, use them to override generated bounds
    if LOWER_BOUND:
        for i in range(min(len(LOWER_BOUND), len(lower))):
            if LOWER_BOUND[i] is not None:
                lower[i] = LOWER_BOUND[i]
        print("Using custom lower bound:", lower)
    if UPPER_BOUND:
        for i in range(min(len(UPPER_BOUND), len(upper))):
            if UPPER_BOUND[i] is not None:
                upper[i] = UPPER_BOUND[i]
        print("Using custom upper bound:", upper)

    # If FIX_VALUE is specified, override _initial_guess, the upper limit, and the lower limit,
    # # to fix the corresponding parameter when fitting.
    if FIX_VALUE:
        for i in range(min(len(_initial_guess), len(FIX_VALUE))):
            if FIX_VALUE[i] is not None:
                _initial_guess[i] = FIX_VALUE[i]
                # Add a tiny value, to ensure the upper bound > the lower bound
                upper[i] = FIX_VALUE[i] + tiny
                lower[i] = FIX_VALUE[i] - tiny
                print(f"Parameter {i+1} is fixed ({FIX_VALUE[i]}).")
    else:
        FIX_VALUE = [None] * len(_initial_guess)

    _bounds = (lower.tolist(), upper.tolist())

    return _initial_guess, _bounds

def evaluate_fitting(y_true, y_pred, k=0):
    """Evaluate the quality of fitting. Returns a tuple: 
    (residuals, R^2, adjusted R^2, chi_squared, RMSE, number of effective points)."""
    residual = y_true - y_pred
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    rmse = np.sqrt(ss_res / len(y_true))
    r_squared = 1 - (ss_res / ss_tot)
    adj_r_squared = r_squared
    if k > 0 and len(y_true) > k + 1:
        adj_r_squared = 1 - (1 - r_squared) * (len(y_true) - 1) / (len(y_true) - k - 1)
    chi_squared = 999999999
    try:
         # Add small value to avoid division by zero
        chi_squared = np.sum((residual)**2 / (y_true + 1e-12))
    except ZeroDivisionError:
        print(f"{Colors.WARNING}!Warning: 0-division error in chi-squared calculation.{Colors.END}")
    return residual, r_squared, adj_r_squared, chi_squared, rmse, len(y_true)

def generate_fitting_info(_popt, _res, fit_func, perr=None, baseline_var=False, reconv=False):
    """Print all fitted values, and evaluation of the model.
    The returning string will be passed to the textbox in the figure."""
    # Print fitted values
    print("All fitted values:", _popt)
    print("Number of points:", _res[-1])
    # Fill error values if not provided
    if perr is None:
        perr = [np.nan] * len(_popt)
    else:
        perr = list(perr)
    # Scale up time-related parameters if they are very small
    idx_to_scale_up = []
    for i, x in enumerate(_popt):
        if abs(x) < 1e-3:
            _popt[i] = x * 1e3
            try:
                perr[i] = perr[i] * 1e3
            except (ValueError, TypeError):
                perr[i] = perr[i]
            idx_to_scale_up.append(i)
    # For fixed values, set the error to NaN
    if FIX_VALUE:
        for i in range(min(len(_popt), len(FIX_VALUE))):
            if FIX_VALUE[i] is not None:
                perr[i] = np.nan

    info_func_name = ""
    info_list = [] # [Symbol_string, order, other_info_string]
    info_sequence = [] # Determine the order of displaying info
    info_text = ''
    # Determine info_list and info_sequence based on fit_func
    if fit_func == linear:
        info_list = [["$y_0$",0,  ""],
                        ["$k$",0,  ""]]
        info_sequence = [1, 0]
        info_func_name = "linear"
    elif fit_func == exp_decay:
        info_list = [["$y_0$",0,  ""],
                        ["$A$",0,  ""],
                        ["$τ$",1,  ""]]
        info_sequence = [2, 1, 0]
        info_func_name = "exp_decay"
    elif fit_func == exp_2decay:
        info_list = [["$y_0$",0,  ""],
                        ["$A_1$",0, f"({abs(_popt[1])*100/(abs(_popt[1])+abs(_popt[2])):.1f}%)"],
                        ["$A_2$",0, f"({abs(_popt[2])*100/(abs(_popt[1])+abs(_popt[2])):.1f}%)"],
                        ["$τ_1$",1,  ""],
                        ["$τ_2$",1,  ""]]
        info_sequence = [3, 4, 1, 2, 0]
        info_func_name = "exp_2decay"
    elif fit_func == exp_3decay:
        info_list = [["$y_0$",0,  ""],
                        ["$A_1$",0, f"({abs(_popt[1])*100/(abs(_popt[1])+abs(_popt[2])+abs(_popt[3])):.1f}%)"],
                        ["$A_2$",0, f"({abs(_popt[2])*100/(abs(_popt[1])+abs(_popt[2])+abs(_popt[3])):.1f}%)"],
                        ["$A_3$",0, f"({abs(_popt[3])*100/(abs(_popt[1])+abs(_popt[2])+abs(_popt[3])):.1f}%)"],
                        ["$τ_1$",1,  ""],
                        ["$τ_2$",1,  ""],
                        ["$τ_3$",1,  ""]]
        info_sequence = [4, 5, 6, 1, 2, 3, 0]
        info_func_name = "exp_3decay"
    elif fit_func == secondary_decay:
        info_list = [["$y_0$",0,  ""],
                        ["$k^'$",-2,  ""],
                        ["$1/A_0$",0,  ""]]
        info_sequence = [1, 2, 0]
        info_func_name = "secondary_decay"
    elif fit_func == primary_secondary_decay:
        info_list = [["$y_0$", 0,  ""],
                        ["$k_1$", -1,  ""],
                        ["$k_2^'$", -2,  ""],
                        ["const", 0,  ""]]
        info_sequence = [1, 2, 3, 0]
        info_func_name = "primary_secondary_decay"
    elif fit_func == primary_secondary_seperate:
        info_list = [["$y_0$", 0,  ""],
                        ["A", 0, ""],
                        ["$k_1$", -1,  ""],
                        ["$k_2^'$", -2,  ""],
                        ["$1/B_0$", 0,  ""]]
        info_sequence = [2, 3, 1, 4, 0]
        info_func_name = "primary_secondary_seperate"
    elif fit_func == primary_rise:
        info_list = [["$y_0$", 0,  ""],
                        ["$A$", 0,  ""],
                        ["$k$", -1,  ""]]
        info_sequence = [2, 1, 0]
        info_func_name = "primary_rise"
    elif fit_func == primary_rise_linear_offset:
        info_list = [["$y_0$", 0,  ""],
                        ["$A$", 0,  ""],
                        ["$k$", -1,  ""],
                        ["$B$", -1,  ""]]
        info_sequence = [2, 1, 0, 3]
        info_func_name = "primary_rise_linear_offset"
    elif fit_func == primary_rise_double_offset:
        info_list = [["$y_0$", 0,  ""],
                        ["$A$", 0,  ""],
                        ["$k$", -1,  ""],
                        ["$B$", -1,  ""],
                        ["$t_0$", 1,  ""]]
        info_sequence = [2, 1, 0, 3, 4]
        info_func_name = "primary_rise_double_offset"
    elif fit_func == primary_rise_triple_offset:
        info_list = [["$y_0$", 0,  ""],
                        ["$A$", 0,  ""],
                        ["$k$", -1,  ""],
                        ["$B$", -1,  ""],
                        ["$C$", 0,  ""],
                        ["$t_0$", 1,  ""]]
        info_sequence = [2, 1, 0, 3, 4, 5]
        info_func_name = "primary_rise_triple_offset"
    elif fit_func == double_primary_rise:
        info_list = [["$y_0$", 0,  ""],
                    ["$A_1$", 0,  ""],
                    ["$k_1$", -1,  ""],
                    ["$A_2$", 0,  ""],
                    ["$k_2$", -1,  ""]]
        info_sequence = [2, 4, 1, 3, 0]
        info_func_name = "double_primary_rise"
    elif fit_func == double_primary_rise_linear_offset:
        info_list = [["$y_0$", 0,  ""],
                    ["$A_1$", 0,  ""],
                    ["$k_1$", -1,  ""],
                    ["$A_2$", 0,  ""],
                    ["$k_2$", -1,  ""],
                    ["$B$", -1,  ""]]
        info_sequence = [2, 4, 1, 3, 0, 5]
        info_func_name = "double_primary_rise_linear_offset"
    elif fit_func == primary_cascade:
        info_list = [["$y_0$", 0,  ""],
                        ["$A$", 0,  ""],
                        ["$k_1$", -1,  ""],
                        ["$k_2$", -1,  ""]]
        info_sequence = [2, 3, 1, 0]
        info_func_name = "primary_cascade"
    elif fit_func == gaussian:
        info_list = [["$y_0$", 0,  ""],
                        ["$A$", 0,  ""],
                        ["$t_0$", 1,  ""],
                        ["$σ$", 0,  ""]]
        info_sequence = [1, 2, 3, 0]
        info_func_name = "gaussian"
    elif fit_func == primary_rise_with_gaussian:
        info_list = [["$y_0$", 0,  ""],
                        ["$A$", 0,  ""],
                        ["$k$", -1, ""],
                        ["$B$", 0,  ""],
                        ["$t_0$", 1,  ""],
                        ["$σ$", 1,  ""],
                        ["C", -1, ""]]
        info_sequence = [1, 2, 3, 4, 5, 0, 6]
        info_func_name = "primary_rise_with_gaussian"

    if baseline_var:
        info_list.append(["$C$", 0, ""])
        info_sequence.append(len(info_sequence))
    if reconv:
        info_list.append(["$t_z$", 1, ""])
        info_sequence.append(len(info_sequence))

    # Generate info lines
    info_lines = [f'Func: \"{info_func_name}\"']
    for i in info_sequence:
        l = 1 if i in idx_to_scale_up else 0
        unit = parse_order_to_unit(info_list[i][1], levelup = l)
        val = popt[i]
        err = perr[i] if i < len(perr) else np.nan
        if np.isnan(err):
            info_lines.append(f"{info_list[i][0]} = {val:.4f} {unit} {info_list[i][2]}".strip())
        else:
            info_lines.append(f"{info_list[i][0]} = {val:.4f} ± {err:.4f} {unit} {info_list[i][2]}".strip())
        if FIX_VALUE[i] is not None:
            info_lines[-1] = info_lines[-1] + " (Fixed)"
    # Add weighting info if applied
    if WEIGHTING_FOR_FIT:
        time_unit = infer_time_unit(x_scale[0])
        info_lines.append(f"Weight({WEIGHT}) when t > {WEIGHTING_AFTER} {time_unit}")
    info_lines.append(f"$R^2$ = {res[1]:.4f}")
    # Then print the part shown in textbox
    info_text = "\n".join(info_lines)
    if SHOW_FITTING_INFO >= 0:
        print(f"{Colors.BLUE}{info_text}{Colors.END}")

    # The following part is shown in console but not in textbox:
    info_lines_extra = [f"Adjusted $R^2$= {_res[2]:.4f}",
                        f"$χ^2$ = {_res[3]:.4e}",
                        f"RMSE = {_res[4]:.4e}",
                        f"Mean of residuals = {np.mean(_res[0]):.4e}",
                        f"Max of residuals = {np.abs(np.max(_res[0])):.4e}"]
    info_text_extra = "\n".join(info_lines_extra)
    # Supplement this part to textbox, if SHOW_FITTING_INFO >= 2:
    if SHOW_FITTING_INFO >= 2:
        info_text += info_text_extra
    if SHOW_FITTING_INFO >= 0:
        print(info_text_extra)

    return info_text

def generate_file_name(copy = None, mode = 0):
    '''Generate file name of saved figs, then copy it to the clipboard.'''
    if copy is None:
        copy = COPY_FILENAME_TO_CLIPBOARD
    # If FILENAME specified, use it directly.
    # Otherwise, generate a filename based on the current configuration.
    if FILENAME:
        filename = FILENAME
    # mode = 0: the same filename as the json file
    elif mode == 0:
        filename = INPUT_JSON_PATH.rsplit('.', 1)[0] # Remove file extension
        if '/' in filename: # Cut the current directory path, keep the last part as filename
            filename = filename.rsplit('/', 1)[-1]
    # mode = 1: a descriptive filename based on the current configuration
    elif mode == 1:
        filename = RUN_DIR + '_' + TYPE + '_'.join(label_list) + '_' + TITLE0.replace(' ', '_')
        if TYPE in ("KinAbs", "KA"):
            filename += f'_{WAVELENGTH_DETECTED}nm'
            if DO_FITTING:
                filename += f'_{FIT_FUNCTION}'
            if WEIGHTING_FOR_FIT:
                filename += f'_WAfter{WEIGHTING_AFTER}_W{WEIGHT}'

    # Substitute illegal filename characters with underscores
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # Replace multiple consecutive underscores with a single underscore
    filename = re.sub(r'_+', '_', filename)
    filename += ".png"

    # Copy the filename to clipboard
    if copy:
        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(filename)
        r.update()
        r.destroy()
    return filename

def show_help():
    """Print all input options in the interaction phase"""
    print("""All options:
[n]: Choose a new .json file, re-initiallize and plot
[r]: Replot using the current configuration
[s]: Save the current plot to the working directory
[v]: Show version
[h]: Show help
[Any key except the listed keys]: Exit the program
          """)
    return 0

#endregion --- End of processing functions ---------------

while LOOP:
    if plot_mode == -1:
        pass
#region ---- TA spectrum plotting ---------------------
    elif plot_mode == 0:
        # Create figure
        pp.figure(figsize=FIGSIZE, constrained_layout=True)

        # Set up the plotting range. REMIND: this may give a redundant blank picture
        if XMIN or XMAX:
            pp.xlim(left=XMIN, right=XMAX)
        if YMIN or YMAX:
            pp.ylim(bottom=YMIN, top=YMAX)

        # Extract data and build data arrays
        data_arrays = []
        for n, ent in enumerate(RUN_LIST):
            exp_data = preprocess_data(n, ent)
            data_arrays.append(exp_data)
            print(f"------Sucessfully read spectrum {n+1}: Run {ent}------")

        # Now plot each prepared data array
        for n, _data in enumerate(data_arrays):
            if _data.size == 0:
                continue
            # Plotting
            plot_kwargs = {'linewidth': LINEWIDTH}
            if label_list:
                plot_kwargs['label'] = label_list[n]
            if COLOR_LIST:
                plot_kwargs['color'] = COLOR_LIST[n % len(COLOR_LIST)]
            pp.plot(_data[:,0], _data[:,1], **plot_kwargs)

        # Finalize the plot
        pp.ylabel(YLABEL)
        pp.xlabel(XLABEL)

        # Constrain the plot by X/YMIN, X/YMAX
        if XMIN or XMAX:
            pp.xlim(left=XMIN, right=XMAX)
        else:
            pp.autoscale(enable = True, axis = 'x', tight = True)

        if YMIN or YMAX:
            pp.ylim(bottom=YMIN, top=YMAX)
        else:
            pp.autoscale(enable = True, axis = 'y')
        if ENABLE_LEGEND and label_list:
            pp.legend(loc = "upper right")

        # Add title. EXAMINE it carefully!!!
        TITLE = rf'{type_title} spectrum of {TITLE0}'
        pp.title(TITLE, fontsize = TITLE_FONTSIZE, fontweight = 'bold')

        # If there are at least two curves, compute their intersections and mark them
        if FIND_INTERSECTIONS and len(data_arrays) >= 2:
            print("Intersections:")
            X1, Y1 = data_arrays[0][:,0], data_arrays[0][:,1]
            X2, Y2 = data_arrays[1][:,0], data_arrays[1][:,1]
            intersections = find_curve_intersections(X1, Y1, X2, Y2, abs_tol=INTERSECTION_ABS_TOL)
            num_intersections = len(intersections)
            if intersections:
                xs = [p[0] for p in intersections]
                ys = [p[1] for p in intersections]
                if SHOW_INTERSECTIONS:
                    # Hollow circle markers: unfilled face, visible edge
                    pp.scatter(xs, ys, marker='o', facecolors='none',
                               edgecolors='k', s=80, linewidths=1.5, zorder=10, label='_nolegend_')
                print(f'{num_intersections} intersection(s) between first two curves:')
                for xi, yi in intersections:
                    print(f'  x = {xi:.6f}, y = {yi:.6e}')
            else:
                print('No intersections found between first two curves.')

#endregion ---- End of TA spectrum plotting ---------------------

#region --- Absorption kinetics plotting and fitting ---------------------

    elif plot_mode == 1:
        # You can change "if RUN_LIST" to a loop, if you have multiple curves to fit.
        for j, ent in enumerate(RUN_LIST):
            DISCARD_FITTING = False
            # Create subplots
            fig, (ax1, ax2) = pp.subplots(
                2, 1,
                figsize=FIGSIZE,
                constrained_layout=True,
                sharex=True,  # Both subplots will share the same x-axis
                gridspec_kw={'height_ratios': [3, 1]} # Main plot is 3x taller than residual plot
            )
            # Set up the plotting range
            if XMIN or XMAX:
                ax1.set_xlim(left=XMIN, right=XMAX)
                ax2.set_xlim(left=XMIN, right=XMAX)
            if YMIN or YMAX:
                ax1.set_ylim(bottom=YMIN, top=YMAX)
                ax2.set_ylim(bottom=YMIN, top=YMAX)

            fig.subplots_adjust(hspace=0.05) # Remove space between plots
            # Ready for data (!!Only pick the first data!!)
            exp_data = preprocess_data(j, ent)
            print(f"------Sucessfully read file {j+1}: Run {ent}------")
            time_data = exp_data[:,0]
            intensity_data = exp_data[:,1]
            # Keep originals for plotting/metrics
            intensity_orig = intensity_data.copy()
            # This finds the index of the time value closest to zero
            time_zero_index = np.argmin(np.abs(time_data))
            # Normalization if needed
            intensity_fit, baseline, scale, snr = normalize_before_fitting(intensity_data,
                                            time_zero_index, conc=CONCENTRATION_LIST[j],
                                        print_statistics=SHOW_STATISTICS, tdata=time_data)
            # Pick the fitting function
            try:
                fit_function = FIT_MODELS[FIT_FUNCTION.lower()]
            except KeyError:
                print(f"{Colors.WARNING}!Warning: Fit function '{FIT_FUNCTION}' not recognized. Defaulting to 'exp_decay'.{Colors.END}")
                fit_function = exp_decay

            BVAR = False
            BCOEFF = 0.0
            # If BASELINE is specified:
            if BASELINE_ID[j] > 0:
                # Read baseline data and normalize the integral to 1
                bdata = preprocess_data(j, BASELINE_ID[j])
                bdata_normalized, _,_,_ = normalize_before_fitting(bdata[:,1], 800,
                                        norm=1, main_data=False, tdata=bdata[:,0],
                        print_statistics=config.get("SHOW_ALL_STATISTICS", False))

                baseline_interp = interp1d(bdata[:,0], bdata_normalized, kind='cubic',
                                           fill_value='extrapolate', assume_sorted=False)
                BVAR = True
                print(f"Baseline data loaded for Run{RUN_LIST[j]}.")
            else:
                baseline_interp = interp1d(time_data, np.zeros_like(time_data), kind='cubic',
                                           fill_value='extrapolate',assume_sorted=False)

            # If IRF is specified, use convolution fitting functions
            if IRF_ID[j] > 0:
                # Read IRF data and normalize the integral to 1
                irf_data = preprocess_data(j, IRF_ID[j])
                irf_time_data = irf_data[:,0]
                irf_intensity_data = irf_data[:,1]
                irf_intensity_fit, _,_,_ = normalize_before_fitting(irf_intensity_data, 800,
                                            norm=1, main_data=False, tdata=irf_time_data,
                                print_statistics=config.get("SHOW_ALL_STATISTICS", False))

                print(f"IRF data loaded for Run{RUN_LIST[j]}.")
            else:
                RECONV_TZ = False

            # --- Perform the Fit --------------------------------------------------------
            while RETRY_FIT_IF_FAIL:
                try:
                    print(f"------Starting fitting for Run{RUN_LIST[j]}------")
                    print("Fit function:", fit_function.__name__,
                          "(reconvolved)" if IRF_ID[j] > 0 else "")
                    # Generate initial guess and bounds using the dedicated function
                    initial_guess, bounds = generate_initial_guess(time_data, intensity_data,
                            fit_function, time_zero_index, _baseline=baseline, _scale=scale,
                                                        baseline_var=BVAR, reconv=RECONV_TZ)
                    # Generate weights
                    weights = generate_weight(time_data, time_zero_index)

                    #region --- Use curve_fit and generate a smooth curve for the fit ---

                    # Generate fine time points for smooth fit curve
                    tnum = len(time_data[time_zero_index:])
                    t_fit = np.linspace(time_data[time_zero_index], time_data.max(), tnum)
                    dpwr = DISCARD_POINTS_WHEN_RECONV
                    orig_fit_function = fit_function

                    # Applying variable C into fit_function if baseline data is specified
                    if BVAR:
                        def combined_model(t, *params):
                            '''Construct function: f(t) + C*b(t)'''
                            core_params = params[:-1]
                            coeff = params[-1]
                            return orig_fit_function(t, *core_params) + coeff * baseline_interp(t) # pylint: disable=cell-var-from-loop
                        fit_function = combined_model

                    # Applying reconvolution fit if IRF is specified
                    if IRF_ID[j] > 0:
                        rfunc = lambda t, *params: reconvolve(
                                                func=fit_function, # pylint: disable=cell-var-from-loop
                                                params=params[:-1] if RECONV_TZ else params,
                                                t=t,
                                                irf_t=irf_time_data, # pylint: disable=cell-var-from-loop
                                                irf_y=irf_intensity_fit, # pylint: disable=cell-var-from-loop
                                                tzero=(params[-1] if RECONV_TZ else 0.0)
                                            )
                        #ax1.plot(irf_time_data, irf_intensity_fit, color="#00aa00")
                        popt, pcov = curve_fit(# pylint: disable=unbalanced-tuple-unpacking
                                            rfunc,
                                            time_data,
                                            intensity_fit,
                                            p0=initial_guess,
                                            bounds=bounds,
                                            sigma=weights if WEIGHTING_FOR_FIT else None,
                                            method='trf',
                                            maxfev=5000,
                                            )
                        tnum = len(time_data[0:])
                        BCOEFF = popt[-2] if RECONV_TZ and BVAR else 0.0
                        tz = popt[-1] if RECONV_TZ else 0.0
                        t_fit = np.linspace(time_data[0], time_data.max(), tnum)
                        y_fit = rfunc(t_fit, *popt)[:-dpwr]
                        Y_FIT_NORECONV = fit_function(t_fit[0:], *popt[:-1 if RECONV_TZ else len(popt)])[:-dpwr] * scale + baseline
                        Y_FIT_NORECONV[:np.argmin(np.abs(time_data - tz))] = 0.0
                        res = evaluate_fitting(intensity_orig[0:],
                                               rfunc(time_data[0:], *popt)*scale+baseline,
                                               k=len(popt))
                        residuals = res[0][:-dpwr]
                        intensity_orig = intensity_orig[:-dpwr]
                        time_data = time_data[:-dpwr]
                        t_fit = t_fit[:-dpwr]
                        time_zero_index = 0
                    else:
                        popt, pcov = curve_fit(# pylint: disable=unbalanced-tuple-unpacking
                                            fit_function,
                                            time_data[time_zero_index:],
                                            intensity_fit[time_zero_index:],
                                            p0=initial_guess,
                                            bounds=bounds,
                                            sigma=weights if WEIGHTING_FOR_FIT else None,
                                            method='trf',
                                            maxfev=5000
                                            )
                        y_fit = fit_function(t_fit, *popt)
                        Y_FIT_NORECONV = None
                        res = evaluate_fitting(intensity_orig[time_zero_index:],
                                               fit_function(t_fit, *popt)*scale + baseline,
                                               k=len(popt))
                        residuals = res[0]

                    BCOEFF = popt[-1] if (not BCOEFF) and BVAR else BCOEFF
                    # y_fit currently is in normalized units -> convert to original units
                    y_fit_plot = y_fit * scale + baseline
                    # Calculate errors:
                    Perr = np.array([np.nan] * len(popt))
                    if ERROR_ANALYSIS and (pcov is not None):
                        try:
                            with np.errstate(invalid='ignore'):
                                Perr = np.sqrt(np.diag(pcov))
                        # If covariance cannot be estimated, set Perr to NaN
                            if not np.isfinite(Perr).all():
                                Perr = np.array([np.nan] * len(popt))
                        except (ValueError, RuntimeError, FloatingPointError):
                            pass

                    # Print the fitted results
                    print(f"------Fit successful for Run{RUN_LIST[j]}------")
                    FIT_INFO_TEXT = generate_fitting_info(popt, res, fit_func=orig_fit_function,
                                                perr=Perr, reconv=RECONV_TZ, baseline_var=BVAR)
                    #endregion

                    #region --- Check R² value for fit quality ---
                    if res[1] < 0.2:
                        print(f"{Colors.WARNING}!Warning: Low R² value ({res[1]:.4f}) indicates poor fit quality.{Colors.END}")
                        if RETRY_FIT_IF_FAIL:
                            retry = input("Retry with adjusted initial guesses? y/n: ").lower()
                            if retry != 'n':
                                # Adjust initial guesses slightly for retry
                                print("Enter new initial guesses. eg: xx yy zz")
                                print(f"Current initial guess: {initial_guess}")
                                inp_initial_guess = input().strip()
                                CUSTOM_INITIAL_GUESS = [float(x) for x in inp_initial_guess.split()]
                                ENABLE_CUSTOM_INITIAL_GUESS = True
                                continue
                    #endregion

                    if RETRY_FIT_IF_FAIL:
                        break

                except RuntimeError:
                    print(f"{Colors.ERROR}!Error: Could not find a fit for Run {RUN_LIST[j]}.{Colors.END}")
                    if RETRY_FIT_IF_FAIL:
                        retry = input("Retry with adjusted initial guesses? y/n: ").lower()
                        if retry != 'y':
                            DISCARD_FITTING = True
                            break
                        # Adjust initial guesses slightly for retry
                        print("Enter new initial guesses. eg: xx yy zz")
                        print(f"Current initial guess: {initial_guess}")
                        input_initial_guess = input().strip()
                        CUSTOM_INITIAL_GUESS = [float(x) for x in input_initial_guess.split()]
                        ENABLE_CUSTOM_INITIAL_GUESS = True

            #region --- Finalize the plot -----------------------------

            if not DISCARD_FITTING:
                # Plot the raw data points on the top subplot (ax1)
                ax1.plot(time_data, intensity_orig,
                        label='Data (Original)' if BVAR else 'Data',
                        color= "#aaaaaa" if BVAR else "#6babd8")

                # If variant baseline is applied, also plot the
                # baseline-subtracted data for better visualization
                if BVAR:
                    ax1.plot(time_data, intensity_orig - BCOEFF*baseline_interp(time_data),
                        label='Data (Subtracted)',
                        color="#6babd8")

                # Plot the fitted curve on ax1
                ax1.plot(t_fit, y_fit_plot - BCOEFF * baseline_interp(t_fit),
                         color='#0000aa', linestyle='-',
                        label='Fit')

                if Y_FIT_NORECONV is not None:
                    ax1.plot(t_fit[0:], Y_FIT_NORECONV[0:], color='#aa0000', linestyle='--',
                            label='Fit (no reconv.)')

                # Add fitting results into a text box on the diagram ---
                if SHOW_FITTING_INFO >= 1:
                    v = 'bottom'
                    y_off = 0.07
                    if 'top' in POSITION_OF_FITTING_INFO:
                        v = 'top'
                        y_off = 0.95
                    h = 'left' if 'left' in POSITION_OF_FITTING_INFO else 'right'
                    ax1.text(0.97, y_off, FIT_INFO_TEXT,
                            transform=ax1.transAxes,
                            fontsize=12,
                            verticalalignment=v,
                            horizontalalignment=h,
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

                # Plot residuals on the bottom subplot (ax2)
                ax2.scatter(time_data[time_zero_index:], residuals,
                            facecolors='none', edgecolors='gray', s=20)
                # Add a zero line
                ax2.axhline(0, color='black', linestyle='--', linewidth=1)

                # Set axis limits if specified
                if XMIN or XMAX:
                    if X_CUT_AFTER is not None and (X_CUT_AFTER < XMAX or XMAX == 0):
                        XMAX = X_CUT_AFTER
                    ax1.set_xlim(XMIN, XMAX)
                    ax2.set_xlim(XMIN, XMAX)
                if YMIN or YMAX:
                    ax1.set_ylim(YMIN, YMAX)

                if YMAX_OF_RESIDUAL:
                    ax2.set_ylim(-YMAX_OF_RESIDUAL, YMAX_OF_RESIDUAL)
                else:
                    ax2.set_ylim(-np.max(np.abs(residuals)) * 1.1, np.max(np.abs(residuals)) * 1.1)

                # Set title
                FINAL_TITLE = rf'{type_title} of {TITLE0}'
                if WAVELENGTH_DETECTED:
                    FINAL_TITLE += rf' at {WAVELENGTH_DETECTED} nm'
                else:
                    print(f"{Colors.WARNING}!Warning: WAVELENGTH_DETECTED is not set properly.{Colors.END}")

                if ENABLE_LEGEND:
                    ax1.legend(loc="best")

                ax1.set_ylabel(YLABEL)
                ax1.set_title(FINAL_TITLE,
                            fontsize = TITLE_FONTSIZE, fontweight='bold')

                # Finalize the bottom subplot (ax2)
                ax2.set_xlabel(XLABEL)
                ax2.set_ylabel('Residuals')
            else:
                print("Fitting is skipped. Nothing is plotted!")

            #endregion --- Finalize the plot -----------------------------

    # Generate the filename and copy it to the clipboard.
    Filename = generate_file_name()
    if plot_mode != -1:
        pp.show()
    #endregion --- End of Absorption kinetics plotting and fitting ---------------------

#region --- Decide the next action --------------

    INTERACT = True
    while INTERACT:
        ans = input('Please input the next action, type "h" to display all options: ').lower()
        # [n]: Choose a new .json file, re-initiallize and plot.
        if ans == 'n':
            pp.close('all')
            print('---------------------------------')
            print('Choosing a new file to plot:')
            config = initialize()
            LOOP = True
            INTERACT = False
        # [r]: Replot with the current configuration.
        elif ans == 'r' and plot_mode != -1:
            print('Reload the current configuration.')
            config = initialize(INPUT_JSON_PATH)
            LOOP = True
            INTERACT = False
        # [s]: Save the current plot to the working directory.
        elif ans == 's':
            # bug here: saved picture is blank
            pp.savefig(f'{os.getcwd()}\\{Filename}.png', bbox_inches="tight", pad_inches=0.02)
            print(f'{os.getcwd()}\\{Filename}.png is saved.')
        # [v]: Show version
        elif ans == 'v':
            print(f"Current version: {VERSION}")
            print(f"Latest update: {LATEST_UPDATE}")
        # [h]: Show help
        elif ans == 'h':
            show_help()
        # Empty input: Just loop
        elif ans == '':
            pass
        # Any key except the listed keys: Exit the program
        else:
            INTERACT = False
            LOOP = False
            break

#endregion --- Decide the next action --------------

pp.close('all')
print('Exited')
sys.exit()
