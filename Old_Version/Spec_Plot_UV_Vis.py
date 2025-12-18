import os
import re
import math
import numpy as np
from matplotlib import pyplot as pp

ROUTE_LIST = [r"ZZL-224"] # r"ZZL-XX\ZZL-YY", in the format of a list.
CONCENTRATION_LIST = [0.040] # Concentration in mmol/L
LABEL_LIST = ['ZZL-136'] # Labels for the legend
BLANK = [] # Blank data, if any, can be added here
COLOR_LIST = [] # Colors for the plots, can be customized
AUTOLABEL = False
TITLE0 = 'ZZL-136'

TYPE = 'UV' # Options: UV, EM, EX, E00, can be downcased

LENGTH = 1.0 # Length of the cuvette in cm, usually 1.0 cm
X_CORR = 0.0
NORMALIZATION_METHOD = 0 # 0: Manual scale, 1: Normalize to ε(L·mol^-1·cm^-1) then scale manually, 
# 2: Normalize the maximum to 1
SCALE_FACTOR = 1.0 # Scale factor for the data when NORMALIZATION_METHOD is 0
SCALE_FACTOR_FLUORE = 1e-4 # Scale factor for fluorescence data

pp.figure(figsize=(8, 6))
pp.rcParams['font.family'] = 'sans-serif'
pp.rcParams['font.sans-serif'] = ['Arial']
pp.rcParams['font.size'] = 15
type_title = '' # Needless to set, it will be set automatically based on TYPE

def find_highest_number_file(directory, pattern = 'EmScan'):
    max_x = -1
    max_file = None
    pattern = re.compile(rf'^{pattern}(\d)\.txt$', re.IGNORECASE)
    for fname in os.listdir(directory):
        match = pattern.match(fname)
        if match:
            x = int(match.group(1))
            if x > max_x:
                max_x = x
                max_file = fname
    return max_file

def get_data_list(type = TYPE, route_list = ROUTE_LIST):
    data_list = []
    for route in route_list:
        target_dir = rf'{os.getcwd()}' + '\\' + route
        if type.upper() == 'UV':
            data_list.append(target_dir + r'\UV.TXT')
        elif type.upper() == 'EX':
            data_list.append(target_dir + '\\' + find_highest_number_file(target_dir, 'ExScan'))
        elif type.upper() == 'EM':
            data_list.append(target_dir + '\\' + find_highest_number_file(target_dir, 'EmScan'))
        elif type.upper() == 'E00':
            data_list.append(target_dir + '\\' + find_highest_number_file(target_dir, 'ExScan'))
            data_list.append(target_dir + '\\' + find_highest_number_file(target_dir, 'EmScan'))
        else:
            raise ValueError("Unsupported type. Use 'UV', 'EX', 'EM', or 'E00'.")
    return data_list

def extract_uv_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the start of the data section
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Data Points'):
            data_start = i + 2  # Skip the header line "nm\tAbs"
            break

    if data_start is None:
        raise ValueError("Data section not found in file.")

    # Read the data lines
    data = []
    for line in lines[data_start:]:
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

# Plotting preprocessing ============================

data_lists = get_data_list()
ylabel = ""
label_list = []
if not AUTOLABEL:
    label_list = LABEL_LIST # Get labels
else:
    label_list = [r.split('\\')[-1] for r in ROUTE_LIST]
name_reserved = ''

# UV-Vis spectrum plotting ============================

if TYPE == "UV":
    type_title = 'UV-Vis'
    ylabel = 'Absorbance (A)'
    data_arrays = [extract_uv_data(data) for data in data_lists]
    for i, data in enumerate(data_arrays):
        # Data pre-processing ============================
        data[:,0] = data[:,0] + X_CORR
        if NORMALIZATION_METHOD == 0:
            # Manual scale
            data[:,1] = data[:,1] * SCALE_FACTOR
        elif NORMALIZATION_METHOD == 1:
            # Normalize to ε(L·mol^-1·cm^-1) and scale
            data[:,1] = data[:,1] / CONCENTRATION_LIST[i] * 1000 / LENGTH * SCALE_FACTOR
            ylabel = "Molar Absorptivity (ε, L·mol$^{-1}$·cm$^{-1}$)"
        elif NORMALIZATION_METHOD == 2:
            # Normalize the maximum to 1
            max_value = np.max(data[:,1])
            if max_value != 0:
                data[:,1] = data[:,1] / max_value
            ylabel = "Normalized Absorbance (A)"

        if not True: # If you want to use a blank data set, set this to True
            pp.plot(data[:,0], data[:,1], label = label_list[i], color = '#b0b0b0')
        else:
            pp.plot(data[:,0], data[:,1], label = label_list[i])

# EX or EM spectrum plotting ============================

if TYPE == "EX" or TYPE == "EM":
    
    if TYPE == "EX":
        type_title = 'Fluorescence excitation'
    elif TYPE == "EM":
        type_title = 'Fluorescence emission'
    
    expo = - math.log10(SCALE_FACTOR_FLUORE)
    ylabel = 'Counts'
    if expo != 0:
        ylabel = f'Counts (x10$^{str(int(expo))}$)'

    data_arrays = [extract_fluore_data(data) for data in data_lists]
    for i, data in enumerate(data_arrays):
        # Data pre-processing ============================
        data[:,0] = data[:,0] + X_CORR
        if NORMALIZATION_METHOD == 0:
            # Manual scale
            data[:,1] = data[:,1] * SCALE_FACTOR_FLUORE
        elif NORMALIZATION_METHOD == 1:
            # Normalize to ε(L·mol^-1·cm^-1) and scale
            data[:,1] = data[:,1] / CONCENTRATION_LIST[i] * 1000 / LENGTH * SCALE_FACTOR_FLUORE
            ylabel = "Molar Absorptivity (ε, L·mol$^{-1}$·cm$^{-1}$)"
        elif NORMALIZATION_METHOD == 2:
            # Normalize the maximum to 1
            max_value = np.max(data[:,1])
            if max_value != 0:
                data[:,1] = data[:,1] / max_value
            ylabel = "Normalized Fluorescence Intensity"

        if not True: # If you want to use a blank data set, set this to True
            pp.plot(data[:,0], data[:,1], label = label_list[i], color = '#b0b0b0')
        else:
            pp.plot(data[:,0], data[:,1], label = label_list[i])

# E00 plotting ============================

if TYPE == "E00":
    type_title = 'Fluorescence'
    ylabel = "Normalized Fluorescence Intensity"
    name_reserved = '_' + ROUTE_LIST[0].split('\\')[-1]

    data_arrays = [extract_fluore_data(data) for data in data_lists]
    if len(data_arrays) > 2:
        print("Warning: More than two data sets found for E00 type.")
        if input('Continue? y/n: ').lower() != 'y':
            print("Plotting aborted.")
            exit()

    for i, data in enumerate(data_arrays):
        # Data pre-processing ============================
        data[:,0] = data[:,0] + X_CORR
        # Normalize the maximum to 1
        max_value = np.max(data[:,1])
        if max_value != 0:
            data[:,1] = data[:,1] / max_value

        label_list = ['Absorp.','Fluores.']
        if not True: # If you want to use a blank data set, set this to True
            pp.plot(data[:,0], data[:,1], label = label_list[i], color = '#b0b0b0')
        else:
            pp.plot(data[:,0], data[:,1], label = label_list[i])

# Finalizing the plot ============================

pp.ylabel(ylabel)
pp.xlabel(fr'Wavelength (nm)')

pp.autoscale(enable = True, axis = 'x', tight = True)
pp.autoscale(enable = True, axis = 'y')
pp.legend(loc = "best")

TITLE = rf'{type_title} spectrum of {TITLE0}' # EXAMINE it carefully!!!

pp.title(TITLE, fontsize = 18, fontweight = 'bold')

if input('Save Figures? y/n: ') == 'y':
    filename = TYPE + '_' + '_'.join(label_list) + name_reserved
    pp.savefig(f'{os.getcwd()}\\{filename}.png')
    print(f'{os.getcwd()}\\{filename}.png is saved.')

pp.show()
pp.close()
