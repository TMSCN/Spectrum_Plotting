# SpecPlot - Spectral Data Processing and Plotting Tool
> **Note**: This program is in early development (version 0.2). Some features may be unstable, and the configuration file format may be adjusted in future versions. It is recommended to retest your configuration after each code update.

## Project Description

SpecPlot is a Python-based tool for visualizing and analyzing spectral data, primarily designed for common laboratory spectroscopic data, currently including:

- **Steady-state spectra**: UV-Vis absorption spectra, fluorescence spectra
- **Transient spectra**: Transient absorption (TA) spectra
- **Kinetic data**: Absorption/emission kinetics, time-correlated single photon counting (TC-SPC)

The program reads a user-provided JSON configuration file to define data processing workflows and plotting styles. It supports multi-dataset overlay, data operations (e.g., difference spectra, averaging), curve fitting (including various exponential decay, rise, and second-order reaction models), baseline correction, and instrument response function (IRF) convolution fitting. Fitting results are output to the terminal and displayed as text boxes on the figure, which can be saved as PNG images.

## Technologies Used

- **Language**: Python 3.6+
- **Core dependencies**:
  - `numpy` ~= 1.21
  - `scipy` ~= 1.7
  - `matplotlib` ~= 3.4
  - `tkinter` (usually included with Python)
- **Optional/indirect dependencies**: No special requirements

## Environment Requirements

### System Requirements
- Windows / Linux / macOS (with tkinter GUI support)
- Monitor resolution of at least 1280×720 recommended for proper display of plot windows

### Python Environment
- Python 3.6 or higher
- It is recommended to use a virtual environment (e.g., `venv` or `conda`) to isolate project dependencies

### Installing Dependencies
Run the following command in the project directory:
```bash
pip install numpy scipy matplotlib
```
(If you wish to use system fonts or additional plotting features, you may install packages like `freetype`, but this is not required.)

## Quick Start

### 1. Prepare Data Files
Data files should be text files (`.txt`) with two numerical values per line (x and y), separated by spaces or tabs. Non-data lines are allowed.
Data files must start with `RunXX`, where XX is a two-digit number; numbers below 10 should be represented as "0X". For example:
 `Run01_UV_SampleA.txt`

### 2. Write a JSON Configuration File
In the same directory, create a JSON file (e.g., `config.json`) defining the data to be processed, parameters, and plotting styles.
A simple example:
```json
{
    "TYPE": "UV",
    "RUN_LIST": [1, 2, 3],
    "CONCENTRATION": [0.95, 0.50, 1.23],
    "LABEL_LIST": ["Sample A", "Sample B", "Sample C"],
    "TITLE0": "My Samples",
    "XMIN": 300,
    "XMAX": 800,
    "NORMALIZATION_METHOD": 1
}
```
For detailed configuration options, refer to the "Keyword Directory" section below.

### 3. Run the Program
In the terminal, execute:
```bash
python spec_plot.py
```
The program will open a file dialog; select your JSON configuration file and the plot will be generated automatically.
You can also specify the input file directly from the command line:
```bash
python spec_plot.py your_file_path/config.json
```
After input, the terminal will display some plotting information, and the generated figure will appear in a plot window. You can interact with the plot (zoom, pan, etc.) or save the current figure. The filename for saving will be automatically copied to your clipboard.

### 4. Interactive Commands
After closing the plot window, you will return to the terminal and can enter the following commands:
- `n`: Select a new JSON file and replot
- `r`: Reload the current JSON file and replot
- `v`: Show version information
- `h`: Show help
- Any other key: Exit the program
Before entering commands, you can modify the currently loaded JSON file and save it to adjust the plot, or prepare a new JSON file.

## Keyword Directory

The following keywords (case-sensitive) can be used in the JSON configuration file to define data processing and plotting parameters. All parameters have default values; if not specified, the built-in defaults will be used.

### Basic Settings
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `TYPE` | string | `""` | Data type: `"UV"` (UV-Vis absorption), `"TA"` (transient absorption), `"FL"/"E00"` (fluorescence), `"EX"` (excitation spectrum), `"EM"` (emission spectrum), `"KinAbs"/"KA"` (absorption kinetics), `"TC-SPC"` (time-correlated single photon counting). Affects axis labels and default behavior. |
| `RUN_LIST` | array | `[]` | **Required**. List of run numbers to process, e.g., `[1,2,3]`. Also supports formula strings like `"1-2"` (difference spectrum) or `"1+2+3"` (averaging). This feature is still under development. |
| `RUN_DIR` | string | `None` | Relative path to the directory containing data files (relative to the current working directory). If not provided, the program automatically uses the directory of the JSON file. |
| `CONCENTRATION` | number or array | `[1.0]` | Concentration value(s) for each run (**unit: mmol/L**), used for normalization to molar absorptivity. If the array is shorter than `RUN_LIST`, the last value is repeated. |
| `LENGTH` | number | `1.0` | Path length (**cm**), used for normalization to molar absorptivity. |
| `WAVELENGTH_DETECTED` | number | `0` | Detection wavelength (used in kinetic plot titles). |

### Data Preprocessing and Correction
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `X_CORR` | number or array | `0.0` | X-axis shift (specified per run). |
| `Y_CORR` | number or array | `0.0` | Y-axis shift. |
| `X_SCALE` | number or array | `1.0` | X-axis scaling factor. |
| `Y_SCALE` | number or array | `1.0` | Y-axis scaling factor. |
| `X_CUT_AFTER` | number | `null` | Truncate data points with X greater than this value. |
| `Y_SCALE_FACTOR_IN_UNIT` | number | `1.0` | Used in combination with concentration normalization to adjust units. |
| `AUTO_FIND_TIME_ZERO` | boolean | `false` | Whether to automatically find time zero (based on maximum y-value; this feature is still under development). |
| `NORMALIZATION_METHOD` | integer | `0` | I. For spectral normalization:<br>`0` - Use only `Y_CORR` correction;<br>`1` - Normalize to molar absorptivity (requires `CONCENTRATION` and `LENGTH`);<br>`2` - Normalize by maximum value;<br>II. For kinetic curves before fitting (does not affect plotting):<br>`0` - No normalization (still applies `Y_CORR` correction);<br>`1` - Subtract baseline mean;<br>`2` - Subtract baseline mean then normalize by maximum;<br>`3` - Normalize by maximum. |
| `SHOW_STATISTICS` | boolean | `true` | Whether to output kinetic curve statistics (noise level, SNR, etc.) to the terminal. |
| `SHOW_ALL_STATISTICS` | boolean | `false` | Whether to output more detailed statistics (e.g., for baseline, IRF SNR). |

### Axes and Ranges
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `XLABEL` | string | `""` | X-axis label. If empty, automatically generated based on `TYPE`. |
| `YLABEL` | string | `""` | Y-axis label. If empty, automatically generated based on `TYPE`. |
| `XMIN` | number | `0.0` | Lower limit of X-axis display. |
| `XMAX` | number | `0.0` | Upper limit of X-axis display (if both `XMIN` and `XMAX` are 0, the X-axis range is auto-determined). |
| `YMIN` | number | `0.0` | Lower limit of Y-axis display. |
| `YMAX` | number | `0.0` | Upper limit of Y-axis display (if both `YMIN` and `YMAX` are 0, the Y-axis range is auto-determined). |
| `YMAX_OF_RESIDUAL` | number | `0.0` | Symmetric Y-axis range for the residual plot. If 0, the range is automatically set to 1.1 × max absolute residual. |

### Fitting Settings
#### Basic Fitting
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `DO_FITTING` | boolean | `false` | Whether to perform fitting on kinetic data. |
| `FIT_FUNCTION` | string | `null` | Name of the fitting function (see code comments or appendix). Common options: `"exp_decay"`, `"exp_2decay"`, `"secondary_decay"`, `"primary_rise"`, etc. |
| `ENABLE_CUSTOM_INITIAL_GUESS` | boolean | `false` | Whether to use custom initial guesses. |
| `CUSTOM_INITIAL_GUESS` | array | `[]` | List of custom initial guess values (order must match the parameters of the fitting function). |
| `UPPER_BOUND` | array | `[]` | Upper bounds for parameters (order must match the initial guess). |
| `LOWER_BOUND` | array | `[]` | Lower bounds for parameters (order must match the initial guess). |
| `FIX_VALUE` | array | `[]` | Fixed parameter values (if specified, the corresponding parameter is held constant during fitting). |
| `ERROR_ANALYSIS` | boolean | `false` | Whether to calculate parameter errors (covariance matrix). |
| `RETRY_FIT_IF_FAIL` | boolean | `true` | Whether to allow immediate retry with adjusted initial guesses if fitting fails. |

#### Weighting
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `WEIGHTING_FOR_FIT` | boolean | `false` | Whether to enable weighting during fitting. |
| `WEIGHT` | number | `0.2` | Weight value (used when `WEIGHTING_AFTER` is enabled). |
| `WEIGHTING_AFTER` | number | `0.0` | Data points with time greater than this value will have their weight reduced from 1.0 to `WEIGHT`. |

#### Variable Baseline and Convolution Fitting
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `BASELINE` | integer or array | `0` | Run number(s) specifying the baseline data (for variable baseline fitting). |
| `IRF` | integer or array | `0` | Run number(s) specifying the instrument response function (IRF) data (for convolution fitting). |
| `IRF_WIDTH` | number | `80.0` | IRF width (used in convolution). |
| `DISCARD_POINTS_WHEN_RECONV` | integer | `50` | Number of points to discard from the end after convolution. |
| `RECONV_TZ` | boolean | `false` | Whether to treat time zero as a free parameter in convolution fitting. |

#### Fitting Result Display
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `SHOW_FITTING_INFO` | integer or boolean | `1` | Determines how fitting results are displayed:<br>`-1` - Hide most output;<br>`0` - Show only in terminal;<br>`1` - Show both in terminal and on the plot;<br>`2` - Show both in terminal and on the plot, with more complete model evaluation metrics. |
| `POSITION_OF_FITTING_INFO` | string | `"top right"` | Position of the fitting information text box, e.g., `"top left"`, `"bottom right"`. |

### Plotting Style
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `TITLE0` | string | `""` | Additional content for the plot title. |
| `TITLE_FONTSIZE` | integer | `18` | Font size of the title. |
| `FONTSIZE` | integer | `16` | Font size of axis labels and ticks. |
| `FIGSIZE` | array | `[10,6]` | Figure size in inches, e.g., `[8,5]`. |
| `LINEWIDTH` | number | `1.5` | Line width of curves. |
| `COLOR_LIST` | array or string | `[]` | List of colors for curves. Can be an array of color names (e.g., `["red","blue"]`) or a matplotlib colormap name (e.g., `"viridis"`). |
| `COLORMAP_REVERSED` | boolean | `false` | When `COLOR_LIST` is a colormap name, whether to reverse the color order. |
| `ENABLE_LEGEND` | boolean | `true` | Whether to display a legend. |
| `SHOW_RUN_NUMBER` | boolean | `false` | Whether to append the run number to legend labels. |
| `AUTOLABEL` | boolean | `false` | Whether to automatically generate legend labels (e.g., `"Run01"`). |
| `LABEL_LIST` | array | `[]` | Custom legend labels. |

### Intersection Search
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `FIND_INTERSECTIONS` | boolean | `false` | Whether to compute intersections between the first two curves. |
| `SHOW_INTERSECTIONS` | boolean | `false` | Whether to mark intersections on the plot. |
| `INTERSECTION_ABS_TOL` | number | `1e-2` | Intersections with y-values below this threshold will be discarded. |

### File Saving and Interaction
| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `FILENAME` | string | `""` | Specifies the output image filename (without path). If empty, the program generates a name based on the configuration. |
| `FILENAME_MODE` | integer | `0` | `0` - Use the JSON filename; `1` - Generate a descriptive filename based on the configuration. |
| `COPY_FILENAME_TO_CLIPBOARD` | boolean | `true` | Whether to copy the generated filename to the clipboard. |
| `LOOP` | boolean | `true` | Whether to enter the interactive command loop after plotting. If set to `false`, the program exits directly. |

> **Note**: Some parameters (e.g., `CONCENTRATION`, `X_CORR`, `Y_CORR`, `X_SCALE`, etc.) can accept either a single number (applied to all runs) or an array of the same length as `RUN_LIST` (applied per run). `BASELINE` and `IRF` can also accept arrays to correspond to each run.

## Fitting Function Descriptions

When analyzing kinetic data (`TYPE` is `"KinAbs"`, `"KA"`, or `"TC-SPC"` and `DO_FITTING` is `true`), the `FIT_FUNCTION` parameter specifies the fitting model. The program supports the following fitting functions (case-insensitive, with multiple aliases):

### Exponential Decay Models

| Function Name | Aliases | Mathematical Expression | Parameter Description |
|---------------|---------|-------------------------|------------------------|
| `linear` | `l`, `x`, `0`, `linear_decay` | $y = y_0 + k \cdot t$ | $y_0$: baseline offset<br>$k$: slope |
| `exp_decay` | `exp`, `decay`, `primary`, `ed`, `pd`, `1` | $y = y_0 + A \cdot e^{-t/\tau}$ | $y_0$: baseline offset<br>$A$: amplitude<br>$\tau$: lifetime |
| `exp_2decay` | `exp2`, `2decay`, `double_decay`, `ded`, `dpd`, `11` | $y = y_0 + A_1 e^{-t/\tau_1} + A_2 e^{-t/\tau_2}$ | $y_0$: baseline offset<br>$A_1,A_2$: amplitudes<br>$\tau_1,\tau_2$: lifetimes |
| `exp_3decay` | `exp3`, `3decay`, `triple_decay`, `ted`, `tpd`, `111` | $y = y_0 + A_1 e^{-t/\tau_1} + A_2 e^{-t/\tau_2} + A_3 e^{-t/\tau_3}$ | $y_0$: baseline offset<br>$A_1,A_2,A_3$: amplitudes<br>$\tau_1,\tau_2,\tau_3$: lifetimes |

### Second-Order Reaction Models and Variants

| Function Name | Aliases | Mathematical Expression | Parameter Description |
|---------------|---------|-------------------------|------------------------|
| `secondary_decay` | `secondary`, `reciprocal`, `sd`, `rd`, `2` | $y = y_0 + \frac{1}{k \cdot t + r_0}$ | $y_0$: baseline offset<br>$k$: second-order rate constant<br>$r_0$: initial reciprocal concentration |
| `primary_secondary_decay` | `primary_secondary`, `psd`, `spd`, `12`, `21` | $y = y_0 + \frac{k_1}{e^{k_1(t - const)} - 2k_2}$ | $y_0$: baseline offset<br>$k_1$: first-order rate constant<br>$k_2$: second-order rate constant<br>$const$: time shift constant |
| `primary_secondary_seperate` | `pss`, `1-2`, `2-1` | $y = y_0 + A e^{-k_1 t} + \frac{1}{k_2 t + r_0}$ | $y_0$: baseline offset<br>$A$: amplitude of first-order process<br>$k_1$: first-order rate constant<br>$k_2$: second-order rate constant<br>$r_0$: initial reciprocal concentration |

### Rise Models

| Function Name | Aliases | Mathematical Expression | Parameter Description |
|---------------|---------|-------------------------|------------------------|
| `primary_rise` | `rise`, `pr`, `er`, `1+` | $y = y_0 + A(1 - e^{-k t})$ | $y_0$: baseline offset<br>$A$: rise amplitude<br>$k$: rise rate constant |
| `primary_rise_linear_offset` | `rise0`, `risel`, `prl`, `erl`, `1+0` | $y = y_0 + A(1 - e^{-k t}) + B t$ | $y_0$: baseline offset<br>$A$: rise amplitude<br>$k$: rise rate constant<br>$B$: linear background slope |
| `primary_rise_double_offset` | `rise2`, `prd`, `erd`, `1+00` | $y = y_0 + A(1 - e^{-k(t-t_0)}) + B t$ | $y_0$: baseline offset<br>$A$: rise amplitude<br>$k$: rise rate constant<br>$B$: linear background slope<br>$t_0$: time shift |
| `primary_rise_triple_offset` | `rise3`, `prt`, `ert`, `1+000` | $y = y_0 + A(C - e^{-k(t-t_0)}) + B t$ | $y_0$: baseline offset<br>$A$: rise amplitude<br>$k$: rise rate constant<br>$B$: linear background slope<br>$C$: constant offset<br>$t_0$: time shift |
| `double_primary_rise` | `double_rise`, `dpr`, `11+` | $y = y_0 + A_1(1 - e^{-k_1 t}) + A_2(1 - e^{-k_2 t})$ | $y_0$: baseline offset<br>$A_1,A_2$: rise amplitudes<br>$k_1,k_2$: rise rate constants |
| `double_primary_rise_linear_offset` | `double_rise0`, `dprl`, `derl`, `110+` | $y = y_0 + A_1(1 - e^{-k_1 t}) + A_2(1 - e^{-k_2 t}) + B t$ | $y_0$: baseline offset<br>$A_1,A_2$: rise amplitudes<br>$k_1,k_2$: rise rate constants<br>$B$: linear background slope |

### Cascade Model

| Function Name | Aliases | Mathematical Expression | Parameter Description |
|---------------|---------|-------------------------|------------------------|
| `primary_cascade` | `cascade`, `pc`, `1-1` | $y = y_0 + A(e^{-k_1 t} - e^{-k_2 t})$ | $y_0$: baseline offset<br>$A$: amplitude<br>$k_1,k_2$: rise and decay rate constants |

### Gaussian Model

| Function Name | Aliases | Mathematical Expression | Parameter Description |
|---------------|---------|-------------------------|------------------------|
| `gaussian` | `g` | $y = y_0 + A e^{\frac{(t-t_0)^2}{2\sigma^2}}$ | $y_0$: baseline offset<br>$A$: peak amplitude<br>$t_0$: peak position<br>$\sigma$: width parameter |

### Usage Notes

1. **Parameter order**: When specifying `CUSTOM_INITIAL_GUESS`, `UPPER_BOUND`, `LOWER_BOUND`, or `FIX_VALUE` in the JSON file, the order must strictly match the parameter order in the table above.

2. **Rate constant units**:
   - First-order rate constants ($k_1$) have units of `time unit⁻¹` (e.g., `ns⁻¹`, `μs⁻¹`).
   - Second-order rate constants ($k_2$) have units of `(time unit·a.u.)⁻¹`. To convert to concentration-based rate constants, the proportionality factor between ΔOD and sample concentration must be determined.

3. **Time units**: The program automatically infers the time unit (ns/μs/ms/s) from `X_SCALE` and displays it correctly in the fitting results.

4. **Fit quality assessment**: After fitting, statistics such as $R^2$, adjusted $R^2$, $\chi^2$, and RMSE are displayed to help evaluate the fit quality.

## Statement and Contact Information
The author writes this code purely for fun; there is no clear development plan—features are added as ideas come. If you have any questions, feel free to raise them. This README file was generated using Deepseek.

- **Feedback**: Please submit issues, suggestions, or feature requests via [GitHub Issues] (link to be added).
- **Email**: zhuzl24@mails.tsinghua.edu.cn (for early-stage contact with the author)

---
