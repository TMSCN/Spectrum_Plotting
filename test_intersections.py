import numpy as np
import scipy

# Reimplementing the helper here to avoid importing the plotting module (which
# runs plotting on import). This is the same algorithm added to Spec_Plot_TA.py.

def find_curve_intersections(x1, y1, x2, y2, xtol=1e-8):
    x1 = np.asarray(x1); y1 = np.asarray(y1)
    x2 = np.asarray(x2); y2 = np.asarray(y2)
    s1 = np.argsort(x1); x1, y1 = x1[s1], y1[s1]
    s2 = np.argsort(x2); x2, y2 = x2[s2], y2[s2]
    x_common = np.unique(np.concatenate([x1, x2]))
    if x_common.size == 0:
        return []
    y1i = np.interp(x_common, x1, y1)
    y2i = np.interp(x_common, x2, y2)
    diff = y1i - y2i
    intersections = []
    zero_idx = np.where(np.isclose(diff, 0.0, atol=1e-12, rtol=0))[0]
    for idx in zero_idx:
        xi = float(x_common[idx]); yi = float(y1i[idx]); intersections.append((xi, yi))
    signs = np.sign(diff)
    sign_change_idx = np.where(signs[:-1] * signs[1:] < 0)[0]
    for idx in sign_change_idx:
        a = float(x_common[idx]); b = float(x_common[idx+1])
        f = lambda x: np.interp(x, x1, y1) - np.interp(x, x2, y2)
        try:
            root = float(scipy.optimize.brentq(f, a, b, xtol=xtol))
            yroot = float(np.interp(root, x1, y1))
            intersections.append((root, yroot))
        except Exception:
            continue
    if not intersections:
        return []
    intersections = sorted(intersections, key=lambda t: t[0])
    merged = [intersections[0]]
    for x,y in intersections[1:]:
        if abs(x - merged[-1][0]) > 1e-8:
            merged.append((x,y))
    return merged

# Path to your attached file
path = r"d:\!AcademicResources\!!Researches\!Jiao\ElectronSpect\ZZL-283\Run10.txt"
# Simple parser to extract numeric (x,y) pairs from the file
xs = []
ys = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0]); y = float(parts[1])
                xs.append(x); ys.append(y)
            except ValueError:
                continue

x = np.array(xs); y = np.array(ys)
print('Loaded', len(x), 'data points from', path)

# Create a second curve by subtracting a smoothed baseline so there will be
# intersections
y2 = y - 0.05 * np.sin(x / 20.0) - 0.02

inter = find_curve_intersections(x, y, x, y2)
print('Found', len(inter), 'intersections:')
for xi, yi in inter:
    print(f'  x={xi:.6f}, y={yi:.6e}')
