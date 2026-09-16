import numpy as np


def compute_r_squared(x, y, a, b):
    x = np.clip(np.array(x, dtype=float), 0.1, None)
    y = np.array(y, dtype=float)
    ln_x = np.log(x)

    y_pred = np.exp(a * ln_x + b)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return None

    return 1 - (ss_res / ss_tot)


def confidence_label(r_squared):
    if r_squared is None:
        return "Insufficient data"
    elif r_squared < 0.33:
        return "Low"
    elif r_squared < 0.66:
        return "Moderate"
    else:
        return "High"