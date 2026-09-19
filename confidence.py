import numpy as np


def compute_r_squared(x, y, L, k, x0):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    y_pred = L / (1 + np.exp(-k * (x - x0)))
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