import os
import csv
import json
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime
from confidence import compute_r_squared, confidence_label

DATA_DIR = "data"
KNOWN_VIDEOS_FILE = "known_videos.json"

HORIZONS_DAYS = [7, 14, 30]
METRICS = ["views", "likes", "comments"]


def load_json(filename, default):
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return default


def load_video_rows(channel_id, video_id):
    path = os.path.join(DATA_DIR, channel_id, f"{video_id}.csv")
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def hours_since_publish(timestamp_str, published_at_str):
    t = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    p = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
    delta = t - p
    return delta.total_seconds() / 3600


def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def fit_logistic_regression(x, y):
    x_arr = np.array(x, dtype=float)
    y_arr = np.maximum.accumulate(np.array(y, dtype=float))

    L_guess = y_arr[-1] * 3
    k_guess = 0.1
    x0_guess = x_arr[len(x_arr) // 2]

    bounds = ([y_arr[-1], 0.0001, -1000], [y_arr[-1] * 100, 10, 10000])

    try:
        params, _ = curve_fit(
            logistic, x_arr, y_arr,
            p0=[L_guess, k_guess, x0_guess],
            bounds=bounds,
            maxfev=10000
        )
        return params[0], params[1], params[2]
    except RuntimeError:
        return y_arr[-1] * 2, 0.05, x_arr[-1]


def predict(L, k, x0, hours):
    days = hours / 24
    value = logistic(days, L, k, x0)
    return max(0.0, value)


def forecast_video(channel_id, video_id, published_at):
    rows = load_video_rows(channel_id, video_id)
    if len(rows) < 3:
        return None

    x = [hours_since_publish(row["timestamp"], published_at) for row in rows]
    x_days = [h / 24 for h in x]

    results = {}
    for metric in METRICS:
        y = [float(row[metric]) for row in rows]
        L, k, x0 = fit_logistic_regression(x_days, y)
        r_squared = compute_r_squared(x_days, y, L, k, x0)

        forecasts = {
            f"day_{days}": round(predict(L, k, x0, days * 24), 1)
            for days in HORIZONS_DAYS
        }

        results[metric] = {
            "r_squared": round(r_squared, 4) if r_squared is not None else None,
            "confidence": confidence_label(r_squared),
            "forecasts": forecasts
        }

    return results


if __name__ == "__main__":
    known_videos = load_json(KNOWN_VIDEOS_FILE, {})

    for video_id, info in known_videos.items():
        if info["status"] == "active":
            result = forecast_video(info["channel_id"], video_id, info["published_at"])
            if result:
                print(f"\n{video_id}")
                for metric, data in result.items():
                    print(f"  {metric}: R²={data['r_squared']} ({data['confidence']}) -> {data['forecasts']}")