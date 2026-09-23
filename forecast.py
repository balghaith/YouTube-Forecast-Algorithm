import os
import csv
import json
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

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


def despike_series(y):
    y = list(y)
    n = len(y)
    cleaned = y[:]
    for i in range(1, n - 1):
        prev_v = cleaned[i - 1]
        curr_v = y[i]
        next_v = y[i + 1]
        moved_away = abs(curr_v - prev_v) > max(5, 0.02 * prev_v)
        reverted = abs(next_v - prev_v) < max(5, 0.01 * prev_v)
        if moved_away and reverted:
            cleaned[i] = prev_v
    return np.array(cleaned, dtype=float)


def prepare_hourly_series(x_hours, y):
    x_arr = np.array(x_hours, dtype=float)
    y_arr = despike_series(y)
    last_hour = x_arr[-1]
    grid = np.arange(0, int(np.floor(last_hour)) + 1)
    y_grid = np.interp(grid, x_arr, y_arr)
    return grid, y_grid


def fit_arima_model(x_hours, y):
    grid, y_grid = prepare_hourly_series(x_hours, y)
    if len(y_grid) < 5:
        raise ValueError("Not enough data points for ARIMA fit")
    model = ARIMA(y_grid, order=(2, 1, 1))
    fitted = model.fit()
    return fitted, float(grid[-1]), y_grid


def predict_arima(fitted, last_hour, target_hours):
    target_hours = np.array(target_hours, dtype=float)
    max_target = float(np.max(target_hours))
    steps = int(np.ceil(max_target - last_hour)) + 1
    steps = max(steps, 1)

    forecast_values = np.array(fitted.forecast(steps=steps))
    forecast_hours = last_hour + np.arange(1, steps + 1)

    known_hours = np.concatenate(([last_hour], forecast_hours))
    known_values = np.concatenate(([fitted.data.endog[-1]], forecast_values))

    predicted = np.interp(target_hours, known_hours, known_values)
    return np.maximum(predicted, 0.0)


def predict_arima_with_interval(fitted, last_hour, target_hours, alpha=0.2):
    target_hours = np.array(target_hours, dtype=float)
    max_target = float(np.max(target_hours))
    steps = int(np.ceil(max_target - last_hour)) + 1
    steps = max(steps, 1)

    forecast_result = fitted.get_forecast(steps=steps)
    forecast_values = np.array(forecast_result.predicted_mean)
    conf_int = np.array(forecast_result.conf_int(alpha=alpha))
    lower = conf_int[:, 0]
    upper = conf_int[:, 1]

    forecast_hours = last_hour + np.arange(1, steps + 1)
    known_hours = np.concatenate(([last_hour], forecast_hours))
    known_values = np.concatenate(([fitted.data.endog[-1]], forecast_values))
    known_lower = np.concatenate(([fitted.data.endog[-1]], lower))
    known_upper = np.concatenate(([fitted.data.endog[-1]], upper))

    predicted = np.maximum(np.interp(target_hours, known_hours, known_values), 0.0)
    predicted_lower = np.maximum(np.interp(target_hours, known_hours, known_lower), 0.0)
    predicted_upper = np.maximum(np.interp(target_hours, known_hours, known_upper), 0.0)

    return predicted, predicted_lower, predicted_upper


def compute_arima_r_squared(fitted, y_grid):
    residuals = np.array(fitted.resid)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_grid - np.mean(y_grid)) ** 2)
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


def forecast_video(channel_id, video_id, published_at):
    rows = load_video_rows(channel_id, video_id)
    if len(rows) < 3:
        return None

    x = [hours_since_publish(row["timestamp"], published_at) for row in rows]

    results = {}
    for metric in METRICS:
        y = [float(row[metric]) for row in rows]
        fitted, last_hour, y_grid = fit_arima_model(x, y)
        r_squared = compute_arima_r_squared(fitted, y_grid)

        target_hours = [days * 24 for days in HORIZONS_DAYS]
        predicted = predict_arima(fitted, last_hour, target_hours)

        forecasts = {
            f"day_{days}": round(float(val), 1)
            for days, val in zip(HORIZONS_DAYS, predicted)
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