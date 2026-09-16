import os
import csv
import json
import numpy as np
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


def fit_log_regression(x, y):
    x_clipped = np.clip(np.array(x, dtype=float), 0.1, None)
    y_arr = np.array(y, dtype=float)
    y_clean = np.maximum.accumulate(y_arr)
    y_clipped = np.clip(y_clean, 0.1, None)
    ln_x = np.log(x_clipped)
    ln_y = np.log(y_clipped)

    a, b = np.polyfit(ln_x, ln_y, 1)
    return a, b


def predict(a, b, hours):
    hours = max(hours, 0.1)
    value = np.exp(a * np.log(hours) + b)
    return max(0.0, value)


def forecast_video(channel_id, video_id, published_at):
    rows = load_video_rows(channel_id, video_id)
    if len(rows) < 3:
        return None

    x = [hours_since_publish(row["timestamp"], published_at) for row in rows]

    results = {}
    for metric in METRICS:
        y = [float(row[metric]) for row in rows]
        a, b = fit_log_regression(x, y)
        r_squared = compute_r_squared(x, y, a, b)

        forecasts = {
            f"day_{days}": round(predict(a, b, days * 24), 1)
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