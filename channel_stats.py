import os
import csv
import json
from datetime import datetime

DATA_DIR = "data"
KNOWN_VIDEOS_FILE = "known_videos.json"


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
    return (t - p).total_seconds() / 3600


def interpolate_value_at_hour(x_hours, y, target_hour):
    if target_hour <= x_hours[0]:
        return y[0]
    if target_hour >= x_hours[-1]:
        return y[-1]
    for i in range(1, len(x_hours)):
        if x_hours[i] >= target_hour:
            x0, x1 = x_hours[i - 1], x_hours[i]
            y0, y1 = y[i - 1], y[i]
            if x1 == x0:
                return y1
            ratio = (target_hour - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return y[-1]


def find_oldest_video_age_hours(channel_id):
    known_videos = load_json(KNOWN_VIDEOS_FILE, {})
    max_age = 0.0

    for video_id, info in known_videos.items():
        if info.get("channel_id") != channel_id:
            continue
        try:
            rows = load_video_rows(channel_id, video_id)
        except FileNotFoundError:
            continue
        if len(rows) < 2:
            continue
        published_at = info["published_at"]
        last_hour = hours_since_publish(rows[-1]["timestamp"], published_at)
        if last_hour > max_age:
            max_age = last_hour

    return max_age


def compute_channel_multipliers(channel_id, metric, exclude_video_id=None):
    known_videos = load_json(KNOWN_VIDEOS_FILE, {})

    oldest_hours = find_oldest_video_age_hours(channel_id)
    mid_checkpoint_hours = oldest_hours / 2
    early_checkpoint_hours = oldest_hours / 4

    ratios_early_mid = []
    ratios_mid_full = []

    for video_id, info in known_videos.items():
        if info.get("channel_id") != channel_id:
            continue
        if video_id == exclude_video_id:
            continue

        try:
            rows = load_video_rows(channel_id, video_id)
        except FileNotFoundError:
            continue

        if len(rows) < 3:
            continue

        published_at = info["published_at"]
        x_hours = [hours_since_publish(r["timestamp"], published_at) for r in rows]

        if metric not in rows[0]:
            continue
        y = [float(r[metric]) for r in rows]

        last_hour = x_hours[-1]

        if last_hour >= mid_checkpoint_hours:
            val_early = interpolate_value_at_hour(x_hours, y, early_checkpoint_hours)
            val_mid = interpolate_value_at_hour(x_hours, y, mid_checkpoint_hours)
            if val_early > 0 and val_mid > 0:
                ratios_early_mid.append(val_mid / val_early)

        if last_hour >= oldest_hours:
            val_mid = interpolate_value_at_hour(x_hours, y, mid_checkpoint_hours)
            val_full = interpolate_value_at_hour(x_hours, y, oldest_hours)
            if val_mid > 0 and val_full > 0:
                ratios_mid_full.append(val_full / val_mid)

    mult_early_mid = sum(ratios_early_mid) / len(ratios_early_mid) if ratios_early_mid else 1.0
    mult_mid_full = sum(ratios_mid_full) / len(ratios_mid_full) if ratios_mid_full else 1.0

    return {
        "mult_7_to_14": mult_early_mid,
        "mult_14_to_30": mult_mid_full,
        "sample_size_7_14": len(ratios_early_mid),
        "sample_size_14_30": len(ratios_mid_full),
        "oldest_video_hours": oldest_hours,
        "checkpoint_early": early_checkpoint_hours,
        "checkpoint_mid": mid_checkpoint_hours,
        "checkpoint_full": oldest_hours,
    }


def apply_channel_adjustment(forecasts, channel_id, metric, video_id):
    multipliers = compute_channel_multipliers(channel_id, metric, exclude_video_id=video_id)

    raw_7 = forecasts.get("day_7")
    raw_14 = forecasts.get("day_14")
    raw_30 = forecasts.get("day_30")

    adjusted_14 = raw_14
    adjusted_30 = raw_30

    if raw_7 is not None and raw_14 is not None:
        projected_14 = raw_7 * multipliers["mult_7_to_14"]
        adjusted_14 = max(raw_14, projected_14)

    if adjusted_14 is not None and raw_30 is not None:
        projected_30 = adjusted_14 * multipliers["mult_14_to_30"]
        adjusted_30 = max(raw_30, projected_30)

    adjusted_forecasts = {
        "day_7": round(raw_7, 3) if raw_7 is not None else None,
        "day_14": round(adjusted_14, 3) if adjusted_14 is not None else None,
        "day_30": round(adjusted_30, 3) if adjusted_30 is not None else None,
    }

    return adjusted_forecasts, multipliers