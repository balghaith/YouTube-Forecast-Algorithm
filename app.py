import os
import json
import csv
import subprocess
from datetime import datetime

import numpy as np

from flask import Flask, render_template, request, jsonify

from forecast import fit_arima_model, predict_arima, compute_arima_r_squared, confidence_label
from checkpoints import update_checkpoints_for_video
from channel_stats import apply_channel_adjustment

app = Flask(__name__)

DATA_DIR = "data"
KNOWN_VIDEOS_FILE = "known_videos.json"


def sync_data():
    if not os.environ.get("RENDER"):
        return
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    token = os.environ.get("GITHUB_TOKEN")
    remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"
    subprocess.run(["git", "remote", "remove", "origin"], cwd=repo_dir)
    subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=repo_dir)
    subprocess.run(["git", "fetch", "origin", "main"], cwd=repo_dir)
    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=repo_dir)


def load_known_videos():
    with open(KNOWN_VIDEOS_FILE, "r") as f:
        return json.load(f)


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


@app.route("/")
def index():
    sync_data()
    known_videos = load_known_videos()
    active_videos = [
        (vid, info) for vid, info in known_videos.items() if info["status"] == "active"
    ]
    active_videos.sort(key=lambda v: v[1]["published_at"], reverse=True)
    return render_template("index.html", videos=active_videos)


@app.route("/forecast/<video_id>")
def forecast(video_id):
    metric = request.args.get("metric", "views")
    return render_template("forecast.html", video_id=video_id, metric=metric)


@app.route("/api/forecast/<video_id>")
def api_forecast(video_id):
    sync_data()
    known_videos = load_known_videos()
    info = known_videos[video_id]
    channel_id = info["channel_id"]
    published_at = info["published_at"]
    metric = request.args.get("metric", "views")

    rows = load_video_rows(channel_id, video_id)
    x = [hours_since_publish(r["timestamp"], published_at) for r in rows]
    y = [float(r[metric]) for r in rows]

    try:
        fitted, last_hour_fit, y_grid = fit_arima_model(x, y)
        r_squared = compute_arima_r_squared(fitted, y_grid)
        confidence = confidence_label(r_squared)
    except Exception:
        r_squared = None
        confidence = confidence_label(None)

    last_hour = x[-1]
    last_actual_value = y[-1]

    first_hour = x[0]
    first_value = y[0]

    if last_hour > first_hour:
        overall_rate = (last_actual_value - first_value) / (last_hour - first_hour)
    else:
        overall_rate = 0.0

    recent_x = [xi for xi in x if xi >= x[-1] - 24]
    recent_y = y[-len(recent_x):]

    if len(recent_x) >= 2 and recent_x[-1] > recent_x[0]:
        recent_rate = (recent_y[-1] - recent_y[0]) / (recent_x[-1] - recent_x[0])
    else:
        recent_rate = overall_rate

    rate_per_hour = 0.2 * overall_rate + 0.8 * recent_rate

    def predict_fn(target_hour):
        hours_ahead = target_hour - last_hour
        value = last_actual_value + rate_per_hour * hours_ahead
        return value

    checkpoint_state = update_checkpoints_for_video(video_id, metric, x, y, predict_fn)

    return jsonify({
        "video_id": video_id,
        "metric": metric,
        "r_squared": round(r_squared, 3) if r_squared is not None else None,
        "confidence": confidence,
        "checkpoints": checkpoint_state,
        "current_value": round(last_actual_value),
        "current_day": round(x[-1] / 24, 2),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)