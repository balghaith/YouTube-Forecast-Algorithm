import os
import json
import csv
import subprocess
from datetime import datetime, timezone

import numpy as np

from flask import Flask, render_template, request, jsonify

from confidence import compute_r_squared, confidence_label
from forecast import fit_logistic_regression, predict

app = Flask(__name__)

DATA_DIR = "data"
KNOWN_VIDEOS_FILE = "known_videos.json"
ORIGINAL_FORECASTS_FILE = "original_forecasts.json"
HORIZONS_DAYS = [7, 14, 30]


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


def load_original_forecasts():
    if not os.path.isfile(ORIGINAL_FORECASTS_FILE):
        return {}
    with open(ORIGINAL_FORECASTS_FILE, "r") as f:
        return json.load(f)


def save_original_forecast_snapshot(video_id, metric, L, k, x0):
    data = load_original_forecasts()
    data.setdefault(video_id, {})
    if metric in data[video_id]:
        return
    data[video_id][metric] = {
        "L": L, "k": k, "x0": x0,
        "recorded_at": datetime.now(timezone.utc).isoformat()
    }
    with open(ORIGINAL_FORECASTS_FILE, "w") as f:
        json.dump(data, f)
    if os.environ.get("RENDER"):
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["git", "add", ORIGINAL_FORECASTS_FILE], cwd=repo_dir)
        subprocess.run(["git", "commit", "-m", f"snapshot original forecast for {video_id}/{metric}"], cwd=repo_dir)
        subprocess.run(["git", "push", "origin", "main"], cwd=repo_dir)


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
    x_days = [h / 24 for h in x]

    L, k, x0 = fit_logistic_regression(x_days, y)
    r_squared = compute_r_squared(x_days, y, L, k, x0)
    confidence = confidence_label(r_squared)

    forecasts = {
        f"day_{d}": round(predict(L, k, x0, d * 24), 1) for d in HORIZONS_DAYS
    }

    last_day = max(x_days)
    offset = y[-1] - predict(L, k, x0, x[-1])
    forecast_x = list(np.linspace(last_day, 30, 50))
    forecast_y = [predict(L, k, x0, d * 24) + offset for d in forecast_x]

    originals = load_original_forecasts()
    existing = originals.get(video_id, {}).get(metric)
    if existing is None:
        save_original_forecast_snapshot(video_id, metric, L, k, x0)
        orig_L, orig_k, orig_x0 = L, k, x0
    else:
        orig_L, orig_k, orig_x0 = existing["L"], existing["k"], existing["x0"]

    original_x = list(np.linspace(0.05, 30, 100))
    original_y = [predict(orig_L, orig_k, orig_x0, d * 24) for d in original_x]

    return jsonify({
        "video_id": video_id,
        "metric": metric,
        "r_squared": round(r_squared, 3) if r_squared is not None else None,
        "confidence": confidence,
        "forecasts": forecasts,
        "actual": {"x": x_days, "y": y},
        "forecast": {"x": forecast_x, "y": forecast_y},
        "original": {"x": original_x, "y": original_y},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, ssl_context='adhoc')