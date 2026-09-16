import os
import io
import base64
import json
import csv
import subprocess
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request

from confidence import compute_r_squared, confidence_label
from forecast import fit_log_regression, predict

app = Flask(__name__)

DATA_DIR = "data"
KNOWN_VIDEOS_FILE = "known_videos.json"
HORIZONS_DAYS = [7, 14, 30]


def sync_data():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    token = os.environ.get("GITHUB_TOKEN")
    remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"
    subprocess.run(["git", "remote", "set-url", "origin", remote_url], cwd=repo_dir)
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


def make_chart(x_hours, y_vals, a, b, metric):
    fig, ax = plt.subplots(figsize=(7, 4))

    x_days = [h / 24 for h in x_hours]
    ax.scatter(x_days, y_vals, label="Actual", color="teal", zorder=3)

    last_day = max(x_days)
    fitted_x = np.linspace(0.05, last_day, 50)
    fitted_y = [predict(a, b, d * 24) for d in fitted_x]
    ax.plot(fitted_x, fitted_y, color="teal", label="Fitted")

    forecast_x = np.linspace(last_day, 30, 50)
    forecast_y = [predict(a, b, d * 24) for d in forecast_x]
    ax.plot(forecast_x, forecast_y, color="orange", linestyle="--", label="Forecast")

    ax.set_xlabel("Days since publish")
    ax.set_ylabel(metric)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


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
    sync_data()
    known_videos = load_known_videos()
    info = known_videos[video_id]
    channel_id = info["channel_id"]
    published_at = info["published_at"]
    metric = request.args.get("metric", "views")

    rows = load_video_rows(channel_id, video_id)
    x = [hours_since_publish(r["timestamp"], published_at) for r in rows]
    y = [float(r[metric]) for r in rows]

    a, b = fit_log_regression(x, y)
    r_squared = compute_r_squared(x, y, a, b)
    confidence = confidence_label(r_squared)

    forecasts = {
        f"day_{d}": round(predict(a, b, d * 24), 1) for d in HORIZONS_DAYS
    }

    chart_b64 = make_chart(x, y, a, b, metric)

    return render_template(
        "forecast.html",
        video_id=video_id,
        metric=metric,
        r_squared=round(r_squared, 3) if r_squared is not None else None,
        confidence=confidence,
        forecasts=forecasts,
        chart_b64=chart_b64,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)