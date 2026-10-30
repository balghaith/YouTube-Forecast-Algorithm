import os
import json
import csv
import subprocess
from datetime import datetime, timezone

import numpy as np

from flask import Flask, render_template, request, jsonify

from forecast import fit_arima_model, predict_arima, compute_arima_r_squared, confidence_label
from checkpoints import update_checkpoints_for_video
from channel_stats import apply_channel_adjustment
from uploads import search_channels

app = Flask(__name__)

DATA_DIR = "data"
KNOWN_VIDEOS_FILE = "known_videos.json"
TRACKED_CHANNELS_FILE = "tracked_channels.json"


def sync_data():
    if os.environ.get("SKIP_SYNC"):
        return
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    token = os.environ.get("GITHUB_TOKEN")
    remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"
    subprocess.run(["git", "remote", "remove", "origin"], cwd=repo_dir)
    subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=repo_dir)
    subprocess.run(["git", "fetch", "origin", "main"], cwd=repo_dir)
    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=repo_dir)


def push_data():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    token = os.environ.get("GITHUB_TOKEN")
    remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"

    subprocess.run(["git", "config", "user.email", "balghaith05@gmail.com"], cwd=repo_dir)
    subprocess.run(["git", "config", "user.name", "balghaith"], cwd=repo_dir)
    subprocess.run(["git", "add", TRACKED_CHANNELS_FILE], cwd=repo_dir)
    commit_result = subprocess.run(
        ["git", "commit", "-m", "Add tracked channel from dashboard"],
        cwd=repo_dir, capture_output=True, text=True
    )
    if commit_result.returncode != 0:
        return

    subprocess.run(["git", "remote", "remove", "origin"], cwd=repo_dir)
    subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=repo_dir)
    subprocess.run(["git", "fetch", "origin", "main"], cwd=repo_dir)
    subprocess.run(["git", "config", "merge.ours.driver", "true"], cwd=repo_dir)
    subprocess.run(["git", "merge", "origin/main", "--no-edit"], cwd=repo_dir)
    subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=repo_dir)


def load_known_videos():
    with open(KNOWN_VIDEOS_FILE, "r") as f:
        return json.load(f)


def load_tracked_channels():
    if not os.path.isfile(TRACKED_CHANNELS_FILE):
        return []
    with open(TRACKED_CHANNELS_FILE, "r") as f:
        return json.load(f)


def save_tracked_channels(data):
    with open(TRACKED_CHANNELS_FILE, "w") as f:
        json.dump(data, f, indent=2)


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


def latest_video_stats(channel_id, video_id):
    try:
        rows = load_video_rows(channel_id, video_id)
    except (FileNotFoundError, OSError):
        return {"views": 0, "likes": 0, "comments": 0}

    if not rows:
        return {"views": 0, "likes": 0, "comments": 0}

    last = rows[-1]
    try:
        views = int(float(last.get("views", 0) or 0))
    except ValueError:
        views = 0
    try:
        likes = int(float(last.get("likes", 0) or 0))
    except ValueError:
        likes = 0
    try:
        comments = int(float(last.get("comments", 0) or 0))
    except ValueError:
        comments = 0

    return {"views": views, "likes": likes, "comments": comments}


@app.route("/classic")
def index_classic():
    sync_data()
    known_videos = load_known_videos()
    active_videos = [
        (vid, info) for vid, info in known_videos.items() if info["status"] == "active"
    ]
    active_videos.sort(key=lambda v: v[1]["published_at"], reverse=True)
    return render_template("index.html", videos=active_videos)


def build_channels_with_counts(tracked_channels, known_videos):
    channels_with_counts = []
    for channel in tracked_channels:
        channel_id = channel["channel_id"]
        channel_videos = [
            {
                "video_id": vid,
                "title": info.get("title") or vid,
                "published_at": info.get("published_at", ""),
                "tracked": info.get("tracked", False)
            }
            for vid, info in known_videos.items()
            if info.get("channel_id") == channel_id and info.get("status") == "active"
        ]
        channel_videos.sort(key=lambda v: v["published_at"], reverse=True)

        channels_with_counts.append({
            "channel_id": channel_id,
            "title": channel.get("title", channel_id),
            "thumbnail": channel.get("thumbnail", ""),
            "added_at": channel.get("added_at", ""),
            "video_count": len(channel_videos),
            "recent_videos": channel_videos[:5]
        })

    channels_with_counts.sort(key=lambda c: c["added_at"], reverse=True)
    return channels_with_counts


@app.route("/")
def dashboard():
    sync_data()
    tracked_channels = load_tracked_channels()
    known_videos = load_known_videos()

    channels_with_counts = build_channels_with_counts(tracked_channels, known_videos)
    total_videos = sum(c["video_count"] for c in channels_with_counts)
    total_channels = len(channels_with_counts)

    return render_template(
        "dashboard.html",
        channels=channels_with_counts[:4],
        total_videos=total_videos,
        total_channels=total_channels
    )


@app.route("/channels")
def all_channels():
    sync_data()
    tracked_channels = load_tracked_channels()
    known_videos = load_known_videos()

    channels_with_counts = build_channels_with_counts(tracked_channels, known_videos)
    total_videos = sum(c["video_count"] for c in channels_with_counts)

    return render_template("channels.html", channels=channels_with_counts, total_videos=total_videos)


@app.route("/channel/<channel_id>")
def channel_detail(channel_id):
    sync_data()
    tracked_channels = load_tracked_channels()
    known_videos = load_known_videos()

    channel = next((c for c in tracked_channels if c["channel_id"] == channel_id), None)
    channel_title = channel.get("title", channel_id) if channel else channel_id
    channel_thumbnail = channel.get("thumbnail", "") if channel else ""

    videos = [
        {
            "video_id": vid,
            "title": info.get("title") or vid,
            "published_at": info.get("published_at", ""),
            "status": info.get("status", "unknown"),
            "tracked": info.get("tracked", False)
        }
        for vid, info in known_videos.items()
        if info.get("channel_id") == channel_id
    ]
    videos.sort(key=lambda v: v["published_at"], reverse=True)

    channel_totals = {"views": 0, "likes": 0, "comments": 0}
    for v in videos:
        stats = latest_video_stats(channel_id, v["video_id"])
        v["views"] = stats["views"]
        v["likes"] = stats["likes"]
        v["comments"] = stats["comments"]
        channel_totals["views"] += stats["views"]
        channel_totals["likes"] += stats["likes"]
        channel_totals["comments"] += stats["comments"]

    if channel_totals["views"] > 0:
        engagement_rate = (channel_totals["likes"] + channel_totals["comments"]) / channel_totals["views"] * 100
    else:
        engagement_rate = 0.0
    channel_totals["engagement_rate"] = round(engagement_rate, 2)

    video_stats = [
        {"title": v["title"], "views": v["views"], "likes": v["likes"], "comments": v["comments"]}
        for v in videos
    ]

    return render_template(
        "channel.html",
        channel_id=channel_id,
        channel_title=channel_title,
        channel_thumbnail=channel_thumbnail,
        videos=videos,
        channel_totals=channel_totals,
        video_stats=video_stats
    )


@app.route("/api/channels/search")
def api_channel_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": []})
    try:
        results = search_channels(query, max_results=8)
    except Exception as e:
        return jsonify({"results": [], "error": str(e)}), 500
    return jsonify({"results": results})


@app.route("/api/channels/add", methods=["POST"])
def api_channel_add():
    sync_data()
    data = request.get_json(force=True)
    channel_id = data.get("channel_id")
    title = data.get("title", channel_id)
    thumbnail = data.get("thumbnail", "")

    if not channel_id:
        return jsonify({"error": "channel_id required"}), 400

    tracked_channels = load_tracked_channels()
    if any(c["channel_id"] == channel_id for c in tracked_channels):
        return jsonify({"status": "already_tracked"}), 200

    tracked_channels.append({
        "channel_id": channel_id,
        "title": title,
        "thumbnail": thumbnail,
        "added_at": datetime.now(timezone.utc).isoformat()
    })
    save_tracked_channels(tracked_channels)
    push_data()

    return jsonify({"status": "added"}), 200


@app.route("/forecast/<video_id>")
def forecast(video_id):
    sync_data()
    known_videos = load_known_videos()
    tracked_channels = load_tracked_channels()
    info = known_videos.get(video_id, {})
    video_title = info.get("title") or video_id

    channel_id = info.get("channel_id", "")
    channel = next((c for c in tracked_channels if c["channel_id"] == channel_id), None)
    channel_title = channel.get("title", channel_id) if channel else channel_id
    channel_thumbnail = channel.get("thumbnail", "") if channel else ""

    metric = request.args.get("metric", "views")
    return render_template(
        "forecast_new.html",
        video_id=video_id,
        video_title=video_title,
        channel_id=channel_id,
        channel_title=channel_title,
        channel_thumbnail=channel_thumbnail,
        metric=metric
    )


@app.route("/classic/forecast/<video_id>")
def forecast_classic(video_id):
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