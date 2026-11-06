import os
import re
import json
import csv
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from flask import Flask, render_template, request, jsonify, redirect, url_for

from forecast import fit_arima_model, predict_arima, compute_arima_r_squared, confidence_label
from checkpoints import update_checkpoints_for_video
from channel_stats import apply_channel_adjustment
from uploads import search_channels

app = Flask(__name__)

DATA_DIR = "data"
KNOWN_VIDEOS_FILE = "known_videos.json"
TRACKED_CHANNELS_FILE = "tracked_channels.json"
SUBSCRIBERS_FILE = "channel_subscribers.json"
MAX_TRACKED_CHANNELS = 15
CHECKPOINTS_STORE_FILE = "forecast_checkpoints.json"
ORIGINAL_FORECASTS_FILE = "original_forecasts.json"
CHANNEL_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


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


def read_json_file(path, default):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (ValueError, OSError):
        return default


def write_json_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def remove_channel_data(channel_id):
    tracked_channels = [c for c in load_tracked_channels() if c.get("channel_id") != channel_id]
    write_json_file(TRACKED_CHANNELS_FILE, tracked_channels)

    known_videos = read_json_file(KNOWN_VIDEOS_FILE, {})
    video_ids = [vid for vid, info in known_videos.items() if info.get("channel_id") == channel_id]
    for vid in video_ids:
        del known_videos[vid]
    write_json_file(KNOWN_VIDEOS_FILE, known_videos)

    for path in (CHECKPOINTS_STORE_FILE, ORIGINAL_FORECASTS_FILE):
        if not os.path.isfile(path):
            continue
        store = read_json_file(path, {})
        for vid in video_ids:
            store.pop(vid, None)
        write_json_file(path, store)

    subscribers = read_json_file(SUBSCRIBERS_FILE, {})
    if channel_id in subscribers:
        del subscribers[channel_id]
        write_json_file(SUBSCRIBERS_FILE, subscribers)

    folder = os.path.join(DATA_DIR, channel_id)
    if os.path.isdir(folder):
        shutil.rmtree(folder)

    return video_ids


def push_channel_deletion(channel_id):
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    token = os.environ.get("GITHUB_TOKEN")

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True)

    tracked_data = git("ls-files", "--", os.path.join(DATA_DIR, channel_id)).stdout.strip()
    paths = [
        TRACKED_CHANNELS_FILE,
        KNOWN_VIDEOS_FILE,
        CHECKPOINTS_STORE_FILE,
        ORIGINAL_FORECASTS_FILE,
        SUBSCRIBERS_FILE
    ]
    if tracked_data:
        paths.append(os.path.join(DATA_DIR, channel_id))
    paths = [p for p in paths if git("ls-files", "--", p).stdout.strip()]

    git("config", "user.email", "balghaith05@gmail.com")
    git("config", "user.name", "balghaith")
    git("add", "-A", "--", *paths)
    commit_result = git("commit", "-m", f"Delete tracked channel {channel_id}", "--", *paths)
    if commit_result.returncode != 0:
        return False

    if token:
        remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"
        git("remote", "remove", "origin")
        git("remote", "add", "origin", remote_url)
    git("fetch", "origin", "main")
    git("config", "merge.ours.driver", "true")
    merge_result = git("merge", "origin/main", "--no-edit")
    if merge_result.returncode != 0:
        git("merge", "--abort")
        return False
    return git("push", "origin", "HEAD:main").returncode == 0


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


def subscriber_text(channel_id):
    if not os.path.isfile(SUBSCRIBERS_FILE):
        return ""
    try:
        with open(SUBSCRIBERS_FILE, "r") as f:
            entry = json.load(f).get(channel_id)
    except (ValueError, OSError):
        return ""
    if not entry or entry.get("subscribers") is None:
        return ""

    count = entry["subscribers"]
    for threshold, divisor, suffix in (
        (999_500_000, 1_000_000_000, "B"),
        (999_500, 1_000_000, "M"),
        (1_000, 1_000, "K")
    ):
        if count >= threshold:
            value = count / divisor
            if value < 10:
                text = f"{value:.2f}"
            elif value < 100:
                text = f"{value:.1f}"
            else:
                text = f"{value:.0f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"{text}{suffix} subscribers"
    label = "subscriber" if count == 1 else "subscribers"
    return f"{count} {label}"


def parse_iso_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def visible_known_videos(known_videos, tracked_channels):
    added_by_channel = {c["channel_id"]: c.get("added_at") for c in tracked_channels}
    visible = {}
    for vid, info in known_videos.items():
        added_at = added_by_channel.get(info.get("channel_id"))
        published_at = info.get("published_at")
        if added_at and published_at and parse_iso_time(published_at) < parse_iso_time(added_at):
            continue
        visible[vid] = info
    return visible


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


TRACKING_HOURS = 30 * 24


def rows_within_tracking(rows, published_at):
    if not published_at:
        return rows
    kept = []
    for row in rows:
        try:
            if hours_since_publish(row["timestamp"], published_at) <= TRACKING_HOURS:
                kept.append(row)
        except (ValueError, KeyError):
            continue
    return kept or rows[:1]


def latest_video_stats(channel_id, video_id, published_at=None):
    try:
        rows = load_video_rows(channel_id, video_id)
    except (FileNotFoundError, OSError):
        return {"views": 0, "likes": 0, "comments": 0}

    if not rows:
        return {"views": 0, "likes": 0, "comments": 0}

    if published_at:
        rows = rows_within_tracking(rows, published_at)

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


LA_TZ = ZoneInfo("America/Los_Angeles")
TODAY_WEIGHTS = {"views": 0.5, "likes": 0.3, "comments": 0.2}


def row_counts(row):
    counts = {}
    for key in ("views", "likes", "comments"):
        try:
            counts[key] = int(float(row.get(key, 0) or 0))
        except ValueError:
            counts[key] = 0
    return counts


def gains_today(channel_id, video_id, published_at, day_start):
    try:
        rows = load_video_rows(channel_id, video_id)
    except (FileNotFoundError, OSError):
        return None

    parsed = []
    for row in rows:
        try:
            parsed.append((parse_iso_time(row["timestamp"]), row))
        except (KeyError, ValueError):
            continue
    if not parsed:
        return None
    parsed.sort(key=lambda item: item[0])

    before = [item for item in parsed if item[0] <= day_start]
    if before:
        baseline = row_counts(before[-1][1])
    elif published_at and parse_iso_time(published_at) >= day_start:
        baseline = {"views": 0, "likes": 0, "comments": 0}
    else:
        baseline = row_counts(parsed[0][1])

    latest = row_counts(parsed[-1][1])
    return {key: max(latest[key] - baseline[key], 0) for key in latest}


def todays_top_videos(channel_id, live_videos, limit=5):
    now_la = datetime.now(LA_TZ)
    day_start = now_la.replace(hour=0, minute=0, second=0, microsecond=0)

    entries = []
    for v in live_videos:
        gains = gains_today(channel_id, v["video_id"], v["published_at"], day_start)
        if gains is None:
            continue
        entries.append({
            "video_id": v["video_id"],
            "title": v["title"],
            "views": gains["views"],
            "likes": gains["likes"],
            "comments": gains["comments"]
        })

    totals = {key: sum(e[key] for e in entries) for key in TODAY_WEIGHTS}
    for e in entries:
        e["score"] = sum(
            weight * (e[key] / totals[key] if totals[key] else 0)
            for key, weight in TODAY_WEIGHTS.items()
        )

    entries.sort(key=lambda e: (e["score"], e["views"]), reverse=True)
    return entries[:limit], now_la


@app.route("/classic")
def index_classic():
    sync_data()
    known_videos = visible_known_videos(load_known_videos(), load_tracked_channels())
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


GROWTH_CHART_MAX_LINES = 6


def build_growth_chart(tracked_channels, known_videos):
    today = datetime.now(LA_TZ).date()
    dates_by_channel = {}
    for info in known_videos.values():
        if info.get("status") != "active":
            continue
        published_at = info.get("published_at")
        channel_id = info.get("channel_id")
        if not published_at or not channel_id:
            continue
        published_day = parse_iso_time(published_at).astimezone(LA_TZ).date()
        dates_by_channel.setdefault(channel_id, []).append(min(published_day, today))

    if not dates_by_channel:
        return None

    start = min(min(days) for days in dates_by_channel.values())
    if start >= today:
        start = today - timedelta(days=1)
    day_list = [start + timedelta(days=n) for n in range((today - start).days + 1)]

    meta = {c["channel_id"]: c for c in tracked_channels}
    series = []
    for channel_id, published_days in dates_by_channel.items():
        channel = meta.get(channel_id)
        if not channel:
            continue
        values = []
        running = 0
        for day in day_list:
            running += sum(1 for d in published_days if d == day)
            values.append(running)
        series.append({
            "id": channel_id,
            "title": channel.get("title", channel_id),
            "thumbnail": channel.get("thumbnail", ""),
            "values": values
        })

    if not series:
        return None

    series.sort(key=lambda s: s["values"][-1], reverse=True)
    shown = series[:GROWTH_CHART_MAX_LINES]
    return {
        "days": [d.isoformat() for d in day_list],
        "series": shown,
        "extra": max(len(tracked_channels) - len(shown), 0)
    }


@app.route("/")
def dashboard():
    sync_data()
    tracked_channels = load_tracked_channels()
    known_videos = visible_known_videos(load_known_videos(), tracked_channels)

    channels_with_counts = build_channels_with_counts(tracked_channels, known_videos)
    total_videos = sum(c["video_count"] for c in channels_with_counts)
    total_channels = len(channels_with_counts)

    return render_template(
        "dashboard.html",
        channels=channels_with_counts[:4],
        total_videos=total_videos,
        total_channels=total_channels,
        growth_chart=build_growth_chart(tracked_channels, known_videos)
    )


@app.route("/channels")
def all_channels():
    sync_data()
    tracked_channels = load_tracked_channels()
    known_videos = visible_known_videos(load_known_videos(), tracked_channels)

    channels_with_counts = build_channels_with_counts(tracked_channels, known_videos)
    total_videos = sum(c["video_count"] for c in channels_with_counts)

    return render_template("channels.html", channels=channels_with_counts, total_videos=total_videos)


@app.route("/channel/<channel_id>")
def channel_detail(channel_id):
    sync_data()
    tracked_channels = load_tracked_channels()
    known_videos = visible_known_videos(load_known_videos(), tracked_channels)

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
    live_videos = []
    for v in videos:
        stats = latest_video_stats(channel_id, v["video_id"])
        v["views"] = stats["views"]
        v["likes"] = stats["likes"]
        v["comments"] = stats["comments"]
        if v["status"] == "active":
            live_videos.append(v)
            channel_totals["views"] += stats["views"]
            channel_totals["likes"] += stats["likes"]
            channel_totals["comments"] += stats["comments"]

    if channel_totals["views"] > 0:
        engagement_rate = (channel_totals["likes"] + channel_totals["comments"]) / channel_totals["views"] * 100
    else:
        engagement_rate = 0.0
    channel_totals["engagement_rate"] = round(engagement_rate, 2)

    video_stats = [
        {
            "video_id": v["video_id"],
            "title": v["title"],
            "views": v["views"],
            "likes": v["likes"],
            "comments": v["comments"]
        }
        for v in live_videos
    ]

    expired_videos = [v for v in videos if v["status"] == "expired"]
    top_today, now_la = todays_top_videos(channel_id, live_videos)
    day_progress = round((now_la.hour * 60 + now_la.minute) / 1440 * 100, 1)

    return render_template(
        "channel.html",
        live_count=len(live_videos),
        total_count=len(videos),
        expired_count=len(expired_videos),
        expired_thumbs=[v["video_id"] for v in expired_videos[:3]],
        top_today=top_today,
        today_prefix=now_la.strftime("%a, %b"),
        today_day=now_la.strftime("%-d"),
        day_progress=day_progress,
        day_time=now_la.strftime("%-I:%M %p"),
        channel_id=channel_id,
        channel_title=channel_title,
        channel_thumbnail=channel_thumbnail,
        subscriber_text=subscriber_text(channel_id),
        videos=live_videos,
        channel_totals=channel_totals,
        video_stats=video_stats
    )


@app.route("/channel/<channel_id>/expired")
def channel_expired(channel_id):
    sync_data()
    tracked_channels = load_tracked_channels()
    known_videos = visible_known_videos(load_known_videos(), tracked_channels)

    channel = next((c for c in tracked_channels if c["channel_id"] == channel_id), None)
    channel_title = channel.get("title", channel_id) if channel else channel_id
    channel_thumbnail = channel.get("thumbnail", "") if channel else ""

    videos = []
    for vid, info in known_videos.items():
        if info.get("channel_id") != channel_id or info.get("status") != "expired":
            continue
        stats = latest_video_stats(channel_id, vid, info.get("published_at"))
        videos.append({
            "video_id": vid,
            "title": info.get("title") or vid,
            "published_at": info.get("published_at", ""),
            "views": stats["views"],
            "likes": stats["likes"],
            "comments": stats["comments"]
        })
    videos.sort(key=lambda v: v["published_at"], reverse=True)

    return render_template(
        "expired.html",
        channel_id=channel_id,
        channel_title=channel_title,
        channel_thumbnail=channel_thumbnail,
        videos=videos
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

    if len(tracked_channels) >= MAX_TRACKED_CHANNELS:
        return jsonify({"error": "limit_reached", "max": MAX_TRACKED_CHANNELS}), 409

    tracked_channels.append({
        "channel_id": channel_id,
        "title": title,
        "thumbnail": thumbnail,
        "added_at": datetime.now(timezone.utc).isoformat()
    })
    save_tracked_channels(tracked_channels)
    push_data()

    return jsonify({"status": "added"}), 200


@app.route("/api/channels/<channel_id>/delete", methods=["POST"])
def api_channel_delete(channel_id):
    if not CHANNEL_ID_PATTERN.fullmatch(channel_id):
        return jsonify({"error": "invalid channel id"}), 400

    sync_data()
    tracked_channels = load_tracked_channels()
    if not any(c.get("channel_id") == channel_id for c in tracked_channels):
        return jsonify({"error": "channel not tracked"}), 404

    video_ids = remove_channel_data(channel_id)
    pushed = push_channel_deletion(channel_id)

    return jsonify({"status": "deleted", "videos": len(video_ids), "pushed": pushed}), 200


@app.route("/forecast/<video_id>")
def forecast(video_id):
    sync_data()
    all_known_videos = load_known_videos()
    tracked_channels = load_tracked_channels()
    known_videos = visible_known_videos(all_known_videos, tracked_channels)

    if video_id in all_known_videos and video_id not in known_videos:
        return redirect(url_for("channel_detail", channel_id=all_known_videos[video_id]["channel_id"]))

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

    try:
        rows = load_video_rows(channel_id, video_id)
    except (FileNotFoundError, OSError):
        rows = []

    if not rows:
        return jsonify({"video_id": video_id, "metric": metric, "pending": True})

    all_hours = [hours_since_publish(r["timestamp"], published_at) for r in rows]
    finished = info.get("status") != "active" or max(all_hours) >= TRACKING_HOURS

    if finished:
        rows = rows_within_tracking(rows, published_at)

    x = [hours_since_publish(r["timestamp"], published_at) for r in rows]
    y = [float(r[metric]) for r in rows]

    if finished and x[-1] < TRACKING_HOURS:
        x.append(float(TRACKING_HOURS))
        y.append(y[-1])

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
        "finished": finished,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)