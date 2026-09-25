import os
import csv
import json
import subprocess
import shutil
from datetime import datetime, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=api_key)

DATA_DIR = "data"
TRACKED_CHANNELS_FILE = "tracked_channels.json"
KNOWN_VIDEOS_FILE = "known_videos.json"
CHECKPOINTS_FILE = "forecast_checkpoints.json"
MAX_TRACKED_CHANNELS = 15


def load_json(filename, default):
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return default


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)


def cleanup_untracked_channels():
    tracked_channels = load_json(TRACKED_CHANNELS_FILE, [])
    tracked_ids = {c["channel_id"] for c in tracked_channels}

    known_videos = load_json(KNOWN_VIDEOS_FILE, {})
    checkpoints = load_json(CHECKPOINTS_FILE, {})

    removed_video_ids = [
        vid for vid, info in known_videos.items()
        if info["channel_id"] not in tracked_ids
    ]

    for video_id in removed_video_ids:
        del known_videos[video_id]
        if video_id in checkpoints:
            del checkpoints[video_id]

    if removed_video_ids:
        save_json(KNOWN_VIDEOS_FILE, known_videos)
        save_json(CHECKPOINTS_FILE, checkpoints)
        print(f"Removed {len(removed_video_ids)} video entries from untracked channels")

    if os.path.isdir(DATA_DIR):
        for entry in os.listdir(DATA_DIR):
            channel_folder = os.path.join(DATA_DIR, entry)
            if os.path.isdir(channel_folder) and entry not in tracked_ids:
                shutil.rmtree(channel_folder)
                print(f"Deleted entire folder for untracked channel {entry}")


def uploads_id(channel_id):
    request = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    )
    response = request.execute()
    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def latest_videos(playlist_id, max_results=50):
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=playlist_id,
        maxResults=max_results
    )
    response = request.execute()
    videos = []
    for item in response["items"]:
        videos.append({
            "video_id": item["contentDetails"]["videoId"],
            "title": item["snippet"]["title"],
            "published_at": item["contentDetails"]["videoPublishedAt"]
        })
    return videos


def poll_video_stats(video_id, channel_id):
    request = youtube.videos().list(
        part="statistics",
        id=video_id
    )
    try:
        response = request.execute()
    except Exception as e:
        print(f"Error fetching stats for {video_id}: {e}")
        return None

    if not response["items"]:
        return None

    stats = response["items"][0]["statistics"]

    row = {
        "channel_id": channel_id,
        "video_id": video_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "views": stats.get("viewCount", 0),
        "likes": stats.get("likeCount", 0),
        "comments": stats.get("commentCount", 0)
    }
    return row


def save_row(row):
    channel_folder = os.path.join(DATA_DIR, row["channel_id"])
    os.makedirs(channel_folder, exist_ok=True)

    video_file = os.path.join(channel_folder, f"{row['video_id']}.csv")
    file_exists = os.path.isfile(video_file)

    with open(video_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "views", "likes", "comments"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": row["timestamp"],
            "views": row["views"],
            "likes": row["likes"],
            "comments": row["comments"]
        })


def is_video_expired(published_at_str, max_days=30):
    published_at = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - published_at).days
    return age_days > max_days


def push_to_github():
    subprocess.run(["git", "config", "--global", "user.email", "balghaith05@gmail.com"])
    subprocess.run(["git", "config", "--global", "user.name", "balghaith"])
    subprocess.run(["git", "add", DATA_DIR, TRACKED_CHANNELS_FILE, KNOWN_VIDEOS_FILE, CHECKPOINTS_FILE])
    subprocess.run(["git", "commit", "-m", "Update tracked data"])

    token = os.getenv("GITHUB_TOKEN")
    remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"
    subprocess.run(["git", "remote", "remove", "origin"])
    subprocess.run(["git", "remote", "add", "origin", remote_url])

    fetch_result = subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, text=True)
    print("FETCH:", fetch_result.returncode, fetch_result.stderr)

    merge_result = subprocess.run(
        ["git", "merge", "-X", "ours", "origin/main", "--no-edit"],
        capture_output=True, text=True
    )
    print("MERGE:", merge_result.returncode, merge_result.stdout, merge_result.stderr)

    push_result = subprocess.run(["git", "push", "origin", "HEAD:main"], capture_output=True, text=True)
    print("PUSH:", push_result.returncode, push_result.stdout, push_result.stderr)


def run_polling_cycle(push=True):
    cleanup_untracked_channels()

    tracked_channels = load_json(TRACKED_CHANNELS_FILE, [])

    if len(tracked_channels) > MAX_TRACKED_CHANNELS:
        tracked_channels = tracked_channels[:MAX_TRACKED_CHANNELS]

    known_videos = load_json(KNOWN_VIDEOS_FILE, {})

    for channel in tracked_channels:
        channel_added_at = channel.get("added_at")
        playlist_id = uploads_id(channel["channel_id"])
        latest = latest_videos(playlist_id, max_results=50)

        for video in latest:
            if video["video_id"] not in known_videos:
                published_at = datetime.fromisoformat(video["published_at"].replace("Z", "+00:00"))
                added_at = datetime.fromisoformat(channel_added_at.replace("Z", "+00:00"))

                if published_at > added_at:
                    known_videos[video["video_id"]] = {
                        "channel_id": channel["channel_id"],
                        "published_at": video["published_at"],
                        "status": "active"
                    }

    active_video_ids = [vid for vid, info in known_videos.items() if info["status"] == "active"]
    polled_ids = active_video_ids[:50]
    skipped_ids = active_video_ids[50:]

    for video_id in polled_ids:
        info = known_videos[video_id]
        if is_video_expired(info["published_at"]):
            info["status"] = "expired"
            info["tracked"] = False
        else:
            row = poll_video_stats(video_id, info["channel_id"])
            if row is None:
                print(f"Video {video_id} no longer available, marking as removed")
                info["status"] = "removed"
                info["tracked"] = False
            else:
                save_row(row)
                print("Saved:", row)
                info["tracked"] = True

    for video_id in skipped_ids:
        known_videos[video_id]["tracked"] = False

    save_json(KNOWN_VIDEOS_FILE, known_videos)

    if push:
        push_to_github()


if __name__ == "__main__":
    run_polling_cycle(push=True)