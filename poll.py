import os
import csv
import json
import subprocess
from datetime import datetime, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=api_key)

DATA_DIR = "data"
TRACKED_CHANNELS_FILE = "tracked_channels.json"
KNOWN_VIDEOS_FILE = "known_videos.json"
MAX_TRACKED_CHANNELS = 15


def load_json(filename, default):
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return default


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)


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
    response = request.execute()
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
    subprocess.run(["git", "add", DATA_DIR, TRACKED_CHANNELS_FILE, KNOWN_VIDEOS_FILE])
    subprocess.run(["git", "commit", "-m", "Update tracked data"])

    token = os.getenv("GITHUB_TOKEN")
    remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"
    subprocess.run(["git", "remote", "remove", "origin"])
    subprocess.run(["git", "remote", "add", "origin", remote_url])

    subprocess.run(["git", "fetch", "origin", "main"])
    subprocess.run(["git", "rebase", "origin/main"])
    subprocess.run(["git", "push", "origin", "HEAD:main"])


def run_polling_cycle(push=True):
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

    for video_id, info in known_videos.items():
        if info["status"] == "active":
            if is_video_expired(info["published_at"]):
                info["status"] = "expired"
            else:
                row = poll_video_stats(video_id, info["channel_id"])
                save_row(row)
                print("Saved:", row)

    save_json(KNOWN_VIDEOS_FILE, known_videos)

    if push:
        push_to_github()


if __name__ == "__main__":
    run_polling_cycle(push=False)