import os
import csv
import json
import subprocess
import shutil
import glob
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
EXCLUDED_VIDEOS_FILE = "excluded_videos.json"
SUBSCRIBERS_FILE = "channel_subscribers.json"
MAX_TRACKED_CHANNELS = 15


def load_json(filename, default):
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return default


def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)


def setup_remote():
    token = os.getenv("GITHUB_TOKEN")
    remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"
    subprocess.run(["git", "remote", "remove", "origin"])
    subprocess.run(["git", "remote", "add", "origin", remote_url])
    subprocess.run(["git", "config", "--global", "user.email", "balghaith05@gmail.com"])
    subprocess.run(["git", "config", "--global", "user.name", "balghaith"])


def fetch_remote_json(filename, default):
    result = subprocess.run(
        ["git", "show", f"origin/main:{filename}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return default
    try:
        return json.loads(result.stdout)
    except Exception:
        return default


def merge_dicts_prefer_local(local, remote):
    merged = dict(remote)
    merged.update(local)
    return merged


def merge_channel_lists_prefer_local(local, remote):
    by_id = {c["channel_id"]: c for c in remote}
    by_id.update({c["channel_id"]: c for c in local})
    return list(by_id.values())


def sync_tracked_state():
    setup_remote()
    fetch_result = subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, text=True)
    print("FETCH:", fetch_result.returncode, fetch_result.stderr)

    remote_tracked = fetch_remote_json(TRACKED_CHANNELS_FILE, [])
    remote_known = fetch_remote_json(KNOWN_VIDEOS_FILE, {})
    remote_checkpoints = fetch_remote_json(CHECKPOINTS_FILE, {})

    local_tracked = load_json(TRACKED_CHANNELS_FILE, [])
    local_known = load_json(KNOWN_VIDEOS_FILE, {})
    local_checkpoints = load_json(CHECKPOINTS_FILE, {})

    merged_tracked = merge_channel_lists_prefer_local(local_tracked, remote_tracked)
    merged_known = merge_dicts_prefer_local(local_known, remote_known)
    merged_checkpoints = merge_dicts_prefer_local(local_checkpoints, remote_checkpoints)

    save_json(TRACKED_CHANNELS_FILE, merged_tracked)
    save_json(KNOWN_VIDEOS_FILE, merged_known)
    save_json(CHECKPOINTS_FILE, merged_checkpoints)


def cleanup_excluded_videos():
    known_videos = load_json(KNOWN_VIDEOS_FILE, {})
    checkpoints = load_json(CHECKPOINTS_FILE, {})
    excluded_videos = set(load_json(EXCLUDED_VIDEOS_FILE, []))

    removed_video_ids = [vid for vid in known_videos if vid in excluded_videos]

    for video_id in removed_video_ids:
        channel_id = known_videos[video_id]["channel_id"]
        del known_videos[video_id]
        if video_id in checkpoints:
            del checkpoints[video_id]

        video_file = os.path.join(DATA_DIR, channel_id, f"{video_id}.csv")
        if os.path.isfile(video_file):
            os.remove(video_file)
            print(f"Deleted CSV for excluded video {video_id}")

    if removed_video_ids:
        save_json(KNOWN_VIDEOS_FILE, known_videos)
        save_json(CHECKPOINTS_FILE, checkpoints)
        print(f"Removed {len(removed_video_ids)} excluded video entries from known_videos")


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


def fetch_channel_info(channel_id):
    request = youtube.channels().list(
        part="contentDetails,statistics",
        id=channel_id
    )
    response = request.execute()
    item = response["items"][0]
    playlist_id = item["contentDetails"]["relatedPlaylists"]["uploads"]
    stats = item.get("statistics", {})
    subscribers = None
    if not stats.get("hiddenSubscriberCount") and "subscriberCount" in stats:
        subscribers = int(stats["subscriberCount"])
    return playlist_id, subscribers


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


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def poll_video_stats_batch(video_ids):
    rows_by_id = {}
    titles_by_id = {}
    for batch in chunks(video_ids, 50):
        request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch)
        )
        try:
            response = request.execute()
        except Exception as e:
            print(f"Error fetching stats for batch {batch}: {e}")
            continue

        now = datetime.now(timezone.utc).isoformat()
        returned_ids = set()
        for item in response.get("items", []):
            stats = item["statistics"]
            rows_by_id[item["id"]] = {
                "video_id": item["id"],
                "timestamp": now,
                "views": stats.get("viewCount", 0),
                "likes": stats.get("likeCount", 0),
                "comments": stats.get("commentCount", 0)
            }
            title = item.get("snippet", {}).get("title", "")
            if title:
                titles_by_id[item["id"]] = title
            returned_ids.add(item["id"])

        for vid in batch:
            if vid not in returned_ids:
                rows_by_id[vid] = None

    return rows_by_id, titles_by_id


def save_row(channel_id, row):
    channel_folder = os.path.join(DATA_DIR, channel_id)
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


def sort_all_data_files():
    for channel_folder in glob.glob(os.path.join(DATA_DIR, "*")):
        if not os.path.isdir(channel_folder):
            continue
        for video_file in glob.glob(os.path.join(channel_folder, "*.csv")):
            with open(video_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames

            if not rows:
                continue

            seen = set()
            unique_rows = []
            for row in rows:
                key = tuple(row.items())
                if key not in seen:
                    seen.add(key)
                    unique_rows.append(row)

            unique_rows.sort(key=lambda r: r["timestamp"])

            with open(video_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(unique_rows)


def is_video_expired(published_at_str, max_days=30):
    published_at = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
    age_days = (datetime.now(timezone.utc) - published_at).days
    return age_days > max_days


def parse_iso_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def published_after_tracking_started(published_at, added_at):
    if not added_at or not published_at:
        return True
    return parse_iso_time(published_at) >= parse_iso_time(added_at)


def push_to_github():
    subprocess.run(["git", "config", "--global", "user.email", "balghaith05@gmail.com"])
    subprocess.run(["git", "config", "--global", "user.name", "balghaith"])
    subprocess.run(["git", "add", DATA_DIR, TRACKED_CHANNELS_FILE, KNOWN_VIDEOS_FILE, CHECKPOINTS_FILE, SUBSCRIBERS_FILE])
    subprocess.run(["git", "commit", "-m", "Update tracked data"])

    token = os.getenv("GITHUB_TOKEN")
    remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"
    subprocess.run(["git", "remote", "remove", "origin"])
    subprocess.run(["git", "remote", "add", "origin", remote_url])

    fetch_result = subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, text=True)
    print("FETCH:", fetch_result.returncode, fetch_result.stderr)

    subprocess.run(["git", "config", "merge.ours.driver", "true"])

    merge_result = subprocess.run(
        ["git", "merge", "origin/main", "--no-edit"],
        capture_output=True, text=True
    )
    print("MERGE:", merge_result.returncode, merge_result.stdout, merge_result.stderr)

    sort_all_data_files()
    amend_result = subprocess.run(
        ["git", "commit", "-a", "--amend", "--no-edit"],
        capture_output=True, text=True
    )
    print("RESORT COMMIT:", amend_result.returncode, amend_result.stdout, amend_result.stderr)

    push_result = subprocess.run(["git", "push", "origin", "HEAD:main"], capture_output=True, text=True)
    print("PUSH:", push_result.returncode, push_result.stdout, push_result.stderr)


def run_polling_cycle(push=True):
    sync_tracked_state()

    cleanup_untracked_channels()
    cleanup_excluded_videos()

    tracked_channels = load_json(TRACKED_CHANNELS_FILE, [])

    if len(tracked_channels) > MAX_TRACKED_CHANNELS:
        tracked_channels = tracked_channels[:MAX_TRACKED_CHANNELS]

    known_videos = load_json(KNOWN_VIDEOS_FILE, {})
    excluded_videos = set(load_json(EXCLUDED_VIDEOS_FILE, []))

    subscribers_by_channel = load_json(SUBSCRIBERS_FILE, {})

    for channel in tracked_channels:
        playlist_id, subscriber_count = fetch_channel_info(channel["channel_id"])
        subscribers_by_channel[channel["channel_id"]] = {
            "subscribers": subscriber_count,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        latest = latest_videos(playlist_id, max_results=50)

        for video in latest:
            if (
                video["video_id"] not in known_videos
                and video["video_id"] not in excluded_videos
                and published_after_tracking_started(video["published_at"], channel.get("added_at"))
            ):
                known_videos[video["video_id"]] = {
                    "channel_id": channel["channel_id"],
                    "published_at": video["published_at"],
                    "status": "active",
                    "title": video.get("title", "")
                }

    for video_id, info in known_videos.items():
        if info["status"] == "active" and is_video_expired(info["published_at"]):
            info["status"] = "expired"
            info["tracked"] = False

    added_by_channel = {c["channel_id"]: c.get("added_at") for c in tracked_channels}
    active_video_ids = [
        vid for vid, info in known_videos.items()
        if info["status"] == "active"
        and published_after_tracking_started(info["published_at"], added_by_channel.get(info["channel_id"]))
    ]

    rows_by_id, titles_by_id = poll_video_stats_batch(active_video_ids)

    for video_id in active_video_ids:
        info = known_videos[video_id]
        if not info.get("title") and video_id in titles_by_id:
            info["title"] = titles_by_id[video_id]

        row = rows_by_id.get(video_id)
        if row is None:
            print(f"Video {video_id} no longer available, marking as removed")
            info["status"] = "removed"
            info["tracked"] = False
        else:
            save_row(info["channel_id"], row)
            print("Saved:", row)
            info["tracked"] = True

    save_json(KNOWN_VIDEOS_FILE, known_videos)
    save_json(SUBSCRIBERS_FILE, subscribers_by_channel)

    sort_all_data_files()

    if push:
        push_to_github()


if __name__ == "__main__":
    run_polling_cycle(push=True)