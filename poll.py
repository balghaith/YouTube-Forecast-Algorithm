import os
import csv
import subprocess
from datetime import datetime, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=api_key)

CSV_FILE = "video_stats.csv"

def poll_video_stats(video_id):
    request = youtube.videos().list(
        part="statistics",
        id=video_id
    )
    response = request.execute()
    stats = response["items"][0]["statistics"]

    row = {
        "video_id": video_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "views": stats.get("viewCount", 0),
        "likes": stats.get("likeCount", 0),
        "comments": stats.get("commentCount", 0)
    }
    return row

def save_row(row):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "timestamp", "views", "likes", "comments"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def push_to_github():
    subprocess.run(["git", "config", "--global", "user.email", "balghaith05@gmail.com"])
    subprocess.run(["git", "config", "--global", "user.name", "balghaith"])
    subprocess.run(["git", "add", CSV_FILE])
    subprocess.run(["git", "commit", "-m", "Update video_stats.csv"])

    token = os.getenv("GITHUB_TOKEN")
    remote_url = f"https://{token}@github.com/balghaith/YouTube-Forecast-Algorithm.git"
    subprocess.run(["git", "remote", "remove", "origin"])
    subprocess.run(["git", "remote", "add", "origin", remote_url])
    subprocess.run(["git", "push", "origin", "HEAD:main"])

if __name__ == "__main__":
    video_id = "EUlE7SY-51s"
    row = poll_video_stats(video_id)
    save_row(row)
    print("Saved:", row)
    push_to_github()