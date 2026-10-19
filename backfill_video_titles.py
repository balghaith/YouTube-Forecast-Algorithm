import json
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

KNOWN_VIDEOS_FILE = "known_videos.json"

def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def main():
    api_key = os.environ["YOUTUBE_API_KEY"]
    youtube = build("youtube", "v3", developerKey=api_key)

    with open(KNOWN_VIDEOS_FILE, "r") as f:
        known_videos = json.load(f)

    missing_ids = [vid for vid, info in known_videos.items() if not info.get("title")]
    print(f"{len(missing_ids)} videos missing a title")

    for batch in chunks(missing_ids, 50):
        resp = youtube.videos().list(part="snippet", id=",".join(batch)).execute()
        for item in resp.get("items", []):
            known_videos[item["id"]]["title"] = item["snippet"]["title"]

    with open(KNOWN_VIDEOS_FILE, "w") as f:
        json.dump(known_videos, f, indent=2)

    print("Done")

if __name__ == "__main__":
    main()