import json
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

TRACKED_CHANNELS_FILE = "tracked_channels.json"

def main():
    api_key = os.environ["YOUTUBE_API_KEY"]
    youtube = build("youtube", "v3", developerKey=api_key)

    with open(TRACKED_CHANNELS_FILE, "r") as f:
        channels = json.load(f)

    changed = False
    for channel in channels:
        if channel.get("thumbnail"):
            continue
        resp = youtube.channels().list(part="snippet", id=channel["channel_id"]).execute()
        items = resp.get("items", [])
        if not items:
            continue
        thumb = items[0]["snippet"]["thumbnails"]["default"]["url"]
        channel["thumbnail"] = thumb
        changed = True
        print(f"Set thumbnail for {channel['title']}")

    if changed:
        with open(TRACKED_CHANNELS_FILE, "w") as f:
            json.dump(channels, f, indent=2)
    else:
        print("Nothing to update")

if __name__ == "__main__":
    main()