import json
from poll import youtube

with open("known_videos.json") as f:
    known_videos = json.load(f)

sorted_videos = sorted(
    known_videos.items(),
    key=lambda item: item[1]["published_at"],
    reverse=True
)

for video_id, info in sorted_videos:
    try:
        response = youtube.videos().list(part="snippet", id=video_id).execute()
        title = response["items"][0]["snippet"]["title"] if response["items"] else "(unavailable)"
    except Exception:
        title = "(error fetching title)"
    print(info["published_at"], video_id, info["status"], "-", title)