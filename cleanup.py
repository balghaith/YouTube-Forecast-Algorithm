import os, json
from datetime import datetime, timezone

EXCLUDED_FILE = 'excluded_videos.json'

known_videos = json.load(open('known_videos.json'))
checkpoints = json.load(open('forecast_checkpoints.json'))
excluded = set(json.load(open(EXCLUDED_FILE))) if os.path.isfile(EXCLUDED_FILE) else set()

now = datetime.now(timezone.utc)
to_delete = [vid for vid, info in known_videos.items() if (now - datetime.fromisoformat(info['published_at'].replace('Z','+00:00'))).total_seconds() > 86400]

for vid in to_delete:
    cid = known_videos[vid]['channel_id']
    path = os.path.join('data', cid, f'{vid}.csv')
    if os.path.isfile(path): os.remove(path)
    del known_videos[vid]
    checkpoints.pop(vid, None)
    excluded.add(vid)

json.dump(known_videos, open('known_videos.json','w'))
json.dump(checkpoints, open('forecast_checkpoints.json','w'))
json.dump(list(excluded), open(EXCLUDED_FILE,'w'))
print(f'Deleted {len(to_delete)} videos')