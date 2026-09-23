import os
import json

CHECKPOINTS_FILE = "forecast_checkpoints.json"
HORIZONS_DAYS = list(range(1, 31))

BLOCK_1 = list(range(8, 15))
BLOCK_2 = list(range(15, 31))


def load_checkpoints():
    if not os.path.isfile(CHECKPOINTS_FILE):
        return {}
    with open(CHECKPOINTS_FILE, "r") as f:
        return json.load(f)


def save_checkpoints(data):
    with open(CHECKPOINTS_FILE, "w") as f:
        json.dump(data, f)


def get_video_checkpoints(all_checkpoints, video_id, metric):
    all_checkpoints.setdefault(video_id, {})
    all_checkpoints[video_id].setdefault(metric, {})
    return all_checkpoints[video_id][metric]


def interpolate_actual_at_hour(x_hours, y, target_hour):
    if target_hour <= x_hours[0]:
        return y[0]
    if target_hour >= x_hours[-1]:
        return y[-1]
    for i in range(1, len(x_hours)):
        if x_hours[i] >= target_hour:
            x0, x1 = x_hours[i - 1], x_hours[i]
            y0, y1 = y[i - 1], y[i]
            if x1 == x0:
                return y1
            ratio = (target_hour - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return y[-1]


def update_checkpoints_for_video(video_id, metric, x_hours, y, predict_fn):
    all_checkpoints = load_checkpoints()
    vid_checkpoints = get_video_checkpoints(all_checkpoints, video_id, metric)
    current_hours = x_hours[-1]
    current_days = current_hours / 24
    changed = True

    for days in HORIZONS_DAYS:
        key = f"day_{days}"
        horizon_hours = days * 24
        day_passed = current_days >= horizon_hours / 24

        if days <= 7:
            vid_checkpoints[key] = {
                "day": days,
                "locked": day_passed,
                "predicted": None,
                "actual": round(interpolate_actual_at_hour(x_hours, y, horizon_hours)) if day_passed else None
            }
            continue

        already_exists = key in vid_checkpoints
        was_locked = already_exists and vid_checkpoints[key].get("locked")

        if was_locked:
            continue

        if days in BLOCK_1:
            block_locked = current_days >= 7
        else:
            block_locked = current_days >= 15

        if block_locked:
            actual_value = round(interpolate_actual_at_hour(x_hours, y, horizon_hours)) if day_passed else None

            if already_exists and vid_checkpoints[key].get("predicted") is not None:
                predicted = vid_checkpoints[key]["predicted"]
            else:
                predicted = round(predict_fn(horizon_hours))

            vid_checkpoints[key] = {
                "day": days,
                "locked": True,
                "predicted": predicted,
                "actual": actual_value
            }
        else:
            vid_checkpoints[key] = {
                "day": days,
                "locked": False,
                "predicted": round(predict_fn(horizon_hours)),
                "actual": None
            }

    save_checkpoints(all_checkpoints)
    return vid_checkpoints