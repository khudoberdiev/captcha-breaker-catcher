"""
features.py - read the shared event log and turn each session into one row of numbers.

The log format is shared by Half A (Breaker) and Half B (Catcher):
one row = one event, one session = all rows with the same session_id.

    session_id,label,t_ms,event,x,y,key
    h_OT_3fa9c2,human,0,mouse_move,30,220,
    h_OT_3fa9c2,human,1820,click,175,146,
    h_OT_3fa9c2,human,2310,key_down,,,K
    h_OT_3fa9c2,human,2405,key_up,,,K
    h_OT_3fa9c2,human,5120,click,201,195,
    h_OT_3fa9c2,human,5121,submit,,,

label: human | bot_naive | bot_evasive.   t_ms: ms since the CAPTCHA appeared.
x, y: px relative to the top-left corner of the CAPTCHA box.

One convention on top of the format: human session ids look like  h_<initials>_<anything>,
so we know WHO solved it and can keep each person entirely on one side of the train/test split.

Every feature is computed here, from the raw events, for humans and bots alike.
"""

import csv
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["session_id", "label", "t_ms", "event", "x", "y", "key"]
LABELS = {"human": "human", "bot_naive": "naive", "bot_evasive": "evasive"}   # label -> kind
EVENTS = {"mouse_move", "click", "key_down", "key_up", "submit"}

# The 9 simple features for round 1 (timing + rough mouse statistics).
BASIC_FEATURES = [
    "time_to_first_mouse_move_ms",
    "time_to_first_keystroke_ms",
    "total_solve_time_ms",
    "num_keystrokes",
    "keystroke_interval_mean_ms",
    "keystroke_interval_variance",
    "num_mouse_samples",
    "mouse_path_length_px",
    "mouse_direction_changes",
]

# Harder-to-fake features for round 3: HOW the hand moves and types, not just how long it takes.
ADVANCED_FEATURES = [
    "speed_mean_px_per_ms",
    "speed_std_px_per_ms",
    "acceleration_std",
    "straightness",              # straight-line distance / path length (1 = ruler-straight)
    "turn_angle_mean_abs",       # average bend between consecutive mouse steps
    "turn_angle_std",
    "pause_count",               # gaps > 100 ms inside the mouse path
    "sample_dt_std_ms",          # irregularity of the time between mouse events inside a movement (see README: logging artifact!)
    "speed_peaks",               # local peaks in the lightly smoothed speed curve (a noisy shape descriptor)
    "keystroke_interval_cv",     # typing rhythm: std / mean of the gaps between key presses
    "keystroke_interval_min_ms",
    "key_dwell_mean_ms",         # how long a key is held down (key_up - key_down)
    "key_dwell_std_ms",
    "num_backspaces",            # typing corrections
    "num_clicks",
    "time_to_first_click_ms",
    "first_click_to_first_key_ms",
    "last_key_to_submit_ms",     # pause between the last key and pressing submit
]

ALL_FEATURES = BASIC_FEATURES + ADVANCED_FEATURES

_HUMAN_ID = re.compile(r"^h_([A-Za-z0-9]+)_[A-Za-z0-9]+$")


def _wrap(angle_diff):
    """Wrap angle differences into [-pi, pi] so 179 degrees -> -181 degrees is a small turn."""
    return (angle_diff + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- reading the log

def build_session(session_id, rows):
    """Rows of one session -> (session dict, None), or (None, reason) if the session is unusable."""
    labels = {r["label"] for r in rows}
    if len(labels) != 1:
        return None, "session has more than one label"
    label = labels.pop()
    if label not in LABELS:
        return None, f"unknown label '{label}'"

    parsed = []
    for r in rows:
        if r["event"] not in EVENTS:
            return None, f"unknown event '{r['event']}'"
        try:
            t = float(r["t_ms"])
            x = float(r["x"]) if r["x"] != "" else None
            y = float(r["y"]) if r["y"] != "" else None
        except ValueError:
            return None, "non-numeric t_ms / x / y"
        if r["event"] in ("mouse_move", "click") and (x is None or y is None):
            return None, "mouse_move / click row without x, y"
        parsed.append((t, r["event"], x, y, r["key"]))
    parsed.sort(key=lambda p: p[0])                      # stable: equal times keep file order

    submits = [p for p in parsed if p[1] == "submit"]
    if not submits:
        return None, "no submit row"

    kind = LABELS[label]
    match = _HUMAN_ID.match(session_id)
    participant = (match.group(1) if match else "unknown") if kind == "human" else "breaker"
    return {
        "session_id": session_id,
        "label": "human" if kind == "human" else "bot",
        "kind": kind,
        "participant": participant,
        "total_solve_time_ms": submits[-1][0],
        "raw_mouse_path": [{"x": x, "y": y, "t": t} for t, e, x, y, _ in parsed if e == "mouse_move"],
        "raw_clicks": [{"x": x, "y": y, "t": t} for t, e, x, y, _ in parsed if e == "click"],
        "raw_key_events": [{"t": t, "event": e, "key": k} for t, e, _, _, k in parsed if e in ("key_down", "key_up")],
    }, None


def read_event_csv(path):
    """CSV file -> {session_id: [row, ...]} (rows as dicts, in file order)."""
    sessions = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; expected {REQUIRED_COLUMNS}")
        for row in reader:
            sessions.setdefault(row["session_id"], []).append(row)
    return sessions


# ---------------------------------------------------------------- features

def extract_features(session):
    """One session dict -> dict of feature name -> number."""
    path = session["raw_mouse_path"]
    clicks = session["raw_clicks"]
    key_events = session["raw_key_events"]
    total_ms = float(session["total_solve_time_ms"])

    t = np.array([p["t"] for p in path], dtype=float)
    x = np.array([p["x"] for p in path], dtype=float)
    y = np.array([p["y"] for p in path], dtype=float)
    downs = [e for e in key_events if e["event"] == "key_down"]
    kt = np.array([e["t"] for e in downs], dtype=float)

    f = {name: 0.0 for name in ALL_FEATURES}
    f["total_solve_time_ms"] = total_ms
    f["num_mouse_samples"] = len(path)
    f["num_keystrokes"] = len(downs)
    f["num_backspaces"] = sum(1 for e in downs if e["key"] == "Backspace")
    f["num_clicks"] = len(clicks)

    # ---------- keyboard ----------
    if len(kt):
        f["time_to_first_keystroke_ms"] = kt[0]
        f["last_key_to_submit_ms"] = max(total_ms - kt[-1], 0.0)
    if len(kt) >= 2:
        gaps = np.diff(kt)
        f["keystroke_interval_mean_ms"] = gaps.mean()
        f["keystroke_interval_variance"] = gaps.var()
        f["keystroke_interval_cv"] = gaps.std() / gaps.mean() if gaps.mean() > 0 else 0.0
        f["keystroke_interval_min_ms"] = gaps.min()

    pending, dwell = {}, []                          # match each key_up to its key_down
    for e in key_events:
        if e["event"] == "key_down":
            pending.setdefault(e["key"], []).append(e["t"])
        elif pending.get(e["key"]):
            dwell.append(e["t"] - pending[e["key"]].pop(0))
    if dwell:
        f["key_dwell_mean_ms"] = float(np.mean(dwell))
        f["key_dwell_std_ms"] = float(np.std(dwell))

    # ---------- clicks ----------
    if clicks:
        f["time_to_first_click_ms"] = clicks[0]["t"]
        if len(kt) and kt[0] > clicks[0]["t"]:
            f["first_click_to_first_key_ms"] = kt[0] - clicks[0]["t"]

    # ---------- mouse ----------
    if len(path) >= 2:
        moved = np.flatnonzero((x != x[0]) | (y != y[0]))    # row 0 is just the starting position
        if len(moved):
            f["time_to_first_mouse_move_ms"] = t[moved[0]]

        dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
        step = np.hypot(dx, dy)                       # pixels moved in each step
        path_len = step.sum()
        f["mouse_path_length_px"] = path_len
        in_burst = dt[dt <= 100]                      # time between samples while the mouse is moving
        f["sample_dt_std_ms"] = in_burst.std() if len(in_burst) else 0.0
        f["pause_count"] = int((dt > 100).sum())

        speed = step / np.maximum(dt, 1.0)            # px per ms (timestamps are whole ms)
        f["speed_mean_px_per_ms"] = speed.mean()
        f["speed_std_px_per_ms"] = speed.std()
        if len(speed) >= 2:
            acc = np.diff(speed) / np.maximum(dt[1:], 1.0)
            f["acceleration_std"] = acc.std()

        chord = np.hypot(x[-1] - x[0], y[-1] - y[0])
        f["straightness"] = chord / path_len if path_len > 0 else 0.0

        moving = step > 0                             # direction is undefined when not moving
        angles = np.arctan2(dy[moving], dx[moving])
        if len(angles) >= 2:
            turns = np.abs(_wrap(np.diff(angles)))
            f["mouse_direction_changes"] = int((turns > 0.5).sum())
            f["turn_angle_mean_abs"] = turns.mean()
            f["turn_angle_std"] = turns.std()

        if len(speed) >= 5:                           # count local peaks in the speed curve
            smooth = np.convolve(speed, np.ones(3) / 3, mode="same")
            floor = 0.3 * smooth.max()
            core = smooth[1:-1]
            peaks = (core > smooth[:-2]) & (core >= smooth[2:]) & (core > floor)
            f["speed_peaks"] = int(peaks.sum())

    return f


def load_sessions(csv_paths):
    """Read the shared-format CSV files into one DataFrame, one row per session.

    Extra columns: session_id, kind ("human" / "naive" / "evasive"), is_bot (0/1),
    participant (humans: initials from the session id; used to split train/test by person).
    Unusable sessions are skipped and counted, never silently.
    """
    rows, problems = [], Counter()
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"File not found (skipped): {path}")
            continue
        for session_id, event_rows in read_event_csv(path).items():
            session, problem = build_session(session_id, event_rows)
            if problem:
                problems[problem] += 1
                continue
            row = extract_features(session)
            row.update(
                session_id=session_id,
                kind=session["kind"],
                is_bot=int(session["label"] == "bot"),
                participant=session["participant"],
            )
            rows.append(row)
    if problems:
        print("Skipped sessions:", dict(problems))
    return pd.DataFrame(rows)
