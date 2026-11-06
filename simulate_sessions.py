"""
simulate_sessions.py - FAKE session logs in the shared format, for testing the Catcher code only.

!!  Nothing produced here is a result.  !!
The "humans" are random numbers shaped like human behaviour, and the bots are stand-ins for the
Breaker's logs. Use this to check that the pipeline runs BEFORE real data exists. For the report,
use real human sessions (capture website) and the Breaker's real solver logs.

    python simulate_sessions.py                 # writes data/synthetic/human_sessions.csv + bot_sessions.csv
    python simulate_sessions.py --human 120 --seed 1

The page layout follows the shared spec (400 x 250 box): CAPTCHA image at x 120-280, y 40-100;
input box at x 120-280, y 130-160; submit button at x 150-250, y 180-210.

Three kinds of session:
  human    several simulated people with their own tempo / typing speed; mouse events every 8-16 ms
  naive    Breaker round 1: cursor teleports, keys typed by a script within milliseconds
  evasive  Breaker round 2: fake human timing, smooth curved mouse moves logged every 16 ms
"""

import argparse
import csv
import os

import numpy as np

CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
HEADER = ["session_id", "label", "t_ms", "event", "x", "y", "key"]


# ---------- movement and typing building blocks ----------

def _min_jerk(s):
    """Smooth 0 -> 1 profile: slow start, fast middle, slow end (how arm movements look)."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _move(rng, start, goal, t0, duration, dt_low, dt_high, tremor, arc):
    """Mouse points for one move: list of (t, x, y). dt_low == dt_high -> perfectly regular sampling."""
    times = [0.0]
    while times[-1] < duration:
        times.append(min(times[-1] + rng.uniform(dt_low, dt_high), duration))
    times = np.array(times)
    s = _min_jerk(times / duration)

    start, goal = np.asarray(start, float), np.asarray(goal, float)
    points = start + np.outer(s, goal - start)
    direction = goal - start
    perp = np.array([-direction[1], direction[0]]) / (np.linalg.norm(direction) + 1e-9)
    points += np.outer(arc * np.sin(np.pi * s), perp)        # slightly curved path
    points += rng.normal(0, tremor, points.shape)            # hand tremor / jitter
    return [(t0 + tt, px, py) for (px, py), tt in zip(points[1:], times[1:])]   # [1:] = skip the start point


def _type(rng, rows, first_down, text, gap_fn, dwell_fn, typo_rate=0.0):
    """Add key_down / key_up rows; returns time of the last key event."""
    keys = []
    for ch in text:
        if rng.random() < typo_rate:
            keys += [rng.choice(list(CHARSET)), "Backspace"]
        keys.append(ch)
    t_down, last = first_down, first_down
    for i, key in enumerate(keys):
        if i:
            t_down += gap_fn()
        t_up = t_down + dwell_fn()
        rows.append((t_down, "key_down", None, None, key))
        rows.append((t_up, "key_up", None, None, key))
        last = max(last, t_up)
    return last


def _point_in(rng, x0, x1, y0, y1):
    return (rng.uniform(x0, x1), rng.uniform(y0, y1))


def _input_point(rng):
    return _point_in(rng, 130, 270, 135, 155)       # inside the input box


def _submit_point(rng):
    return _point_in(rng, 160, 240, 185, 205)       # inside the submit button


def _random_text(rng):
    return "".join(rng.choice(list(CHARSET), 5))


# ---------- the three kinds of session (each returns rows: (t, event, x, y, key)) ----------

def _human_reach(rng, rows, pos, target, t, tempo):
    """Reach for a target in 1-3 sub-movements (people correct their aim). Returns (pos, time of arrival)."""
    n_parts = rng.choice([1, 2, 3], p=[0.30, 0.45, 0.25])
    for i in range(n_parts):
        last = i == n_parts - 1
        goal = np.asarray(target) + rng.normal(0, 3 if last else 30, 2)
        dist = np.linalg.norm(goal - np.asarray(pos))
        dur = max(rng.normal(220 + dist, 50) * tempo, 120)
        rows += [(pt, "mouse_move", px, py, "") for pt, px, py in
                 _move(rng, pos, goal, t, dur, 8, 16, tremor=0.8, arc=rng.normal(0, 10))]
        pos, t = goal, t + dur + (0 if last else rng.uniform(50, 300))
    return pos, t


def human_rows(rng, style):
    tempo = style["tempo"]
    start = (rng.uniform(0, 400), rng.uniform(150, 300))
    rows = [(0, "mouse_move", start[0], start[1], "")]
    t = rng.lognormal(np.log(400 * tempo), 0.4)                     # reaction time before the first move

    pos, t = _human_reach(rng, rows, start, _input_point(rng), t, tempo)
    t += rng.uniform(60, 160)
    rows.append((t, "click", pos[0], pos[1], ""))

    last = _type(rng, rows, t + rng.lognormal(np.log(550 * tempo), 0.4), _random_text(rng),
                 gap_fn=lambda: float(np.clip(rng.lognormal(np.log(style["key_mean"]), 0.35), 60, 900)),
                 dwell_fn=lambda: float(np.clip(rng.lognormal(np.log(95), 0.25), 40, 220)),
                 typo_rate=0.10)

    pos, t = _human_reach(rng, rows, pos, _submit_point(rng), last + rng.lognormal(np.log(450), 0.5), tempo)
    t += rng.uniform(60, 160)
    rows.append((t, "click", pos[0], pos[1], ""))
    rows.append((t + 1, "submit", None, None, ""))
    return rows


def naive_rows(rng):
    start = (rng.uniform(0, 400), rng.uniform(150, 300))
    rows = [(0, "mouse_move", start[0], start[1], "")]
    t = rng.uniform(10, 50)                                         # model inference time
    p = _input_point(rng)
    rows += [(t, "mouse_move", p[0], p[1], ""), (t + 1, "click", p[0], p[1], "")]
    last = _type(rng, rows, t + 2 + rng.integers(0, 3), _random_text(rng),
                 gap_fn=lambda: float(rng.integers(0, 3)), dwell_fn=lambda: float(rng.integers(0, 3)))
    p = _submit_point(rng)
    t = last + rng.integers(1, 4)
    rows += [(t, "mouse_move", p[0], p[1], ""), (t + 1, "click", p[0], p[1], ""), (t + 2, "submit", None, None, "")]
    return rows


def evasive_rows(rng):
    start = (rng.uniform(0, 400), rng.uniform(150, 300))
    rows = [(0, "mouse_move", start[0], start[1], "")]

    def curve(a, b, t0):
        pts = _move(rng, a, b, t0, rng.normal(650, 100), 16, 16, tremor=1.2, arc=rng.normal(0, 8))
        rows.extend((pt, "mouse_move", px, py, "") for pt, px, py in pts)
        return pts[-1][0]

    p = _input_point(rng)
    t = curve(start, p, rng.uniform(150, 450)) + rng.uniform(40, 120)
    rows.append((t, "click", p[0], p[1], ""))
    last = _type(rng, rows, t + rng.lognormal(np.log(600), 0.3), _random_text(rng),
                 gap_fn=lambda: float(max(rng.normal(190, 45), 70)),
                 dwell_fn=lambda: float(max(rng.normal(95, 20), 40)))
    q = _submit_point(rng)
    t = curve(p, q, last + rng.uniform(300, 700)) + rng.uniform(40, 120)
    rows += [(t, "click", q[0], q[1], ""), (t + 1, "submit", None, None, "")]
    return rows


# ---------- writing the shared-format CSV ----------

def _write_sessions(writer, session_id, label, rows):
    for t, event, x, y, key in sorted(rows, key=lambda r: round(r[0])):     # stable sort by whole ms
        writer.writerow([session_id, label, int(round(t)), event,
                         "" if x is None else int(round(x)), "" if y is None else int(round(y)), key])


def generate(out_dir="data/synthetic", n_human=90, n_naive=60, n_evasive=60, seed=0, n_people=6):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    people = {
        f"S{i + 1:02d}": {"tempo": rng.uniform(0.7, 1.4), "key_mean": rng.uniform(150, 280)}
        for i in range(n_people)
    }
    ids = list(people)

    with open(os.path.join(out_dir, "human_sessions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i in range(n_human):
            pid = ids[i % n_people]
            _write_sessions(w, f"h_{pid}_{i:04d}", "human", human_rows(rng, people[pid]))

    with open(os.path.join(out_dir, "bot_sessions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i in range(n_naive):
            _write_sessions(w, f"b_{i:03d}", "bot_naive", naive_rows(rng))
        for i in range(n_evasive):
            _write_sessions(w, f"b_{n_naive + i:03d}", "bot_evasive", evasive_rows(rng))
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate FAKE sessions (shared log format) for testing the pipeline.")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--human", type=int, default=90)
    ap.add_argument("--naive", type=int, default=60)
    ap.add_argument("--evasive", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = generate(args.out, args.human, args.naive, args.evasive, args.seed)
    print(f"Wrote human_sessions.csv and bot_sessions.csv to {out}/  (TEST DATA - not results)")
