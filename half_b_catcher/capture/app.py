"""
Capture website (server side): shows a CAPTCHA in the browser, records HOW a person solves it,
and appends the events to data/human_sessions.csv in the shared log format.
(This file is the small server behind the page; people just open the link in a browser.)

Run (from the half_b_catcher/ folder):
    python capture/generate_captchas.py      # once, makes the CAPTCHA images
    python capture/app.py                    # then open http://localhost:5000

Classmates on the same Wi-Fi open  http://<your-ip>:5000  (ifconfig / ipconfig shows it).
The server listens on the whole network, so only run it on a Wi-Fi you trust and stop it
when you are done collecting.

Only human sessions are accepted here. The Breaker writes its own bot_sessions.csv.
Session ids look like  h_<initials>_<random>, so the Catcher can keep each person on one side of the
train/test split.
"""

import csv
import os
import random
import re
import threading
import uuid

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CAPTCHA_DIR = os.path.join(STATIC_DIR, "captchas")
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
HUMAN_CSV = os.path.join(DATA_DIR, "human_sessions.csv")
HEADER = ["session_id", "label", "t_ms", "event", "x", "y", "key"]
EVENTS = {"mouse_move", "click", "key_down", "key_up", "submit"}
MAX_EVENTS = 20000                                     # ~ 5 minutes of mouse movement; far above a normal solve

for folder in (CAPTCHA_DIR, DATA_DIR):
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024    # refuse anything huge
write_lock = threading.Lock()                           # several people may submit at the same moment


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


@app.route("/api/next-captcha")
def next_captcha():
    """A random CAPTCHA. The file name (without .png) is the correct answer."""
    files = [f for f in os.listdir(CAPTCHA_DIR) if f.lower().endswith(".png")]
    if not files:
        return jsonify({
            "image_url": None,
            "answer": None,
            "message": "No CAPTCHA images yet. Run: python capture/generate_captchas.py",
        })
    chosen = random.choice(files)
    return jsonify({"image_url": f"/static/captchas/{chosen}", "answer": os.path.splitext(chosen)[0]})


def _clean_events(events):
    """Validate the events from the browser; return CSV rows (without session_id/label) or an error string."""
    if not isinstance(events, list) or not 2 <= len(events) <= MAX_EVENTS:
        return f"events must be a list of 2 to {MAX_EVENTS} items"
    rows = []
    for e in events:
        if not isinstance(e, dict) or e.get("event") not in EVENTS:
            return "bad event"
        t = e.get("t_ms")
        if not isinstance(t, int) or isinstance(t, bool) or not 0 <= t <= 3_600_000:
            return "bad t_ms"
        x = y = ""
        if e["event"] in ("mouse_move", "click"):
            x, y = e.get("x"), e.get("y")
            if not all(isinstance(v, int) and not isinstance(v, bool) and -10000 <= v <= 10000 for v in (x, y)):
                return "bad x / y"
        key = ""
        if e["event"] in ("key_down", "key_up"):
            raw = str(e.get("key", ""))
            key = raw if raw == "Backspace" or re.fullmatch(r"[A-Za-z0-9]", raw) else "other"   # never store free text
        rows.append((t, e["event"], x, y, key))
    rows.sort(key=lambda r: r[0])                      # stable: equal times keep the order they happened in
    if rows[0][1] != "mouse_move" or rows[-1][1] != "submit":
        return "session must start with mouse_move and end with submit"
    return rows


@app.route("/api/log-session", methods=["POST"])
def log_session():
    """Save one human solve session (see features.py for the format)."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "body must be a JSON object"}), 400

    participant = re.sub(r"[^A-Za-z0-9]", "", str(data.get("participant", "")))[:12]
    if not participant:
        return jsonify({"status": "error", "message": "participant initials are required"}), 400

    rows = _clean_events(data.get("events"))
    if isinstance(rows, str):
        return jsonify({"status": "error", "message": rows}), 400

    session_id = f"h_{participant}_{uuid.uuid4().hex[:6]}"
    with write_lock:
        is_new = not os.path.exists(HUMAN_CSV) or os.path.getsize(HUMAN_CSV) == 0
        with open(HUMAN_CSV, "a", newline="") as fh:
            writer = csv.writer(fh)
            if is_new:
                writer.writerow(HEADER)
            for t, event, x, y, key in rows:
                writer.writerow([session_id, "human", t, event, x, y, key])
    return jsonify({"status": "ok", "session_id": session_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
