# half_b_catcher: human or bot?

The Breaker (`half_a_breaker/`) solves CAPTCHAs with a CNN. The Catcher looks at **how** a CAPTCHA was solved
(timing, typing rhythm, mouse movement) and decides whether a human or a bot did it.

```
half_b_catcher/
  capture/                small website that records people solving CAPTCHAs
    app.py                  the server behind the page
    static/index.html       the page itself (the 400 x 250 box from the log spec)
    generate_captchas.py    makes the CAPTCHA images (same style as the Breaker's)
  data/
    human_sessions.csv      written by the website (shared log format)
  features.py             log -> sessions -> numbers (same code for humans and bots)
  simulate_sessions.py    FAKE sessions for testing the code before real data exists
  catcher.ipynb           the experiment: 3 rounds, 4 rows, 3 models
  results/                tables and charts for the report
```

Bot logs are written by the Breaker to `half_a_breaker/logs/bot_sessions.csv`; the notebook reads them from there.

## Quick start

```
pip install -r requirements.txt
python capture/generate_captchas.py     # once
python capture/app.py                   # then open http://localhost:5000
```

Classmates on the same Wi-Fi open `http://<your-ip>:5000` (`ifconfig` on Mac, `ipconfig` on Windows).
The server is open to everyone on that network, so use a Wi-Fi you trust and stop it when you are done.

**Collecting human data:** laptop or desktop with a mouse or trackpad (phones have no mouse path and would look like
bots). Each person types their initials once, then solves about 15 CAPTCHAs. Aim for 100+ sessions from as many
different people as possible. Rows are appended to `data/human_sessions.csv`.

**Running the experiment:** `jupyter notebook catcher.ipynb`. It starts with `USE_SYNTHETIC = True`, which uses **fake**
sessions just to prove the code works. Change it to `False` to use the two real CSV files.

## Log format (shared by Half A and Half B)

One row = one event. One session = all rows with the same `session_id`.

| column | meaning | example |
|---|---|---|
| `session_id` | unique id per solve | `h_OT_3fa9c2`, `b_042` |
| `label` | `human` / `bot_naive` / `bot_evasive` | `human` |
| `t_ms` | milliseconds since the CAPTCHA appeared (int) | `1450` |
| `event` | `mouse_move` / `click` / `key_down` / `key_up` / `submit` | `key_down` |
| `x`, `y` | mouse position in px (empty for key and submit rows) | `182`, `145` |
| `key` | typed character or `Backspace` (key rows only) | `K` |

Rules:

- `t_ms` starts at 0 when the CAPTCHA is shown.
- `x`, `y` are relative to the top-left corner of the CAPTCHA box, not the screen.
- The first row of each session is a `mouse_move` with the starting mouse position. The last row is `submit`.
- `mouse_move`: browsers fire it about every 8-16 ms; bots should log at ~16 ms intervals.
- Every key press has a `key_down` row and a matching `key_up` row (same key).
- `click` is recorded on mouse button press, with the x, y where it happened.

Page layout (px, inside a 400 x 250 box): CAPTCHA image x 120-280, y 40-100; input box x 120-280, y 130-160;
submit button x 150-250, y 180-210. The capture website uses exactly this layout, so humans and bots live in the
same coordinate frame.

```
session_id,label,t_ms,event,x,y,key
h_OT_3fa9c2,human,0,mouse_move,30,220,
h_OT_3fa9c2,human,16,mouse_move,34,216,
h_OT_3fa9c2,human,1820,click,175,146,
h_OT_3fa9c2,human,2310,key_down,,,K
h_OT_3fa9c2,human,2405,key_up,,,K
h_OT_3fa9c2,human,5120,click,201,195,
h_OT_3fa9c2,human,5121,submit,,,
```

One convention added by the Catcher: human session ids are `h_<initials>_<anything>` (the website does this). The notebook
reads the person from the id, so every person stays entirely on one side of the train/test split. Bot ids can be anything.

The website only stores letters, digits and `Backspace` as keys (anything else is saved as `other`) and ignores Enter:
people must click Submit, like the bots do.

## Honest limits

- A few people and one CAPTCHA style: say so in the report. Test people are never in training.
- **Logging rhythm trap.** Browsers report the mouse every 8-16 ms. If the bot logs are exactly 16 ms apart, the Catcher can
  win by spotting that rhythm, which says nothing about behaviour. Ask the Breaker to log bots at irregular 8-16 ms steps,
  or set `EXCLUDE = ["sample_dt_std_ms"]` in the notebook and report both.
- The evasive bot is only as smart as the evasion layer we wrote; a better one could beat the final round.
- `data/human_sessions.csv` records how real people use a mouse. It is git-ignored by default; commit it only with the participants' OK.
