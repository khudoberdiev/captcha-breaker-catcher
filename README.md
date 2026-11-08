# Breaker vs. Catcher

A classroom AI arms race over CAPTCHAs.

- **Half A · Breaker:** a PyTorch CNN reads the CAPTCHA, then a bot types the answer like a person.
- **Half B · Catcher:** a Random Forest decides who typed it: human or bot.

Everything runs on our own generated CAPTCHAs and our own catcher, built for a class. It is never pointed at real websites.

## Results

| | |
|---|---|
| Characters read correctly | **95.1%** |
| CAPTCHAs fully solved | **80.3%** |
| Bots caught (final round) | **99.9–100%** |
| Humans wrongly flagged (final round) | **0%** |

Tested on 2,000 unseen CAPTCHAs (Half A) and person-split human sessions (Half B).

## Structure

```
half_a_breaker/    breaker.ipynb · presentation/ (Manim videos)
half_b_catcher/    catcher.ipynb · features.py · capture/
LOG_FORMAT.md      shared session log format
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

1. `half_a_breaker/breaker.ipynb`: Run All
2. `half_b_catcher/catcher.ipynb`: Run All

## Videos (optional)

```bash
brew install cairo pkg-config ffmpeg
cd half_a_breaker/presentation
manim -qh -a act0.py && manim -qh -a scenes.py && manim -qh -a act3.py
```

## Privacy

Human session data stays local (`data/` is git-ignored).

## Team

Ulmasbek (Half A) · Oybek (Half B)