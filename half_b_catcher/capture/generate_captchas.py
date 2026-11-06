"""
Makes CAPTCHA images for the capture app, named <ANSWER>.png (the app reads the answer from the name).

Uses the same settings as the Breaker's notebook (160x60, the same character set, 5 characters),
so people solve exactly what the solver solves.

    pip install captcha pillow
    python capture/generate_captchas.py            # 60 images
    python capture/generate_captchas.py --n 200
"""

import argparse
import os
import random

CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"   # same as breaker.ipynb (no 0, 1, I, O)
LENGTH = 5
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "captchas")

try:
    from captcha.image import ImageCaptcha
    _generator = ImageCaptcha(width=160, height=60)

    def make_image(text):
        return _generator.generate_image(text)

except ImportError:
    print("WARNING: package 'captcha' not found (pip install captcha). "
          "Using a simplified look-alike, fine for a quick test but not the Breaker's style.")
    from PIL import Image, ImageDraw, ImageFont

    def make_image(text):
        img = Image.new("RGB", (160, 60), "white")
        draw = ImageDraw.Draw(img)
        for _ in range(5):
            draw.line([(random.randint(0, 160), random.randint(0, 60)),
                       (random.randint(0, 160), random.randint(0, 60))], fill=(150, 150, 150))
        font = ImageFont.load_default()
        for i, ch in enumerate(text):
            draw.text((12 + i * 28, random.randint(18, 30)), ch, fill=(20, 20, 20), font=font)
        return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    made = 0
    while made < args.n:
        text = "".join(random.choice(CHARSET) for _ in range(LENGTH))
        path = os.path.join(OUT_DIR, f"{text}.png")
        if os.path.exists(path):
            continue
        make_image(text).save(path)
        made += 1
    print(f"Wrote {made} CAPTCHA images to {OUT_DIR}")
