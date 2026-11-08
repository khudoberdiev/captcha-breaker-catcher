"""
Act 1 — how the CNN reads one CAPTCHA.
Manim Community v0.22.1. Uses only Text (no LaTeX).

Render one scene (fast preview):   manim -pql scenes.py Act1_Zoom
Render all scenes in HD:           manim -qh scenes.py Act1_Zoom Act1_Conv Act1_Deeper Act1_Flatten Act1_Heads

All data comes from ./data/ (written by "Step 13: Export for animation" in breaker.ipynb).
"""
from pathlib import Path

import numpy as np
from PIL import Image
from manim import *

# ---------------------------------------------------------------- data & style
DATA = Path(__file__).parent / "data"

PIXELS = np.load(DATA / "pixels.npy")        # (60, 160)   values 0–1
BLOCK1 = np.load(DATA / "block1.npy")        # (32, 30, 80)
BLOCK2 = np.load(DATA / "block2.npy")        # (64, 15, 40)
BLOCK3 = np.load(DATA / "block3.npy")        # (128, 7, 20)
HEADS = np.load(DATA / "heads.npy")          # (5, 32)     probabilities
ANSWER = (DATA / "answer.txt").read_text().strip()
CHARSET = (DATA / "charset.txt").read_text().strip()
CAPTCHA_PNG = str(DATA / "captcha.png")

BG = "#0e1117"
TEAL = "#5DCAA5"
PURPLE = "#AFA9EC"
GOLD = "#FFD866"
GREY = "#8a94a6"

config.background_color = BG


def to_uint8(a):
    """Scale any array to 0–255 for display."""
    a = np.asarray(a, dtype=np.float32)
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo + 1e-8)
    return (a * 255).astype(np.uint8)


def pixel_image(a, width=None, height=None, raw=False):
    """Array → crisp (nearest-neighbour) grayscale image.
    raw=True shows the true 0–1 brightness (no contrast stretch)."""
    data = (np.clip(a, 0, 1) * 255).astype(np.uint8) if raw else to_uint8(a)
    img = ImageMobject(data)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    if width is not None:
        img.width = width
    if height is not None:
        img.height = height
    return img


def pixel_image_rgb(rgb, width=None):
    """RGB uint8 array → crisp colour image."""
    img = ImageMobject(np.asarray(rgb, dtype=np.uint8))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    if width is not None:
        img.width = width
    return img


def top_channels(block, k):
    """The k most 'active' channels (largest variance) — the most readable maps."""
    order = np.argsort([-block[c].std() for c in range(block.shape[0])])
    return [block[c] for c in order[:k]]


def caption(text, size=30):
    return Text(text, font_size=size, color=GREY).to_edge(DOWN, buff=0.45)


def tile_grid(maps, tile_w, cols=4, buff=0.15, color=TEAL):
    tiles = Group(*[
        Group(pixel_image(m, width=tile_w),
              SurroundingRectangle(pixel_image(m, width=tile_w), color=color,
                                   stroke_width=1.5, buff=0))
        for m in maps
    ])
    tiles.arrange_in_grid(cols=cols, buff=buff)
    return tiles


# ================================================================ Scene 1
class Act1_Zoom(MovingCameraScene):
    """Colour CAPTCHA → 3 planes (R, G, B) → 1 gray plane → zoom in → just numbers."""

    def construct(self):
        frame = self.camera.frame
        frame.save_state()

        rgb = np.asarray(Image.open(CAPTCHA_PNG).convert("RGB"))       # (60, 160, 3), 0–255

        # ---- 1. the real CAPTCHA, in colour
        title = Text("One real CAPTCHA", font_size=40).to_edge(UP, buff=0.6)
        color_img = pixel_image_rgb(rgb, width=10)
        size = Text("160 × 60 pixels", font_size=28, color=GREY).next_to(color_img, DOWN, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.play(FadeIn(color_img, scale=0.9), run_time=1.5)
        self.play(Write(size))
        self.wait(1)

        # ---- 2. a colour pixel is 3 numbers: one per plane
        planes_title = Text("A colour image is 3 planes stacked", font_size=36).to_edge(UP, buff=0.6)
        names = [("Red", "#E24B4A", 0), ("Green", "#639922", 1), ("Blue", "#378ADD", 2)]
        planes = Group()
        for _, _, k in names:
            only = np.zeros_like(rgb)
            only[..., k] = rgb[..., k]
            planes.add(pixel_image_rgb(only, width=6))
        for i, p in enumerate(planes):
            p.move_to(LEFT * 1.2 + UP * 0.9 + (RIGHT * 0.9 + DOWN * 0.7) * i)
        labels = VGroup()
        for i, (n, col, _) in enumerate(names):
            t = Text(n, font_size=24, color=WHITE)
            t.move_to(planes[i].get_corner(UL) + RIGHT * (t.width / 2 + 0.2) + DOWN * 0.3)
            labels.add(VGroup(BackgroundRectangle(t, color=BLACK, fill_opacity=0.6, buff=0.08), t))

        self.play(FadeOut(title), FadeOut(size))
        self.play(FadeIn(planes_title))
        self.play(ReplacementTransform(color_img, planes[0]),
                  FadeIn(planes[1], shift=DOWN * 0.3), FadeIn(planes[2], shift=DOWN * 0.6),
                  run_time=1.5)
        self.play(LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.2))

        # one dark stroke pixel → its 3 values
        r, c = np.unravel_index(np.argmin(PIXELS), PIXELS.shape)
        vals = rgb[r, c]
        readout = VGroup(*[
            Text(f"{n}: {int(v)}", font_size=28, color=col)
            for (n, col, _), v in zip(names, vals)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.8)
        head = Text("one pixel =", font_size=28).next_to(readout, UP, buff=0.3)
        markers = Group()
        for p in planes:
            px = p.width / 160
            dot = Circle(radius=0.12, color=GOLD, stroke_width=3).move_to(
                p.get_corner(UL) + RIGHT * (c + 0.5) * px + DOWN * (r + 0.5) * px)
            markers.add(dot)
        self.play(*[Create(m) for m in markers])
        self.play(Write(head), LaggedStart(*[FadeIn(t, shift=LEFT * 0.2) for t in readout], lag_ratio=0.2))
        cap = caption("Each colour pixel = 3 numbers (0–255): red, green, blue")
        self.play(FadeIn(cap))
        self.wait(2.5)

        # ---- 3. the model uses ONE plane: brightness (grayscale)
        gray_title = Text("Our model uses 1 plane: brightness", font_size=36).to_edge(UP, buff=0.6)
        gray_img = pixel_image(PIXELS, width=10, raw=True)
        gray_val = float(PIXELS[r, c])
        formula = Text(f"R, G, B  →  one brightness value: {gray_val:.2f}",
                       font_size=28, color=GOLD).next_to(gray_img, DOWN, buff=0.4)
        self.play(FadeOut(cap), FadeOut(head), FadeOut(readout), FadeOut(markers), FadeOut(labels),
                  FadeOut(planes_title))
        self.play(FadeIn(gray_title))
        self.play(FadeOut(planes[1]), FadeOut(planes[2]),
                  ReplacementTransform(planes[0], gray_img), run_time=1.5)
        self.play(Write(formula))

        scale = VGroup(
            Square(0.45, fill_color=BLACK, fill_opacity=1, stroke_color=GREY, stroke_width=1),
            Text("0 = black", font_size=24),
            Square(0.45, fill_color=WHITE, fill_opacity=1, stroke_color=GREY, stroke_width=1),
            Text("1 = white", font_size=24),
        ).arrange(RIGHT, buff=0.25).next_to(formula, DOWN, buff=0.35)
        self.play(FadeIn(scale))
        self.wait(2)
        self.play(FadeOut(formula), FadeOut(scale), FadeOut(gray_title))

        # ---- 4. zoom in on the gray image until pixels are numbers
        n = 8
        best, r0, c0 = -1, 0, 0
        for rr in range(0, 60 - n, 2):
            for cc in range(0, 160 - n, 2):
                v = PIXELS[rr:rr + n, cc:cc + n].std()
                if v > best:
                    best, r0, c0 = v, rr, cc

        px = gray_img.width / 160
        top_left = gray_img.get_corner(UL)

        def pixel_center(rr, cc):
            return top_left + RIGHT * (cc + 0.5) * px + DOWN * (rr + 0.5) * px

        patch_center = pixel_center(r0 + n / 2 - 0.5, c0 + n / 2 - 0.5)
        for target_width in (6, 2.6, n * px * 2.8):
            self.play(frame.animate.set(width=target_width).move_to(patch_center),
                      run_time=1.6, rate_func=smooth)
            self.wait(0.3)

        cells = VGroup()
        for rr in range(n):
            for cc in range(n):
                v = float(PIXELS[r0 + rr, c0 + cc])
                sq = Square(side_length=px, stroke_width=0.6, stroke_color=GREY,
                            fill_opacity=0).move_to(pixel_center(r0 + rr, c0 + cc))
                num = Text(f"{v:.2f}", font_size=48, color=BLACK if v > 0.5 else WHITE)
                num.scale_to_fit_width(px * 0.72).move_to(sq)
                cells.add(VGroup(sq, num))
        self.play(LaggedStart(*[FadeIn(cell) for cell in cells], lag_ratio=0.02), run_time=2.5)

        zcap = Text("Dark stroke → small number · white background → close to 1",
                    font_size=48, color=GOLD)
        zcap.scale_to_fit_width(frame.width * 0.85)
        zcap.next_to(frame.get_top(), DOWN, buff=frame.height * 0.03)
        zbg = BackgroundRectangle(zcap, color=BG, fill_opacity=0.85, buff=zcap.height * 0.3)
        self.play(FadeIn(zbg), FadeIn(zcap))
        self.wait(2.5)

        # ---- 5. back out: the whole picture is numbers
        self.play(FadeOut(zcap), FadeOut(zbg), FadeOut(cells))
        self.play(Restore(frame), run_time=2)
        final = Text("To the model, this picture is 60 × 160 = 9,600 numbers",
                     font_size=34, color=GOLD).next_to(gray_img, DOWN, buff=0.5)
        self.play(Write(final))
        self.wait(2)


# ================================================================ Scene 2
class Act1_Conv(Scene):
    """A 3×3 filter slides over the image → a feature map appears → pooling halves it."""

    def construct(self):
        heading = Text("Block 1: filters look for strokes", font_size=36).to_edge(UP, buff=0.5)
        self.play(FadeIn(heading))

        img = pixel_image(PIXELS, width=6, raw=True).to_edge(LEFT, buff=0.6).shift(UP * 0.6)
        img_label = Text("input  60 × 160", font_size=24, color=GREY).next_to(img, DOWN, buff=0.25)

        fmap_data = top_channels(BLOCK1, 1)[0]                  # one real feature map
        fmap = pixel_image(fmap_data, width=6).to_edge(RIGHT, buff=0.6).shift(UP * 0.6)
        fmap_frame = SurroundingRectangle(fmap, color=TEAL, buff=0, stroke_width=2)
        fmap_label = Text("one of 32 feature maps", font_size=24, color=TEAL).next_to(fmap, DOWN, buff=0.25)

        self.play(FadeIn(img), Write(img_label))
        self.play(Create(fmap_frame), Write(fmap_label))

        # sliding 3×3 window on the input, map fills in left → right
        px = img.width / 160
        window = Square(side_length=3 * px * 4, color=GOLD, stroke_width=3)  # drawn larger to be visible
        progress = ValueTracker(0)

        def window_pos():
            p = progress.get_value()
            x = img.get_left()[0] + window.width / 2 + p * (img.width - window.width)
            row = int(p * 6) % 3                                 # hop between 3 rows for a "scan" feel
            y = img.get_top()[1] - window.height / 2 - row * (img.height - window.height) / 2
            return np.array([x, y, 0])

        window.add_updater(lambda m: m.move_to(window_pos()))

        cover = always_redraw(lambda: Rectangle(
            width=max(fmap.width * (1 - progress.get_value()), 0.001),
            height=fmap.height, fill_color=BG, fill_opacity=1, stroke_width=0,
        ).align_to(fmap, RIGHT).align_to(fmap, UP))

        self.add(fmap, cover, window)
        self.play(progress.animate.set_value(1), run_time=5, rate_func=linear)
        window.clear_updaters()
        self.remove(cover)
        self.play(FadeOut(window))

        cap1 = caption("Each filter slides over the image and lights up where it finds its pattern")
        self.play(FadeIn(cap1))
        self.wait(2)
        self.play(FadeOut(cap1))

        # pooling: 4×4 numbers → 2×2 (keep the max)
        self.play(FadeOut(img), FadeOut(img_label), FadeOut(heading))
        pool_title = Text("Max pooling: keep the strongest value in each 2 × 2", font_size=32).to_edge(UP, buff=0.5)
        self.play(FadeIn(pool_title))

        block = fmap_data[10:14, 30:34]
        block = (block - fmap_data.min()) / (fmap_data.max() - fmap_data.min() + 1e-8)
        s = 0.9
        grid = VGroup()
        for r in range(4):
            for c in range(4):
                sq = Square(side_length=s, stroke_color=GREY, stroke_width=1.5,
                            fill_color=interpolate_color(ManimColor(BG), ManimColor(TEAL), float(block[r, c])),
                            fill_opacity=1)
                sq.move_to(np.array([(c - 1.5) * s, (1.5 - r) * s, 0]))
                t = Text(f"{block[r, c]:.2f}", font_size=22).move_to(sq)
                grid.add(VGroup(sq, t))
        grid.shift(LEFT * 3)

        self.play(FadeOut(fmap), FadeOut(fmap_frame), FadeOut(fmap_label), FadeIn(grid))

        pooled = VGroup()
        quads = [(0, 0), (0, 2), (2, 0), (2, 2)]
        colors = [GOLD, PURPLE, TEAL, "#F0997B"]
        for (qr, qc), col in zip(quads, colors):
            idx = [(qr + i) * 4 + (qc + j) for i in range(2) for j in range(2)]
            hl = SurroundingRectangle(VGroup(*[grid[k] for k in idx]), color=col, buff=0.03, stroke_width=4)
            best = max(idx, key=lambda k: block[k // 4, k % 4])
            out_sq = Square(side_length=s, stroke_color=col, stroke_width=3,
                            fill_color=grid[best][0].get_fill_color(), fill_opacity=1)
            out_sq.move_to(RIGHT * (3 + (qc // 2 - 0.5) * s) + UP * ((0.5 - qr // 2) * s))
            out_t = Text(f"{block[best // 4, best % 4]:.2f}", font_size=22).move_to(out_sq)
            self.play(Create(hl), run_time=0.4)
            self.play(Indicate(grid[best], color=col), run_time=0.5)
            self.play(TransformFromCopy(grid[best], VGroup(out_sq, out_t)), run_time=0.7)
            self.play(FadeOut(hl), run_time=0.2)
            pooled.add(VGroup(out_sq, out_t))

        arrow = Arrow(grid.get_right(), pooled.get_left(), color=GREY, buff=0.4)
        self.play(GrowArrow(arrow))
        cap2 = caption("4 × 4 → 2 × 2: the map shrinks to half, the strongest signals survive")
        self.play(FadeIn(cap2))
        self.wait(2)

        # the whole map halves
        self.play(FadeOut(grid), FadeOut(pooled), FadeOut(arrow), FadeOut(cap2), FadeOut(pool_title))
        big = pixel_image(fmap_data, width=8)
        big_label = Text("60 × 160", font_size=28, color=GREY).next_to(big, DOWN)
        self.play(FadeIn(big), Write(big_label))
        small = pixel_image(fmap_data, width=4)
        small_label = Text("30 × 80", font_size=28, color=TEAL).next_to(small, DOWN)
        self.play(big.animate.set(width=4), Transform(big_label, small_label), run_time=1.5)
        self.wait(1.5)


# ================================================================ Scene 3
class Act1_Deeper(Scene):
    """Deeper blocks: smaller, more numerous, more abstract maps."""

    def construct(self):
        stages = [
            ("Block 1", BLOCK1, "32 maps · 30 × 80", "edges & strokes", 2.6),
            ("Block 2", BLOCK2, "64 maps · 15 × 40", "curves & corners", 1.9),
            ("Block 3", BLOCK3, "128 maps · 7 × 20", "parts of characters", 1.3),
        ]
        thumb = ImageMobject(CAPTCHA_PNG)
        thumb.width = 2.4
        thumb.to_corner(DL, buff=0.4)
        self.play(FadeIn(thumb))

        prev = None
        for name, block, shape, meaning, tile_w in stages:
            title = Text(name, font_size=40).to_edge(UP, buff=0.5)
            info = Text(f"{shape}  —  {meaning}", font_size=28, color=TEAL).next_to(title, DOWN, buff=0.2)
            tiles = tile_grid(top_channels(block, 8), tile_w).next_to(info, DOWN, buff=0.5)
            note = Text("(showing 8 of them)", font_size=22, color=GREY).next_to(tiles, DOWN, buff=0.3)

            if prev is None:
                self.play(FadeIn(title), FadeIn(info))
                self.play(LaggedStart(*[FadeIn(t, scale=0.8) for t in tiles], lag_ratio=0.08))
                self.play(FadeIn(note))
            else:
                self.play(
                    ReplacementTransform(prev[0], title),
                    ReplacementTransform(prev[1], info),
                    FadeOut(prev[2], scale=0.7),
                    ReplacementTransform(prev[3], note),
                )
                self.play(LaggedStart(*[FadeIn(t, scale=1.2) for t in tiles], lag_ratio=0.08))
            self.wait(1.5)
            prev = (title, info, tiles, note)

        cap = Text("Each block: smaller maps, more of them, more abstract patterns",
                   font_size=24, color=GREY)
        cap.next_to(thumb, RIGHT, buff=0.4).align_to(thumb, DOWN).shift(UP * 0.1)
        self.play(FadeIn(cap))
        self.wait(2)


# ================================================================ Scene 4
class Act1_Flatten(Scene):
    """128 small maps unroll into one row of 17,920 numbers."""

    def construct(self):
        title = Text("Flatten", font_size=40).to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        tiles = tile_grid(top_channels(BLOCK3, 8), 1.6, color=PURPLE).shift(UP * 0.6)
        label = Text("128 maps × 7 × 20", font_size=28, color=PURPLE).next_to(tiles, DOWN, buff=0.4)
        self.play(FadeIn(tiles), Write(label))
        self.wait(1)

        flat = BLOCK3.reshape(1, -1)                            # (1, 17920)
        strip = pixel_image(flat)
        strip.stretch_to_fit_width(13).stretch_to_fit_height(0.35)
        strip_frame = SurroundingRectangle(strip, color=PURPLE, buff=0, stroke_width=2)
        strip_group = Group(strip, strip_frame).move_to(DOWN * 1.2)

        self.play(tiles.animate.arrange(RIGHT, buff=0).stretch_to_fit_width(13)
                  .stretch_to_fit_height(0.35).move_to(DOWN * 1.2),
                  FadeOut(label), run_time=2)
        self.play(FadeIn(strip_group), FadeOut(tiles))

        count = Text("17,920 numbers", font_size=36, color=GOLD).next_to(strip_group, UP, buff=0.5)
        self.play(Write(count))
        cap = caption("One long list that describes the whole CAPTCHA")
        self.play(FadeIn(cap))
        self.wait(2.5)


# ================================================================ Scene 5
class Act1_Heads(Scene):
    """5 heads read the same 17,920 numbers; each picks 1 of 32 characters."""

    def construct(self):
        strip = pixel_image(BLOCK3.reshape(1, -1))
        strip.stretch_to_fit_width(12).stretch_to_fit_height(0.3)
        strip_frame = SurroundingRectangle(strip, color=PURPLE, buff=0, stroke_width=2)
        strip_group = Group(strip, strip_frame).to_edge(UP, buff=0.7)
        strip_label = Text("17,920 numbers", font_size=24, color=PURPLE).next_to(strip_group, UP, buff=0.15)
        self.play(FadeIn(strip_group), FadeIn(strip_label))

        box_w, box_h = 2.3, 2.0
        heads = VGroup()
        bars_all, picks = [], []
        for i in range(5):
            box = RoundedRectangle(width=box_w, height=box_h, corner_radius=0.12,
                                   stroke_color=PURPLE, stroke_width=2)
            name = Text(f"Head {i + 1}", font_size=22).next_to(box, UP, buff=0.1)
            probs = HEADS[i]
            bw = (box_w - 0.3) / 32
            bars = VGroup()
            for k, p in enumerate(probs):
                h = max(float(p) * (box_h - 0.5), 0.02)
                bar = Rectangle(width=bw * 0.8, height=h, stroke_width=0,
                                fill_color=GREY, fill_opacity=0.9)
                bar.move_to(box.get_bottom() + UP * (0.2 + h / 2)
                            + RIGHT * (-box_w / 2 + 0.15 + bw * (k + 0.5)))
                bars.add(bar)
            heads.add(VGroup(box, name, bars))
            bars_all.append(bars)
            picks.append(int(np.argmax(probs)))
        heads.arrange(RIGHT, buff=0.3).shift(DOWN * 0.3)

        lines = VGroup(*[
            Line(strip_group.get_bottom(), h[1].get_top(), stroke_width=1.5, color=GREY)
            for h in heads
        ])
        self.play(Create(lines), LaggedStart(*[FadeIn(h[0]) for h in heads], lag_ratio=0.1),
                  LaggedStart(*[FadeIn(h[1]) for h in heads], lag_ratio=0.1))

        cap = caption("Each head scores all 32 characters for its own position")
        self.play(FadeIn(cap))
        for bars in bars_all:
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.01), run_time=0.8)
        self.wait(1)

        # highlight the winner in each head
        letters = VGroup()
        for i, (bars, k) in enumerate(zip(bars_all, picks)):
            self.play(bars[k].animate.set_fill(GOLD), run_time=0.3)
            ch = Text(CHARSET[k], font_size=44, color=GOLD)
            pct = Text(f"{HEADS[i][k] * 100:.0f}%", font_size=20, color=GOLD)
            VGroup(ch, pct).arrange(DOWN, buff=0.08).next_to(heads[i], DOWN, buff=0.2)
            self.play(FadeIn(ch, shift=UP * 0.2), FadeIn(pct), run_time=0.4)
            letters.add(VGroup(ch, pct))
        self.wait(1)

        # letters fly under the original CAPTCHA and form the answer
        self.play(FadeOut(cap), FadeOut(heads), FadeOut(lines),
                  FadeOut(strip_group), FadeOut(strip_label),
                  *[FadeOut(l[1]) for l in letters])
        img = ImageMobject(CAPTCHA_PNG)
        img.width = 8
        img.shift(UP * 1.2)
        self.play(FadeIn(img))

        answer = VGroup(*[Text(c, font_size=72, color=GOLD) for c in ANSWER]).arrange(RIGHT, buff=0.35)
        answer.next_to(img, DOWN, buff=0.6)
        self.play(*[ReplacementTransform(letters[i][0], answer[i]) for i in range(5)], run_time=1.5)

        done = Text("The model's answer", font_size=30, color=GREY).next_to(answer, DOWN, buff=0.4)
        self.play(FadeIn(done))
        self.wait(2.5)


# ################################################################ ACT 2
# The answer becomes actions: mouse moves, clicks, typing, submit.
# Data: data/naive.csv and data/evasive.csv (same format as LOG_FORMAT.md).

import csv

PAGE_W, PAGE_H = 400, 250                     # the page box, in pixels (LOG_FORMAT.md)


def load_session(name):
    rows = list(csv.DictReader(open(DATA / name)))
    for r in rows:
        r["t_ms"] = int(r["t_ms"])
    return rows


class Page:
    """The 400 × 250 page drawn on screen, plus a pixel → screen converter."""

    def __init__(self, width=7.0, center=DOWN * 0.05):
        self.s = width / PAGE_W                                   # screen units per pixel
        self.box = Rectangle(width=width, height=PAGE_H * self.s,
                             stroke_color=GREY, stroke_width=2,
                             fill_color="#161b24", fill_opacity=1).move_to(center)
        self.tl = self.box.get_corner(UL)

        img = ImageMobject(CAPTCHA_PNG)
        img.width = 160 * self.s
        img.move_to(self.pt(200, 70))                             # x 120–280, y 40–100

        self.input_box = Rectangle(width=160 * self.s, height=30 * self.s,
                                   stroke_color=GREY, stroke_width=2,
                                   fill_color=WHITE, fill_opacity=0.95).move_to(self.pt(200, 145))
        self.button = RoundedRectangle(width=100 * self.s, height=30 * self.s, corner_radius=0.08,
                                       fill_color="#378ADD", fill_opacity=1,
                                       stroke_width=0).move_to(self.pt(200, 195))
        btn_text = Text("Submit", font_size=22, color=WHITE).move_to(self.button)
        self.size = Text("400 × 250 page", font_size=20, color=GREY).next_to(self.box, UP, buff=0.12).align_to(self.box, LEFT)
        self.group = Group(self.box, img, self.input_box, self.button, btn_text, self.size)

    def pt(self, x, y):
        return self.tl + RIGHT * float(x) * self.s + DOWN * float(y) * self.s


def cursor_shape():
    """A small mouse pointer; its tip is the first point."""
    arrow = Polygon([0, 0, 0], [0, -0.42, 0], [0.11, -0.31, 0], [0.2, -0.5, 0],
                    [0.27, -0.46, 0], [0.18, -0.28, 0], [0.32, -0.28, 0],
                    fill_color=WHITE, fill_opacity=1, stroke_color=BLACK, stroke_width=1.5)
    return arrow


def replay(scene, page, rows, speed=1.0, trail_color=GOLD, timeline=None):
    """Play a logged session in (scaled) real time.
    timeline: (x_left, x_right, y) on screen → key presses appear as ticks."""
    moves = [(r["t_ms"], float(r["x"]), float(r["y"])) for r in rows if r["event"] == "mouse_move"]
    clicks = [(r["t_ms"], float(r["x"]), float(r["y"])) for r in rows if r["event"] == "click"]
    keys = [(r["t_ms"], r["key"]) for r in rows if r["event"] == "key_down"]
    total = rows[-1]["t_ms"]
    now = ValueTracker(0)

    def mouse_at(t):
        prev = moves[0]
        for m in moves:
            if m[0] > t:
                t0, x0, y0 = prev
                t1, x1, y1 = m
                a = 0 if t1 == t0 else (t - t0) / (t1 - t0)
                return page.pt(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
            prev = m
        return page.pt(prev[1], prev[2])

    cursor = cursor_shape()
    cursor.add_updater(lambda m: m.shift(mouse_at(now.get_value()) - m.get_vertices()[0]))

    trail = TracedPath(lambda: mouse_at(now.get_value()), stroke_color=trail_color,
                       stroke_width=3, stroke_opacity=0.8)

    def typed_text():
        txt = "".join(k for t, k in keys if t <= now.get_value())
        if not txt:                                   # nothing typed yet → invisible placeholder
            return Dot(radius=0.001, fill_opacity=0).move_to(page.input_box)
        return (Text(txt, font_size=26, color=BLACK).move_to(page.input_box)
                .align_to(page.input_box, LEFT).shift(RIGHT * 0.15))

    typed = always_redraw(typed_text)

    def ripples():
        g = VGroup()
        for t, x, y in clicks:
            age = now.get_value() - t
            if 0 <= age <= 350:
                a = age / 350
                g.add(Circle(radius=0.08 + 0.4 * a, color=GOLD, stroke_width=4 * (1 - a) + 0.5)
                      .set_stroke(opacity=1 - a).move_to(page.pt(x, y)))
        return g

    ripple = always_redraw(ripples)
    clock = always_redraw(lambda: Text(f"{now.get_value() / 1000:.2f} s", font_size=34, color=GOLD)
                          .next_to(page.box, RIGHT, buff=0.35).align_to(page.box, UP))

    extras = []
    if timeline is not None:
        xl, xr, y = timeline
        axis = Line([xl, y, 0], [xr, y, 0], color=GREY, stroke_width=2)
        lab = Text("key presses", font_size=20, color=GREY).next_to(axis, LEFT, buff=0.25)
        end = Text(f"{total / 1000:.1f} s", font_size=20, color=GREY).next_to(axis, RIGHT, buff=0.25)

        def ticks():
            g = VGroup()
            for t, k in keys:
                if t <= now.get_value():
                    x = xl + (xr - xl) * t / total
                    g.add(VGroup(Line([x, y - 0.12, 0], [x, y + 0.25, 0], color=GOLD, stroke_width=4),
                                 Text(k, font_size=20, color=GOLD).move_to([x, y + 0.45, 0])))
            return g

        tick_marks = always_redraw(ticks)
        scene.add(axis, lab, end, tick_marks)
        extras = [axis, lab, end, tick_marks]

    scene.add(trail, typed, ripple, clock, cursor)
    scene.play(now.animate.set_value(total), run_time=total / 1000 / speed, rate_func=linear)
    scene.wait(0.3)
    return Group(trail, typed, ripple, clock, cursor, *extras), now


# ================================================================ Act 2 · Scene 1
class Act2_Naive(Scene):
    """The naive bot: straight line, instant typing — blink and it's done."""

    def construct(self):
        title = Text(f"The answer is ready: {ANSWER}. Now the bot has to act.", font_size=32).to_edge(UP, buff=0.4)
        page = Page()
        self.play(FadeIn(title))
        self.play(FadeIn(page.group))
        self.wait(1)

        rows = load_session("naive.csv")
        sub = Text("Naive bot · real speed", font_size=28, color=TEAL).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(sub))
        played, _ = replay(self, page, rows, speed=1.0)
        cap = caption(f"Done in {rows[-1]['t_ms'] / 1000:.2f} seconds. Did you even see it?", size=26)
        self.play(FadeIn(cap))
        self.wait(2)

        # again, slowed down, with the path drawn
        self.play(FadeOut(cap), FadeOut(played))
        sub2 = Text("Same bot · slowed down 6×", font_size=28, color=TEAL).move_to(sub)
        self.play(ReplacementTransform(sub, sub2))
        tl = (page.box.get_left()[0], page.box.get_right()[0], page.box.get_bottom()[1] - 0.55)
        played, _ = replay(self, page, rows, speed=1 / 6, timeline=tl)
        cap = caption("Perfectly straight line · constant speed · same gap between every key", size=24)
        self.play(FadeIn(cap))
        self.wait(3)


# ================================================================ Act 2 · Scene 2
class Act2_Evasive(Scene):
    """The evasive bot: reading pause, curved path, uneven typing."""

    def construct(self):
        title = Text("The evasive bot: same answer, human-like behaviour", font_size=32).to_edge(UP, buff=0.4)
        page = Page()
        self.play(FadeIn(title), FadeIn(page.group))

        rows = load_session("evasive.csv")
        sub = Text("real speed", font_size=28, color=TEAL).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(sub))
        tl = (page.box.get_left()[0], page.box.get_right()[0], page.box.get_bottom()[1] - 0.55)
        played, _ = replay(self, page, rows, speed=1.0, trail_color=TEAL, timeline=tl)

        first_move = next(r["t_ms"] for r in rows[1:] if r["event"] == "mouse_move")
        cap = caption(f"Waits {first_move / 1000:.1f} s to 'read' · curved, wobbly path · uneven typing — "
                      f"{rows[-1]['t_ms'] / 1000:.1f} s in total", size=24)
        self.play(FadeIn(cap))
        self.wait(3.5)


# ================================================================ Act 2 · Scene 3
class Act2_Handoff(Scene):
    def construct(self):
        a = Text("It reads the CAPTCHA.", font_size=40)
        b = Text("It moves and types like a person.", font_size=40)
        c = Text("Can anything still tell it's a bot?", font_size=44, color=GOLD)
        VGroup(a, b, c).arrange(DOWN, buff=0.6)
        self.play(FadeIn(a, shift=UP * 0.2))
        self.wait(0.8)
        self.play(FadeIn(b, shift=UP * 0.2))
        self.wait(0.8)
        self.play(Write(c), run_time=1.5)
        self.wait(2.5)