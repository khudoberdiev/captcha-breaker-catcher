"""
Act 3 — the catcher: how it tells a person from a bot.
Manim Community v0.22.1. Uses only Text (no LaTeX). Reuses helpers from scenes.py.

Preview one scene:   manim -pql act3.py Act3_Who
Render all in HD:    manim -qh act3.py Act3_Who Act3_Numbers Act3_Cloud Act3_Rounds Act3_Close

Data (in ./data/):
  human_a.csv     one anonymised human session (typed keys already masked by the export cell)
  evasive_v2.csv  one evasive-v2 session (from breaker.ipynb Step 13)
  features.csv    kind + 3 feature columns for every session (no IDs, no names)
Round results below are copied from half_b_catcher/results/results.csv.
"""
import csv

import numpy as np
from manim import *

import scenes as S          # Page, cursor_shape, load_session, colours, background

BG, TEAL, PURPLE, GOLD, GREY = S.BG, S.TEAL, S.PURPLE, S.GOLD, S.GREY
RED = "#F07178"
ORANGE = "#F5A05A"
KIND_COLOR = {"human": TEAL, "naive": RED, "evasive": ORANGE, "evasive_v2": GOLD}
KIND_NAME = {"human": "humans", "naive": "naive bot", "evasive": "evasive bot", "evasive_v2": "evasive bot v2"}

HUMAN = S.load_session("human_a.csv")
BOT = S.load_session("evasive_v2.csv")

# catch rates (%), from results.csv — Round 5 there repeats Round 4, so it is left out.
#            label in video                 humans flagged, naive, evasive, v2
ROUNDS = [("trained on\nnaive bots",        0.0, 100, 0.25, 3.4),
          ("+ better\nfeatures",            0.0, 100, 0.0, 0.0),
          ("+ evasive\nexamples",           0.96, 100, 100, 69.4),
          ("features +\nexamples",          0.0, 100, 100, 66.2),
          ("+ v2\nexamples",                0.0, 100, 100, 99.9)]

# top features, relative bar lengths read off results/feature_importance.png
TOP_FEATURES = [("time between key presses (mean)", 0.250),
                ("time between key presses (variance)", 0.232),
                ("click → first key", 0.160),
                ("typing rhythm (CV)", 0.107),
                ("mouse path straightness", 0.075)]

N_FEATURES = 27


# ---------------------------------------------------------------- session numbers
def key_downs(rows):
    return [r["t_ms"] for r in rows if r["event"] == "key_down"]


def session_stats(rows):
    """The same three numbers features.py computes, for one session."""
    kt = np.array(key_downs(rows), dtype=float)
    gaps = np.diff(kt)
    xy = np.array([(float(r["x"]), float(r["y"])) for r in rows if r["event"] == "mouse_move"])
    steps = np.hypot(*np.diff(xy, axis=0).T)
    chord = np.hypot(*(xy[-1] - xy[0]))
    return {"gap": gaps.mean(), "cv": gaps.std() / gaps.mean(),
            "straight": chord / steps.sum() if steps.sum() > 0 else 0.0}


# ---------------------------------------------------------------- side-by-side replay
def live_session(page, rows, now, color):
    """Cursor, trail, click ripples and masked typing for one page, driven by a shared clock."""
    moves = [(r["t_ms"], float(r["x"]), float(r["y"])) for r in rows if r["event"] == "mouse_move"]
    clicks = [(r["t_ms"], float(r["x"]), float(r["y"])) for r in rows if r["event"] == "click"]
    keys = [(r["t_ms"], r["key"]) for r in rows if r["event"] == "key_down"]

    def mouse_at(t):
        prev = moves[0]
        for m in moves:
            if m[0] > t:
                a = 0 if m[0] == prev[0] else (t - prev[0]) / (m[0] - prev[0])
                return page.pt(prev[1] + (m[1] - prev[1]) * a, prev[2] + (m[2] - prev[2]) * a)
            prev = m
        return page.pt(prev[1], prev[2])

    cursor = S.cursor_shape().scale(0.85)
    cursor.add_updater(lambda m: m.shift(mouse_at(now.get_value()) - m.get_vertices()[0]))
    trail = TracedPath(lambda: mouse_at(now.get_value()), stroke_color=color,
                       stroke_width=3, stroke_opacity=0.8)

    def typed():
        n = 0
        for t, k in keys:
            if t <= now.get_value():
                n = max(n - 1, 0) if k == "Backspace" else n + 1
        if n == 0:
            return Dot(radius=0.001, fill_opacity=0).move_to(page.input_box)
        return (Text("•" * n, font_size=26, color=BLACK).move_to(page.input_box)
                .align_to(page.input_box, LEFT).shift(RIGHT * 0.12))

    def ripples():
        g = VGroup()
        for t, x, y in clicks:
            age = now.get_value() - t
            if 0 <= age <= 350:
                a = age / 350
                g.add(Circle(radius=0.07 + 0.35 * a, color=color, stroke_width=4 * (1 - a) + 0.5)
                      .set_stroke(opacity=1 - a).move_to(page.pt(x, y)))
        return g

    return Group(trail, always_redraw(typed), always_redraw(ripples), cursor)


def two_pages():
    left = S.Page(width=6.0, center=LEFT * 3.4 + DOWN * 0.35)
    right = S.Page(width=6.0, center=RIGHT * 3.4 + DOWN * 0.35)
    for p in (left, right):
        p.group.remove(p.size)              # drop the "400 × 250 page" label
    return left, right


# ================================================================ Act 3 · Scene 1
class Act3_Who(Scene):
    """Two sessions side by side. Which one is the person?"""

    def construct(self):
        tag = Text("Half B · The Catcher", font_size=26, color=TEAL).to_edge(UP, buff=0.35)
        title = Text("One of these is a person. One is evasive bot v2.", font_size=32).next_to(tag, DOWN, buff=0.2)
        left, right = two_pages()
        la = Text("A", font_size=34, color=WHITE).next_to(left.box, UP, buff=0.2)
        lb = Text("B", font_size=34, color=WHITE).next_to(right.box, UP, buff=0.2)

        self.play(FadeIn(tag, shift=DOWN * 0.2))
        self.play(FadeIn(title))
        self.play(FadeIn(left.group), FadeIn(right.group), FadeIn(la), FadeIn(lb))
        self.wait(0.8)

        # A = bot, B = human (the audience usually guesses the other way)
        now = ValueTracker(0)
        a = live_session(left, BOT, now, GREY)
        b = live_session(right, HUMAN, now, GREY)
        self.add(a, b)
        total = max(BOT[-1]["t_ms"], HUMAN[-1]["t_ms"])
        self.play(now.animate.set_value(total), run_time=total / 1000, rate_func=linear)

        q = Text("Which one is human?", font_size=40, color=GOLD).to_edge(DOWN, buff=0.5)
        self.play(Write(q))
        self.wait(4)                                    # let the room vote

        ra = Text("A · bot v2", font_size=34, color=GOLD).move_to(la)
        rb = Text("B · Human A", font_size=34, color=TEAL).move_to(lb)
        self.play(ReplacementTransform(la, ra), ReplacementTransform(lb, rb))
        a[0].set_stroke(color=GOLD)
        b[0].set_stroke(color=TEAL)
        self.wait(1)
        cap = S.caption("Hard to tell by eye. So the catcher doesn't use eyes.", size=28)
        self.play(ReplacementTransform(q, cap))
        self.wait(3)


# ================================================================ Act 3 · Scene 2
class Act3_Numbers(Scene):
    """What the catcher actually sees: a row of numbers per session."""

    def construct(self):
        title = Text("What the catcher sees: not a video, a row of numbers", font_size=32).to_edge(UP, buff=0.45)
        self.play(FadeIn(title))

        cols = []
        for i, (name, rows, color) in enumerate([("Human A", HUMAN, TEAL), ("bot v2", BOT, GOLD)]):
            x = -3.4 if i == 0 else 3.4
            head = Text(name, font_size=32, color=color).move_to([x, 2.3, 0])

            # key presses on a shared 0–3 s axis, measured from the first key
            kt = key_downs(rows)
            span = 3000
            xl, xr, y = x - 2.6, x + 2.6, 1.2
            axis = Line([xl, y, 0], [xr, y, 0], color=GREY, stroke_width=2)
            ticks = VGroup(*[Line([xl + (xr - xl) * min((t - kt[0]) / span, 1), y - 0.15, 0],
                                  [xl + (xr - xl) * min((t - kt[0]) / span, 1), y + 0.3, 0],
                                  color=color, stroke_width=5) for t in kt])
            lab = Text("key presses (first 3 s)", font_size=18, color=GREY).next_to(axis, DOWN, buff=0.15)

            st = session_stats(rows)
            lines = VGroup(
                Text(f"average gap between keys   {st['gap']:.0f} ms", font_size=24),
                Text(f"rhythm unevenness (CV)       {st['cv']:.2f}", font_size=24),
                Text(f"mouse path straightness      {st['straight']:.2f}", font_size=24),
                Text(f"… + {N_FEATURES - 3} more numbers", font_size=24, color=GREY),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([x, -0.9, 0])
            cols.append((head, axis, ticks, lab, lines))

        for head, axis, ticks, lab, lines in cols:
            self.play(FadeIn(head), Create(axis), FadeIn(lab))
            self.play(LaggedStart(*[GrowFromPoint(t, t.get_bottom()) for t in ticks], lag_ratio=0.25), run_time=1.5)
        self.wait(0.5)
        for _, _, _, _, lines in cols:
            self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in lines], lag_ratio=0.3), run_time=1.8)

        cap = S.caption(f"Every session becomes {N_FEATURES} numbers. That row is all the catcher ever sees.", size=26)
        self.play(FadeIn(cap))
        self.wait(3)

        # which numbers matter most
        self.play(*[FadeOut(m) for m in self.mobjects])
        t2 = Text("Which numbers does the Random Forest rely on most?", font_size=32).to_edge(UP, buff=0.6)
        self.play(FadeIn(t2))
        bars = VGroup()
        top = TOP_FEATURES[0][1]
        for i, (name, v) in enumerate(TOP_FEATURES):
            y = 1.5 - i * 0.85
            lab = Text(name, font_size=24).move_to([-0.5, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=5.4 * v / top, height=0.5, fill_color=TEAL if i != 3 else GOLD,
                            fill_opacity=0.9, stroke_width=0).move_to([-0.2, y, 0], aligned_edge=LEFT)
            bars.add(VGroup(lab, bar))
        for b in bars:
            self.play(FadeIn(b[0]), GrowFromEdge(b[1], LEFT), run_time=0.6)
        cap = S.caption("Mostly typing rhythm. A person never presses keys like a metronome.", size=26)
        self.play(FadeIn(cap))
        self.wait(3)


# ================================================================ Act 3 · Scene 3
def load_features():
    pts = {}
    for r in csv.DictReader(open(S.DATA / "features.csv")):
        pts.setdefault(r["kind"], []).append((float(r["keystroke_interval_mean_ms"]),
                                              float(r["keystroke_interval_cv"])))
    return pts


class Act3_Cloud(Scene):
    """Every session as one dot: average key gap vs. rhythm unevenness."""

    def construct(self):
        pts = load_features()
        allx = np.concatenate([[p[0] for p in v] for v in pts.values()])
        ally = np.concatenate([[p[1] for p in v] for v in pts.values()])
        xmax = float(np.percentile(allx, 98)) * 1.1
        ymax = max(float(np.percentile(ally, 98)) * 1.1, 1.0)

        W, H = 9.0, 4.3
        o = np.array([-4.6, -2.1, 0])
        to = lambda x, y: o + RIGHT * W * min(x / xmax, 1.02) + UP * H * min(y / ymax, 1.02)

        title = Text("Every session = one dot", font_size=34).to_edge(UP, buff=0.4)
        xa = Arrow(o, o + RIGHT * (W + 0.3), buff=0, color=GREY, stroke_width=2, max_tip_length_to_length_ratio=0.02)
        ya = Arrow(o, o + UP * (H + 0.3), buff=0, color=GREY, stroke_width=2, max_tip_length_to_length_ratio=0.03)
        xl = Text("average gap between key presses (ms) →", font_size=22, color=GREY).next_to(xa, DOWN, buff=0.2)
        yl = Text("rhythm unevenness (CV) →", font_size=22, color=GREY).rotate(PI / 2).next_to(ya, LEFT, buff=0.2)
        xt = VGroup(*[Text(f"{v:.0f}", font_size=18, color=GREY).next_to(to(v, 0), DOWN, buff=0.08)
                      for v in np.linspace(0, xmax, 5)[1:]])
        self.play(FadeIn(title), Create(xa), Create(ya), FadeIn(xl), FadeIn(yl), FadeIn(xt))

        legend = VGroup()
        captions = {
            "human": "Real people: uneven rhythm, all over the place.",
            "naive": "Naive bot: every gap identical, so CV = 0. One tight pile.",
            "evasive": "Evasive bot v1: some jitter, but still too regular.",
            "evasive_v2": "Evasive bot v2: moves into the human cloud, but not all the way.",
        }
        cap = None
        rng = np.random.default_rng(0)
        for kind in ["human", "naive", "evasive", "evasive_v2"]:
            if kind not in pts:
                continue
            p = pts[kind]
            if len(p) > 150:
                p = [p[i] for i in rng.choice(len(p), 150, replace=False)]
            dots = VGroup(*[Dot(to(x, y), radius=0.045, color=KIND_COLOR[kind], fill_opacity=0.75) for x, y in p])
            item = VGroup(Dot(radius=0.07, color=KIND_COLOR[kind]),
                          Text(KIND_NAME[kind], font_size=20, color=KIND_COLOR[kind])).arrange(RIGHT, buff=0.12)
            legend.add(item)
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UR, buff=0.5).shift(DOWN * 0.6)
            new_cap = S.caption(captions[kind], size=24)
            anims = [LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.01, run_time=1.6),
                     FadeIn(item)]
            anims.append(ReplacementTransform(cap, new_cap) if cap else FadeIn(new_cap))
            self.play(*anims)
            cap = new_cap
            self.wait(2.2)

        end = S.caption(f"Two numbers can't separate them. The forest uses all {N_FEATURES} at once.", size=24)
        self.play(ReplacementTransform(cap, end))
        self.wait(3)


# ================================================================ Act 3 · Scene 4
class Act3_Rounds(Scene):
    """The arms race, round by round, from results.csv."""

    def construct(self):
        title = Text("Catcher vs. breaker, round by round", font_size=34).to_edge(UP, buff=0.35)
        self.play(FadeIn(title))

        series = [("humans wrongly flagged", GREY), ("naive caught", RED),
                  ("evasive caught", ORANGE), ("v2 caught", GOLD)]
        legend = VGroup(*[VGroup(Square(0.22, fill_color=c, fill_opacity=1, stroke_width=0),
                                 Text(n, font_size=20, color=c)).arrange(RIGHT, buff=0.12) for n, c in series])
        legend.arrange(RIGHT, buff=0.45).next_to(title, DOWN, buff=0.3)
        unit = Text("bar labels in %", font_size=18, color=GREY).next_to(legend, DOWN, buff=0.15)

        base_y, Hmax, bw = -1.75, 3.2, 0.5
        floor = Line([-6.6, base_y, 0], [6.6, base_y, 0], color=GREY, stroke_width=1.5)
        self.play(Create(floor), FadeIn(legend[:3]), FadeIn(unit))

        def column(i, values, which):
            cx = -5.2 + i * 2.6
            g = VGroup()
            for j in which:
                v = values[j]
                h = max(Hmax * v / 100, 0.03)
                bar = Rectangle(width=bw, height=h, fill_color=series[j][1], fill_opacity=0.95, stroke_width=0)
                bar.move_to([cx + (j - 1.5) * (bw + 0.06), base_y + h / 2, 0])
                num = Text(f"{v:.0f}" if v >= 1 or v == 0 else f"{v:.1f}", font_size=17,
                           color=series[j][1]).next_to(bar, UP, buff=0.08)
                g.add(VGroup(bar, num))
            return g

        labels = [Text(f"{i + 1}\n{r[0]}", font_size=18, color=GREY, line_spacing=0.9)
                  .move_to([-5.2 + i * 2.6, base_y - 0.6, 0]) for i, r in enumerate(ROUNDS)]

        steps = [
            "Round 1: trained on naive bots only. Naive: 100% caught. Evasive: walks right through.",
            "Round 2: smarter features alone change nothing. It has never seen an evasive bot.",
            "Round 3: show it evasive examples: 100% caught. Cost: 1% of real people flagged.",
            "Round 4: better features + examples: 100% caught, 0% of people flagged.",
        ]
        cap = None
        for i in range(4):
            col = column(i, ROUNDS[i][1:], [0, 1, 2])
            new_cap = S.caption(steps[i], size=22)
            self.play(FadeIn(labels[i]),
                      LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in col], lag_ratio=0.15),
                      ReplacementTransform(cap, new_cap) if cap else FadeIn(new_cap), run_time=1.2)
            self.play(*[FadeIn(b[1]) for b in col], run_time=0.4)
            cap = new_cap
            self.wait(2.2)

        # the breaker answers
        new_cap = S.caption("Then the breaker answered with v2. The best catcher still misses 1 in 3.", size=22)
        v2 = [column(i, ROUNDS[i][1:], [3]) for i in range(4)]
        self.play(FadeIn(legend[3]), ReplacementTransform(cap, new_cap))
        self.play(LaggedStart(*[GrowFromEdge(c[0][0], DOWN) for c in v2], lag_ratio=0.2), run_time=1.5)
        self.play(*[FadeIn(c[0][1]) for c in v2], run_time=0.4)
        cap = new_cap
        self.wait(3)

        # the catcher answers back
        col = column(4, ROUNDS[4][1:], [0, 1, 2, 3])
        new_cap = S.caption("Round 5: retrain on v2: 99.9% caught, still 0% of people flagged.", size=22)
        self.play(FadeIn(labels[4]),
                  LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in col], lag_ratio=0.15),
                  ReplacementTransform(cap, new_cap), run_time=1.4)
        self.play(*[FadeIn(b[1]) for b in col], run_time=0.4)
        self.wait(3.5)


# ================================================================ Act 3 · Scene 5
class Act3_Close(Scene):
    """The loop, and the takeaway."""

    def construct(self):
        br = VGroup(Circle(radius=1.1, color=GOLD, stroke_width=4),
                    Text("Breaker", font_size=32, color=GOLD)).move_to(LEFT * 3.5 + UP * 0.6)
        ca = VGroup(Circle(radius=1.1, color=TEAL, stroke_width=4),
                    Text("Catcher", font_size=32, color=TEAL)).move_to(RIGHT * 3.5 + UP * 0.6)
        top = CurvedArrow(br.get_top() + RIGHT * 0.4, ca.get_top() + LEFT * 0.4, angle=-PI / 3, color=GOLD)
        bot = CurvedArrow(ca.get_bottom() + LEFT * 0.4, br.get_bottom() + RIGHT * 0.4, angle=-PI / 3, color=TEAL)
        tl = Text("a new bot gets through", font_size=24, color=GOLD).next_to(top, UP, buff=0.1)
        bl = Text("retrain on it, catch it again", font_size=24, color=TEAL).next_to(bot, DOWN, buff=0.1)

        self.play(FadeIn(br), FadeIn(ca))
        self.play(Create(top), FadeIn(tl))
        self.play(Create(bot), FadeIn(bl))
        for _ in range(2):
            self.play(Indicate(br, color=GOLD), run_time=0.7)
            self.play(Indicate(ca, color=TEAL), run_time=0.7)
        line = Text("A catcher only catches what it has already seen.", font_size=32).to_edge(DOWN, buff=1.0)
        self.play(Write(line), run_time=1.5)
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in self.mobjects])
        a = Text("No genius required.", font_size=48)
        b = Text("Numbers. Filters. Practice.", font_size=48, color=GOLD)
        c = Text("Built on our own CAPTCHAs and our own catcher, for this class only.", font_size=22, color=GREY)
        VGroup(a, b).arrange(DOWN, buff=0.6)
        c.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(a, shift=UP * 0.2))
        self.wait(0.8)
        self.play(Write(b), run_time=1.6)
        self.play(FadeIn(c))
        self.wait(3)