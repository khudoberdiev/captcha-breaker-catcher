"""
Act 0 — intro: what a CAPTCHA is, and what Half A and Half B each do.
Manim Community v0.22.1. Uses only Text (no LaTeX). Reuses helpers from scenes.py.

Preview one scene:   manim -pql act0.py Act0_Captcha
Render all in HD:    manim -qh act0.py Act0_Captcha Act0_HalfA Act0_HalfB
"""
from manim import *

import scenes as S          # Page, cursor_shape, CAPTCHA_PNG, ANSWER, colours, background

TEAL, PURPLE, GOLD, GREY = S.TEAL, S.PURPLE, S.GOLD, S.GREY
RED = "#F07178"


# ---------------------------------------------------------------- little figures
def person(color=TEAL, h=1.4):
    head = Circle(radius=0.22, color=color, fill_color=color, fill_opacity=1, stroke_width=0)
    body = Sector(radius=0.42, angle=PI, start_angle=0, color=color, fill_opacity=1, stroke_width=0)
    body.next_to(head, DOWN, buff=0.08)
    g = VGroup(head, body)
    return g.scale_to_fit_height(h)


def robot(color=GOLD, h=1.4):
    head = RoundedRectangle(width=0.8, height=0.6, corner_radius=0.1, color=color,
                            fill_color=color, fill_opacity=1, stroke_width=0)
    eyes = VGroup(*[Square(0.12, fill_color=S.BG, fill_opacity=1, stroke_width=0).move_to(head.get_center() + RIGHT * dx + UP * 0.04)
                    for dx in (-0.18, 0.18)])
    mouth = Line(LEFT * 0.16, RIGHT * 0.16, color=S.BG, stroke_width=4).move_to(head.get_center() + DOWN * 0.16)
    ant = Line(head.get_top(), head.get_top() + UP * 0.18, color=color, stroke_width=4)
    knob = Dot(ant.get_end(), radius=0.06, color=color)
    body = RoundedRectangle(width=0.95, height=0.55, corner_radius=0.1, color=color,
                            fill_color=color, fill_opacity=1, stroke_width=0).next_to(head, DOWN, buff=0.08)
    g = VGroup(ant, knob, head, eyes, mouth, body)
    return g.scale_to_fit_height(h)


def captcha_img(width):
    img = ImageMobject(S.CAPTCHA_PNG)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.width = width
    return img


def tag(text, color):
    return Text(text, font_size=26, color=color).to_edge(UP, buff=0.35)


# ================================================================ Act 0 · Scene 1
class Act0_Captcha(Scene):
    """What a CAPTCHA is and its one job."""

    def construct(self):
        t = tag("Act 0 · Before we start", GREY)
        self.play(FadeIn(t))

        page = S.Page(width=6.4, center=DOWN * 0.2)
        page.group.remove(page.size)
        q = Text("You've seen this a thousand times.", font_size=34).next_to(page.box, UP, buff=0.4)
        self.play(FadeIn(page.group), FadeIn(q))
        self.wait(1.5)

        # the name
        self.play(FadeOut(q), page.group.animate.scale(0.75).to_edge(LEFT, buff=0.7))
        words = ["Completely", "Automated", "Public", "Turing test to tell",
                 "Computers and", "Humans", "Apart"]
        lines = VGroup(*[Text(w, font_size=34, t2c={"[0:1]": GOLD}) for w in words])
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=1.6).shift(DOWN * 0.35)
        name = Text("CAPTCHA", font_size=44, color=GOLD).next_to(lines, UP, buff=0.4).align_to(lines, LEFT)
        self.play(Write(name))
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in lines], lag_ratio=0.25), run_time=2.2)
        self.wait(2)

        # its one job
        self.play(FadeOut(lines), FadeOut(name), FadeOut(page.group))
        gate = VGroup(Rectangle(width=0.35, height=3.0, fill_color=GREY, fill_opacity=0.35, stroke_color=GREY),
                      Text("CAPTCHA", font_size=24, color=GOLD).rotate(PI / 2)).move_to(ORIGIN + DOWN * 0.2)
        gate[1].move_to(gate[0])
        hum = person().move_to(LEFT * 4.5 + UP * 0.6)
        bot = robot().move_to(LEFT * 4.5 + DOWN * 1.2)
        self.play(FadeIn(gate), FadeIn(hum), FadeIn(bot))
        self.play(hum.animate.move_to(RIGHT * 3.8 + UP * 0.6), run_time=1.6)
        ok = Text("✓", font_size=60, color=TEAL).next_to(hum, RIGHT, buff=0.3)
        self.play(FadeIn(ok, scale=1.4))
        self.play(bot.animate.move_to(LEFT * 0.75 + DOWN * 1.2), run_time=1.0, rate_func=rush_into)
        self.play(bot.animate.shift(LEFT * 0.5), run_time=0.3, rate_func=rush_from)
        no = Text("✗", font_size=60, color=RED).next_to(bot, LEFT, buff=0.3)
        self.play(FadeIn(no, scale=1.4))
        cap = S.caption("One job: let people in, keep bots out.", size=32)
        self.play(FadeIn(cap))
        self.wait(2.5)


# ================================================================ Act 0 · Scene 2
class Act0_HalfA(Scene):
    """Half A: read the CAPTCHA, then type the answer like a person."""

    def construct(self):
        t = tag("Half A · The Breaker", TEAL)
        self.play(FadeIn(t))

        # ---- Task 1: image -> CNN -> answer
        y1 = 1.5
        l1 = Text("Task 1 · read it", font_size=28, color=TEAL).move_to([-4.6, y1 + 1.15, 0], aligned_edge=LEFT)
        img = captcha_img(3.0).move_to([-3.6, y1, 0])
        cnn_box = RoundedRectangle(width=2.9, height=1.3, corner_radius=0.15, stroke_color=TEAL, stroke_width=3,
                                   fill_color=S.BG, fill_opacity=1).move_to([0.4, y1, 0])
        bars = VGroup(*[Rectangle(width=0.16, height=h, fill_color=TEAL, fill_opacity=0.8, stroke_width=0)
                        for h in (0.8, 0.6, 0.45, 0.3)]).arrange(RIGHT, buff=0.12).move_to(cnn_box).shift(LEFT * 0.75)
        cnn_lab = Text("CNN", font_size=30).move_to(cnn_box).shift(RIGHT * 0.55)
        cnn = VGroup(cnn_box, bars, cnn_lab)
        ans = Text(S.ANSWER, font_size=48, color=GOLD).move_to([4.3, y1, 0])
        a1 = Arrow(img.get_right(), cnn_box.get_left(), buff=0.2, color=GREY, stroke_width=4)
        a2 = Arrow(cnn_box.get_right(), ans.get_left(), buff=0.3, color=GREY, stroke_width=4)

        self.play(FadeIn(l1), FadeIn(img))
        self.play(GrowArrow(a1), FadeIn(cnn))
        self.play(LaggedStart(*[Indicate(b, color=GOLD, scale_factor=1.3) for b in bars], lag_ratio=0.2), run_time=1.2)
        self.play(GrowArrow(a2), Write(ans))
        self.wait(1.2)

        # ---- Task 2: the bot types it in, acting human
        y2 = -1.85
        l2 = Text("Task 2 · type it in, like a person", font_size=28, color=TEAL).move_to([-4.6, y2 + 1.7, 0], aligned_edge=LEFT)
        page = S.Page(width=3.9, center=[1.3, y2, 0])
        page.group.remove(page.size)
        bot = robot(h=1.3).move_to([-4.0, y2, 0])
        self.play(FadeIn(l2), FadeIn(page.group), FadeIn(bot))

        cursor = S.cursor_shape().scale(0.7)
        start = page.pt(30, 225)
        cursor.shift(start - cursor.get_vertices()[0])
        self.play(FadeIn(cursor))
        target = page.pt(200, 145)
        path = CubicBezier(start, page.pt(60, 120), page.pt(120, 215), target)
        tip = Dot(start, radius=0.001, fill_opacity=0)
        cursor.add_updater(lambda m: m.shift(tip.get_center() - m.get_vertices()[0]))
        trail = TracedPath(tip.get_center, stroke_color=GOLD, stroke_width=3, stroke_opacity=0.8)
        self.add(trail, tip)
        self.play(MoveAlongPath(tip, path), run_time=1.6, rate_func=smooth)

        typed = VGroup()
        pauses = [0.25, 0.45, 0.15, 0.6, 0.3]          # uneven on purpose
        for i, ch in enumerate(S.ANSWER):
            c = Text(ch, font_size=24, color=BLACK)
            if typed:
                c.next_to(typed, RIGHT, buff=0.05)
            else:
                c.move_to(page.input_box).align_to(page.input_box, LEFT).shift(RIGHT * 0.12)
            typed.add(c)
            self.add(c)
            self.wait(pauses[i % len(pauses)])

        btn = page.pt(200, 195)
        self.play(tip.animate.move_to(btn), run_time=0.7, rate_func=smooth)
        self.play(Indicate(page.button, color=GOLD), run_time=0.6)
        cursor.clear_updaters()

        cap = S.caption("The goal: beat the CAPTCHA at its one job, telling humans from bots.", size=26)
        self.play(FadeIn(cap))
        self.wait(2.5)


# ================================================================ Act 0 · Scene 3
class Act0_HalfB(Scene):
    """Half B: who typed that answer, a person or a bot?"""

    def construct(self):
        t = tag("Half B · The Catcher", PURPLE)
        self.play(FadeIn(t))

        # a session: mouse squiggle + key presses
        sess_box = RoundedRectangle(width=3.6, height=2.4, corner_radius=0.15, stroke_color=GREY, stroke_width=2,
                                    fill_color="#161b24", fill_opacity=1).move_to(LEFT * 4.3 + UP * 0.3)
        squig = VMobject(stroke_color=GOLD, stroke_width=3).set_points_smoothly(
            [sess_box.get_center() + np.array(p) for p in
             [(-1.3, -0.6, 0), (-0.7, 0.3, 0), (-0.1, -0.1, 0), (0.5, 0.5, 0), (1.2, 0.1, 0)]])
        ticks = VGroup(*[Line(UP * 0.12, DOWN * 0.12, color=GOLD, stroke_width=4).move_to(sess_box.get_bottom() + UP * 0.45 + RIGHT * x)
                         for x in (-1.2, -0.7, -0.45, 0.2, 0.9)])
        sess_lab = Text("one solve: mouse + keys", font_size=20, color=GREY).next_to(sess_box, DOWN, buff=0.2)
        self.play(FadeIn(sess_box), Create(squig), LaggedStart(*[FadeIn(k) for k in ticks], lag_ratio=0.2), FadeIn(sess_lab))

        catcher = VGroup(RoundedRectangle(width=2.6, height=1.4, corner_radius=0.15, stroke_color=PURPLE, stroke_width=3,
                                          fill_color=S.BG, fill_opacity=1),
                         Text("Catcher", font_size=32, color=PURPLE)).move_to(UP * 0.3)
        catcher[1].move_to(catcher[0])
        a1 = Arrow(sess_box.get_right(), catcher.get_left(), buff=0.2, color=GREY, stroke_width=4)
        self.play(GrowArrow(a1), FadeIn(catcher))

        card = RoundedRectangle(width=3.6, height=3.2, corner_radius=0.15, stroke_color=PURPLE, stroke_width=2,
                                fill_color="#161b24", fill_opacity=1).move_to(RIGHT * 4.3 + UP * 0.3)
        hum = person(h=1.1).move_to(card.get_center() + LEFT * 0.8 + UP * 0.35)
        bot = robot(h=1.1).move_to(card.get_center() + RIGHT * 0.8 + UP * 0.35)
        q = Text("human or bot?", font_size=28).move_to(card.get_center() + DOWN * 0.95)
        a2 = Arrow(catcher.get_right(), card.get_left(), buff=0.2, color=GREY, stroke_width=4)
        self.play(GrowArrow(a2), FadeIn(card), FadeIn(hum), FadeIn(bot), Write(q))
        for _ in range(2):
            self.play(Indicate(hum, color=TEAL), run_time=0.6)
            self.play(Indicate(bot, color=GOLD), run_time=0.6)

        cap = S.caption("It doesn't check the answer. It checks how the answer was typed.", size=28)
        self.play(FadeIn(cap))
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in self.mobjects])
        a = Text("Breaker", font_size=56, color=TEAL)
        vs = Text("vs.", font_size=36, color=GREY)
        b = Text("Catcher", font_size=56, color=PURPLE)
        VGroup(a, vs, b).arrange(RIGHT, buff=0.5)
        nxt = Text("First: how does a model read a CAPTCHA?", font_size=30, color=GREY).shift(DOWN * 1.3)
        self.play(FadeIn(a, shift=RIGHT * 0.3), FadeIn(b, shift=LEFT * 0.3), FadeIn(vs))
        self.play(FadeIn(nxt))
        self.wait(2.5)