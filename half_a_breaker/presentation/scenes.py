from manim import *

class Hello(Scene):
    def construct(self):
        title = Text("How a CNN reads a CAPTCHA")
        self.play(Write(title))
        self.wait(1)