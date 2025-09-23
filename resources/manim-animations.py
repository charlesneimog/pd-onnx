from manim import *

class FolderScene(Scene):
    def construct(self):
        # --- First part: note about the folder ---
        lines_note = [
            "There is a folder called 'Flute'",
            "in the same directory as the main patch.",
            "It contains subfolders for each audio category",
            "we want to train."
        ]
        note_objects = [Text(line, font_size=42, color=WHITE) for line in lines_note]
        group_note = VGroup(*note_objects).arrange(DOWN, center=True, buff=0.6)
        group_note.move_to(ORIGIN)
        
        # Animate the note lines
        for txt in note_objects:
            self.play(FadeIn(txt, shift=UP), run_time=0.8)
            self.wait(0.1)
        
        # Fade out the note before showing example
        self.play(*[FadeOut(txt) for txt in note_objects], run_time=1)
        self.wait(0.3)
        
        # --- Second part: example subfolders ---
        lines_example = [
            "For example:",
            "attack-percussive",
            "jet-whistle",
            "silence",
            "som+ar",
        ]
        example_objects = [Text(line, font_size=40, color=WHITE) for line in lines_example]
        group_example = VGroup(*example_objects).arrange(DOWN, center=True, buff=0.5)
        group_example.move_to(ORIGIN)
        
        # Animate the example lines
        for txt in example_objects:
            self.play(FadeIn(txt, shift=UP), run_time=0.8)
            self.wait(0.2)
        
        # Fade out the example
        self.play(*[FadeOut(txt) for txt in example_objects], run_time=1)

class FancyTextScene(Scene):
    def construct(self):
        # --- Multi-line text lines (shorter and concise) ---
        lines = [
            "Timbre Recognition Training",
            "Using Pd, py4pd, and pd-onnx",
            "Code and resources in the description!"
        ]
               
        # --- Create Text objects with style ---
        text_objects = []
        for i, line in enumerate(lines):
            txt = Text(
                line,
                font_size=50 if i == 0 else 40,
                color=WHITE
            )
            text_objects.append(txt)
        
        # --- Position lines vertically centered ---
        group = VGroup(*text_objects).arrange(DOWN, center=True, buff=0.7)
        group.move_to(ORIGIN)  # center of the screen
        
        # --- Animate lines in sequentially, faster ---
        for txt in text_objects:
            self.play(FadeIn(txt, shift=UP), run_time=1)
            self.wait(0.3)
        
        # --- Fade everything out smoothly ---
        self.play(*[FadeOut(txt) for txt in text_objects], run_time=1.5)


class InferenceScene(Scene):
    def construct(self):
        # --- Multi-line text ---
        lines = [
            "After training and saving the model,",
            "load it into the onnx object,",
            "and use timbreIDLib's mfcc~",
            "to perform inference."
        ]
                      
        # --- Create Text objects ---
        text_objects = []
        for i, line in enumerate(lines):
            txt = Text(
                line,
                font_size=45 if i == 0 else 40,
                color=WHITE
            )
            text_objects.append(txt)
        
        # --- Arrange and center ---
        group = VGroup(*text_objects).arrange(DOWN, center=True, buff=0.7)
        group.move_to(ORIGIN)
        
        # --- Animate lines sequentially ---
        for txt in text_objects:
            self.play(FadeIn(txt, shift=UP), run_time=1)
            self.wait(0.3)
        
        # --- Fade out smoothly ---
        self.play(*[FadeOut(txt) for txt in text_objects], run_time=1.5)

from manim import *

class GitHubNoteScene(Scene):
    def construct(self):
        # --- Final note ---
        final_note = Text(
            "Check more on GitHub:",
            font_size=40,
            color=WHITE
        ).move_to(UP*1)  # slightly above center

        self.play(FadeIn(final_note, shift=UP))
        self.wait(0.5)

        # --- GitHub repository links ---
        links = [
            "https://github.com/charlesneimog/pd-onnx",
            "https://github.com/charlesneimog/py4pd"
        ]
        repo_links_text = []

        # Start links further down for bigger space
        previous_text = final_note
        for i, link in enumerate(links):
            link_text = Text(
                link,
                font_size=36,
                color=WHITE,
                slant=ITALIC
            ).next_to(previous_text, DOWN, buff=1.2 if i == 0 else 0.8)
            repo_links_text.append(link_text)
            self.play(FadeIn(link_text, shift=UP))
            previous_text = link_text
            self.wait(0.5)

        # --- Fade everything out smoothly ---
        self.wait(1)
        self.play(FadeOut(final_note), *[FadeOut(txt) for txt in repo_links_text], run_time=1)

