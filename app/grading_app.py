"""Grading app: shows one answer at a time and saves Sara's grades to the CSV sheet.

    pip install -e ".[app]"
    python -m evals.grading_sheet          # create evals/human_grades.csv first
    python app/grading_app.py              # then open http://127.0.0.1:7860
    python app/grading_app.py --dry-run    # practice on the fake dry-run answers

The sheet is blind: no model name and no judge score are shown. Every click on
"Save and next" writes to disk, so you can stop and come back at any time.
Excel works too: fill the human_* columns with 1-5 and save as CSV UTF-8.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage pings from a local tool
import gradio as gr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # so the app also runs without `pip install -e .`

from multieval.grading import is_graded, read_sheet, save_grade  # noqa: E402
from multieval.judge import CRITERIA  # noqa: E402

SHEET = REPO / "evals" / "human_grades.csv"  # practice: python app/grading_app.py --dry-run
RUBRIC = (REPO / "docs" / "rubric.md").read_text(encoding="utf-8")
LABELS = {
    "accuracy": "Accuracy",
    "policy": "Policy compliance",
    "tone": "Tone and helpfulness",
    "language": "Language quality",
    "format": "Format",
}


def first_ungraded() -> int:
    sheet = read_sheet(SHEET)
    for i, row in sheet.iterrows():
        if not is_graded(row):
            return int(i)
    return len(sheet) - 1


def show(index: int):
    sheet = read_sheet(SHEET)
    index = max(0, min(int(index), len(sheet) - 1))
    row = sheet.iloc[index]
    done = sum(is_graded(r) for _, r in sheet.iterrows())
    rtl = row["language"] == "ar"  # right-to-left boxes for Arabic
    header = (
        f"**{done} / {len(sheet)} graded** · answer `{row['answer_id']}` · "
        f"{row['language']} ({row['variety']}) · {row['category']}"
    )
    scores = [int(row[f"human_{c}"]) if str(row[f"human_{c}"]).strip() else None for c in CRITERIA]
    return [
        index,
        header,
        gr.update(value=row["prompt"], rtl=rtl),
        row["reference_facts"].replace(" | ", "\n"),
        row["expected_behavior"],
        gr.update(value=row["answer"], rtl=rtl),
        *scores,
        row["human_notes"],
    ]


def save_and_next(index: int, *values):
    *scores, notes = values
    if any(s is None for s in scores):
        gr.Warning("Give all five scores before saving.")
        return show(index)
    answer_id = read_sheet(SHEET).iloc[int(index)]["answer_id"]
    save_grade(SHEET, answer_id, dict(zip(CRITERIA, scores, strict=True)), notes or "")
    return show(int(index) + 1)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Lumi Skin answer grading") as app:
        gr.Markdown(
            "# Grade assistant answers (blind)\nScore 1-5 per criterion. "
            "Pass rule: accuracy ≥ 4 and policy = 5."
        )
        index = gr.Number(value=first_ungraded(), visible=False, precision=0)
        header = gr.Markdown()
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Customer message", interactive=False, lines=3)
                facts = gr.Textbox(label="Reference facts", interactive=False, lines=4)
                expected = gr.Textbox(label="Expected behaviour", interactive=False, lines=2)
            with gr.Column():
                answer = gr.Textbox(label="Assistant answer", interactive=False, lines=12)
        with gr.Row():
            radios = [gr.Radio([1, 2, 3, 4, 5], label=LABELS[c]) for c in CRITERIA]
        notes = gr.Textbox(label="Notes (why you disagree, what was wrong)", lines=2)
        with gr.Row():
            back = gr.Button("Previous")
            save = gr.Button("Save and next", variant="primary")
        with gr.Accordion("Rubric", open=False):
            gr.Markdown(RUBRIC)

        outputs = [index, header, prompt, facts, expected, answer, *radios, notes]
        app.load(show, inputs=index, outputs=outputs)
        save.click(save_and_next, inputs=[index, *radios, notes], outputs=outputs)
        back.click(lambda i: show(int(i) - 1), inputs=index, outputs=outputs)
    return app


if __name__ == "__main__":
    if "--dry-run" in sys.argv:  # practice on the fake dry-run answers
        SHEET = REPO / "evals" / "dry_run" / "human_grades.csv"
    if not SHEET.exists():
        raise SystemExit(f"{SHEET} not found. Run: python -m evals.grading_sheet")
    build_app().launch()
