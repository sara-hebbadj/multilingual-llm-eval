import csv
import importlib.util

import pandas as pd
import pytest

from evals import review
from evals.run import EVALS_DIR
from multieval.grading import SHEET_COLUMNS, read_sheet, write_sheet


def test_review_sheet_pairs_every_arabic_and_french_prompt_with_english(tmp_path):
    sheet = tmp_path / "review.csv"
    assert review.export(sheet) == 150  # 120 quality + 30 red-team items
    with sheet.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert {r["language"] for r in rows} == {"ar", "fr"}
    assert all(r["english_version"] for r in rows)


def test_grading_app_builds_and_saves(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    spec = importlib.util.spec_from_file_location("grading_app", EVALS_DIR.parent / "app" / "grading_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    sheet = pd.DataFrame([{c: "" for c in SHEET_COLUMNS} | {
        "answer_id": "g001", "item_id": "ar-pol-001", "language": "ar", "variety": "gulf",
        "category": "policy_fact", "prompt": "كم يوم؟", "reference_facts": "a | b",
        "expected_behavior": "x", "answer": "١٤ يومًا"}])
    path = tmp_path / "human_grades.csv"
    write_sheet(sheet, path)
    monkeypatch.setattr(app, "SHEET", path)

    assert app.build_app() is not None
    shown = app.show(0)
    assert shown[3] == "a\nb"  # reference facts, one per line
    app.save_and_next(0, 5, 4, 5, 3, 5, "fine")
    saved = read_sheet(path).iloc[0]
    assert (saved["human_accuracy"], saved["human_format"], saved["human_notes"]) == ("5", "5", "fine")
