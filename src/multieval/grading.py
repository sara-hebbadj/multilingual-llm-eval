"""The blind human-grading sheet (Sara grades 60 answers: 20 per language).

Blind means the sheet shows the question, the reference facts and the answer,
but NOT which model wrote it and NOT the judge's scores, so neither can sway
the human grade. The model name is kept in a separate key file and joined back
only when agreement is computed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from multieval.dataset import EvalItem
from multieval.judge import CRITERIA

HUMAN_COLUMNS = [f"human_{c}" for c in CRITERIA]
SHEET_COLUMNS = (["answer_id", "item_id", "language", "variety", "category", "prompt",
                  "reference_facts", "expected_behavior", "answer"]
                 + HUMAN_COLUMNS + ["human_notes"])


def make_sheet(results: pd.DataFrame, items: list[EvalItem], per_language: int = 20,
               seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick `per_language` judged answers per language at random (fixed seed).

    Returns (sheet, key). Only answers the judge scored are eligible, because the
    point is to compare the two.
    """
    judged = results[results["score_accuracy"].notna()]
    parts = [group.sample(n=min(per_language, len(group)), random_state=seed)
             for _, group in judged.groupby("language")]
    # Shuffle across languages and models so the grading order gives nothing away.
    picked = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    picked["answer_id"] = [f"g{n:03d}" for n in range(1, len(picked) + 1)]

    by_id = {item.id: item for item in items}
    picked["reference_facts"] = [" | ".join(by_id[i].reference_facts) for i in picked["item_id"]]
    picked["expected_behavior"] = [by_id[i].expected_behavior for i in picked["item_id"]]
    for column in HUMAN_COLUMNS + ["human_notes"]:
        picked[column] = ""
    key = picked[["answer_id", "model", "run_id", "item_id"]]
    return picked[SHEET_COLUMNS], key


def read_sheet(path: Path) -> pd.DataFrame:
    # Read everything as text so an empty grade stays "" (not NaN) and Arabic survives.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def write_sheet(sheet: pd.DataFrame, path: Path) -> None:
    sheet.to_csv(path, index=False, encoding="utf-8-sig")  # BOM: Excel shows Arabic correctly


def is_graded(row: pd.Series) -> bool:
    return all(str(row[c]).strip() in {"1", "2", "3", "4", "5"} for c in HUMAN_COLUMNS)


def has_any_grade(sheet: pd.DataFrame) -> bool:
    return bool((sheet[HUMAN_COLUMNS].astype(str).apply(lambda col: col.str.strip()) != "").any().any())


def save_grade(path: Path, answer_id: str, scores: dict[str, int], notes: str = "") -> None:
    """Write one answer's grades straight to disk, so nothing is lost if the app closes."""
    sheet = read_sheet(path)
    row = sheet.index[sheet["answer_id"] == answer_id][0]
    for criterion, score in scores.items():
        sheet.loc[row, f"human_{criterion}"] = str(int(score))
    sheet.loc[row, "human_notes"] = notes
    write_sheet(sheet, path)
