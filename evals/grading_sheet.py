"""Create the blind grading sheet for Sara: 60 answers, 20 per language.

    python -m evals.grading_sheet              # from evals/results.csv (after a real run)
    python -m evals.grading_sheet --dry-run    # practice sheet from evals/dry_run/

Writes human_grades.csv (what Sara fills in, in Excel or with app/grading_app.py)
and human_grades_key.csv (which model wrote each answer; do not open while grading).
An existing sheet with grades in it is never overwritten unless --force is given.
"""

from __future__ import annotations

import argparse

from evals.report import load_latest
from evals.run import EVALS_DIR
from multieval.dataset import EvalItem, load_jsonl
from multieval.grading import has_any_grade, make_sheet, read_sheet, write_sheet


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--per-language", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="overwrite a sheet that has grades")
    args = p.parse_args(argv)

    folder = EVALS_DIR / "dry_run" if args.dry_run else EVALS_DIR
    sheet_path, key_path = folder / "human_grades.csv", folder / "human_grades_key.csv"
    if sheet_path.exists() and has_any_grade(read_sheet(sheet_path)) and not args.force:
        print(f"{sheet_path} already has grades. Use --force to replace it (grades would be lost).")
        return 1

    results = load_latest(folder / "results.csv")
    items = load_jsonl(EVALS_DIR / "dataset.jsonl", EvalItem)
    sheet, key = make_sheet(results, items, args.per_language, args.seed)
    write_sheet(sheet, sheet_path)
    key.to_csv(key_path, index=False, encoding="utf-8-sig")
    counts = sheet["language"].value_counts().sort_index().to_dict()
    print(f"Wrote {len(sheet)} answers to grade {counts} -> {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
