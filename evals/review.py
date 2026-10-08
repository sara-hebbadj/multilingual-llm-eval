"""Native-speaker review of the Arabic and French prompts (Sara's job before any live run).

    python -m evals.review export   # -> evals/review_ar_fr.csv, open it in Excel
    python -m evals.review apply    # copy the fixes back into dataset.jsonl / redteam.jsonl

In the sheet, each Arabic or French prompt sits next to its English version.
Fill in `natural` with yes (fine as it is) or no, and when it is "no" write a
better prompt in `new_prompt`. `apply` swaps in every non-empty new_prompt and
marks every row with a yes/no answer as reviewed=true. `python -m evals.stats`
then shows how many items have been reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evals.run import EVALS_DIR
from multieval.dataset import EvalItem, RedTeamItem

FILES = {"dataset": (EVALS_DIR / "dataset.jsonl", EvalItem),
         "redteam": (EVALS_DIR / "redteam.jsonl", RedTeamItem)}
SHEET = EVALS_DIR / "review_ar_fr.csv"
COLUMNS = ["file", "id", "language", "variety", "category", "english_version", "prompt",
           "natural", "new_prompt", "comment"]


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def export(sheet: Path = SHEET) -> int:
    rows = []
    for name, (path, _model) in FILES.items():
        lines = read_lines(path)
        english = {d["parallel_id"]: d["prompt"] for d in lines if d["language"] == "en"}
        for d in lines:
            if d["language"] in ("ar", "fr"):
                rows.append({
                    "file": name, "id": d["id"], "language": d["language"],
                    "variety": d["variety"], "category": d.get("category") or d.get("attack_type"),
                    "english_version": english.get(d["parallel_id"], ""), "prompt": d["prompt"],
                    "natural": "yes" if d.get("reviewed") else "", "new_prompt": "", "comment": "",
                })
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel shows Arabic
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Arabic/French prompts to {sheet}")
    return len(rows)


def apply(sheet: Path = SHEET) -> int:
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        decisions = {(r["file"], r["id"]): r for r in csv.DictReader(f)}
    changed = reviewed = 0
    for name, (path, model) in FILES.items():
        lines = read_lines(path)
        for d in lines:
            row = decisions.get((name, d["id"]))
            if row is None:
                continue
            if row["natural"].strip().lower() in ("yes", "no", "y", "n"):
                d["reviewed"] = True
                reviewed += 1
            if row["new_prompt"].strip():
                d["prompt"] = row["new_prompt"].strip()
                changed += 1
        for d in lines:  # refuse to save anything that breaks the schema
            model.model_validate(d)
        path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in lines),
                        encoding="utf-8")
    print(f"Marked {reviewed} items reviewed, replaced {changed} prompts. "
          "Run python -m evals.stats to re-check the sets.")
    return changed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("action", choices=["export", "apply"])
    args = p.parse_args(argv)
    export() if args.action == "export" else apply()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
