"""How well does the LLM judge agree with Sara (the human grader)?

- Exact match: share of answers where both gave the same score.
- Within one: share where the scores differ by at most 1 point.
- Cohen's kappa: agreement corrected for chance. 1 = perfect, 0 = no better
  than two graders guessing with the same score habits, below 0 = worse than chance.
- Quadratic-weighted kappa: like kappa, but a 4-vs-5 disagreement counts much
  less than a 1-vs-5 one. It suits 1-5 scales, where "close" is meaningful.

Kappa is written out by hand (not imported) so every step can be explained.
tests/test_agreement.py checks it against textbook values and scikit-learn.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from multieval.judge import CRITERIA


def exact_match(a: Sequence, b: Sequence) -> float:
    _check_same_length(a, b)
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def within_one(a: Sequence[int], b: Sequence[int]) -> float:
    _check_same_length(a, b)
    return sum(abs(x - y) <= 1 for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: Sequence, b: Sequence, weights: str | None = None) -> float:
    """Cohen's kappa = 1 - observed disagreement / disagreement expected by chance.

    weights=None treats every disagreement the same (classic kappa);
    "linear" or "quadratic" weigh a disagreement by the distance between scores.
    Returns NaN when chance disagreement is zero (both graders always gave the
    same single label), because kappa is undefined there.
    """
    _check_same_length(a, b)
    n = len(a)
    labels = sorted(set(a) | set(b))
    pair_counts = Counter(zip(a, b, strict=True))
    count_a, count_b = Counter(a), Counter(b)

    observed = expected = 0.0
    for i in labels:
        for j in labels:
            w = _disagreement_weight(i, j, weights)
            observed += w * pair_counts[(i, j)] / n
            expected += w * (count_a[i] / n) * (count_b[j] / n)
    if expected == 0:
        return math.nan
    return 1 - observed / expected


def _disagreement_weight(i, j, weights: str | None) -> float:
    if weights is None:
        return 0.0 if i == j else 1.0
    if weights == "linear":
        return abs(i - j)
    if weights == "quadratic":
        return (i - j) ** 2
    raise ValueError(f"unknown weights: {weights!r}")


def _check_same_length(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("need two non-empty lists of the same length")


def agreement_rows(pairs: list[dict], group: str = "all") -> list[dict]:
    """One row per criterion (plus pass/fail) for paired human/judge grades.

    Each item in `pairs` has human_<criterion> and judge_<criterion> integers
    and human_pass / judge_pass booleans.
    """
    rows = []
    for criterion in CRITERIA + ["pass"]:
        human = [p[f"human_{criterion}"] for p in pairs]
        judge = [p[f"judge_{criterion}"] for p in pairs]
        is_score = criterion != "pass"
        rows.append({
            "group": group,
            "criterion": criterion,
            "n": len(pairs),
            "exact_match_pct": round(100 * exact_match(human, judge), 1),
            "within_one_pct": round(100 * within_one(human, judge), 1) if is_score else None,
            "cohen_kappa": _round(cohen_kappa(human, judge)),
            "quadratic_kappa": _round(cohen_kappa(human, judge, "quadratic")) if is_score else None,
        })
    return rows


def disagreements(pairs: list[dict], min_gap: int = 2) -> list[dict]:
    """Answers where judge and human differ by >= min_gap on a criterion, or on pass/fail."""
    found = []
    for p in pairs:
        gaps = {c: p[f"judge_{c}"] - p[f"human_{c}"] for c in CRITERIA}
        big = {c: g for c, g in gaps.items() if abs(g) >= min_gap}
        if big or p["human_pass"] != p["judge_pass"]:
            found.append({
                "item_id": p["item_id"],
                "model": p["model"],
                "language": p["language"],
                "criteria_judge_minus_human": "; ".join(f"{c} {g:+d}" for c, g in big.items()),
                "human_pass": p["human_pass"],
                "judge_pass": p["judge_pass"],
                "judge_rationale": p.get("judge_rationale", ""),
                "human_notes": p.get("human_notes", ""),
            })
    return found


def _round(value: float) -> float | None:
    return None if math.isnan(value) else round(value, 3)
