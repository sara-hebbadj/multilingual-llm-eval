import math
import random

import pytest

from multieval.agreement import agreement_rows, cohen_kappa, disagreements, exact_match, within_one


def test_textbook_example():
    # Classic 50-item example: yes/yes 20, yes/no 5, no/yes 10, no/no 15.
    # Observed agreement 0.70, chance agreement 0.5*0.6 + 0.5*0.4 = 0.50,
    # so kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40.
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert exact_match(a, b) == pytest.approx(0.70)
    assert cohen_kappa(a, b) == pytest.approx(0.40)


def test_perfect_agreement_is_one():
    scores = [1, 2, 3, 4, 5, 5, 4]
    assert cohen_kappa(scores, scores) == pytest.approx(1.0)
    assert cohen_kappa(scores, scores, "quadratic") == pytest.approx(1.0)


def test_systematic_disagreement_is_negative():
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)


def test_agreement_no_better_than_chance_is_zero():
    # Rater b says "pass" on half of each of rater a's groups: pure chance.
    a = [True, True, False, False]
    b = [True, False, True, False]
    assert cohen_kappa(a, b) == pytest.approx(0.0)


def test_single_shared_label_is_undefined():
    # Both graders always gave 5: there is no chance disagreement to correct for.
    assert math.isnan(cohen_kappa([5, 5, 5], [5, 5, 5]))


def test_kappa_is_symmetric():
    a, b = [1, 2, 3, 3, 5, 4], [1, 3, 3, 2, 5, 5]
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa(b, a))
    assert cohen_kappa(a, b, "quadratic") == pytest.approx(cohen_kappa(b, a, "quadratic"))


def test_quadratic_weights_forgive_near_misses():
    human = [5, 4, 3, 2, 1, 5, 4, 3]
    near = [4, 5, 2, 3, 2, 4, 5, 2]  # every grade off by exactly one point
    assert cohen_kappa(human, near, "quadratic") > cohen_kappa(human, near)
    assert within_one(human, near) == 1.0
    assert exact_match(human, near) == 0.0


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_matches_scikit_learn(weights):
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = random.Random(0)
    for _ in range(20):
        a = [rng.randint(1, 5) for _ in range(40)]
        b = [min(5, max(1, x + rng.choice([-2, -1, 0, 0, 0, 1]))) for x in a]
        expected = sklearn.cohen_kappa_score(a, b, labels=[1, 2, 3, 4, 5], weights=weights)
        assert cohen_kappa(a, b, weights) == pytest.approx(expected)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1])
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1, 2], weights="cubic")


def _pair(item_id, human, judge, language="en"):
    from multieval.judge import CRITERIA, passes

    h = dict(zip(CRITERIA, human, strict=True))
    j = dict(zip(CRITERIA, judge, strict=True))
    return {"item_id": item_id, "model": "m", "language": language,
            **{f"human_{c}": h[c] for c in CRITERIA}, **{f"judge_{c}": j[c] for c in CRITERIA},
            "human_pass": passes(h), "judge_pass": passes(j)}


def test_agreement_rows_and_disagreements():
    pairs = [
        _pair("a", [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]),
        _pair("b", [4, 5, 4, 5, 5], [4, 5, 5, 5, 4]),
        _pair("c", [2, 5, 4, 5, 5], [5, 5, 4, 5, 5]),  # judge says pass, human says fail
    ]
    rows = {r["criterion"]: r for r in agreement_rows(pairs)}
    assert rows["accuracy"]["n"] == 3
    assert rows["accuracy"]["exact_match_pct"] == pytest.approx(66.7)
    assert rows["pass"]["exact_match_pct"] == pytest.approx(66.7)
    found = disagreements(pairs)
    assert [d["item_id"] for d in found] == ["c"]
    assert "accuracy +3" in found[0]["criteria_judge_minus_human"]
