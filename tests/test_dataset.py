"""The real test sets must stay valid: CI fails if an edit breaks them."""

import pytest
from pydantic import ValidationError

from evals.run import EVALS_DIR
from evals.stats import run_checks
from multieval.dataset import EvalItem, RedTeamItem, find_problems, load_jsonl, source_problems

QUALITY = load_jsonl(EVALS_DIR / "dataset.jsonl", EvalItem)
REDTEAM = load_jsonl(EVALS_DIR / "redteam.jsonl", RedTeamItem)
CATEGORIES = {"policy_fact", "product_advice", "complaint_tone", "refusal", "clarification",
              "formatting"}


def test_real_sets_pass_every_check():
    assert run_checks(QUALITY, REDTEAM) == []


def test_sizes_match_the_spec():
    for lang in ("ar", "en", "fr"):
        assert sum(i.language == lang for i in QUALITY) == 60
        assert sum(i.language == lang for i in REDTEAM) == 15
        assert {i.category for i in QUALITY if i.language == lang} == CATEGORIES


def test_arabic_covers_dialects_and_arabizi():
    varieties = {i.variety for i in QUALITY if i.language == "ar"}
    assert {"msa", "gulf", "levantine", "maghrebi", "arabizi"} <= varieties


def _item(**changes):
    base = dict(id="en-pol-001", language="en", variety="standard", category="policy_fact",
                prompt="How many days do I have to return a product?",
                reference_facts=["Products can be returned within 14 days of delivery."],
                expected_behavior="States the 14-day window.", source="policies#5",
                parallel_id="return-window")
    return EvalItem(**(base | changes))


def test_schema_rejects_mismatched_language_and_variety():
    with pytest.raises(ValidationError):
        _item(id="fr-pol-001")  # id says French, language says English
    with pytest.raises(ValidationError):
        _item(variety="gulf")  # a dialect label on an English item
    with pytest.raises(ValidationError):
        _item(unknown_field=1)


def test_script_check_catches_wrong_script():
    arabic_in_latin = _item(id="ar-pol-001", language="ar", variety="msa")
    assert any("no Arabic letters" in p for p in find_problems([arabic_in_latin]))


def test_grounding_check_catches_invented_numbers():
    policy = "You can return products within 14 days of delivery."
    wrong = _item(reference_facts=["Products can be returned within 30 days."])
    assert find_problems([_item()], knowledge_text=policy) == [
        "parallel_id return-window: languages ['en'] (expected ar, en, fr once each)"]
    assert any("number 30" in p for p in find_problems([wrong], knowledge_text=policy))


def test_source_check_catches_missing_section():
    policy = "## 5. Returns\n..."
    assert source_problems([_item()], policy, "", set()) == []
    assert source_problems([_item(source="policies#12")], policy, "", set())
    assert source_problems([_item(source="products:P999")], policy, "", {"P001"})
