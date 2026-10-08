import pytest

from evals.run import resolve_models, take_balanced
from multieval.config import Settings, model_family, resolve_model

SETTINGS = Settings(api_key="sk-secret", base_url="https://example.invalid", model_main="openai/main",
                    model_cheap="qwen/cheap", model_judge="google/judge",
                    max_cost_per_run_usd=3, max_cost_per_project_usd=10)


def test_key_is_never_shown():
    assert "sk-secret" not in repr(SETTINGS)


def test_resolve_aliases_and_full_ids():
    assert resolve_model("main", SETTINGS) == "openai/main"
    assert resolve_model("anthropic/some-model", SETTINGS) == "anthropic/some-model"
    with pytest.raises(ValueError):
        resolve_model("gpt", SETTINGS)  # neither alias nor provider/model


def test_unset_alias_is_an_error():
    empty = Settings(None, "x", None, None, None, 3, 10)
    with pytest.raises(ValueError, match="MODEL_CHEAP"):
        resolve_model("cheap", empty)


def test_model_family():
    assert model_family("anthropic/claude-x") == "anthropic"
    assert model_family("~Google/gemini") == "google"


def test_judge_must_be_from_another_family():
    models, judge = resolve_models("main,cheap", SETTINGS, dry_run=False)
    assert models == ["openai/main", "qwen/cheap"] and judge == "google/judge"
    with pytest.raises(SystemExit, match="same family"):
        resolve_models("google/other-model", SETTINGS, dry_run=False)


def test_dry_run_uses_placeholder_names():
    models, judge = resolve_models("main", SETTINGS, dry_run=True)
    assert models == ["dry-run/main"] and judge.startswith("dry-run-judge/")


class _Item:
    def __init__(self, language):
        self.language = language


def test_take_balanced_spreads_the_limit_over_languages():
    items = [_Item(lang) for lang in ["ar"] * 5 + ["en"] * 5 + ["fr"] * 5]
    picked = take_balanced(items, ["ar", "en", "fr"], limit=4)
    assert [i.language for i in picked] == ["ar", "en", "fr", "ar"]
    assert len(take_balanced(items, ["fr"], limit=None)) == 5
