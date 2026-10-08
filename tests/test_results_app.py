"""Smoke test for the results explorer in app/app.py (skipped without the [app] extra, as in CI).

No network: "Try one prompt" is tested in demo mode only.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("gradio")
APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("results_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explorer_builds_on_the_main_run_only(app):
    assert app.build() is not None
    assert set(app.QUALITY["run_id"]) == {app.MAIN_RUN} and len(app.QUALITY) == 540  # 180 items x 3 models
    assert len(app.REDTEAM) == 135  # 45 attacks x 3 models


def test_quality_filters_and_rationale(app):
    summary, table, pick = app.filter_quality(app.ALL, "ar", app.ALL, "gulf", app.ALL)
    assert set(table["language"]) == {"ar"} and set(table["variety"]) == {"gulf"}
    assert f"{len(table)} answers" in summary
    assert "Judge rationale" in app.quality_details(pick["value"])
    _, failed, _ = app.filter_quality(app.ALL, app.ALL, app.ALL, app.ALL, "failed")
    assert set(failed["passed"]) == {"False"}


def test_redteam_filters(app):
    summary, table, pick = app.filter_redteam(app.ALL, app.ALL, "prompt_injection", app.ALL)
    assert set(table["attack_type"]) == {"prompt_injection"} and "blocked" in summary
    assert "Judge rationale" in app.redteam_details(pick["value"])


def test_try_prompt_without_key_shows_demo_mode(app, monkeypatch):
    monkeypatch.setattr(app, "LIVE", False)
    answer, checks, count = app.try_prompt("How many days do I have to return a product?", app.MODELS[0], 0)
    assert "Demo mode" in answer and count == 0
