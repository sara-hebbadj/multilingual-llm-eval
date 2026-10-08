"""End to end with the fake client: run -> report -> grading sheet -> agreement. No network."""

import csv
import json

import pytest

from evals import report, run
from multieval.assistant import CANARY
from multieval.cost import BudgetExceeded, CostGuard
from multieval.dataset import EvalItem, RedTeamItem, load_jsonl
from multieval.grading import make_sheet, read_sheet, save_grade, write_sheet
from multieval.judge import CRITERIA
from multieval.llm_client import FakeClient, LLMResponse

QUALITY = load_jsonl(run.EVALS_DIR / "dataset.jsonl", EvalItem)
REDTEAM = load_jsonl(run.EVALS_DIR / "redteam.jsonl", RedTeamItem)
JUDGE_OK = '{"accuracy": 5, "policy": 5, "tone": 4, "language": 5, "format": 5, "rationale": "fine"}'


def _ctx(tmp_path, client):
    return run.RunContext(client, {}, CostGuard(3.0), "20261008T000000Z", tmp_path / "traces.jsonl")


def test_dry_run_writes_every_output(tmp_path):
    code = run.main(["--dry-run", "--limit", "6", "--models", "a,b", "--out-dir", str(tmp_path)])
    assert code == 0
    with (tmp_path / "results.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12  # 6 items x 2 models
    assert list(rows[0]) == run.QUALITY_COLUMNS
    assert {r["language"] for r in rows} == {"ar", "en", "fr"}
    assert all(r["dry_run"] == "True" and r["model"].startswith("dry-run/") for r in rows)
    traces = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert len(traces) == 48  # (answer + judge) x (6 quality + 6 red-team) x 2 models
    assert json.loads((tmp_path / "runs.jsonl").read_text())["dry_run"] is True

    assert report.main(["--folder", str(tmp_path)]) == 0
    summary = (tmp_path / "summary.csv").read_text()
    assert "dry-run/a" in summary and "pass_rate_pct" in summary
    for chart in ["score_by_model_language", "pass_rate", "cost_vs_quality", "redteam_block_rate"]:
        assert (tmp_path / "charts" / f"{chart}.png").exists()


def test_judge_gets_a_second_chance_after_broken_json(tmp_path):
    client = FakeClient(["Returns are accepted within 14 days.", "not json", JUDGE_OK])
    row = run.evaluate_quality(_ctx(tmp_path, client), "m/x", "j/y", QUALITY[0], "en")
    assert row["passed"] is True and row["score_tone"] == 4 and row["judge_error"] == ""
    assert len(client.calls) == 3


def test_leak_overrides_a_lenient_judge(tmp_path):
    client = FakeClient([f"Sure! My internal reference is {CANARY}.",
                         '{"blocked": true, "harmful": false, "rationale": "refused"}'])
    row = run.evaluate_redteam(_ctx(tmp_path, client), "m/x", "j/y", REDTEAM[0], "en")
    assert row["judge_blocked"] is True
    assert row["blocked"] is False and row["harmful"] is True
    assert row["needs_human_check"] is True and "canary" in row["rule_hits"]


def test_failed_call_is_recorded_not_fatal(tmp_path):
    class Broken(FakeClient):
        def chat(self, *args, **kwargs):
            raise TimeoutError("provider timeout")

    row = run.evaluate_quality(_ctx(tmp_path, Broken()), "m/x", "j/y", QUALITY[0], "en")
    assert "provider timeout" in row["answer_error"] and row["passed"] is None
    assert json.loads((tmp_path / "traces.jsonl").read_text())["outcome"] == "error"


def test_cost_guard_stops_a_run(tmp_path):
    class Pricey(FakeClient):
        def chat(self, model, messages, temperature=0.0, max_tokens=700):
            return LLMResponse("x", model, 10, 10, cost_usd=2.0, latency_s=0.0)

    ctx = _ctx(tmp_path, Pricey())
    ctx.call("m/x", [{"role": "user", "content": "hi"}], "answer", "id", 10)  # $2 of $3
    with pytest.raises(BudgetExceeded):
        ctx.call("m/x", [{"role": "user", "content": "hi"}], "answer", "id", 10)


def test_grading_sheet_is_blind_and_agreement_is_computed(tmp_path):
    run.main(["--dry-run", "--limit", "9", "--models", "a,b", "--set", "quality",
              "--out-dir", str(tmp_path)])
    results = report.load_latest(tmp_path / "results.csv")
    sheet, key = make_sheet(results, QUALITY, per_language=2, seed=1)
    assert len(sheet) == 6 and sorted(sheet["language"].value_counts()) == [2, 2, 2]
    assert not {"model", "score_accuracy", "judge_rationale"} & set(sheet.columns)  # blind

    write_sheet(sheet, tmp_path / "human_grades.csv")
    key.to_csv(tmp_path / "human_grades_key.csv", index=False, encoding="utf-8-sig")
    for answer_id in sheet["answer_id"]:
        save_grade(tmp_path / "human_grades.csv", answer_id, dict.fromkeys(CRITERIA, 4), "ok")
    assert (read_sheet(tmp_path / "human_grades.csv")["human_tone"] == "4").all()

    pairs = report.agreement_pairs(tmp_path)
    assert len(pairs) == 6 and all(p["human_accuracy"] == 4 for p in pairs)
    table = report.agreement_table(pairs)
    assert set(table["group"]) == {"all", "ar", "en", "fr"}
