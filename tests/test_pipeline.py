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


def test_report_has_attack_grid_status_and_keeps_written_findings(tmp_path):
    run.main(["--dry-run", "--limit", "15", "--models", "a", "--out-dir", str(tmp_path)])
    assert report.main(["--folder", str(tmp_path)]) == 0
    draft_path = tmp_path / "REPORT_DRAFT.md"
    draft = draft_path.read_text(encoding="utf-8")
    assert report.FINDINGS_TODO in draft
    # Write findings between the markers, regenerate: they must survive.
    draft_path.write_text(draft.replace(report.FINDINGS_TODO, "- Model a blocked every attack."),
                          encoding="utf-8")
    assert report.main(["--folder", str(tmp_path)]) == 0
    with (tmp_path / "redteam_by_attack_language.csv").open(encoding="utf-8") as f:
        long = list(csv.DictReader(f))
    assert {r["language"] for r in long} == {"ar", "en", "fr"}
    assert sum(int(r["n_attacks"]) for r in long) == 15
    draft = (tmp_path / "REPORT_DRAFT.md").read_text(encoding="utf-8")
    assert "By attack type and language" in draft and "Model a blocked every attack." in draft
    assert "uncalibrated LLM-judge score" in draft  # no human grades in this folder
    assert (tmp_path / "summary_by_variety.csv").exists()


def test_judge_gets_a_second_chance_after_broken_json(tmp_path):
    client = FakeClient(["Returns are accepted within 14 days.", "not json", JUDGE_OK])
    row = run.evaluate_quality(_ctx(tmp_path, client), "m/x", "j/y", QUALITY[0], "en")
    assert row["passed"] is True and row["score_tone"] == 4 and row["judge_error"] == ""
    assert len(client.calls) == 3


def test_judge_reply_cut_off_at_max_tokens_is_labelled(tmp_path):
    # Seen in the first live smoke run: a reasoning judge used its whole token limit
    # and the visible JSON stopped at '{"accuracy": 5, "'.
    class CutOff(FakeClient):
        def chat(self, model, messages, temperature=0.0, max_tokens=700):
            if model == "m/x":
                return LLMResponse("Returns: 14 days.", model, 10, 5, 0.0, 0.0, "stop")
            return LLMResponse('{"accuracy": 5, "', model, 10, max_tokens, 0.0, 0.0, "length", 290)

    row = run.evaluate_quality(_ctx(tmp_path, CutOff()), "m/x", "j/y", QUALITY[0], "en")
    assert row["passed"] is None and "cut off at max_tokens" in row["judge_error"]
    traces = [json.loads(line) for line in (tmp_path / "traces.jsonl").read_text().splitlines()]
    assert traces[-1]["finish_reason"] == "length" and traces[-1]["reasoning_tokens"] == 290
    assert run.JUDGE_MAX_TOKENS >= 1000  # room for hidden reasoning plus the JSON


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


def test_ctrl_c_is_recorded_in_the_run_log(tmp_path, monkeypatch):
    calls = {"n": 0}

    def chat_then_stop(self, model, messages, temperature=0.0, max_tokens=700):
        calls["n"] += 1
        if calls["n"] > 2:  # one answer + one judge call, then Ctrl+C
            raise KeyboardInterrupt
        return LLMResponse(JUDGE_OK, model, 10, 10, 0.0, 0.0, "stop")

    monkeypatch.setattr(run.FakeClient, "chat", chat_then_stop)
    code = run.main(["--dry-run", "--limit", "3", "--set", "quality", "--out-dir", str(tmp_path)])
    log = json.loads((tmp_path / "runs.jsonl").read_text())
    assert code == 1 and "interrupted" in log["stopped"] and log["quality_rows"] == 1
    assert log["answer_max_tokens"] == run.ANSWER_MAX_TOKENS


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
