import json

import pytest

from multieval.cost import (
    FALLBACK_PRICE,
    BudgetExceeded,
    CostGuard,
    call_cost,
    check_budget,
    estimate_run_cost,
    estimate_tokens,
    load_prices,
    project_spend,
)


def test_estimate_tokens_is_pessimistic():
    assert estimate_tokens("a" * 300) == 100  # 3 characters per token
    assert estimate_tokens("") == 1


def test_call_cost_uses_price_table_and_fallback():
    prices = {"cheap/model": (0.5, 1.5)}
    assert call_cost("cheap/model", 1_000_000, 1_000_000, prices) == pytest.approx(2.0)
    unknown = call_cost("new/model", 1_000_000, 0, prices)
    assert unknown == pytest.approx(FALLBACK_PRICE[0])


def test_load_prices_skips_comment_lines(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("# note\nmodel_id,input_usd_per_mtok,output_usd_per_mtok\na/b,0.1,0.4\n")
    assert load_prices(path) == {"a/b": (0.1, 0.4)}
    assert load_prices(tmp_path / "missing.csv") == {}


def test_estimate_run_cost_sums_calls():
    prices = {"m/x": (1.0, 2.0)}
    planned = [("m/x", 1_000_000, 0), ("m/x", 0, 1_000_000)]
    assert estimate_run_cost(planned, prices) == pytest.approx(3.0)


def test_check_budget_blocks_expensive_runs():
    check_budget(estimate=2.0, already_spent=0, run_limit=3, project_limit=10)  # fine
    with pytest.raises(BudgetExceeded, match="MAX_COST_PER_RUN_USD"):
        check_budget(estimate=3.5, already_spent=0, run_limit=3, project_limit=10)
    with pytest.raises(BudgetExceeded, match="MAX_COST_PER_PROJECT_USD"):
        check_budget(estimate=2.0, already_spent=9.0, run_limit=3, project_limit=10)


def test_cost_guard_stops_at_the_limit():
    guard = CostGuard(limit_usd=1.0)
    guard.add(0.6)
    with pytest.raises(BudgetExceeded):
        guard.add(0.6)


def test_project_spend_reads_traces(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps({"cost_usd": c}) for c in [0.1, 0.25, None]) + "\n")
    assert project_spend(path) == pytest.approx(0.35)
    assert project_spend(tmp_path / "none.jsonl") == 0.0
