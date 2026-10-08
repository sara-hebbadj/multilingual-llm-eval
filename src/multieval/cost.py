"""Cost estimate before a run, and a guard that stops a run that gets too expensive.

Two checks protect the budget (limits come from .env):
1. Before the run: a worst-case estimate. If it is above MAX_COST_PER_RUN_USD,
   or would push the project total above MAX_COST_PER_PROJECT_USD, the run
   does not start.
2. During the run: `CostGuard` adds up the real cost of every call and stops
   the run as soon as the total passes the per-run limit.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

# Pessimistic on purpose: English is about 4 characters per token, Arabic often
# 2-3. Over-estimating tokens makes the guard stop too early, never too late.
CHARS_PER_TOKEN = 3

# Price used for a model missing from evals/model_prices.csv (US$ per million
# tokens, input and output). Deliberately expensive, so an unknown model makes
# the guard stop until Sara adds its real price.
FALLBACK_PRICE = (15.0, 75.0)


class BudgetExceeded(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def load_prices(path: Path) -> dict[str, tuple[float, float]]:
    """Read model_id -> (input, output) US$ per million tokens."""
    if not path.exists():
        return {}
    prices = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            prices[row["model_id"].strip()] = (
                float(row["input_usd_per_mtok"]),
                float(row["output_usd_per_mtok"]),
            )
    return prices


def call_cost(model: str, input_tokens: int, output_tokens: int, prices: dict) -> float:
    price_in, price_out = prices.get(model, FALLBACK_PRICE)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def estimate_run_cost(planned_calls: list[tuple[str, int, int]], prices: dict) -> float:
    """planned_calls: one (model, input_tokens, max_output_tokens) per call."""
    return sum(call_cost(m, tin, tout, prices) for m, tin, tout in planned_calls)


def project_spend(traces_path: Path) -> float:
    """Total US$ already spent on real runs, from the trace log."""
    if not traces_path.exists():
        return 0.0
    total = 0.0
    with traces_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += json.loads(line).get("cost_usd") or 0.0
    return total


def check_budget(estimate: float, already_spent: float, run_limit: float, project_limit: float):
    if estimate > run_limit:
        raise BudgetExceeded(
            f"Estimated run cost ${estimate:.2f} is above MAX_COST_PER_RUN_USD=${run_limit:.2f}. "
            "Use --limit, fewer models, or a cheaper model."
        )
    if already_spent + estimate > project_limit:
        raise BudgetExceeded(
            f"Spent so far ${already_spent:.2f} + estimate ${estimate:.2f} would pass "
            f"MAX_COST_PER_PROJECT_USD=${project_limit:.2f}. Ask before spending more."
        )


class CostGuard:
    """Keeps a running total during a run and stops it at the limit."""

    def __init__(self, limit_usd: float):
        self.limit_usd = limit_usd
        self.spent_usd = 0.0

    def add(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        if self.spent_usd > self.limit_usd:
            raise BudgetExceeded(
                f"Run stopped: spent ${self.spent_usd:.2f}, limit ${self.limit_usd:.2f}."
            )
