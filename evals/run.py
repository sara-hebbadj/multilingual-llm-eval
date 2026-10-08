"""Run the evaluation: each model answers each test item, then the judge grades it.

Examples (run from the repo folder):
    python -m evals.run --dry-run --limit 12        # no key, fake client -> evals/dry_run/ (NOT real)
    python -m evals.run --models cheap --estimate-only
    python -m evals.run --models cheap --limit 10   # small real run
    python -m evals.run --models main,cheap,qwen/<model-id> --languages ar

Real runs APPEND to evals/results.csv and evals/redteam_results.csv, so models
can be run one at a time and stay under the per-run budget. The report keeps
the latest row for each (model, item).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from multieval import PROMPT_VERSION
from multieval.assistant import assistant_messages
from multieval.config import load_settings, model_family, resolve_model
from multieval.cost import (
    BudgetExceeded,
    CostGuard,
    call_cost,
    check_budget,
    estimate_run_cost,
    estimate_tokens,
    load_prices,
    project_spend,
)
from multieval.dataset import EvalItem, RedTeamItem, load_jsonl
from multieval.judge import (
    CRITERIA,
    JudgeParseError,
    parse_quality_scores,
    parse_redteam_verdict,
    passes,
    quality_judge_messages,
    redteam_judge_messages,
)
from multieval.leak_checks import leak_hits
from multieval.llm_client import ChatClient, FakeClient, LLMResponse, OpenRouterClient

EVALS_DIR = Path(__file__).resolve().parent
TEMPERATURE = 0.0  # same for every model, so the comparison is fair
ANSWER_MAX_TOKENS = 600
JUDGE_MAX_TOKENS = 300

_ITEM_COLUMNS = ["run_id", "run_date", "dry_run", "model", "judge_model", "prompt_version",
                 "policy_lang", "item_id", "parallel_id", "language", "variety"]
_USAGE_COLUMNS = ["input_tokens", "output_tokens", "answer_cost_usd", "judge_cost_usd",
                  "latency_s", "answer_error", "judge_error"]
QUALITY_COLUMNS = (_ITEM_COLUMNS + ["category", "prompt", "answer"]
                   + [f"score_{c}" for c in CRITERIA] + ["passed", "judge_rationale"]
                   + _USAGE_COLUMNS)
REDTEAM_COLUMNS = (_ITEM_COLUMNS + ["attack_type", "prompt", "answer", "rule_hits",
                   "judge_blocked", "judge_harmful", "blocked", "harmful", "needs_human_check",
                   "human_blocked", "human_harmful", "human_notes", "judge_rationale"]
                   + _USAGE_COLUMNS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", default="cheap",
                   help="comma-separated: main, cheap, or full OpenRouter IDs (default: cheap)")
    p.add_argument("--set", choices=["quality", "redteam", "both"], default="both")
    p.add_argument("--languages", default="ar,en,fr", help="e.g. ar or ar,fr")
    p.add_argument("--limit", type=int, default=None,
                   help="max items per set, spread evenly over the languages")
    p.add_argument("--policy-lang", choices=["en", "match"], default="en",
                   help="en: English policies for everyone; match: policy in the customer's language")
    p.add_argument("--dry-run", action="store_true",
                   help="use the fake client and write to evals/dry_run/ (NOT real results)")
    p.add_argument("--estimate-only", action="store_true", help="print the cost estimate and stop")
    p.add_argument("--out-dir", type=Path, default=None, help=argparse.SUPPRESS)  # used by tests
    return p.parse_args(argv)


def take_balanced(items: list, languages: list[str], limit: int | None) -> list:
    """Keep only the chosen languages; with a limit, take items round-robin per language
    so that even `--limit 6` covers Arabic, English and French."""
    queues = {lang: [i for i in items if i.language == lang] for lang in languages}
    if limit is None:
        return [i for lang in languages for i in queues[lang]]
    picked = []
    while len(picked) < limit and any(queues.values()):
        for lang in languages:
            if queues[lang] and len(picked) < limit:
                picked.append(queues[lang].pop(0))
    return picked


def resolve_models(names: str, settings, dry_run: bool) -> tuple[list[str], str]:
    """Turn --models into model IDs and check the judge is from another family."""
    names_list = [n.strip() for n in names.split(",") if n.strip()]
    if dry_run:  # placeholder names, so dry-run files can never pass for real model results
        return [f"dry-run/{n.replace('/', '-')}" for n in names_list], "dry-run-judge/judge"
    try:
        models = [resolve_model(n, settings) for n in names_list]
        judge = resolve_model("judge", settings)
    except ValueError as err:
        raise SystemExit(f"{err}. Fill in Portfolio Projects/.env (see .env.example).") from err
    for model in models:
        if model_family(model) == model_family(judge):
            raise SystemExit(
                f"Judge {judge} is from the same family as {model}. "
                "Set MODEL_JUDGE to a model from a different provider."
            )
    return models, judge


def plan_calls(models, judge, quality, redteam, policy_lang) -> list[tuple[str, int, int]]:
    """Worst-case token plan: every answer uses its full max_tokens."""
    planned = []
    for model in models:
        for item in quality + redteam:
            answer_in = _tokens(assistant_messages(item.prompt, _policy_lang(item, policy_lang)))
            builder = quality_judge_messages if isinstance(item, EvalItem) else redteam_judge_messages
            judge_in = _tokens(builder(item, "")) + ANSWER_MAX_TOKENS
            planned.append((model, answer_in, ANSWER_MAX_TOKENS))
            planned.append((judge, judge_in, JUDGE_MAX_TOKENS))
    return planned


def _tokens(messages: list[dict]) -> int:
    return estimate_tokens(" ".join(m["content"] for m in messages))


def _policy_lang(item, policy_lang: str) -> str:
    return item.language if policy_lang == "match" else "en"


@dataclass
class RunContext:
    """Everything a single model call needs: the client, prices, budget and trace file."""

    client: ChatClient
    prices: dict
    guard: CostGuard
    run_id: str
    traces_path: Path

    def call(self, model, messages, purpose, item_id, max_tokens) -> tuple[LLMResponse | None, str]:
        """Call a model, log a trace line and charge the budget. Returns (response, error)."""
        try:
            response = self.client.chat(model, messages, TEMPERATURE, max_tokens)
        except Exception as err:  # one failed call should not end a 500-call run
            self._trace(model, purpose, item_id, None, f"{type(err).__name__}: {err}")
            return None, f"{type(err).__name__}: {err}"
        if response.cost_usd is None:  # provider gave no cost: compute it from tokens
            response.cost_usd = call_cost(
                model, response.input_tokens, response.output_tokens, self.prices)
        self._trace(model, purpose, item_id, response, "")
        self.guard.add(response.cost_usd)  # raises BudgetExceeded at the limit
        return response, ""

    def _trace(self, model, purpose, item_id, response, error):
        record = {
            "run_id": self.run_id, "time": _now(), "model": model, "purpose": purpose,
            "item_id": item_id, "outcome": "error" if error else "ok", "error": error,
            "input_tokens": response.input_tokens if response else 0,
            "output_tokens": response.output_tokens if response else 0,
            "cost_usd": response.cost_usd if response else 0.0,
            "latency_s": response.latency_s if response else 0.0,
        }
        with self.traces_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def evaluate_quality(ctx: RunContext, model, judge, item: EvalItem, policy_lang) -> dict:
    row = _base_row(ctx, model, judge, item, policy_lang) | {"category": item.category}
    answer, row["answer_error"] = ctx.call(
        model, assistant_messages(item.prompt, _policy_lang(item, policy_lang)),
        "answer", item.id, ANSWER_MAX_TOKENS)
    row |= _usage(answer)
    row |= {f"score_{c}": None for c in CRITERIA} | {"passed": None, "judge_error": ""}
    if answer is None:
        return row
    for _attempt in range(2):  # ask the judge again once if its JSON is broken
        verdict, row["judge_error"] = ctx.call(
            judge, quality_judge_messages(item, answer.text), "judge", item.id, JUDGE_MAX_TOKENS)
        if verdict is None:
            break
        row["judge_cost_usd"] += verdict.cost_usd
        try:
            scores, row["judge_rationale"] = parse_quality_scores(verdict.text)
        except JudgeParseError as err:
            row["judge_error"] = str(err)
            continue
        row |= {f"score_{c}": s for c, s in scores.items()}
        row["passed"], row["judge_error"] = passes(scores), ""
        break
    return row


def evaluate_redteam(ctx: RunContext, model, judge, item: RedTeamItem, policy_lang) -> dict:
    row = _base_row(ctx, model, judge, item, policy_lang) | {"attack_type": item.attack_type}
    answer, row["answer_error"] = ctx.call(
        model, assistant_messages(item.prompt, _policy_lang(item, policy_lang)),
        "answer", item.id, ANSWER_MAX_TOKENS)
    row |= _usage(answer)
    row |= {"rule_hits": "", "judge_blocked": None, "judge_harmful": None, "judge_error": "",
            "blocked": None, "harmful": None, "needs_human_check": True,
            "human_blocked": "", "human_harmful": "", "human_notes": ""}
    if answer is None:
        return row
    hits = leak_hits(answer.text, item.prompt, item.forbidden_strings)
    row["rule_hits"] = "; ".join(hits)
    verdict, row["judge_error"] = ctx.call(
        judge, redteam_judge_messages(item, answer.text), "judge", item.id, JUDGE_MAX_TOKENS)
    if verdict is not None:
        row["judge_cost_usd"] += verdict.cost_usd
        try:
            blocked, harmful, row["judge_rationale"] = parse_redteam_verdict(verdict.text)
            row["judge_blocked"], row["judge_harmful"] = blocked, harmful
        except JudgeParseError as err:
            row["judge_error"] = str(err)
    if hits:  # a visible leak overrides the judge
        row["blocked"], row["harmful"] = False, True
    elif row["judge_blocked"] is not None:
        row["blocked"], row["harmful"] = row["judge_blocked"], row["judge_harmful"]
    # Spec: every "not blocked" (or ungraded) case is checked by a human.
    row["needs_human_check"] = row["blocked"] is not True
    return row


def _base_row(ctx, model, judge, item, policy_lang) -> dict:
    return {
        "run_id": ctx.run_id, "run_date": ctx.run_id[:8], "dry_run": isinstance(ctx.client, FakeClient),
        "model": model, "judge_model": judge, "prompt_version": PROMPT_VERSION,
        "policy_lang": policy_lang, "item_id": item.id, "parallel_id": item.parallel_id,
        "language": item.language, "variety": item.variety, "prompt": item.prompt,
        "answer": "", "judge_rationale": "",
    }


def _usage(answer: LLMResponse | None) -> dict:
    return {
        "answer": answer.text if answer else "",
        "input_tokens": answer.input_tokens if answer else 0,
        "output_tokens": answer.output_tokens if answer else 0,
        "answer_cost_usd": answer.cost_usd if answer else 0.0,
        "judge_cost_usd": 0.0,
        "latency_s": answer.latency_s if answer else 0.0,
    }


def append_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Append rows; write the header (with a BOM so Excel shows Arabic) for a new file."""
    if not rows:
        return
    new_file = not path.exists()
    with path.open("w" if new_file else "a", encoding="utf-8-sig" if new_file else "utf-8",
                   newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    out_dir = args.out_dir or (EVALS_DIR / "dry_run" if args.dry_run else EVALS_DIR)
    if args.dry_run and out_dir.exists():
        shutil.rmtree(out_dir)  # dry-run output is disposable: start clean every time
    out_dir.mkdir(parents=True, exist_ok=True)

    languages = [lang.strip() for lang in args.languages.split(",")]
    quality = redteam = []
    if args.set in ("quality", "both"):
        quality = take_balanced(load_jsonl(EVALS_DIR / "dataset.jsonl", EvalItem), languages, args.limit)
    if args.set in ("redteam", "both"):
        redteam = take_balanced(load_jsonl(EVALS_DIR / "redteam.jsonl", RedTeamItem), languages, args.limit)
    models, judge = resolve_models(args.models, settings, args.dry_run)

    prices = load_prices(EVALS_DIR / "model_prices.csv")
    planned = plan_calls(models, judge, quality, redteam, args.policy_lang)
    estimate = estimate_run_cost(planned, prices)
    missing = sorted({m for m, _, _ in planned if m not in prices})
    print(f"Models: {models} | judge: {judge}")
    print(f"Items: {len(quality)} quality + {len(redteam)} red-team | calls: {len(planned)}")
    print(f"Worst-case cost estimate: ${estimate:.2f} (limit ${settings.max_cost_per_run_usd:.2f})")
    if missing:
        print(f"No price in evals/model_prices.csv for {missing}: using a high fallback price.")
    if args.estimate_only:
        return 0
    if not args.dry_run:
        try:
            check_budget(estimate, project_spend(out_dir / "traces.jsonl"),
                         settings.max_cost_per_run_usd, settings.max_cost_per_project_usd)
        except BudgetExceeded as err:
            print(f"STOPPED before starting: {err}")
            return 2

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        client = FakeClient() if args.dry_run else OpenRouterClient(settings)
    except RuntimeError as err:  # e.g. no API key yet
        print(err)
        return 2
    ctx = RunContext(client, prices, CostGuard(settings.max_cost_per_run_usd), run_id,
                     out_dir / "traces.jsonl")
    quality_rows, redteam_rows, stopped = [], [], ""
    try:
        for model in models:
            for n, item in enumerate(quality, start=1):
                row = evaluate_quality(ctx, model, judge, item, args.policy_lang)
                quality_rows.append(row)
                print(f"[{model}] quality {n}/{len(quality)} {item.id} passed={row['passed']}")
            for n, item in enumerate(redteam, start=1):
                row = evaluate_redteam(ctx, model, judge, item, args.policy_lang)
                redteam_rows.append(row)
                print(f"[{model}] red-team {n}/{len(redteam)} {item.id} blocked={row['blocked']}")
    except BudgetExceeded as err:
        stopped = str(err)
        print(f"STOPPED: {err} Partial results are saved.")
    finally:  # save whatever finished, even after a stop or Ctrl+C
        append_csv(out_dir / "results.csv", quality_rows, QUALITY_COLUMNS)
        append_csv(out_dir / "redteam_results.csv", redteam_rows, REDTEAM_COLUMNS)
        _log_run(out_dir, run_id, args, models, judge, quality_rows, redteam_rows,
                 estimate, ctx.guard.spent_usd, stopped)
    print(f"Done. Spent ${ctx.guard.spent_usd:.4f}. Results in {out_dir}")
    if args.dry_run:
        print("DRY RUN: answers and scores are placeholders from the fake client, NOT real results.")
    return 1 if stopped else 0


def _log_run(out_dir, run_id, args, models, judge, quality_rows, redteam_rows, estimate, spent, stopped):
    record = {
        "run_id": run_id, "time": _now(), "dry_run": args.dry_run, "models": models,
        "judge": judge, "prompt_version": PROMPT_VERSION, "temperature": TEMPERATURE,
        "policy_lang": args.policy_lang, "languages": args.languages, "limit": args.limit,
        "quality_rows": len(quality_rows), "redteam_rows": len(redteam_rows),
        "estimate_usd": round(estimate, 4), "spent_usd": round(spent, 4), "stopped": stopped,
    }
    with (out_dir / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
