# Notes for coding agents (multilingual-llm-eval)

This repository is public portfolio evidence for Sara Hebbadj. She must be able to explain every
line in an interview: keep functions short, names plain, and comment only non-obvious decisions.

## Layout

- `src/multieval/`: library code (config, client, assistant prompt, dataset schema, judge, leak
  rules, cost guard, agreement statistics, grading sheet). Prompts live in `src/multieval/prompts/`.
- `evals/`: command-line scripts (`python -m evals.<name>`), the test sets (`dataset.jsonl`,
  `redteam.jsonl`), prices (`model_prices.csv`) and real results. `evals/dry_run/` holds fake
  output and is git-ignored.
- `docs/rubric.md` is read by the judge at run time: editing it changes the judge.
- `app/grading_app.py`: Gradio grading UI (optional extra `[app]`).

## Commands

```bash
pip install -e ".[dev]"
ruff check . && pytest -q
python -m evals.stats                     # must report 0 validation problems
python -m evals.run --dry-run --limit 12  # fake client, writes evals/dry_run/
python -m evals.report --dry-run
```

## Rules

1. **No network in tests.** Use `FakeClient` (pass `replies=[...]` for exact responses). Only
   `OpenRouterClient` may call a model, and only from `evals.run` without `--dry-run`.
2. **Never invent numbers.** README and RESULTS tables show only measured values with the date,
   denominator, model IDs and command. LLM numbers stay "pending live run" until a real run exists
   in `evals/results.csv`. Dry-run numbers are never results.
3. **Secrets.** Keys come from `Portfolio Projects/.env` or the environment via `config.py`.
   Never print, log or commit them. `Settings` hides the key from `repr`.
4. **Test sets are the source of truth.** Edit `evals/dataset.jsonl` / `redteam.jsonl` (or use
   `python -m evals.review`), then run `python -m evals.stats`. Keep every `parallel_id` present
   once per language, reference facts in English, and every number in a fact present in
   `data/lumi-skin/`. Arabic and French wording changes need Sara's review (`reviewed: true`).
5. **Prompt changes.** If you edit anything in `src/multieval/prompts/` or `docs/rubric.md`, bump
   `PROMPT_VERSION` in `src/multieval/__init__.py` so results stay traceable.
6. **Judge independence.** The judge must come from a different provider family than every judged
   model; `evals.run.resolve_models` enforces it. Do not remove that check.
7. **Budget.** Do not raise `MAX_COST_PER_RUN_USD` (US$3) or `MAX_COST_PER_PROJECT_USD` (US$10)
   without asking Sara. Fill `evals/model_prices.csv` from openrouter.ai/models before full runs.
8. **Blind grading.** `evals/human_grades.csv` must never show the model name or judge scores; the
   mapping lives in `human_grades_key.csv`. Never overwrite a sheet that already has grades.
9. **Publishing** (GitHub repo, push, Hugging Face Space) needs Sara's explicit OK each time.

## Known gaps

- Reasoning models may spend part of `ANSWER_MAX_TOKENS` (600) on hidden reasoning; if answers come
  back empty or cut off, raise it in `evals/run.py` and re-estimate the cost.
- OpenRouter normally returns `usage.cost`; if it does not, cost is computed from
  `model_prices.csv`.
