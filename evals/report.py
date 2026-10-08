"""Summary tables, charts and a report draft from the run results.

    python -m evals.report              # real results in evals/  -> charts in docs/charts/
    python -m evals.report --dry-run    # evals/dry_run/ (placeholder numbers, testing only)

Writes summary.csv (model x language), summary_by_category.csv,
redteam_summary.csv, agreement.csv + disagreements.csv (once Sara has graded),
five charts, and REPORT_DRAFT.md with every table filled in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from multieval.agreement import agreement_rows, disagreements  # noqa: E402
from multieval.dataset import EvalItem, RedTeamItem, load_jsonl  # noqa: E402
from multieval.grading import is_graded, read_sheet  # noqa: E402
from multieval.judge import CRITERIA, passes  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
SCORE_COLUMNS = [f"score_{c}" for c in CRITERIA]
LANGUAGES = ["ar", "en", "fr"]
LANGUAGE_NAMES = {"ar": "Arabic", "en": "English", "fr": "French"}
HUMAN_TARGET = 60  # answers Sara grades by hand (20 per language), from the build spec
# Findings written between these markers in REPORT_DRAFT.md survive a re-run of this script.
FINDINGS_START, FINDINGS_END = "<!-- findings:start -->", "<!-- findings:end -->"
FINDINGS_TODO = "> TODO (Sara): factual findings, each backed by a table below or an item ID."
# One fixed colour per model, in this order (a colour-blind-checked categorical palette).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
                 "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- loading
def load_latest(path: Path) -> pd.DataFrame:
    """Results file, keeping only the newest row for each (model, item)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.sort_values("run_id", kind="stable")
    return df.drop_duplicates(["model", "item_id"], keep="last").reset_index(drop=True)


def to_bool(series: pd.Series) -> pd.Series:
    """'True'/'yes' -> True, 'False'/'no' -> False, anything else -> missing."""
    mapping = {"true": True, "yes": True, "y": True, "false": False, "no": False, "n": False}
    return series.map(lambda v: mapping.get(str(v).strip().lower())).astype("boolean")


# ---------------------------------------------------------------- quality
def quality_summary(df: pd.DataFrame, by: str = "language") -> pd.DataFrame:
    """One row per model and value of `by`, plus an "all" row per model."""
    rows = []
    for model, part in df.groupby("model"):
        rows.append(_quality_row(model, by, "all", part))
        rows += [_quality_row(model, by, key, sub) for key, sub in part.groupby(by)]
    return pd.DataFrame(rows)


def _quality_row(model, by, key, part: pd.DataFrame) -> dict:
    graded = part[part["score_accuracy"].notna()]
    n = len(part)
    row = {"model": model, by: key, "n_answers": n, "n_graded": len(graded)}
    for c in CRITERIA:
        row[f"mean_{c}"] = _round(graded[f"score_{c}"].mean())
    row["mean_overall"] = _round(graded[SCORE_COLUMNS].mean(axis=1).mean())
    row["pass_rate_pct"] = _round(100 * to_bool(graded["passed"]).mean(), 1)
    row["answer_cost_per_100_usd"] = _round(100 * part["answer_cost_usd"].sum() / n, 4)
    row["judge_cost_per_100_usd"] = _round(100 * part["judge_cost_usd"].sum() / n, 4)
    row["tokens_per_100_answers"] = int(100 * (part["input_tokens"] + part["output_tokens"]).sum() / n)
    row["mean_latency_s"] = _round(part["latency_s"].mean())
    return row


def _round(value, digits: int = 2):
    return None if pd.isna(value) else round(float(value), digits)


# ---------------------------------------------------------------- red team
def final_redteam_verdicts(rt: pd.DataFrame) -> pd.DataFrame:
    """Sara's hand check (human_blocked / human_harmful = yes/no) overrides the automatic verdict."""
    rt = rt.copy()
    for col in ("blocked", "harmful"):
        human = to_bool(rt[f"human_{col}"]) if f"human_{col}" in rt else pd.Series(pd.NA, index=rt.index)
        rt[f"final_{col}"] = human.where(human.notna(), to_bool(rt[col]))
    rt["open_human_check"] = to_bool(rt["needs_human_check"]).fillna(True) & to_bool(
        rt["human_blocked"]).isna()
    return rt


def redteam_summary(rt: pd.DataFrame, by: str = "language") -> pd.DataFrame:
    rt = final_redteam_verdicts(rt)
    rows = []
    for model, part in rt.groupby("model"):
        for key, sub in [("all", part)] + list(part.groupby(by)):
            decided = sub[sub["final_blocked"].notna()]
            rows.append({
                "model": model, by: key, "n_attacks": len(sub), "n_decided": len(decided),
                "blocked_pct": _round(100 * decided["final_blocked"].astype(float).mean(), 1),
                "harmful_n": int(sub["final_harmful"].fillna(False).sum()),
                "rule_hit_n": int(sub["rule_hits"].fillna("").astype(str).str.len().gt(0).sum()),
                "open_human_checks": int(sub["open_human_check"].sum()),
            })
    return pd.DataFrame(rows)


def redteam_by_attack_language(rt: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per model x attack type x language (3 attacks per cell)."""
    rt = final_redteam_verdicts(rt)
    rows = []
    for (model, attack, lang), sub in rt.groupby(["model", "attack_type", "language"]):
        decided = sub[sub["final_blocked"].notna()]
        rows.append({
            "model": model, "attack_type": attack, "language": lang, "n_attacks": len(sub),
            "n_decided": len(decided), "blocked_n": int(decided["final_blocked"].astype(bool).sum()),
            "blocked_pct": _round(100 * decided["final_blocked"].astype(float).mean(), 1),
        })
    return pd.DataFrame(rows)


def attack_language_grid(long: pd.DataFrame) -> pd.DataFrame:
    """Readable grid for the report: 'blocked/decided' per language, e.g. '2/3'."""
    if long.empty:
        return long
    long = long.assign(cell=long["blocked_n"].astype(str) + "/" + long["n_decided"].astype(str))
    grid = long.pivot_table(index=["model", "attack_type"], columns="language", values="cell",
                            aggfunc="first")
    return grid.reindex(columns=[c for c in LANGUAGES if c in grid.columns]).reset_index()


def failed_examples(results: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Failed answers, lowest accuracy first, for the "failures with examples" section."""
    failed = results[to_bool(results["passed"]).eq(False).fillna(False)]
    failed = failed.sort_values(["score_accuracy", "score_policy"]).head(n)
    out = failed[["item_id", "model", "language", "category", "score_accuracy", "score_policy",
                  "judge_rationale"]].copy()
    out["judge_rationale"] = out["judge_rationale"].fillna("").astype(str).str.slice(0, 120)
    return out


def not_blocked_examples(rt: pd.DataFrame) -> pd.DataFrame:
    rt = final_redteam_verdicts(rt)
    missed = rt[rt["final_blocked"].eq(False).fillna(False)]
    return missed[["item_id", "model", "language", "attack_type", "rule_hits", "human_blocked"]]


# ---------------------------------------------------------------- judge vs human
def agreement_pairs(folder: Path) -> list[dict]:
    """Join Sara's graded rows with the judge's scores for exactly the same answer."""
    sheet_path, key_path = folder / "human_grades.csv", folder / "human_grades_key.csv"
    if not sheet_path.exists() or not key_path.exists():
        return []
    sheet = read_sheet(sheet_path)
    graded = sheet[sheet.apply(is_graded, axis=1)] if len(sheet) else sheet
    if graded.empty:
        return []
    key = pd.read_csv(key_path, dtype=str, encoding="utf-8-sig")
    results = pd.read_csv(folder / "results.csv", dtype={"run_id": str}, encoding="utf-8-sig")
    judge_cols = ["run_id", "model", "item_id", "judge_rationale"] + SCORE_COLUMNS
    merged = graded.merge(key, on=["answer_id", "item_id"]).merge(
        results[judge_cols], on=["run_id", "model", "item_id"])
    pairs = []
    for _, r in merged.iterrows():
        human = {c: int(r[f"human_{c}"]) for c in CRITERIA}
        judge = {c: int(r[f"score_{c}"]) for c in CRITERIA}
        pairs.append({
            "item_id": r["item_id"], "model": r["model"], "language": r["language"],
            **{f"human_{c}": human[c] for c in CRITERIA},
            **{f"judge_{c}": judge[c] for c in CRITERIA},
            "human_pass": passes(human), "judge_pass": passes(judge),
            "judge_rationale": r["judge_rationale"], "human_notes": r["human_notes"],
        })
    return pairs


def agreement_table(pairs: list[dict]) -> pd.DataFrame:
    rows = agreement_rows(pairs, "all")
    for lang in LANGUAGES:
        subset = [p for p in pairs if p["language"] == lang]
        if subset:
            rows += agreement_rows(subset, lang)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts
def _style(ax, title: str, ylabel: str, note: str = ""):
    """Quiet axes and grid; `note` (e.g. DRY RUN) is printed in red under the title."""
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=18 if note else 6)
    if note:
        ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=8, color="#e34948", va="bottom")
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def grouped_bars(table: pd.DataFrame, value: str, title: str, ylabel: str, path: Path,
                 ymax: float | None = None, note: str = "") -> None:
    """Languages on the x axis, one bar per model."""
    models = sorted(table["model"].unique())[: len(SERIES_COLORS)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        part = table[table["model"] == model].set_index("language").reindex(LANGUAGES)
        xs = [x + (i - (len(models) - 1) / 2) * width for x in range(len(LANGUAGES))]
        bars = ax.bar(xs, part[value].astype(float).fillna(0), width * 0.92, label=model,
                      color=SERIES_COLORS[i])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=MUTED, padding=2)
    ax.set_xticks(range(len(LANGUAGES)), [LANGUAGE_NAMES[lang] for lang in LANGUAGES])
    if ymax:
        ax.set_ylim(0, ymax)
    _style(ax, title, ylabel, note)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_vs_quality(summary: pd.DataFrame, path: Path, note: str = "") -> None:
    overall = summary[summary["language"] == "all"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(overall["answer_cost_per_100_usd"], overall["pass_rate_pct"], s=64,
               color=SERIES_COLORS[0], edgecolor="white", linewidth=2, zorder=3)
    # Label each model directly (no legend), alternating above/below in cost order so
    # neighbouring labels do not overlap.
    by_cost = overall.sort_values("answer_cost_per_100_usd")
    for i, (_, r) in enumerate(by_cost.iterrows()):
        ax.annotate(r["model"], (r["answer_cost_per_100_usd"], r["pass_rate_pct"]),
                    textcoords="offset points", xytext=(6, [8, -14][i % 2]), fontsize=8, color=INK)
    ax.set_xlabel("US$ per 100 answers (assistant only)", color=MUTED)
    ax.set_xlim(0, 1.7 * max(overall["answer_cost_per_100_usd"].max(), 1e-9))  # room for labels
    ax.set_ylim(0, 115)
    _style(ax, "Cost vs pass rate, per model", "Pass rate (%)", note)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def agreement_chart(table: pd.DataFrame, path: Path, note: str = "") -> None:
    overall = table[(table["group"] == "all")]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(overall["criterion"], overall["cohen_kappa"].astype(float).fillna(0),
                  color=SERIES_COLORS[0], width=0.6)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, color=MUTED, padding=2)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylim(min(-0.2, overall["cohen_kappa"].astype(float).min() - 0.1), 1.05)
    _style(ax, f"Judge vs human: Cohen's kappa per criterion (n={overall['n'].iloc[0]})",
           "Cohen's kappa", note)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report draft
def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def native_review_counts() -> tuple[int, int]:
    """(reviewed, total) Arabic and French items across both test sets."""
    items = (load_jsonl(EVALS_DIR / "dataset.jsonl", EvalItem)
             + load_jsonl(EVALS_DIR / "redteam.jsonl", RedTeamItem))
    ar_fr = [i for i in items if i.language in ("ar", "fr")]
    return sum(i.reviewed for i in ar_fr), len(ar_fr)


def status_note(n_human: int) -> str:
    """Says plainly how far the numbers can be trusted yet (review and calibration status)."""
    reviewed, total = native_review_counts()
    lines = [f"- Arabic/French test items reviewed by a native speaker: **{reviewed} of {total}**.",
             f"- Human calibration (Sara's blind grades): **{n_human} of {HUMAN_TARGET}** answers."]
    if reviewed < total or n_human < HUMAN_TARGET:
        lines.append("- Until both are complete, every score below is an **uncalibrated LLM-judge "
                     "score**, not a human score, and Arabic/French results may partly reflect "
                     "test wording that a native speaker would change.")
    return "**Status of this report**\n\n" + "\n".join(lines) + "\n"


def write_report_draft(folder: Path, chart_dir: Path, dry_run: bool, tables: dict,
                       n_human: int = 0) -> Path:
    """The two-page report template: every table filled from the CSVs, the prose left as
    TODOs. Sara writes the findings and saves the result as REPORT.md in the repo root.
    Text written between the findings markers is kept when the draft is regenerated."""
    runs = []
    if (folder / "runs.jsonl").exists():
        runs = [json.loads(line) for line in (folder / "runs.jsonl").read_text().splitlines() if line]
    warning = ("> **DRY RUN: every number below comes from the fake client. These are NOT real "
               "results. Never copy them into REPORT.md.**\n" if dry_run else "")
    run_lines = "\n".join(
        f"- `{r['run_id']}`: models {r['models']}, judge `{r['judge']}`, prompt {r['prompt_version']}, "
        f"temperature {r['temperature']}, max tokens {r.get('answer_max_tokens', '?')} (answer) / "
        f"{r.get('judge_max_tokens', '?')} (judge), {r['quality_rows']} quality + "
        f"{r['redteam_rows']} red-team rows, spent ${r['spent_usd']}"
        + (f", STOPPED: {r['stopped']}" if r["stopped"] else "")
        + (f" ({r['note']})" if r.get("note") else "")
        for r in runs)
    charts = _display_path(chart_dir)
    path = folder / "REPORT_DRAFT.md"
    findings = kept_findings(path)
    reviewed, total = native_review_counts()
    agreement_text = (md_table(tables["agreement"]) if not tables["agreement"].empty else
                      "_Pending: no human grades yet. Grade `evals/human_grades.csv` (or use "
                      "`app/grading_app.py`), then re-run `python -m evals.report`._")
    parts = [
        "# Which model should answer Lumi Skin's customers in Arabic, English and French?\n",
        warning,
        "_Report template generated by `python -m evals.report`. Tables are filled from the "
        "results; write the TODO parts yourself, keep it to about two pages, and save it as "
        "`REPORT.md`._\n",
        status_note(n_human),
        "## 1. Summary\n\n> TODO (Sara): which model for which language and budget, the biggest "
        "failure, and how far the judge can be trusted (3-4 sentences).\n",
        f"### Findings\n\n{FINDINGS_START}\n{findings or FINDINGS_TODO}\n{FINDINGS_END}\n",
        f"## 2. Setup\n\n{run_lines or '_No runs logged._'}\n\n"
        "Test set: 180 quality items (60 per language) and 45 red-team attacks (15 per language). "
        "Native-speaker review status: see `evals/dataset_stats.md`. Pass rule: accuracy ≥ 4 and "
        "policy = 5 (`docs/rubric.md`).\n",
        f"## 3. Quality by model and language\n\n{md_table(tables['summary'])}\n\n"
        f"Charts: `{charts}/pass_rate.png`, `{charts}/score_by_model_language.png`.\n\n"
        "> TODO (Sara): hardest language and why, with two example answers (item IDs).\n",
        f"### By category\n\n{md_table(tables['by_category'])}\n",
        f"### Arabic by dialect\n\n{md_table(tables['by_variety'])}\n\n"
        "Each dialect has only 5-26 items: treat differences as hints, not rankings.\n",
        "## 4. Failures with examples\n\nFailed answers with the lowest accuracy (pick the "
        f"instructive ones and explain them):\n\n{md_table(tables['failures'])}\n\n"
        "> TODO (Sara): group the failures (wrong fact, invented policy, wrong language, ignored "
        "format...) and say what you would change in the prompt or the model choice.\n",
        f"## 5. Red team\n\n{md_table(tables['redteam'])}\n\n### By attack type\n\n"
        f"{md_table(tables['redteam_by_attack'])}\n\n### By attack type and language "
        f"(blocked / decided)\n\n{md_table(tables['redteam_grid'])}\n\n"
        "### Not blocked (check each by hand)\n\n"
        f"{md_table(tables['not_blocked'])}\n\n"
        "> TODO (Sara): one attack that worked, and how you would fix it.\n",
        f"## 6. Can the judge be trusted?\n\n{agreement_text}\n\n"
        f"### Largest disagreements\n\n{md_table(tables['disagreements'].head(15))}\n\n"
        "> TODO (Sara): where the judge disagreed with you, and whether it is too lenient or too "
        "strict in a particular language.\n",
        f"## 7. Recommendation under a budget\n\nChart: `{charts}/cost_vs_quality.png`.\n\n"
        "> TODO (Sara): e.g. 'for Arabic-first support under US$X per 100 answers, use ...'.\n",
        "## 8. Limitations\n\n"
        "- 60 items per language and 20 human grades per language: kappa on 20 items is noisy.\n"
        "- One judge model, run once; its biases are only partly measured by 60 human grades.\n"
        f"- The prompts were drafted by an AI model; {reviewed} of {total} Arabic/French items "
        "have been reviewed by a native speaker so far.\n"
        "- Single-turn questions; the assistant has no tools or order lookup.\n",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def kept_findings(path: Path) -> str:
    """The findings text between the markers in an existing draft ("" if there is none).
    Regenerating the tables (for example after Sara's grading) must not erase it."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    start, end = text.find(FINDINGS_START), text.find(FINDINGS_END)
    if start == -1 or end <= start:
        return ""
    kept = text[start + len(FINDINGS_START):end].strip()
    return "" if kept == FINDINGS_TODO else kept


def _display_path(path: Path) -> str:
    """Path relative to the repo when possible (keeps local folder names out of the report)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--folder", type=Path, default=None, help=argparse.SUPPRESS)  # for tests
    args = p.parse_args(argv)
    folder = args.folder or (EVALS_DIR / "dry_run" if args.dry_run else EVALS_DIR)
    chart_dir = folder / "charts" if (args.dry_run or args.folder) else REPO_ROOT / "docs" / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    if not (folder / "results.csv").exists():
        print(f"No results in {folder}. Run python -m evals.run first.")
        return 1

    results = load_latest(folder / "results.csv")
    dry = results["dry_run"].astype(str).eq("True").any()
    note = "DRY RUN: fake client, NOT real results" if dry else ""
    tables = {
        "summary": quality_summary(results, "language"),
        "by_category": quality_summary(results, "category")[
            ["model", "category", "n_graded", "pass_rate_pct", "mean_accuracy", "mean_policy"]],
        "by_variety": quality_summary(results[results["language"] == "ar"], "variety")[
            ["model", "variety", "n_graded", "pass_rate_pct", "mean_language"]],
        "failures": failed_examples(results),
        "redteam": pd.DataFrame(), "redteam_by_attack": pd.DataFrame(), "not_blocked": pd.DataFrame(),
        "redteam_grid": pd.DataFrame(),
        "agreement": pd.DataFrame(), "disagreements": pd.DataFrame(),
    }
    tables["summary"].to_csv(folder / "summary.csv", index=False)
    tables["by_category"].to_csv(folder / "summary_by_category.csv", index=False)
    tables["by_variety"].to_csv(folder / "summary_by_variety.csv", index=False)
    by_lang = tables["summary"][tables["summary"]["language"] != "all"]
    grouped_bars(by_lang, "mean_overall", "Mean score (1-5) by model and language", "Mean score",
                 chart_dir / "score_by_model_language.png", ymax=5.5, note=note)
    grouped_bars(by_lang, "pass_rate_pct", "Pass rate by model and language",
                 "Pass rate (%)", chart_dir / "pass_rate.png", ymax=110, note=note)
    cost_vs_quality(tables["summary"], chart_dir / "cost_vs_quality.png", note)

    if (folder / "redteam_results.csv").exists():
        rt = load_latest(folder / "redteam_results.csv")
        tables["redteam"] = redteam_summary(rt, "language")
        tables["redteam_by_attack"] = redteam_summary(rt, "attack_type")
        tables["not_blocked"] = not_blocked_examples(rt)
        long = redteam_by_attack_language(rt)
        tables["redteam_grid"] = attack_language_grid(long)
        tables["redteam"].to_csv(folder / "redteam_summary.csv", index=False)
        tables["redteam_by_attack"].to_csv(folder / "redteam_by_attack.csv", index=False)
        long.to_csv(folder / "redteam_by_attack_language.csv", index=False)
        rt_lang = tables["redteam"][tables["redteam"]["language"] != "all"]
        grouped_bars(rt_lang, "blocked_pct", "Red-team attacks blocked, by model and language",
                     "Blocked (%)", chart_dir / "redteam_block_rate.png", ymax=110, note=note)

    pairs = agreement_pairs(folder)
    if pairs:
        tables["agreement"] = agreement_table(pairs)
        tables["disagreements"] = pd.DataFrame(disagreements(pairs))
        tables["agreement"].to_csv(folder / "agreement.csv", index=False)
        tables["disagreements"].to_csv(folder / "disagreements.csv", index=False, encoding="utf-8-sig")
        agreement_chart(tables["agreement"], chart_dir / "judge_human_agreement.png", note)

    draft = write_report_draft(folder, chart_dir, dry, tables, n_human=len(pairs))
    print(f"Wrote summaries to {folder}, charts to {chart_dir}, draft {draft.name}")
    if dry:
        print("DRY RUN: placeholder numbers, NOT real results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
