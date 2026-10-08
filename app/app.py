"""Results explorer for the evaluation (can run as a Hugging Face Space).

    pip install -e ".[app]"
    python app/app.py        then open http://127.0.0.1:7860

- Browse the saved results of the main live run (evals/results.csv, evals/redteam_results.csv): filter by
  model, language, category, Arabic variety (dialect) or attack type, and read the judge's rationale.
- "Try one prompt" sends one customer message to one of the compared models with the same system prompt as
  the evaluation, then runs the rule-based leak checks. It needs OPENROUTER_API_KEY (on a Hugging Face Space:
  a secret in the Space settings); without it the app runs in demo mode and everything else still works.
Every score is an uncalibrated LLM-judge score (see the README); all data is synthetic.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)  # works without installing (e.g. a Space)

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from multieval.assistant import assistant_messages  # noqa: E402
from multieval.config import REPO_ROOT, load_settings  # noqa: E402
from multieval.leak_checks import leak_hits  # noqa: E402
from multieval.llm_client import OpenRouterClient  # noqa: E402

EVALS = REPO_ROOT / "evals"
CHARTS = REPO_ROOT / "docs" / "charts"
MAIN_RUN = "20261008T124509Z"  # the full live run reported in the README (smoke runs are left out)
ALL = "(all)"
SCORES = ["score_accuracy", "score_policy", "score_tone", "score_language", "score_format"]
TRY_LIMIT = int(os.getenv("DEMO_MESSAGE_LIMIT", "10"))  # live calls per browser session

SETTINGS = load_settings()
LIVE = bool(SETTINGS.api_key)
DEMO_MODE_NOTE = (
    "**Demo mode — live AI is off; add OPENROUTER_API_KEY in Space settings to enable** the "
    '"Try one prompt" tab. The results explorer below shows the saved results and works now.'
)


def load_results(name: str) -> pd.DataFrame:
    frame = pd.read_csv(EVALS / name, encoding="utf-8-sig", dtype=str).fillna("")
    return frame[frame["run_id"] == MAIN_RUN].reset_index(drop=True)


QUALITY = load_results("results.csv")
REDTEAM = load_results("redteam_results.csv")
MODELS = sorted(QUALITY["model"].unique())
DATASET = [
    json.loads(line) for line in (EVALS / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line
]
SAMPLE_PROMPTS = {
    f"{item['id']} · {item['category']} · {item['prompt'][:60]}": item["prompt"] for item in DATASET
}


def options(frame: pd.DataFrame, column: str) -> list[str]:
    return [ALL, *sorted(frame[column].unique())]


def apply_filters(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    for column, value in filters.items():
        if value and value != ALL:
            frame = frame[frame[column] == value]
    return frame


def row_label(row: pd.Series) -> str:
    return f"{row['item_id']} · {row['model']}"


def filter_quality(model: str, language: str, category: str, variety: str, outcome: str):
    rows = apply_filters(
        QUALITY, {"model": model, "language": language, "category": category, "variety": variety}
    )
    if outcome != ALL:
        rows = rows[rows["passed"] == ("True" if outcome == "passed" else "False")]
    passed = int((rows["passed"] == "True").sum())
    summary = f"**{len(rows)} answers** match the filters; **{passed}/{len(rows)} passed** the judge's rule."
    table = rows[
        ["item_id", "model", "language", "variety", "category", *SCORES, "passed", "prompt", "answer"]
    ]
    labels = [row_label(r) for _, r in rows.iterrows()]
    return summary, table, gr.update(choices=labels, value=labels[0] if labels else None)


def quality_details(label: str) -> str:
    match = [r for _, r in QUALITY.iterrows() if row_label(r) == label]
    if not match:
        return "Pick an answer above to read the judge's rationale."
    r = match[0]
    scores = ", ".join(f"{s.removeprefix('score_')} {r[s]}" for s in SCORES)
    return (
        f"### {r['item_id']} ({r['language']}, {r['variety']}, {r['category']}) · `{r['model']}`\n"
        f"**Customer:** {r['prompt']}\n\n**Answer:** {r['answer']}\n\n**Judge scores (1-5):** {scores} · "
        f"**passed:** {r['passed']}\n\n**Judge rationale** (`{r['judge_model']}`): {r['judge_rationale']}"
    )


def filter_redteam(model: str, language: str, attack: str, outcome: str):
    rows = apply_filters(REDTEAM, {"model": model, "language": language, "attack_type": attack})
    if outcome != ALL:
        rows = rows[rows["blocked"] == ("True" if outcome == "blocked" else "False")]
    blocked = int((rows["blocked"] == "True").sum())
    summary = f"**{len(rows)} attacks** match the filters; **{blocked}/{len(rows)} blocked**."
    table = rows[
        [
            "item_id",
            "model",
            "language",
            "attack_type",
            "blocked",
            "harmful",
            "rule_hits",
            "needs_human_check",
            "prompt",
            "answer",
        ]
    ]
    labels = [row_label(r) for _, r in rows.iterrows()]
    return summary, table, gr.update(choices=labels, value=labels[0] if labels else None)


def redteam_details(label: str) -> str:
    match = [r for _, r in REDTEAM.iterrows() if row_label(r) == label]
    if not match:
        return "Pick an attack above to read the judge's rationale."
    r = match[0]
    return (
        f"### {r['item_id']} ({r['language']}, {r['attack_type']}) · `{r['model']}`\n"
        f"**Attack:** {r['prompt']}\n\n**Answer:** {r['answer']}\n\n**Blocked:** {r['blocked']} · "
        f"**harmful:** {r['harmful']} · **rule hits:** {r['rule_hits'] or 'none'} · **needs a human check:** "
        f"{r['needs_human_check']}\n\n**Judge rationale** (`{r['judge_model']}`): {r['judge_rationale']}"
    )


def try_prompt(message: str, model: str, count: int):
    if not LIVE:
        return DEMO_MODE_NOTE, "", count
    if not message.strip():
        return "Type a customer message first.", "", count
    if count >= TRY_LIMIT:
        return (
            f"Limit reached ({TRY_LIMIT} live messages per session). Refresh the page to start again.",
            "",
            count,
        )
    try:
        reply = OpenRouterClient(SETTINGS).chat(
            model, assistant_messages(message), temperature=0.0, max_tokens=600
        )
    except Exception as error:  # show the problem instead of crashing the demo
        return f"The model call failed: {type(error).__name__}.", "", count + 1
    hits = leak_hits(reply.text, message, [])
    checks = (
        f"**Leak checks (code):** {'; '.join(hits) if hits else 'no canary, staff code, new phone or email'}"
        f" · {reply.output_tokens} output tokens · US${reply.cost_usd or 0:.5f} · {reply.latency_s} s"
    )
    return reply.text, checks, count + 1


def table_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(EVALS / name, encoding="utf-8-sig")


def build() -> gr.Blocks:
    with gr.Blocks(title="Multilingual LLM evaluation") as demo:
        gr.Markdown(
            "# Multilingual LLM evaluation and red-team suite (Arabic · English · French)\n"
            "Three models as the customer-service assistant of **Lumi Skin**, a fictional skincare shop: 180 "
            "quality items (60 per language, including Gulf, Levantine and Maghrebi Arabic and Arabizi) "
            "and 45 red-team attacks, graded by an LLM judge from a fourth company. "
            f"Run `{MAIN_RUN}`, 8 October 2026. "
            "**Scores are uncalibrated LLM-judge scores**: Sara's blind human grades and the native-speaker "
            "review of the Arabic and French items are pending. All data is synthetic."
        )
        gr.Markdown(DEMO_MODE_NOTE if not LIVE else 'Live AI is on for the "Try one prompt" tab.')
        with gr.Tab("Summary"):
            gr.Dataframe(table_csv("summary.csv"), label="Quality by model and language", interactive=False)
            gr.Dataframe(
                table_csv("redteam_summary.csv"), label="Red team by model and language", interactive=False
            )
            with gr.Row():
                for chart in sorted(CHARTS.glob("*.png")):
                    gr.Image(str(chart), label=chart.stem.replace("_", " "))
        with gr.Tab("Quality results"):
            with gr.Row():
                q_model = gr.Dropdown(options(QUALITY, "model"), value=ALL, label="Model")
                q_language = gr.Dropdown(options(QUALITY, "language"), value=ALL, label="Language")
                q_category = gr.Dropdown(options(QUALITY, "category"), value=ALL, label="Category")
                q_variety = gr.Dropdown(options(QUALITY, "variety"), value=ALL, label="Variety (dialect)")
                q_outcome = gr.Radio([ALL, "passed", "failed"], value=ALL, label="Judge verdict")
            q_summary = gr.Markdown()
            q_table = gr.Dataframe(interactive=False, wrap=True, max_height=420)
            q_pick = gr.Dropdown([], label="Answer to inspect")
            q_details = gr.Markdown()
            q_filters = [q_model, q_language, q_category, q_variety, q_outcome]
            for control in q_filters:
                control.change(filter_quality, q_filters, [q_summary, q_table, q_pick])
            q_pick.change(quality_details, q_pick, q_details)
            demo.load(filter_quality, q_filters, [q_summary, q_table, q_pick])
            gr.Dataframe(
                table_csv("summary_by_variety.csv"), label="Pass rate by Arabic variety", interactive=False
            )
            gr.Dataframe(
                table_csv("summary_by_category.csv"), label="Pass rate by category", interactive=False
            )
        with gr.Tab("Red-team results"):
            with gr.Row():
                r_model = gr.Dropdown(options(REDTEAM, "model"), value=ALL, label="Model")
                r_language = gr.Dropdown(options(REDTEAM, "language"), value=ALL, label="Language")
                r_attack = gr.Dropdown(options(REDTEAM, "attack_type"), value=ALL, label="Attack type")
                r_outcome = gr.Radio([ALL, "blocked", "not blocked"], value=ALL, label="Verdict")
            r_summary = gr.Markdown()
            r_table = gr.Dataframe(interactive=False, wrap=True, max_height=420)
            r_pick = gr.Dropdown([], label="Attack to inspect")
            r_details = gr.Markdown()
            r_filters = [r_model, r_language, r_attack, r_outcome]
            for control in r_filters:
                control.change(filter_redteam, r_filters, [r_summary, r_table, r_pick])
            r_pick.change(redteam_details, r_pick, r_details)
            demo.load(filter_redteam, r_filters, [r_summary, r_table, r_pick])
            gr.Dataframe(
                table_csv("redteam_by_attack.csv"), label="Blocked by attack type", interactive=False
            )
        with gr.Tab("Try one prompt"):
            gr.Markdown(
                "Same system prompt as the evaluation (shop policies, FAQ and catalogue; no tools). "
                "The reply is checked by the same leak rules; it is not judged here."
                + ("" if LIVE else "\n\n" + DEMO_MODE_NOTE)
            )
            count = gr.State(0)
            sample = gr.Dropdown(list(SAMPLE_PROMPTS), value=None, label="Start from a test item (optional)")
            message = gr.Textbox(label="Customer message (Arabic, English or French)", lines=3)
            model = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
            send = gr.Button("Send", variant="primary", interactive=LIVE)
            answer = gr.Textbox(label="Assistant answer", lines=6)
            checks = gr.Markdown()
            sample.change(lambda label: SAMPLE_PROMPTS.get(label, ""), sample, message)
            send.click(try_prompt, [message, model, count], [answer, checks, count])
    return demo


if __name__ == "__main__":
    build().launch()
