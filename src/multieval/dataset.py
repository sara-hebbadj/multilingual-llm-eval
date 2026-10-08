"""Test-set schema, loading and quality checks.

Every line of `evals/dataset.jsonl` is an `EvalItem`; every line of
`evals/redteam.jsonl` is a `RedTeamItem`. Pydantic rejects missing fields,
unknown fields and wrong values, and `find_problems()` adds checks that need
the whole file (duplicate IDs, the right script for the language, numbers in
the reference facts that really appear in the shop policy).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Language = Literal["ar", "en", "fr"]
Variety = Literal["standard", "informal", "msa", "gulf", "levantine", "maghrebi", "arabizi"]
Category = Literal[
    "policy_fact", "product_advice", "complaint_tone", "refusal", "clarification", "formatting"
]
AttackType = Literal[
    "prompt_injection", "pii_extraction", "jailbreak_roleplay", "unsafe_advice", "discount_fraud"
]

ARABIC_VARIETIES = {"msa", "gulf", "levantine", "maghrebi", "arabizi"}
LATIN_VARIETIES = {"standard", "informal"}
ARABIC_LETTER = re.compile(r"[؀-ۿ]")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


class _Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^(ar|en|fr)-[a-z]+-\d{3}$")
    language: Language
    variety: Variety
    prompt: str = Field(min_length=5)
    expected_behavior: str = Field(min_length=10)
    # The same question asked in the other languages shares a parallel_id,
    # so score differences between languages are not caused by different questions.
    parallel_id: str
    # True once a native speaker (Sara) has checked that the prompt sounds natural.
    reviewed: bool = False

    @model_validator(mode="after")
    def _id_and_variety_match_language(self):
        if not self.id.startswith(self.language + "-"):
            raise ValueError(f"id {self.id} does not start with its language '{self.language}-'")
        allowed = ARABIC_VARIETIES if self.language == "ar" else LATIN_VARIETIES
        if self.variety not in allowed:
            raise ValueError(f"variety '{self.variety}' is not valid for language {self.language}")
        return self


class EvalItem(_Item):
    category: Category
    # Facts a correct answer must state. Written in English for every language,
    # so one judge prompt (and any reviewer) can check them the same way.
    reference_facts: list[str] = Field(min_length=1)
    # Where the facts come from, e.g. "policies#5" (section 5), "products:P009", "faq#22".
    source: str


class RedTeamItem(_Item):
    attack_type: AttackType
    # If any of these strings appears in the answer, the attack worked.
    forbidden_strings: list[str] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


def load_jsonl(path: Path, model: type[T]) -> list[T]:
    """Load and validate a JSONL file; the error names the bad line."""
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                items.append(model.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as err:
                raise ValueError(f"{path}:{line_no}: {err}") from err
    return items


def find_problems(items: list[_Item], knowledge_text: str | None = None) -> list[str]:
    """Checks across the whole file. An empty list means the file is clean.

    knowledge_text: the shop policies + FAQ + product catalogue. When given,
    every number in a reference fact must appear there (or in the prompt).
    """
    problems = []
    for item_id, n in Counter(i.id for i in items).items():
        if n > 1:
            problems.append(f"{item_id}: duplicate id ({n} times)")
    prompts = Counter(i.prompt.strip().lower() for i in items)
    problems += [f"duplicate prompt: {p[:60]!r}" for p, n in prompts.items() if n > 1]
    problems += _parallel_problems(items)
    for item in items:
        problems += _script_problems(item)
        if knowledge_text and isinstance(item, EvalItem):
            problems += _ungrounded_numbers(item, knowledge_text)
    return problems


def _parallel_problems(items: list[_Item]) -> list[str]:
    """Each parallel_id should be asked exactly once in each of the three languages."""
    by_id: dict[str, list[str]] = {}
    for item in items:
        by_id.setdefault(item.parallel_id, []).append(item.language)
    return [
        f"parallel_id {pid}: languages {sorted(langs)} (expected ar, en, fr once each)"
        for pid, langs in by_id.items()
        if sorted(langs) != ["ar", "en", "fr"]
    ]


def _script_problems(item: _Item) -> list[str]:
    """Arabic prompts must be in Arabic script (except Arabizi); others must not be."""
    has_arabic = bool(ARABIC_LETTER.search(item.prompt))
    if item.language == "ar" and item.variety != "arabizi" and not has_arabic:
        return [f"{item.id}: Arabic item has no Arabic letters"]
    if item.variety == "arabizi" and has_arabic:
        return [f"{item.id}: Arabizi item should be written in Latin letters"]
    if item.language in ("en", "fr") and has_arabic:
        return [f"{item.id}: {item.language} item contains Arabic letters"]
    return []


def _ungrounded_numbers(item: EvalItem, knowledge_text: str) -> list[str]:
    """Every number in a reference fact (days, AED prices...) must exist in the shop data.

    This catches the most common test-set bug: a reference answer that quotes
    a number the shop never says. Numbers the customer wrote are also allowed.
    """
    known = _numbers(knowledge_text) | _numbers(item.prompt)
    problems = []
    for fact in item.reference_facts:
        for number in sorted(_numbers(fact) - known):
            problems.append(f"{item.id}: number {number} in a reference fact is not in the shop data")
    return problems


def source_problems(items: list[EvalItem], policy_text: str, faq_text: str,
                    product_ids: set[str]) -> list[str]:
    """Every `source` must point at something that exists: a policy section
    ("policies#5"), an FAQ entry ("faq#22") or product IDs ("products:P009,P020")."""
    sections = set(re.findall(r"^## (\d+)\.", policy_text, re.MULTILINE))
    faq_numbers = set(re.findall(r"^### (\d+)\.", faq_text, re.MULTILINE))
    problems = []
    for item in items:
        src = item.source
        if src in ("none", "products"):
            continue
        if src.startswith("policies#"):
            missing = [s for s in re.findall(r"#(\d+)", src) if s not in sections]
        elif src.startswith("faq#"):
            missing = [s for s in re.findall(r"#(\d+)", src) if s not in faq_numbers]
        elif src.startswith("products:"):
            missing = [p for p in src.split(":", 1)[1].split(",") if p not in product_ids]
        else:
            missing = [f"unknown format {src!r}"]
        problems += [f"{item.id}: source {src} -> {m} not found" for m in missing]
    return problems


def _numbers(text: str) -> set[str]:
    # "1,000" -> "1000" so thousands separators do not split numbers.
    return set(NUMBER.findall(re.sub(r"(\d),(\d{3})", r"\1\2", text)))


def coverage_rows(items: list[_Item], column: str) -> list[dict]:
    """Count items per language for each value of `column` (e.g. category)."""
    values = sorted({getattr(i, column) for i in items})
    rows = []
    for value in values:
        row = {column: value}
        for lang in ("ar", "en", "fr"):
            row[lang] = sum(1 for i in items if i.language == lang and getattr(i, column) == value)
        row["total"] = row["ar"] + row["en"] + row["fr"]
        rows.append(row)
    total = {column: "TOTAL"} | {
        lang: sum(r[lang] for r in rows) for lang in ("ar", "en", "fr", "total")
    }
    return rows + [total]
