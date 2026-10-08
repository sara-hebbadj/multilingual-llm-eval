"""Settings: API key, model IDs and budget limits.

Values are read from `Portfolio Projects/.env` (two folders above this repo, where
Sara keeps her keys), then from a local `repo/.env` (for a fresh clone), and
finally from ordinary environment variables. Real environment variables always win.
The API key is never printed: `Settings` hides it from repr().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_ENV_FILE = REPO_ROOT.parent.parent / ".env"  # Portfolio Projects/.env
LOCAL_ENV_FILE = REPO_ROOT / ".env"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    api_key: str | None = field(repr=False)
    base_url: str
    model_main: str | None
    model_cheap: str | None
    model_judge: str | None
    max_cost_per_run_usd: float
    max_cost_per_project_usd: float


def load_settings() -> Settings:
    # override=False: a variable already set in the shell is not replaced.
    load_dotenv(SHARED_ENV_FILE, override=False)
    load_dotenv(LOCAL_ENV_FILE, override=False)
    return Settings(
        api_key=_get("OPENROUTER_API_KEY"),
        base_url=_get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
        model_main=_get("MODEL_MAIN"),
        model_cheap=_get("MODEL_CHEAP"),
        model_judge=_get("MODEL_JUDGE"),
        max_cost_per_run_usd=float(_get("MAX_COST_PER_RUN_USD") or 3),
        max_cost_per_project_usd=float(_get("MAX_COST_PER_PROJECT_USD") or 10),
    )


def _get(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def resolve_model(name: str, settings: Settings) -> str:
    """Turn an alias ("main", "cheap", "judge") into a model ID.

    Anything else is treated as a full OpenRouter ID such as "provider/model-name",
    so a third comparison model can be passed on the command line directly.
    """
    aliases = {
        "main": settings.model_main,
        "cheap": settings.model_cheap,
        "judge": settings.model_judge,
    }
    if name in aliases:
        model_id = aliases[name]
        if not model_id:
            raise ValueError(f"MODEL_{name.upper()} is not set in .env")
        return model_id
    if "/" not in name:
        raise ValueError(f"'{name}' is not an alias or an OpenRouter ID (provider/model)")
    return name


def model_family(model_id: str) -> str:
    """The provider part of an OpenRouter ID: "anthropic/x" -> "anthropic".

    Used to make sure the judge never grades a model from its own family,
    because judges tend to prefer answers written in their own style.
    """
    return model_id.split("/", 1)[0].lstrip("~").lower()
