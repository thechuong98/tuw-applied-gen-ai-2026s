"""Load config.yaml + .env.

Model and temperature selection live entirely in `.env` (see `.env.example`):
  - MODEL / MODEL_<ROLE>            — which "provider:model" each agent uses
  - TEMPERATURE / TEMPERATURE_<ROLE> — sampling temperature per agent
config.yaml holds the rest (structured_output_method, loop thresholds, weights,
default attributes).
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root (one level above backend/) and from cwd, if present.
_HERE = Path(__file__).resolve()
_BACKEND_DIR = _HERE.parent.parent
load_dotenv(_BACKEND_DIR.parent / ".env")  # project-root/.env
load_dotenv()  # also cwd/.env, without overriding already-set vars

# Roles the pipeline can address. `default` is the fallback for any unset role.
MODEL_ROLES = ("default", "defender", "attacker", "judge", "matcher")
TEMPERATURE_ROLES = ("default", "defender", "attacker", "judge")
DEFAULT_MODEL = "google_vertexai:gemini-2.5-flash"


def _parse_temperature(raw: str):
    """'null'/'none'/'' (any case, whitespace ok) -> None (don't send the param); else float."""
    s = raw.strip().lower()
    if s in ("", "null", "none"):
        return None
    return float(raw)


def _models_from_env() -> dict:
    """Build the per-role model map from env.

    MODEL_<ROLE> overrides MODEL (which sets every role); `default` falls back
    to DEFAULT_MODEL so get_llm always has a usable spec.
    """
    all_override = os.getenv("MODEL")
    models = {}
    for role in MODEL_ROLES:
        val = os.getenv(f"MODEL_{role.upper()}") or all_override
        if val:
            models[role] = val
    models.setdefault("default", all_override or DEFAULT_MODEL)
    return models


def _temperature_from_env() -> dict:
    """Build the per-role temperature map from env.

    TEMPERATURE_<ROLE> overrides TEMPERATURE (which sets every role). `default`
    is None when unset, meaning the temperature parameter is not sent at all
    (required by models like the gpt-5 line that only accept their default temp).
    """
    all_override = os.getenv("TEMPERATURE")
    temps = {}
    for role in TEMPERATURE_ROLES:
        raw = os.getenv(f"TEMPERATURE_{role.upper()}")
        if raw is None:
            raw = all_override
        if raw is not None:
            temps[role] = _parse_temperature(raw)
    temps.setdefault("default", None)
    return temps


def load_config() -> dict:
    path = os.getenv("CONFIG_PATH") or str(_BACKEND_DIR / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Model + temperature selection comes from .env, not config.yaml.
    cfg["models"] = _models_from_env()
    cfg["temperature"] = _temperature_from_env()

    return cfg
