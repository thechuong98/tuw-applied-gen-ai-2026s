"""Load the single config.yaml + .env. No hardcoded paths."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root (one level above backend/) and from cwd, if present.
_HERE = Path(__file__).resolve()
_BACKEND_DIR = _HERE.parent.parent
load_dotenv(_BACKEND_DIR.parent / ".env")  # project-root/.env
load_dotenv()  # also cwd/.env, without overriding already-set vars


def load_config() -> dict:
    path = os.getenv("CONFIG_PATH") or str(_BACKEND_DIR / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Optional: override every model role from a single env var, handy for quick swaps.
    model_override = os.getenv("MODEL")
    if model_override:
        cfg["models"] = {role: model_override for role in cfg["models"]}

    return cfg
