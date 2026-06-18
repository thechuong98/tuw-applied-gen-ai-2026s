"""Unit tests for config.py — model + temperature config sourced from .env."""
import pytest

from app.config import (
    DEFAULT_MODEL,
    _models_from_env,
    _parse_temperature,
    _temperature_from_env,
    load_config,
)

_MODEL_ENV = ["MODEL", "MODEL_DEFAULT", "MODEL_DEFENDER", "MODEL_ATTACKER",
              "MODEL_JUDGE", "MODEL_MATCHER"]
_TEMP_ENV = ["TEMPERATURE", "TEMPERATURE_DEFAULT", "TEMPERATURE_DEFENDER",
             "TEMPERATURE_ATTACKER", "TEMPERATURE_JUDGE"]


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every model/temperature env var so tests assert from a known baseline."""
    for key in _MODEL_ENV + _TEMP_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class TestModelsFromEnv:
    def test_default_when_nothing_set(self, clean_env):
        assert _models_from_env() == {"default": DEFAULT_MODEL}

    def test_MODEL_overrides_all_roles(self, clean_env):
        clean_env.setenv("MODEL", "ollama:llama3.1")
        m = _models_from_env()
        assert m["default"] == "ollama:llama3.1"
        assert m["attacker"] == "ollama:llama3.1"
        assert m["judge"] == "ollama:llama3.1"

    def test_per_role_overrides_MODEL(self, clean_env):
        clean_env.setenv("MODEL", "openai:gpt-4o")
        clean_env.setenv("MODEL_ATTACKER", "ollama:llama3.1")
        m = _models_from_env()
        assert m["attacker"] == "ollama:llama3.1"
        assert m["default"] == "openai:gpt-4o"
        assert m["judge"] == "openai:gpt-4o"

    def test_per_role_without_MODEL_falls_back_to_default_model(self, clean_env):
        clean_env.setenv("MODEL_DEFENDER", "ollama:mistral")
        m = _models_from_env()
        assert m["defender"] == "ollama:mistral"
        assert m["default"] == DEFAULT_MODEL


class TestTemperatureFromEnv:
    def test_default_none_when_unset(self, clean_env):
        assert _temperature_from_env()["default"] is None

    def test_TEMPERATURE_overrides_all_roles(self, clean_env):
        clean_env.setenv("TEMPERATURE", "0.1")
        t = _temperature_from_env()
        assert t["default"] == 0.1
        assert t["attacker"] == 0.1

    def test_per_role_overrides_TEMPERATURE(self, clean_env):
        clean_env.setenv("TEMPERATURE", "0.1")
        clean_env.setenv("TEMPERATURE_DEFENDER", "0.5")
        t = _temperature_from_env()
        assert t["defender"] == 0.5
        assert t["attacker"] == 0.1

    def test_null_string_is_none(self, clean_env):
        clean_env.setenv("TEMPERATURE", "null")
        assert _temperature_from_env()["default"] is None


class TestParseTemperature:
    @pytest.mark.parametrize("raw,expected", [
        ("null", None), ("none", None), ("", None), ("  ", None),
        ("0.0", 0.0), ("0.7", 0.7), ("1", 1.0),
    ])
    def test_parse(self, raw, expected):
        assert _parse_temperature(raw) == expected


class TestLoadConfig:
    def test_injects_env_models_and_temperature_and_keeps_yaml_sections(self, clean_env):
        cfg = load_config()
        assert cfg["models"]["default"] == DEFAULT_MODEL
        assert "default" in cfg["temperature"]
        # yaml-sourced sections must survive
        assert "structured_output_method" in cfg
        assert "loop" in cfg
        assert "defaults" in cfg

    def test_MODEL_env_reflected_in_load_config(self, clean_env):
        clean_env.setenv("MODEL", "ollama:llama3.1")
        cfg = load_config()
        assert cfg["models"]["default"] == "ollama:llama3.1"
        assert cfg["models"]["attacker"] == "ollama:llama3.1"
