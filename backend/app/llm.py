"""LLM-agnostic factory. Swap providers by changing the "provider:model" string in config.yaml."""
from functools import lru_cache

from langchain.chat_models import init_chat_model


@lru_cache(maxsize=None)
def _build(spec: str, temperature):
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    # spec e.g. "openai:gpt-5-mini", "anthropic:claude-...", "ollama:llama3.1"
    return init_chat_model(spec, **kwargs)


def get_llm(config: dict, role: str):
    models = config["models"]
    spec = models.get(role) or models["default"]
    temps = config.get("temperature", {}) or {}
    temp = temps.get(role, temps.get("default"))
    return _build(spec, temp)
