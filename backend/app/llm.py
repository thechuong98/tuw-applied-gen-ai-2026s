"""LLM-agnostic factory. Swap providers by changing the "provider:model" string in config.yaml."""
import os
from functools import lru_cache

from langchain.chat_models import init_chat_model


def resolve_method(config: dict, spec: str) -> str:
    """Structured-output method for a model spec, keyed by provider prefix.

    Ollama models default to json_schema (robust for local models); cloud
    providers keep function_calling. Resolution order: provider key -> default
    -> "function_calling".
    """
    provider = spec.split(":", 1)[0]
    methods = config.get("structured_output_method") or {}
    return methods.get(provider) or methods.get("default") or "function_calling"


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@lru_cache(maxsize=None)
def _build(spec: str, temperature, base_url):
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if base_url is not None:
        kwargs["base_url"] = base_url
    # spec e.g. "openai:gpt-5-mini", "google_vertexai:gemini-2.5-flash", "ollama:llama3.1"
    return init_chat_model(spec, **kwargs)


def get_llm(config: dict, role: str):
    models = config["models"]
    spec = models.get(role) or models["default"]
    temps = config.get("temperature", {}) or {}
    temp = temps.get(role, temps.get("default"))
    base_url = None
    if spec.split(":", 1)[0] == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    return _build(spec, temp, base_url)


def safe_structured_invoke(chain, inputs: dict, schema_name: str, max_retries: int = 1):
    """Invoke structured output chain with retry on None or exception.

    Args:
        chain: LangChain chain with structured output
        inputs: Dict of prompt variables
        schema_name: Name of the schema for error messages
        max_retries: Number of retries after initial attempt (default 1)

    Returns:
        The structured output object

    Raises:
        ValueError: If all attempts fail (None result or exception)
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = chain.invoke(inputs)
            if result is not None:
                return result
        except Exception as e:
            last_error = e
    if last_error:
        raise ValueError(
            f"Structured output failed for {schema_name} after {max_retries + 1} attempts: {last_error}"
        )
    raise ValueError(
        f"Structured output returned None for {schema_name} after {max_retries + 1} attempts"
    )
