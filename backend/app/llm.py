"""LLM-agnostic factory. Swap providers by changing the "provider:model" string in config.yaml."""
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
