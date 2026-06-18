"""Unit tests for llm.py — safe_structured_invoke with mocked chains."""
import pytest
from unittest.mock import MagicMock, patch
import app.llm as llm_mod
from app.llm import safe_structured_invoke, resolve_method, get_structured_llm


class TestSafeStructuredInvoke:
    def test_success_on_first_try(self):
        chain = MagicMock()
        chain.invoke.return_value = {"answer": "test"}
        result = safe_structured_invoke(chain, {}, "TestSchema")
        assert result == {"answer": "test"}
        assert chain.invoke.call_count == 1

    def test_retry_on_none_then_success(self):
        chain = MagicMock()
        chain.invoke.side_effect = [None, {"answer": "retry_success"}]
        result = safe_structured_invoke(chain, {}, "TestSchema", max_retries=1)
        assert result == {"answer": "retry_success"}
        assert chain.invoke.call_count == 2

    def test_retry_on_exception_then_success(self):
        chain = MagicMock()
        chain.invoke.side_effect = [RuntimeError("API error"), {"answer": "recovered"}]
        result = safe_structured_invoke(chain, {}, "TestSchema", max_retries=1)
        assert result == {"answer": "recovered"}
        assert chain.invoke.call_count == 2

    def test_all_none_raises_valueerror(self):
        chain = MagicMock()
        chain.invoke.return_value = None
        with pytest.raises(ValueError) as exc_info:
            safe_structured_invoke(chain, {}, "MySchema", max_retries=1)
        assert "MySchema" in str(exc_info.value)
        assert "2 attempts" in str(exc_info.value)
        assert "None" in str(exc_info.value)
        assert chain.invoke.call_count == 2

    def test_all_exceptions_raises_valueerror_with_original_message(self):
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("Persistent API failure")
        with pytest.raises(ValueError) as exc_info:
            safe_structured_invoke(chain, {}, "FailSchema", max_retries=1)
        assert "FailSchema" in str(exc_info.value)
        assert "Persistent API failure" in str(exc_info.value)
        assert chain.invoke.call_count == 2

    def test_zero_retries(self):
        chain = MagicMock()
        chain.invoke.return_value = {"answer": "first"}
        result = safe_structured_invoke(chain, {}, "TestSchema", max_retries=0)
        assert result == {"answer": "first"}
        assert chain.invoke.call_count == 1

    def test_zero_retries_fail_on_none(self):
        chain = MagicMock()
        chain.invoke.return_value = None
        with pytest.raises(ValueError):
            safe_structured_invoke(chain, {}, "TestSchema", max_retries=0)
        assert chain.invoke.call_count == 1

    def test_passes_inputs_to_chain(self):
        chain = MagicMock()
        chain.invoke.return_value = {"result": "ok"}
        inputs = {"text": "hello", "attr": "name"}
        safe_structured_invoke(chain, inputs, "TestSchema")
        chain.invoke.assert_called_once_with(inputs)

    def test_multiple_retries(self):
        chain = MagicMock()
        chain.invoke.side_effect = [None, None, None, {"answer": "finally"}]
        result = safe_structured_invoke(chain, {}, "TestSchema", max_retries=3)
        assert result == {"answer": "finally"}
        assert chain.invoke.call_count == 4


class TestResolveMethod:
    CFG = {"structured_output_method": {"default": "function_calling", "ollama": "json_schema"}}

    def test_ollama_uses_json_schema(self):
        assert resolve_method(self.CFG, "ollama:llama3.1") == "json_schema"

    def test_openai_uses_default(self):
        assert resolve_method(self.CFG, "openai:gpt-4o") == "function_calling"

    def test_vertex_uses_default(self):
        assert resolve_method(self.CFG, "google_vertexai:gemini-2.5-flash") == "function_calling"

    def test_missing_block_falls_back_to_function_calling(self):
        assert resolve_method({}, "openai:gpt-4o") == "function_calling"

    def test_missing_default_key_falls_back(self):
        cfg = {"structured_output_method": {"ollama": "json_schema"}}
        assert resolve_method(cfg, "openai:gpt-4o") == "function_calling"


class TestGetLlmBaseUrl:
    def test_ollama_passes_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://myhost:9999")
        llm_mod._build.cache_clear()
        cfg = {"models": {"default": "ollama:llama3.1"}, "temperature": {}}
        with patch.object(llm_mod, "init_chat_model", return_value=MagicMock()) as m:
            llm_mod.get_llm(cfg, "default")
        args, kwargs = m.call_args
        assert args[0] == "ollama:llama3.1"
        assert kwargs.get("base_url") == "http://myhost:9999"

    def test_ollama_defaults_base_url_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        llm_mod._build.cache_clear()
        cfg = {"models": {"default": "ollama:mistral"}, "temperature": {}}
        with patch.object(llm_mod, "init_chat_model", return_value=MagicMock()) as m:
            llm_mod.get_llm(cfg, "default")
        _, kwargs = m.call_args
        assert kwargs.get("base_url") == "http://localhost:11434"

    def test_openai_does_not_pass_base_url(self):
        llm_mod._build.cache_clear()
        cfg = {"models": {"default": "openai:gpt-4o"}, "temperature": {}}
        with patch.object(llm_mod, "init_chat_model", return_value=MagicMock()) as m:
            llm_mod.get_llm(cfg, "default")
        _, kwargs = m.call_args
        assert "base_url" not in kwargs


class TestGetStructuredLlm:
    BASE_CFG = {
        "models": {"default": "ollama:llama3.1"},
        "temperature": {},
        "structured_output_method": {"default": "function_calling", "ollama": "json_schema"},
    }

    def test_ollama_role_uses_json_schema(self):
        llm_mod._build.cache_clear()
        schema = object()
        fake_llm = MagicMock()
        with patch.object(llm_mod, "init_chat_model", return_value=fake_llm):
            get_structured_llm(self.BASE_CFG, "default", schema)
        fake_llm.with_structured_output.assert_called_once_with(schema, method="json_schema")

    def test_openai_role_uses_function_calling(self):
        llm_mod._build.cache_clear()
        cfg = dict(self.BASE_CFG, models={"default": "openai:gpt-4o"})
        schema = object()
        fake_llm = MagicMock()
        with patch.object(llm_mod, "init_chat_model", return_value=fake_llm):
            get_structured_llm(cfg, "default", schema)
        fake_llm.with_structured_output.assert_called_once_with(schema, method="function_calling")

    def test_per_role_spec_resolves_independently(self):
        llm_mod._build.cache_clear()
        cfg = {
            "models": {"default": "openai:gpt-4o", "attacker": "ollama:llama3.1"},
            "temperature": {},
            "structured_output_method": {"default": "function_calling", "ollama": "json_schema"},
        }
        schema = object()
        fake_llm = MagicMock()
        with patch.object(llm_mod, "init_chat_model", return_value=fake_llm):
            get_structured_llm(cfg, "attacker", schema)
        fake_llm.with_structured_output.assert_called_once_with(schema, method="json_schema")
