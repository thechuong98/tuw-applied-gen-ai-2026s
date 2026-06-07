"""Unit tests for llm.py — safe_structured_invoke with mocked chains."""
import pytest
from unittest.mock import MagicMock
from app.llm import safe_structured_invoke


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
