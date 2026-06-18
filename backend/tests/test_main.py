"""Unit tests for main.py — Ollama health probe (no real network)."""
from unittest.mock import MagicMock, patch
import app.main as main_mod


class TestOllamaStatus:
    def test_none_when_no_ollama_model(self):
        cfg = {"models": {"default": "openai:gpt-4o", "judge": "openai:gpt-4o"}}
        assert main_mod._ollama_status(cfg) is None

    def test_reachable_and_model_present(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cfg = {"models": {"default": "ollama:llama3.1"}}
        fake_resp = MagicMock()
        fake_resp.read.return_value = b'{"models": [{"name": "llama3.1:latest"}]}'
        cm = MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with patch.object(main_mod.urllib.request, "urlopen", return_value=cm):
            status = main_mod._ollama_status(cfg)
        assert status["reachable"] is True
        assert status["base_url"] == "http://localhost:11434"
        assert status["models"]["llama3.1"] == "present"

    def test_model_missing_when_not_pulled(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cfg = {"models": {"default": "ollama:llama3.1"}}
        fake_resp = MagicMock()
        fake_resp.read.return_value = b'{"models": [{"name": "mistral:latest"}]}'
        cm = MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with patch.object(main_mod.urllib.request, "urlopen", return_value=cm):
            status = main_mod._ollama_status(cfg)
        assert status["reachable"] is True
        assert status["models"]["llama3.1"] == "missing"

    def test_unreachable_server(self):
        cfg = {"models": {"default": "ollama:llama3.1"}}
        with patch.object(main_mod.urllib.request, "urlopen", side_effect=OSError("refused")):
            status = main_mod._ollama_status(cfg)
        assert status["reachable"] is False
        assert status["models"]["llama3.1"] == "unknown"
