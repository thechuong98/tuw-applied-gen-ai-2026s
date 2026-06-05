"""Unit tests for ner.py — NER/regex pre-scan for direct identifiers."""
import pytest
from app.ner import (
    _regex_findings,
    _dedupe_findings,
    detect_identifiers,
    format_ner_hints,
    PATTERNS,
)


class TestRegexFindings:
    def test_email_detection(self):
        text = "Contact john.smith@acme.com for details"
        findings = _regex_findings(text)
        emails = [f for f in findings if f["label"] == "EMAIL"]
        assert len(emails) == 1
        assert emails[0]["text"] == "john.smith@acme.com"
        assert emails[0]["source"] == "regex"

    def test_phone_detection(self):
        text = "Call me at 555-123-4567 or (555) 987-6543"
        findings = _regex_findings(text)
        phones = [f for f in findings if f["label"] == "PHONE"]
        assert len(phones) == 2

    def test_url_detection(self):
        text = "Visit https://example.com/page for more info"
        findings = _regex_findings(text)
        urls = [f for f in findings if f["label"] == "URL"]
        assert len(urls) == 1
        assert "https://example.com" in urls[0]["text"]

    def test_ip_address_detection(self):
        text = "Server at 192.168.1.1 is down"
        findings = _regex_findings(text)
        ips = [f for f in findings if f["label"] == "IP_ADDRESS"]
        assert len(ips) == 1
        assert ips[0]["text"] == "192.168.1.1"

    def test_ssn_detection(self):
        text = "SSN: 123-45-6789"
        findings = _regex_findings(text)
        ssns = [f for f in findings if f["label"] == "SSN"]
        assert len(ssns) == 1
        assert ssns[0]["text"] == "123-45-6789"

    def test_credit_card_detection(self):
        text = "Card: 1234-5678-9012-3456"
        findings = _regex_findings(text)
        cards = [f for f in findings if f["label"] == "CREDIT_CARD"]
        assert len(cards) == 1

    def test_no_findings_in_clean_text(self):
        text = "This is a normal sentence without identifiers."
        findings = _regex_findings(text)
        assert len(findings) == 0

    def test_multiple_emails(self):
        text = "Email alice@test.com or bob@test.org"
        findings = _regex_findings(text)
        emails = [f for f in findings if f["label"] == "EMAIL"]
        assert len(emails) == 2


class TestDedupeFindings:
    def test_no_overlap(self):
        findings = [
            {"text": "a", "label": "A", "start": 0, "end": 1, "source": "regex"},
            {"text": "b", "label": "B", "start": 5, "end": 6, "source": "regex"},
        ]
        result = _dedupe_findings(findings)
        assert len(result) == 2

    def test_overlapping_keeps_first(self):
        findings = [
            {"text": "abc", "label": "A", "start": 0, "end": 3, "source": "regex"},
            {"text": "bc", "label": "B", "start": 1, "end": 3, "source": "spacy"},
        ]
        result = _dedupe_findings(findings)
        assert len(result) == 1
        assert result[0]["text"] == "abc"

    def test_prefers_regex_over_spacy(self):
        findings = [
            {"text": "test", "label": "A", "start": 0, "end": 4, "source": "spacy"},
            {"text": "test", "label": "B", "start": 0, "end": 4, "source": "regex"},
        ]
        result = _dedupe_findings(findings)
        assert len(result) == 1
        assert result[0]["source"] == "regex"

    def test_empty_list(self):
        assert _dedupe_findings([]) == []


class TestDetectIdentifiers:
    def test_combined_detection(self):
        text = "John Smith (john@acme.com) works at 555-123-4567"
        findings = detect_identifiers(text)
        labels = [f["label"] for f in findings]
        assert "EMAIL" in labels
        assert "PHONE" in labels

    def test_returns_list(self):
        text = "Just some text"
        findings = detect_identifiers(text)
        assert isinstance(findings, list)


class TestFormatNerHints:
    def test_empty_returns_none_detected(self):
        result = format_ner_hints([])
        assert result == "(none detected)"

    def test_single_finding(self):
        findings = [{"text": "john@test.com", "label": "EMAIL", "start": 0, "end": 13, "source": "regex"}]
        result = format_ner_hints(findings)
        assert "Direct identifiers detected:" in result
        assert "john@test.com" in result
        assert "EMAIL" in result
        assert "0-13" in result

    def test_multiple_findings(self):
        findings = [
            {"text": "john@test.com", "label": "EMAIL", "start": 0, "end": 13, "source": "regex"},
            {"text": "555-123-4567", "label": "PHONE", "start": 20, "end": 32, "source": "regex"},
        ]
        result = format_ner_hints(findings)
        assert "john@test.com" in result
        assert "555-123-4567" in result


class TestGracefulFallback:
    def test_detect_works_without_spacy(self):
        text = "Contact alice@example.com or call 555-123-4567"
        findings = detect_identifiers(text)
        assert len(findings) >= 2
        regex_findings = [f for f in findings if f["source"] == "regex"]
        assert len(regex_findings) >= 2
