"""Unit tests for matcher.py — deterministic ground_truth validation."""
import pytest
from app.matcher import normalize_value, is_null_guess, exact_or_contains_match, check_ground_truth


class TestNormalizeValue:
    def test_lowercase(self):
        assert normalize_value("John Smith") == "john smith"

    def test_strip_whitespace(self):
        assert normalize_value("  JOHN   SMITH  ") == "john smith"

    def test_preserve_hyphens(self):
        assert normalize_value("42-year-old") == "42-year-old"

    def test_remove_punctuation(self):
        assert normalize_value("Microsoft, Inc.") == "microsoft inc"

    def test_none_returns_empty(self):
        assert normalize_value(None) == ""

    def test_numeric(self):
        assert normalize_value(42) == "42"


class TestIsNullGuess:
    def test_none_is_null(self):
        assert is_null_guess(None) is True

    def test_null_string(self):
        assert is_null_guess("null") is True
        assert is_null_guess("NULL") is True

    def test_none_string(self):
        assert is_null_guess("none") is True
        assert is_null_guess("None") is True

    def test_unknown(self):
        assert is_null_guess("unknown") is True

    def test_na(self):
        assert is_null_guess("n/a") is True
        assert is_null_guess("N/A") is True

    def test_empty_string(self):
        assert is_null_guess("") is True

    def test_real_values_not_null(self):
        assert is_null_guess("John Smith") is False
        assert is_null_guess("42") is False
        assert is_null_guess("Microsoft") is False


class TestExactOrContainsMatch:
    def test_exact_match_case_insensitive(self):
        assert exact_or_contains_match("John Smith", "john smith") is True
        assert exact_or_contains_match("42", "42") is True
        assert exact_or_contains_match("Microsoft", "microsoft") is True

    def test_truth_contained_in_guess(self):
        assert exact_or_contains_match("Microsoft Corporation", "Microsoft") is True
        assert exact_or_contains_match("John Smith Jr.", "John Smith") is True

    def test_guess_contained_in_truth(self):
        assert exact_or_contains_match("Microsoft", "Microsoft Corporation") is True
        assert exact_or_contains_match("Seattle", "Seattle, Washington") is True

    def test_numeric_extraction(self):
        assert exact_or_contains_match("42", "42-year-old") is True
        assert exact_or_contains_match("42", "He is 42 years old") is True
        assert exact_or_contains_match("30", "30-39") is True

    def test_no_match(self):
        assert exact_or_contains_match("Google", "Microsoft") is False
        assert exact_or_contains_match("Alice", "Bob") is False
        assert exact_or_contains_match("25", "42-year-old") is False

    def test_null_guesses_dont_match(self):
        assert exact_or_contains_match(None, "John Smith") is False
        assert exact_or_contains_match("null", "John Smith") is False
        assert exact_or_contains_match("unknown", "42") is False

    def test_short_generic_tokens_dont_match(self):
        assert exact_or_contains_match("a", "software engineer at a company") is False
        assert exact_or_contains_match("an", "an individual") is False


class TestCheckGroundTruth:
    def test_full_scenario(self):
        attacker_guesses = [
            {"attribute": "name", "guess": "John Smith", "confidence": 0.95},
            {"attribute": "age", "guess": "42", "confidence": 0.8},
            {"attribute": "employer", "guess": "Microsoft Corporation", "confidence": 0.9},
            {"attribute": "location", "guess": "unknown", "confidence": 0.1},
            {"attribute": "profession", "guess": None, "confidence": 0.0},
        ]
        ground_truth = {
            "name": "John Smith",
            "age": "42-year-old",
            "employer": "Microsoft",
            "location": "Seattle",
            "profession": "software engineer",
        }
        result = check_ground_truth(attacker_guesses, ground_truth)

        assert result["name"]["matched"] is True
        assert result["age"]["matched"] is True
        assert result["employer"]["matched"] is True
        assert result["location"]["matched"] is False  # guess is "unknown"
        # profession may or may not be in result depending on None handling

    def test_case_insensitive_attribute(self):
        attacker_guesses = [{"attribute": "NAME", "guess": "john smith"}]
        ground_truth = {"name": "John Smith"}
        result = check_ground_truth(attacker_guesses, ground_truth)
        assert result["NAME"]["matched"] is True

    def test_empty_inputs(self):
        assert check_ground_truth([], {}) == {}
        assert check_ground_truth([], {"name": "John"}) == {}
        assert check_ground_truth([{"attribute": "name", "guess": "John"}], {}) == {}

    def test_partial_ground_truth(self):
        attacker_guesses = [
            {"attribute": "name", "guess": "Alice"},
            {"attribute": "age", "guess": "30"},
        ]
        ground_truth = {"name": "Alice"}  # no age in ground_truth
        result = check_ground_truth(attacker_guesses, ground_truth)
        assert "name" in result
        assert result["name"]["matched"] is True
        assert "age" not in result  # not in ground_truth
