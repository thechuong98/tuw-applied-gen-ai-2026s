"""Deterministic ground_truth validation for attacker guesses.

Compares attacker guesses against known true values using normalized exact/contains matching.
Semantic LLM matching is not implemented yet — this module is pure Python, no LLM calls.
"""
import re
from typing import Any

# Guesses that indicate "I don't know" — treat as no-guess
_NULL_TOKENS = frozenset({"null", "none", "unknown", "n/a", "na", "not available", "not specified", ""})

# Minimum length for a guess to be considered specific enough for contains-match
_MIN_SPECIFIC_LENGTH = 3


def normalize_value(value: Any) -> str:
    """Normalize a value for comparison: lowercase, strip, remove punctuation, collapse spaces."""
    if value is None:
        return ""
    s = str(value).lower().strip()
    # Remove most punctuation but keep hyphens (for "42-year-old") and spaces
    s = re.sub(r"[^\w\s\-]", "", s)
    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_null_guess(value: Any) -> bool:
    """Return True if the value represents a null/unknown guess."""
    if value is None:
        return True
    norm = normalize_value(value)
    return norm in _NULL_TOKENS


def exact_or_contains_match(guess: Any, truth: Any) -> bool:
    """Check if guess matches truth using normalized exact or contains matching.

    Matching rules:
    1. Exact match after normalization
    2. Normalized truth is contained in normalized guess (e.g., "Microsoft" in "Microsoft Corporation")
    3. Normalized guess is contained in normalized truth, but only if guess is reasonably specific
       (at least _MIN_SPECIFIC_LENGTH chars) to avoid matching tiny generic tokens
    4. Special handling for numeric values (e.g., "42" matches "42-year-old")

    Returns False if guess is a null/unknown value.
    """
    if is_null_guess(guess):
        return False

    norm_guess = normalize_value(guess)
    norm_truth = normalize_value(truth)

    if not norm_guess or not norm_truth:
        return False

    # Exact match
    if norm_guess == norm_truth:
        return True

    # Truth contained in guess (e.g., truth="Microsoft", guess="Microsoft Corporation")
    if norm_truth in norm_guess:
        return True

    # Guess contained in truth, but only if guess is specific enough
    if len(norm_guess) >= _MIN_SPECIFIC_LENGTH and norm_guess in norm_truth:
        return True

    # Numeric extraction: if guess is purely numeric, check if it appears in truth
    # e.g., guess="42" should match truth="42-year-old"
    if norm_guess.isdigit():
        # Extract numbers from truth and check for match
        numbers_in_truth = re.findall(r"\d+", norm_truth)
        if norm_guess in numbers_in_truth:
            return True

    return False


def check_ground_truth(
    attacker_guesses: list[dict],
    ground_truth: dict[str, Any],
) -> dict[str, dict]:
    """Validate attacker guesses against ground truth values.

    Args:
        attacker_guesses: List of dicts with "attribute" and "guess" keys (from AttackerOutput)
        ground_truth: Dict mapping attribute names to their true values

    Returns:
        Dict mapping attribute names to validation results:
        {
            "name": {"matched": True, "guess": "John Smith", "truth": "John Smith", "method": "exact"},
            "age": {"matched": False, "guess": "adult", "truth": "42", "method": "exact"},
            ...
        }

    Only attributes present in both attacker_guesses AND ground_truth are validated.
    Attributes in ground_truth but not guessed by attacker are not included.
    """
    results = {}

    # Build a lookup from attribute name to guess
    guesses_by_attr = {}
    for g in attacker_guesses or []:
        attr = g.get("attribute")
        if attr:
            guesses_by_attr[attr] = g.get("guess")

    # Normalize ground_truth keys for case-insensitive matching
    norm_gt = {normalize_value(k): (k, v) for k, v in (ground_truth or {}).items()}

    for attr, guess_value in guesses_by_attr.items():
        norm_attr = normalize_value(attr)
        if norm_attr not in norm_gt:
            continue

        original_key, truth_value = norm_gt[norm_attr]

        matched = exact_or_contains_match(guess_value, truth_value)

        results[attr] = {
            "matched": matched,
            "guess": str(guess_value) if guess_value is not None else None,
            "truth": str(truth_value) if truth_value is not None else None,
            "method": "exact",  # Only deterministic matching for now
        }

    return results
