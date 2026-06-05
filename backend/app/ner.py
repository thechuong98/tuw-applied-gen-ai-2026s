"""NER/regex pre-scan for direct identifiers.

Detects names, emails, phones, etc. and returns findings as advisory hints for the Defender LLM.
Does NOT replace text — the LLM decides how to handle each identifier.

Uses:
  - Regex for structured patterns: EMAIL, PHONE, URL, IP_ADDRESS, SSN, CREDIT_CARD
  - spaCy for named entities: PERSON, GPE, LOC, ORG

If spaCy model is missing, falls back to regex-only detection (no crash).
"""
import re
from typing import TypedDict

# Regex patterns for structured identifiers
PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "URL": re.compile(r"https?://[^\s]+"),
    "IP_ADDRESS": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
}

# spaCy entity labels we care about
SPACY_LABELS = {"PERSON", "GPE", "LOC", "ORG"}


class NERFinding(TypedDict):
    """A detected identifier. JSON-serializable for state storage."""
    text: str
    label: str
    start: int
    end: int
    source: str  # "regex" or "spacy"


# Lazy-loaded spaCy model (None = not yet attempted, False = failed to load)
_nlp_cache: object = None
_nlp_load_attempted: bool = False


def _get_nlp():
    """Lazily load spaCy model. Returns None if unavailable."""
    global _nlp_cache, _nlp_load_attempted
    if _nlp_load_attempted:
        return _nlp_cache
    _nlp_load_attempted = True
    try:
        import spacy
        _nlp_cache = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        # spaCy not installed or model not downloaded — fall back to regex-only.
        # To enable spaCy NER: pip install spacy && python -m spacy download en_core_web_sm
        _nlp_cache = None
    return _nlp_cache


def _regex_findings(text: str) -> list[NERFinding]:
    """Detect structured identifiers via regex."""
    findings: list[NERFinding] = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({
                "text": match.group(),
                "label": label,
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
            })
    return findings


def _spacy_findings(text: str) -> list[NERFinding]:
    """Detect named entities via spaCy. Returns empty if model unavailable."""
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    findings: list[NERFinding] = []
    for ent in doc.ents:
        if ent.label_ in SPACY_LABELS:
            findings.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy",
            })
    return findings


def _dedupe_findings(findings: list[NERFinding]) -> list[NERFinding]:
    """Remove overlapping findings, keeping the more specific (regex) or first match."""
    if not findings:
        return []
    # Sort by start position, then prefer regex over spacy for same span
    sorted_findings = sorted(findings, key=lambda f: (f["start"], f["source"] != "regex"))
    result: list[NERFinding] = []
    last_end = -1
    for f in sorted_findings:
        if f["start"] >= last_end:
            result.append(f)
            last_end = f["end"]
    return result


def detect_identifiers(text: str) -> list[NERFinding]:
    """
    Run regex + spaCy NER on text to detect direct identifiers.
    Returns a deduplicated list of findings (JSON-serializable dicts).
    Does NOT modify the text.
    """
    regex_hits = _regex_findings(text)
    spacy_hits = _spacy_findings(text)
    all_findings = regex_hits + spacy_hits
    return _dedupe_findings(all_findings)


def format_ner_hints(findings: list[NERFinding]) -> str:
    """
    Format NER findings as a human-readable hint string for the Defender prompt.
    Returns "(none detected)" if empty.
    """
    if not findings:
        return "(none detected)"
    parts = []
    for f in findings:
        parts.append(f"'{f['text']}' ({f['label']} at {f['start']}-{f['end']})")
    return "Direct identifiers detected: " + ", ".join(parts)


def spacy_available() -> bool:
    """Check if spaCy model loaded successfully (for diagnostics)."""
    return _get_nlp() is not None
