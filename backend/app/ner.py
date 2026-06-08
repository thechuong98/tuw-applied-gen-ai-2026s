"""NER/regex pre-scan for direct identifiers.

Detects names, emails, phones, dates, numbers, locations, etc. and returns
findings as advisory hints for the Defender LLM.
Does NOT replace text — the LLM decides how to handle each identifier.

Detection stack (priority order, highest first):
  1. Regex  — structured patterns: EMAIL, PHONE, URL, IP, SSN, CREDIT_CARD,
               DATE, AGE, NUMBER
  2. GLiNER — general-purpose transformer NER, works across domains; finds
               names, dates, locations, orgs, quantities, professions, etc.
               Model: knowledgator/gliner-multitask-large-v0.5 (downloaded on
               first use, ~600 MB, cached in ~/.cache/huggingface).
  3. spaCy  — fallback if GLiNER is not installed; covers PERSON, GPE, LOC,
               ORG, DATE, TIME, QUANTITY, MONEY, NORP, LAW, EVENT.
  4. Regex-only — fallback if neither GLiNER nor spaCy is available.

Install for full coverage:
    pip install gliner
    # or for spaCy fallback:
    pip install spacy && python -m spacy download en_core_web_sm
"""
from __future__ import annotations

import re
from typing import TypedDict


# --------------------------------------------------------------------------- #
# Regex patterns — structured / unambiguous identifiers
# --------------------------------------------------------------------------- #

PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(
        r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    ),
    "URL": re.compile(r"https?://[^\s]+"),
    "IP_ADDRESS": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    # Dates — multiple formats
    "DATE": re.compile(
        r"\b(?:"
        # 29 November 1996 / 29 Nov 1996
        r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{2,4}"
        r"|"
        # November 1996 / Nov 1996
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{2,4}"
        r"|"
        # 29/11/1996, 11-29-1996, 1996-11-29, 29.11.1996
        r"\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
        r")\b",
        re.IGNORECASE,
    ),
    # Standalone years — 1900–2099
    "YEAR": re.compile(r"\b(19|20)\d{2}\b"),
    # Age expressions
    "AGE": re.compile(
        r"\b(?:age[sd]?\s+\d{1,3}|\d{1,3}[\s-]year[\s-]old|\d{1,3}\s+years?\s+old)\b",
        re.IGNORECASE,
    ),
    # Generic numbers with separators: "36110/97", "A-1234-56", "ref 12345"
    "NUMBER": re.compile(
        r"\b(?:[A-Z]{0,4}[-/]?)?\d{3,}(?:[\/\-\.]\d{2,})+\b"
    ),
}

# --------------------------------------------------------------------------- #
# GLiNER entity labels — generic, domain-agnostic
# --------------------------------------------------------------------------- #

GLINER_LABELS = [
    "person name",
    "date",
    "time",
    "year",
    "age",
    "location",
    "address",
    "organization",
    "reference number",
    "quantity",
    "nationality",
    "profession",
    "medical condition",
    "ethnic group",
]

# spaCy labels used when GLiNER is unavailable
SPACY_LABELS = {
    "PERSON", "GPE", "LOC", "ORG",
    "DATE", "TIME", "QUANTITY", "MONEY",
    "NORP", "LAW", "EVENT",
}


# --------------------------------------------------------------------------- #
# TypedDict
# --------------------------------------------------------------------------- #

class NERFinding(TypedDict):
    """A detected identifier. JSON-serializable for state storage."""
    text: str
    label: str
    start: int
    end: int
    source: str  # "regex" | "gliner" | "spacy"


# --------------------------------------------------------------------------- #
# Lazy model loaders
# --------------------------------------------------------------------------- #

_gliner_cache: object = None
_gliner_attempted: bool = False

_spacy_cache: object = None
_spacy_attempted: bool = False


def _get_gliner():
    """Lazily load GLiNER model. Returns None if unavailable."""
    global _gliner_cache, _gliner_attempted
    if _gliner_attempted:
        return _gliner_cache
    _gliner_attempted = True
    try:
        from gliner import GLiNER  # type: ignore
        _gliner_cache = GLiNER.from_pretrained(
            "knowledgator/gliner-multitask-large-v0.5"
        )
    except Exception:
        _gliner_cache = None
    return _gliner_cache


def _get_spacy():
    """Lazily load spaCy model. Returns None if unavailable."""
    global _spacy_cache, _spacy_attempted
    if _spacy_attempted:
        return _spacy_cache
    _spacy_attempted = True
    try:
        import spacy  # type: ignore
        _spacy_cache = spacy.load("en_core_web_sm")
    except Exception:
        _spacy_cache = None
    return _spacy_cache


# --------------------------------------------------------------------------- #
# Detection functions
# --------------------------------------------------------------------------- #

def _regex_findings(text: str) -> list[NERFinding]:
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


def _gliner_findings(text: str, chunk_size: int = 400, overlap: int = 50) -> list[NERFinding]:
    """Detect entities via GLiNER with chunking for long texts.

    GLiNER has an internal token limit (~512). For long texts we split into
    overlapping character chunks, run prediction on each, adjust offsets back
    to the full text, and deduplicate.
    """
    model = _get_gliner()
    if model is None:
        return []

    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws
        chunks.append((start, text[start:end]))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    raw_findings: list[NERFinding] = []
    seen_spans: set[tuple[int, int]] = set()

    for offset, chunk in chunks:
        try:
            entities = model.predict_entities(chunk, GLINER_LABELS, threshold=0.4)
        except Exception:
            continue
        for ent in entities:
            abs_start = offset + ent["start"]
            abs_end = offset + ent["end"]
            span_key = (abs_start, abs_end)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            raw_findings.append({
                "text": ent["text"],
                "label": ent["label"].upper().replace(" ", "_"),
                "start": abs_start,
                "end": abs_end,
                "source": "gliner",
            })

    return raw_findings


def _spacy_findings(text: str) -> list[NERFinding]:
    """Detect entities via spaCy. Returns empty list if model unavailable."""
    nlp = _get_spacy()
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
    """Remove overlapping spans; prefer regex > gliner > spacy for same span."""
    if not findings:
        return []
    source_priority = {"regex": 0, "gliner": 1, "spacy": 2}
    sorted_f = sorted(
        findings,
        key=lambda f: (f["start"], source_priority.get(f["source"], 9)),
    )
    result: list[NERFinding] = []
    last_end = -1
    for f in sorted_f:
        if f["start"] >= last_end:
            result.append(f)
            last_end = f["end"]
    return result


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def detect_identifiers(text: str) -> list[NERFinding]:
    """
    Run regex + GLiNER (or spaCy fallback) on text to detect identifiers.
    Returns a deduplicated list of NERFinding dicts. Does NOT modify the text.

    Detection priority: regex > GLiNER > spaCy.
    If GLiNER is unavailable, falls back to spaCy.
    If neither is available, returns regex findings only.
    """
    regex_hits = _regex_findings(text)

    if _get_gliner() is not None:
        ml_hits = _gliner_findings(text)
    else:
        ml_hits = _spacy_findings(text)

    return _dedupe_findings(regex_hits + ml_hits)


def format_ner_hints(findings: list[NERFinding]) -> str:
    """
    Format NER findings as a hint string for the Defender prompt.
    Returns "(none detected)" if empty.
    """
    if not findings:
        return "(none detected)"
    parts = [f"'{f['text']}' ({f['label']} at {f['start']}-{f['end']})"
             for f in findings]
    return "Detected identifiers: " + ", ".join(parts)


def gliner_available() -> bool:
    """True if GLiNER model loaded successfully."""
    return _get_gliner() is not None


def spacy_available() -> bool:
    """True if spaCy model loaded successfully."""
    return _get_spacy() is not None


def check_verbatim_leaks(
    original_findings: list[NERFinding],
    rewritten_text: str,
) -> list[NERFinding]:
    """Check which NER findings from the original text are still verbatim
    in the rewritten text.

    Returns a list of findings whose text still appears word-for-word.
    Used by nodes.py to build targeted feedback for the Defender.
    """
    import unicodedata

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKC", s).replace("\u00ad", "")
        return " ".join(s.casefold().split())

    norm_rewritten = _norm(rewritten_text)
    still_present = []
    seen_texts: set[str] = set()
    for f in original_findings:
        normed = _norm(f["text"])
        if not normed or normed in seen_texts:
            continue
        if normed in norm_rewritten:
            seen_texts.add(normed)
            still_present.append(f)
    return still_present


def format_verbatim_feedback(leaks: list[NERFinding]) -> str:
    """Format verbatim leaks as a feedback string for the Defender prompt."""
    if not leaks:
        return ""
    spans = ", ".join(f"'{f['text']}' ({f['label']})" for f in leaks)
    return (
        "VERBATIM LEAK DETECTED: the following spans from the original text "
        "are still word-for-word in your rewrite and MUST be changed or generalised: "
        + spans
    )
