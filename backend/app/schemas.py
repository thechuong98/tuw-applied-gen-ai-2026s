"""Structured-output contracts between graph nodes."""
from typing import Optional

from pydantic import BaseModel, Field


class AttrGuess(BaseModel):
    attribute: str
    guess: Optional[str] = Field(None, description="Best guess for the attribute, or null if not inferable")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = Field("", description="Chain-of-thought used to reach the guess")
    evidence_spans: list[str] = Field(default_factory=list, description="Exact spans in the text that leaked it")


class AttackerOutput(BaseModel):
    guesses: list[AttrGuess] = Field(default_factory=list)


class DefenderOutput(BaseModel):
    reasoning: str = Field(
        "",
        description="Brief chain-of-thought BEFORE rewriting: which clues are risky and how you neutralize "
                    "them while preserving meaning (and how you acted on any feedback)",
    )
    rewritten_text: str
    strategy_log: dict[str, str] = Field(
        default_factory=dict,
        description="attribute -> 'abstraction|shifting|omission' + short note",
    )


class AttrLeak(BaseModel):
    attribute: str
    leaked: bool = Field(False, description="True if this attribute is still inferable from the rewritten text")
    inferred_value: Optional[str] = Field(None, description="Value the judge believes is recoverable, if any")
    rationale: str = Field("", description="Why the judge ruled this leaked / not leaked")


class PrivacyVerdict(BaseModel):
    """Judge stage 1 — did any sensitive attribute still leak?"""
    leaks: list[AttrLeak] = Field(
        default_factory=list,
        description="Assess EVERY listed attribute (leaked true or false), each with a rationale",
    )
    summary: str = Field("", description="One-line reason WHY the rewrite is (or is not) safe — fed back to the Defender")


class UtilityScores(BaseModel):
    """Judge stage 2 — scored ONLY when stage 1 found no leak."""
    task_utility: float = Field(0.0, ge=0.0, le=1.0)
    informational_completeness: float = Field(0.0, ge=0.0, le=1.0)
    factual_consistency: float = Field(0.0, ge=0.0, le=1.0)
    fluency: float = Field(0.0, ge=0.0, le=1.0)
    format_preserved: float = Field(1.0, ge=0.0, le=1.0)
    notes: str = ""
