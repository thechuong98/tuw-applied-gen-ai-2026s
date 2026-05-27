"""Shared LangGraph state for the per-record adversarial loop (docs/architecture.md §5.1)."""
import operator
from typing import Annotated, Optional, TypedDict


class AnonState(TypedDict, total=False):
    # --- input (fixed) ---
    original_text: str
    attributes_to_hide: list[str]
    ground_truth: dict          # optional; reserved for eval mode (leak verdict now comes from the Judge)
    utility_to_preserve: list[str]
    channel: str
    config: dict

    # --- working ---
    current_text: str           # latest rewrite (Defender rewrites from original_text)
    iteration: int
    feedback: Optional[str]
    strategy_log: dict
    defender_reasoning: str     # Defender's chain-of-thought for the latest rewrite

    # --- per-component results ---
    attacker_result: Optional[dict]
    judge_result: Optional[dict]
    leaked_attrs: list[str]

    # --- bookkeeping / audit ---
    history: Annotated[list, operator.add]   # reducer = append (audit trail)
    best_candidate: Optional[dict]
    verdict: Optional[str]      # "PASS" | "MAX_ITERS"
    final_text: Optional[str]
    rounds: int
