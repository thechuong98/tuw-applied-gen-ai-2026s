"""Score candidates across rounds and build retry feedback.

The leak/utility verdicts themselves now come from the Judge LLM (see nodes.judge); this module only
ranks candidates and turns the Judge's findings into actionable feedback for the next Defender rewrite.
"""


def candidate_score(config: dict, leaked_count: int, total: int, task_utility: float) -> float:
    w = config["weights"]
    priv = 1.0 - (leaked_count / total) if total else 1.0
    return w["privacy"] * priv + w["utility"] * (task_utility or 0.0)


def update_best(prev: dict | None, cand: dict) -> dict:
    if prev is None or cand["score"] > prev.get("score", float("-inf")):
        return cand
    return prev


def build_leak_feedback(details: list[dict]) -> str:
    lines = []
    for d in details:
        ev = "; ".join(d.get("evidence") or []) or "(no specific span)"
        why = f" — judge: {d['rationale']}" if d.get("rationale") else ""
        lines.append(f"- '{d['attribute']}' is still inferable as '{d['guess']}' from clue(s): {ev}{why}")
    return (
        "The Judge ruled these attributes STILL inferable. Rewrite HARDER — break or remove the listed clues "
        "(stronger abstraction, shifting, or omission), while keeping the non-sensitive meaning:\n"
        + "\n".join(lines)
    )


def build_utility_feedback(j: dict, privacy_note: str = "") -> str:
    safe = f"No attribute leaked — keep it that way ({privacy_note}). " if privacy_note \
        else "No attribute leaked — keep it that way. "
    return (
        safe
        + f"But utility dropped (task_utility={j['task_utility']:.2f}, "
        f"factual_consistency={j['factual_consistency']:.2f}, format_preserved={j['format_preserved']:.2f}). "
        f"Reason: {j.get('notes', '')}. "
        "Rewrite more LIGHTLY: restore exactly the lost task-relevant signal and structure while still "
        "hiding the sensitive attributes."
    )
