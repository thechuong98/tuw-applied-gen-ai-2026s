"""LangGraph node functions: defender -> attacker -> judge -> finalize.

Pure-LLM: no deterministic prepass — the Defender LLM handles identifiers and semantic clues alike.

The Judge is the single verdict authority, run in TWO sequential stages:
  1. PRIVACY  — decide whether any sensitive attribute still leaked.
  2. UTILITY  — ONLY if stage 1 found no leak, score each utility dimension.
A leak short-circuits the judge: utility is never scored and the Defender is asked to rewrite harder.
There is no separate confidence-threshold scoring node anymore.

Routing:
  judge -> finalize(PASS)            if no leak AND utility ok
        -> retry(defender)           if (leak or low utility) and under the iteration cap
        -> finalize(best candidate)  at the iteration cap
"""
from .llm import get_llm
from .prompts import ATTACKER_PROMPT, DEFENDER_PROMPT, PRIVACY_PROMPT, UTILITY_PROMPT
from .schemas import AttackerOutput, DefenderOutput, PrivacyVerdict, UtilityScores
from .scoring import (
    build_leak_feedback,
    build_utility_feedback,
    candidate_score,
    update_best,
)
from .state import AnonState


def defender(state: AnonState) -> dict:
    llm = get_llm(state["config"], "defender")
    chain = DEFENDER_PROMPT | llm.with_structured_output(DefenderOutput, method="function_calling")
    out: DefenderOutput = chain.invoke({
        "text": state["original_text"],
        "attrs": ", ".join(state["attributes_to_hide"]),
        "channel": state.get("channel", "text"),
        "feedback": state.get("feedback") or "(none)",
    })
    new_iter = state["iteration"] + 1
    return {
        "current_text": out.rewritten_text,
        "strategy_log": out.strategy_log,
        "defender_reasoning": out.reasoning,
        "iteration": new_iter,
        "feedback": None,
        "history": [{"round": new_iter, "rewrite": out.rewritten_text,
                     "strategy": out.strategy_log, "reasoning": out.reasoning}],
    }


def attacker(state: AnonState) -> dict:
    llm = get_llm(state["config"], "attacker")
    chain = ATTACKER_PROMPT | llm.with_structured_output(AttackerOutput, method="function_calling")
    out: AttackerOutput = chain.invoke({
        "text": state["current_text"],
        "attrs": ", ".join(state["attributes_to_hide"]),
    })
    return {"attacker_result": out.model_dump()}


def _format_guesses(attacker_result: dict) -> str:
    """Render the Attacker's guesses as evidence for the Judge prompt."""
    rows = []
    for g in (attacker_result or {}).get("guesses", []):
        val = g.get("guess")
        if val is None:
            continue
        ev = "; ".join(g.get("evidence_spans") or []) or "(none)"
        rows.append(
            f"- {g.get('attribute')}: guess='{val}' (attacker confidence {g.get('confidence', 0.0):.2f}); "
            f"evidence: {ev}"
        )
    return "\n".join(rows) or "(the attacker made no concrete guesses)"


def judge(state: AnonState) -> dict:
    cfg = state["config"]
    attrs = state["attributes_to_hide"]
    llm = get_llm(cfg, "judge")

    # --- Stage 1: privacy. Decide "no leak" before scoring anything. ---
    priv_chain = PRIVACY_PROMPT | llm.with_structured_output(PrivacyVerdict, method="function_calling")
    priv: PrivacyVerdict = priv_chain.invoke({
        "attrs": ", ".join(attrs),
        "guesses": _format_guesses(state["attacker_result"]),
        "original": state["original_text"],
        "rewritten": state["current_text"],
    })
    priv_dump = priv.model_dump()
    leaks_dump = priv_dump["leaks"]
    privacy_summary = priv_dump.get("summary", "")

    # Pair each leak with the Attacker's evidence spans for actionable feedback.
    guesses_by_attr = {g.get("attribute"): g for g in (state["attacker_result"] or {}).get("guesses", [])}
    details = []
    for lk in leaks_dump:
        if lk.get("leaked") and lk.get("attribute") in attrs:
            ag = guesses_by_attr.get(lk["attribute"], {})
            details.append({
                "attribute": lk["attribute"],
                "guess": lk.get("inferred_value") or ag.get("guess") or "(unspecified)",
                "evidence": ag.get("evidence_spans", []),
                "rationale": lk.get("rationale", ""),
            })
    leaked = [d["attribute"] for d in details]

    # Leak found -> short-circuit: do NOT score utility. Hand the Defender the reasons it leaked.
    if leaked:
        sc = candidate_score(cfg, len(leaked), len(attrs), 0.0)
        best = update_best(state.get("best_candidate"),
                           {"text": state["current_text"], "score": sc, "round": state["iteration"],
                            "leaked_attrs": leaked, "verdict": "MAX_ITERS"})
        return {
            "judge_result": {"leaks": leaks_dump, "summary": privacy_summary},
            "leaked_attrs": leaked,
            "best_candidate": best,
            "feedback": build_leak_feedback(details),
        }

    # --- Stage 2: utility. Reached only when nothing leaked. ---
    util_chain = UTILITY_PROMPT | llm.with_structured_output(UtilityScores, method="function_calling")
    util: UtilityScores = util_chain.invoke({
        "utility": ", ".join(state.get("utility_to_preserve") or []) or "(preserve general meaning)",
        "original": state["original_text"],
        "rewritten": state["current_text"],
    })
    j = util.model_dump()

    lc = cfg["loop"]
    utility_ok = (j["task_utility"] >= lc["min_task_utility"]
                  and j["factual_consistency"] >= lc["min_factual"]
                  and j["format_preserved"] >= lc.get("min_format", 0.0))

    sc = candidate_score(cfg, 0, len(attrs), j["task_utility"])
    best = update_best(state.get("best_candidate"),
                       {"text": state["current_text"], "score": sc, "round": state["iteration"],
                        "leaked_attrs": [], "verdict": "PASS" if utility_ok else "MAX_ITERS"})

    ret = {"judge_result": {"leaks": leaks_dump, "summary": privacy_summary, **j},
           "leaked_attrs": [], "best_candidate": best}
    if utility_ok:
        ret["verdict"] = "PASS"
    else:
        # Pass utility scores + reason AND the "why it's safe" note so the rewrite keeps privacy intact.
        ret["feedback"] = build_utility_feedback(j, privacy_summary)
    return ret


def finalize(state: AnonState) -> dict:
    if state.get("verdict") == "PASS":
        return {"final_text": state["current_text"], "rounds": state["iteration"]}
    bc = state.get("best_candidate") or {}
    return {
        "verdict": bc.get("verdict", "MAX_ITERS"),
        "final_text": bc.get("text", state.get("current_text")),
        "rounds": state["iteration"],
    }


# --- router (conditional edge) ---

def route_after_judge(state: AnonState) -> str:
    if state.get("verdict") == "PASS":
        return "finalize"
    return "retry" if state["iteration"] < state["config"]["loop"]["max_iters"] else "finalize"
