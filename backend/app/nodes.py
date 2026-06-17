"""LangGraph node functions: defender -> attacker -> judge -> finalize.

The Judge is the single verdict authority, run in TWO sequential stages:
  1. PRIVACY  — decide whether any sensitive attribute still leaked.
  2. UTILITY  — ONLY if stage 1 found no leak, score each utility dimension.

Important implementation detail:
The Judge's structured output uses free-text attribute names. It may write
"duration", "application number", or "name of organisation" instead of the
exact target label. Therefore, any leaked=true finding from the Judge is
treated as a privacy failure. This avoids false PASS outcomes caused by
attribute-name mismatch.
"""
from .llm import get_llm, safe_structured_invoke
from .matcher import check_ground_truth
from .ner import detect_identifiers, format_ner_hints, check_verbatim_leaks, format_verbatim_feedback
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
    # NER/regex pre-scan: detect direct identifiers and pass as hints to the LLM.
    ner_findings = state.get("ner_findings")
    if ner_findings is None:
        ner_findings = detect_identifiers(state["original_text"])
    ner_hints = format_ner_hints(ner_findings)

    llm = get_llm(state["config"], "defender")
    chain = DEFENDER_PROMPT | llm.with_structured_output(DefenderOutput, method="function_calling")
    feedback = state.get("feedback") or "(none)"
    if state["iteration"] > 0 and state.get("current_text") and ner_findings:
        verbatim_leaks = check_verbatim_leaks(ner_findings, state["current_text"])
        if verbatim_leaks:
            verbatim_note = format_verbatim_feedback(verbatim_leaks)
            feedback = verbatim_note + "\n" + feedback

    out: DefenderOutput = safe_structured_invoke(chain, {
        "text": state["original_text"],
        "attrs": ", ".join(state["attributes_to_hide"]),
        "channel": state.get("channel", "text"),
        "feedback": feedback,
        "ner_hints": ner_hints,
    }, "DefenderOutput")

    new_iter = state["iteration"] + 1
    return {
        "current_text": out.rewritten_text,
        "strategy_log": out.strategy_log,
        "defender_reasoning": out.reasoning,
        "iteration": new_iter,
        "feedback": None,
        "ner_findings": ner_findings,
        "history": [{
            "round": new_iter,
            "rewrite": out.rewritten_text,
            "strategy": out.strategy_log,
            "reasoning": out.reasoning,
        }],
    }


def attacker(state: AnonState) -> dict:
    llm = get_llm(state["config"], "attacker")
    chain = ATTACKER_PROMPT | llm.with_structured_output(AttackerOutput, method="function_calling")
    out: AttackerOutput = safe_structured_invoke(chain, {
        "text": state["current_text"],
        "attrs": ", ".join(state["attributes_to_hide"]),
    }, "AttackerOutput")
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
            f"- {g.get('attribute')}: guess='{val}' "
            f"(attacker confidence {g.get('confidence', 0.0):.2f}); evidence: {ev}"
        )
    return "\n".join(rows) or "(the attacker made no concrete guesses)"


def _build_leak_details(leaks_dump: list[dict], attacker_result: dict) -> list[dict]:
    """Turn Judge leaked=true findings into actionable Defender feedback.

    We intentionally do NOT require exact attribute-name equality here. The Judge
    may output free-text sub-attributes such as "duration", "application number",
    or "name of organisation". If leaked=true, the privacy gate should fail.
    """
    guesses_by_attr = {
        g.get("attribute"): g
        for g in (attacker_result or {}).get("guesses", [])
        if g.get("attribute")
    }

    details: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for lk in leaks_dump or []:
        if not lk.get("leaked"):
            continue

        attr = lk.get("attribute") or "(unspecified sensitive clue)"
        inferred = lk.get("inferred_value") or "(unspecified)"
        rationale = lk.get("rationale", "")
        key = (str(attr), str(inferred), str(rationale))

        if key in seen:
            continue
        seen.add(key)

        ag = guesses_by_attr.get(attr, {})
        details.append({
            "attribute": attr,
            "guess": inferred or ag.get("guess") or "(unspecified)",
            "evidence": ag.get("evidence_spans", []),
            "rationale": rationale,
        })

    return details


def judge(state: AnonState) -> dict:
    cfg = state["config"]
    attrs = state["attributes_to_hide"]
    llm = get_llm(cfg, "judge")

    # --- Stage 1: privacy. Decide "no leak" before scoring anything. ---
    priv_chain = PRIVACY_PROMPT | llm.with_structured_output(PrivacyVerdict, method="function_calling")
    priv: PrivacyVerdict = safe_structured_invoke(priv_chain, {
        "attrs": ", ".join(attrs),
        "guesses": _format_guesses(state["attacker_result"]),
        "original": state["original_text"],
        "rewritten": state["current_text"],
    }, "PrivacyVerdict")

    priv_dump = priv.model_dump()
    leaks_dump = priv_dump.get("leaks", [])
    privacy_summary = priv_dump.get("summary", "")

    details = _build_leak_details(leaks_dump, state.get("attacker_result") or {})
    leaked = [d["attribute"] for d in details]

    # --- Ground truth validation: can prove positive leaks, not negatives ---
    ground_truth = state.get("ground_truth") or {}
    gt_validation = None
    if ground_truth:
        attacker_guesses = (state["attacker_result"] or {}).get("guesses", [])
        gt_validation = check_ground_truth(attacker_guesses, ground_truth)

        guesses_by_attr = {
            g.get("attribute"): g
            for g in attacker_guesses
            if g.get("attribute")
        }

        for attr, result in gt_validation.items():
            if result["matched"] and attr not in leaked:
                leaked.append(attr)
                ag = guesses_by_attr.get(attr, {})
                gt_rationale = (
                    f"Ground truth match: attacker guessed '{result['guess']}' "
                    f"which matches true value"
                )
                details.append({
                    "attribute": attr,
                    "guess": result["guess"] or "(unspecified)",
                    "evidence": ag.get("evidence_spans", []),
                    "rationale": gt_rationale,
                })
                leaks_dump.append({
                    "attribute": attr,
                    "leaked": True,
                    "inferred_value": result["guess"],
                    "rationale": gt_rationale,
                })

    # Leak found -> short-circuit: do NOT score utility.
    if leaked:
        leak_count_for_score = min(len(leaked), len(attrs))
        sc = candidate_score(cfg, leak_count_for_score, len(attrs), 0.0)
        best = update_best(
            state.get("best_candidate"),
            {
                "text": state["current_text"],
                "score": sc,
                "round": state["iteration"],
                "leaked_attrs": leaked,
                "verdict": "MAX_ITERS",
            },
        )

        judge_result = {"leaks": leaks_dump, "summary": privacy_summary}
        if gt_validation is not None:
            judge_result["ground_truth_validation"] = gt_validation

        return {
            "judge_result": judge_result,
            "leaked_attrs": leaked,
            "best_candidate": best,
            "feedback": build_leak_feedback(details),
        }

    # --- Stage 2: utility. Reached only when nothing leaked. ---
    util_chain = UTILITY_PROMPT | llm.with_structured_output(UtilityScores, method="function_calling")
    util: UtilityScores = safe_structured_invoke(util_chain, {
        "utility": ", ".join(state.get("utility_to_preserve") or []) or "(preserve general meaning)",
        "original": state["original_text"],
        "rewritten": state["current_text"],
    }, "UtilityScores")
    j = util.model_dump()

    lc = cfg["loop"]
    utility_ok = (
        j["task_utility"] >= lc["min_task_utility"]
        and j["factual_consistency"] >= lc["min_factual"]
        and j["format_preserved"] >= lc.get("min_format", 0.0)
    )

    sc = candidate_score(cfg, 0, len(attrs), j["task_utility"])
    best = update_best(
        state.get("best_candidate"),
        {
            "text": state["current_text"],
            "score": sc,
            "round": state["iteration"],
            "leaked_attrs": [],
            "verdict": "PASS" if utility_ok else "MAX_ITERS",
        },
    )

    judge_result = {"leaks": leaks_dump, "summary": privacy_summary, **j}
    if gt_validation is not None:
        judge_result["ground_truth_validation"] = gt_validation

    ret = {"judge_result": judge_result, "leaked_attrs": [], "best_candidate": best}
    if utility_ok:
        ret["verdict"] = "PASS"
    else:
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


def route_after_judge(state: AnonState) -> str:
    if state.get("verdict") == "PASS":
        return "finalize"
    return "retry" if state["iteration"] < state["config"]["loop"]["max_iters"] else "finalize"