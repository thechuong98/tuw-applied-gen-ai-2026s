"""Evaluate the Semantic Anonymizer against gold TAB/ECHR records — focus: the JUDGE.

We use one OBJECTIVE ground-truth signal: the verbatim presence of a gold
identifier span in the text. An identifier you can still read off the page has
plainly not been hidden. Against that signal we report three things:

  SYSTEM privacy  — does the DELIVERED text (final_text) still contain any gold
                    identifier span? (broken down per attribute label)
  JUDGE accuracy  — when the Judge ruled an attribute "not leaked", was its gold
                    span actually gone? A gold span still present + Judge says
                    "safe"  =>  a JUDGE FALSE NEGATIVE (the critical failure).
  UTILITY         — the Judge's utility scores on records that passed.

Why verbatim, and its limit
---------------------------
Verbatim match is a *lower bound* on leakage: it catches identifiers copied
through unchanged, not paraphrased or inferential leaks. Hence the asymmetry we
rely on when scoring the Judge:

  gold span PRESENT  -> definitely leaked   -> the Judge MUST flag it
  gold span ABSENT   -> not verbatim-leaked -> the Judge MAY still flag a
                                               semantic leak (we can't refute it)

So the Judge is scored mainly on RECALL over verbatim-present spans. A Judge
flag on an absent span is counted as an *unverified positive*, not a mistake —
it may well be catching a real semantic leak this harness cannot see.

The Judge verdict and the text it judged must be the SAME pair, so per-attribute
Judge scoring uses `current_text` (the last rewrite) together with `judge_result`
(the verdict on that rewrite). SYSTEM privacy uses `final_text` (what the user
actually receives, which at MAX_ITERS may be an earlier best candidate).

Labels
------
Records store a SHORT label (the TAB entity type, e.g. "PERSON"). At run time
eval.py expands each via `type_labels.json` (sitting next to the records, or
passed with --label-map) into the detailed description actually fed to the
defender/attacker/judge — while every metric still aggregates by the short label.

Run:
    cd backend
    python -m app.eval ../data/eval/tab_example.json
    python -m app.eval ../data/eval/                 # a dir of .json/.jsonl records
    python -m app.eval ../data/eval/tab.jsonl --limit 20 --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from .config import load_config
# NOTE: `.graph` (langgraph + langchain) is imported lazily inside main(), so the
# pure scoring helpers below can be imported and unit-tested without the LLM stack.


# --------------------------------------------------------------------------- #
# Loading records (tolerant: single object / JSON array / JSONL / directory)
# --------------------------------------------------------------------------- #

def _load_one(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    data = json.loads(raw)
    return data if isinstance(data, list) else [data]


def load_records(path: Path) -> list[dict]:
    if path.is_dir():
        recs: list[dict] = []
        for p in sorted(path.iterdir()):
            if p.suffix in (".json", ".jsonl"):
                recs.extend(_load_one(p))
        return recs
    return _load_one(path)


def load_label_map(records_path: Path, override: str | None) -> dict:
    """Short label (TAB type) -> detailed description fed to the system.

    Looks for `type_labels.json` next to the records file unless --label-map
    overrides. Absent -> empty map (short labels are fed through unchanged).
    """
    p = Path(override) if override else (
        (records_path if records_path.is_dir() else records_path.parent) / "type_labels.json")
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in m.items() if not k.startswith("_")}  # drop _comment keys
    print(f"note: no label map at {p} — feeding short labels as-is")
    return {}


def normalize_attribute(a) -> dict:
    """Accept both the rich object form and the legacy 'TYPE: span' string form.

    Rich:   {"label", "mentions":[...]}
    Legacy: "PERSON: Mr Galip Yalman"  -> label is the whole string, the part
            after ': ' is the single gold mention.
    """
    if isinstance(a, dict):
        return {"label": a["label"], "mentions": a.get("mentions") or []}
    label = str(a)
    mention = label.split(": ", 1)[1] if ": " in label else label
    return {"label": label, "mentions": [mention]}


# --------------------------------------------------------------------------- #
# Verbatim ground-truth matching
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    """Casefold + NFKC + whitespace-collapse so trivial formatting never hides a leak."""
    s = unicodedata.normalize("NFKC", s).replace("­", "")  # drop soft hyphen
    return " ".join(s.casefold().split())


def is_present(text_norm: str, mentions: list[str]) -> bool:
    """True if ANY gold surface form of the entity appears verbatim in the text."""
    return any(_norm(m) and _norm(m) in text_norm for m in mentions)


# --------------------------------------------------------------------------- #
# Run one record through the graph
# --------------------------------------------------------------------------- #

def run_record(graph, config: dict, rec: dict, label_map: dict) -> dict:
    attrs = [normalize_attribute(a) for a in rec.get("attributes_to_hide", [])]
    # The data keeps a SHORT label (the TAB type, e.g. "PERSON"); expand it to the
    # detailed description the defender/attacker/judge actually reason over. The
    # Judge then echoes this detailed string back as the leak key, so we look its
    # verdict up by `sys_label` but still report/aggregate by the short label.
    sys_labels = [label_map.get(a["label"], a["label"]) for a in attrs]

    init = {
        "original_text": rec["text"],
        "attributes_to_hide": sys_labels,
        "utility_to_preserve": rec.get("utility_to_preserve", []),
        "channel": rec.get("channel", "text"),
        "config": config,
        "iteration": 0,
        "history": [],
    }
    final = graph.invoke(init, config={"recursion_limit": 60})

    delivered = final.get("final_text", "") or ""
    judged = final.get("current_text", delivered) or ""
    judge_result = final.get("judge_result", {}) or {}
    # detailed attribute label -> the Judge's raw verdict on `judged`.
    judge_by_attr = {lk.get("attribute"): lk for lk in judge_result.get("leaks", [])}
    judge_leaked = {k: bool(v.get("leaked")) for k, v in judge_by_attr.items()}

    nd, nj = _norm(delivered), _norm(judged)
    per_attr = []
    for a, sys_label in zip(attrs, sys_labels):
        present_delivered = is_present(nd, a["mentions"])
        present_judged = is_present(nj, a["mentions"])
        flagged = judge_leaked.get(sys_label, False)
        # Confusion category for JUDGE accuracy, evaluated on the judged text:
        if present_judged and flagged:
            cat = "TP"            # present and the Judge caught it
        elif present_judged and not flagged:
            cat = "FN"            # present but the Judge passed it  <-- critical
        elif (not present_judged) and flagged:
            cat = "FP_unverified"  # flagged a span we can't see verbatim (maybe semantic)
        else:
            cat = "TN"
        lk = judge_by_attr.get(sys_label) or {}
        per_attr.append({
            "label": a["label"],
            "present_delivered": present_delivered,
            "present_judged": present_judged,
            "judge_flagged": flagged,
            # False => the Judge did not echo this attribute key back verbatim, so the
            # leak verdict could not be matched (a silent miss, not a real "safe").
            "judge_matched_key": sys_label in judge_by_attr,
            "judge_inferred_value": lk.get("inferred_value"),
            "judge_rationale": lk.get("rationale", ""),
            "category": cat,
        })

    # Utility scores are only meaningful when the Judge actually reached stage 2.
    utility = {k: judge_result.get(k) for k in
               ("task_utility", "informational_completeness", "factual_consistency",
                "fluency", "format_preserved")
               if judge_result.get(k) is not None}

    # Attacker guesses, trimmed — evidence for why each attribute did/didn't leak.
    attacker_guesses = [
        {"attribute": g.get("attribute"), "guess": g.get("guess"),
         "confidence": g.get("confidence"), "evidence_spans": g.get("evidence_spans", [])}
        for g in (final.get("attacker_result") or {}).get("guesses", [])
    ]

    leaked_delivered = [p for p in per_attr if p["present_delivered"]]
    return {
        "id": rec.get("id", rec.get("label", "?")),
        "verdict": final.get("verdict", "MAX_ITERS"),
        "rounds": final.get("rounds", 0),
        "n_attrs": len(attrs),
        "per_attr": per_attr,
        "leaked_delivered": [p["label"] for p in leaked_delivered],
        # system FALSE PASS: the system said PASS yet shipped a verbatim identifier
        "false_pass": final.get("verdict") == "PASS" and bool(leaked_delivered),
        "utility": utility,
        # --- diagnostics: the Judge's raw verdict + BOTH texts, so the written JSON is
        # self-sufficient for inspecting misses / false-passes without re-running. ---
        "judge_summary": judge_result.get("summary", ""),
        "judge_leaks": judge_result.get("leaks", []),
        "attacker_guesses": attacker_guesses,
        "judged_text": judged,      # the text the Judge verdict above applies to (last rewrite)
        "final_text": delivered,    # what the user actually receives (may be an earlier best candidate)
    }


# --------------------------------------------------------------------------- #
# Aggregate + report
# --------------------------------------------------------------------------- #

def _rate(n: int, d: int) -> float:
    return n / d if d else 0.0


def aggregate(results: list[dict]) -> dict:
    cats = {"TP": 0, "FN": 0, "FP_unverified": 0, "TN": 0}
    leaked_by_label: dict[str, int] = {}
    total_by_label: dict[str, int] = {}
    util_sum: dict[str, float] = {}
    util_n = 0
    clean_records = 0
    false_pass = 0

    for r in results:
        for p in r["per_attr"]:
            cats[p["category"]] += 1
            lbl = p["label"]
            total_by_label[lbl] = total_by_label.get(lbl, 0) + 1
            if p["present_delivered"]:
                leaked_by_label[lbl] = leaked_by_label.get(lbl, 0) + 1
        if not r["leaked_delivered"]:
            clean_records += 1
        if r["false_pass"]:
            false_pass += 1
        if r["utility"]:
            util_n += 1
            for k, v in r["utility"].items():
                util_sum[k] = util_sum.get(k, 0.0) + v

    tp, fn, fp = cats["TP"], cats["FN"], cats["FP_unverified"]
    total_gold_leak = sum(leaked_by_label.values())
    total_gold = sum(total_by_label.values())
    return {
        "n_records": len(results),
        "judge": {
            # recall over verbatim-present spans: the headline Judge metric
            "leak_recall": _rate(tp, tp + fn),
            "missed_leaks": fn,            # Judge false negatives (let an identifier through)
            "caught_leaks": tp,
            "unverified_flags": fp,        # flagged where we couldn't confirm verbatim
            "confusion": cats,
        },
        "system_privacy": {
            "gold_leak_rate": _rate(total_gold_leak, total_gold),
            "leak_rate_by_label": {lbl: _rate(leaked_by_label.get(lbl, 0), n)
                                   for lbl, n in sorted(total_by_label.items())},
            "fully_clean_records": clean_records,
            "false_pass_records": false_pass,  # said PASS but shipped an identifier
        },
        "utility_mean": {k: v / util_n for k, v in util_sum.items()} if util_n else {},
        "utility_scored_records": util_n,
    }


def print_report(agg: dict, results: list[dict]) -> None:
    j, s = agg["judge"], agg["system_privacy"]
    print("\n" + "=" * 64)
    print(f"EVAL  —  {agg['n_records']} record(s)")
    print("=" * 64)

    print("\nJUDGE (vs verbatim ground truth)")
    print(f"  leak recall .............. {j['leak_recall']:.0%}  "
          f"(caught {j['caught_leaks']}, missed {j['missed_leaks']})")
    print(f"  MISSED leaks (FN) ........ {j['missed_leaks']}   <-- critical: Judge passed a verbatim identifier")
    print(f"  unverified flags ......... {j['unverified_flags']}   (flagged; no verbatim span — may be a semantic catch)")

    print("\nSYSTEM PRIVACY (delivered text)")
    print(f"  gold leak rate ........... {s['gold_leak_rate']:.0%}")
    for lbl, rate in s["leak_rate_by_label"].items():
        print(f"    {(lbl + ' ').ljust(20, '.')} {rate:.0%}")
    print(f"  fully-clean records ...... {s['fully_clean_records']}/{agg['n_records']}")
    print(f"  FALSE PASS records ....... {s['false_pass_records']}   (verdict=PASS but identifier shipped)")

    if agg["utility_mean"]:
        print(f"\nUTILITY (mean over {agg['utility_scored_records']} scored record(s))")
        for k, v in agg["utility_mean"].items():
            print(f"  {k:.<26} {v:.2f}")

    # Per-record one-liners + any leaked attributes, so failures are inspectable.
    print("\nPER-RECORD")
    for r in results:
        flag = "  ⚠ FALSE PASS" if r["false_pass"] else ""
        print(f"  {r['id']}: verdict={r['verdict']} rounds={r['rounds']} "
              f"leaked={len(r['leaked_delivered'])}/{r['n_attrs']}{flag}")
        for lbl in r["leaked_delivered"]:
            print(f"      leaked: {lbl}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the anonymizer (judge-focused) on TAB records")
    ap.add_argument("path", help="record file (.json/.jsonl) or a directory of them")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N records")
    ap.add_argument("--label-map", help="type->description JSON (default: type_labels.json next to the records)")
    ap.add_argument("--out", help="write full per-record results + aggregate as JSON here")
    args = ap.parse_args()

    records = load_records(Path(args.path))
    if args.limit:
        records = records[: args.limit]
    if not records:
        sys.exit(f"no records found at {args.path}")

    label_map = load_label_map(Path(args.path), args.label_map)
    if label_map:
        print(f"label map: {len(label_map)} type(s) -> detailed descriptions")

    config = load_config()
    from .graph import build_graph  # lazy: pulls in langgraph/langchain only when actually running
    graph = build_graph()

    results = []
    for i, rec in enumerate(records, 1):
        rid = rec.get("id", rec.get("label", f"#{i}"))
        print(f"[{i}/{len(records)}] running {rid} ...", flush=True)
        try:
            results.append(run_record(graph, config, rec, label_map))
        except Exception as e:  # one bad record shouldn't sink the whole run
            print(f"    ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            results.append({"id": rid, "error": f"{type(e).__name__}: {e}",
                            "verdict": "ERROR", "rounds": 0, "n_attrs": 0,
                            "per_attr": [], "leaked_delivered": [], "false_pass": False,
                            "utility": {}, "final_text": ""})

    ok = [r for r in results if r.get("verdict") != "ERROR"]
    agg = aggregate(ok)
    print_report(agg, ok)

    if args.out:
        Path(args.out).write_text(
            json.dumps({"aggregate": agg, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
