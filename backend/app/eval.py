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
  NER BASELINE    — same verbatim privacy metric but using plain spaCy NER
                    masking (no LLM, no tokens). Shows what a simple masker
                    achieves so we can compare against our adversarial system.

Extra metrics vs the original eval
-----------------------------------
  - attacker_confidence: mean confidence of Attacker guesses (all / leaked only)
  - rounds_distribution: histogram of how many rounds each record needed
  - leak_rate_by_label: already present, now also shown for NER baseline
  - original_text / final_text saved per record for inspection

Batching & resuming
-------------------
Records are processed in batches of --batch-size (default 10). After each
batch the partial results are flushed to --out (if given), so a crash or
API quota error does not lose all prior work. Re-run with --skip N to skip
the first N records (or use --resume to auto-detect from an existing --out).

Run:
    cd backend
    python -m app.eval ../data/eval/tab.json --limit 20 --batch-size 10 --out results.json
    python -m app.eval ../data/eval/tab.json --resume --out results.json   # continue after crash
    python -m app.eval ../data/eval/tab.json --ner-only --out ner_baseline.json  # free, no LLM
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from .config import load_config


# --------------------------------------------------------------------------- #
# Loading records
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
    p = Path(override) if override else (
        (records_path if records_path.is_dir() else records_path.parent) / "type_labels.json")
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in m.items() if not k.startswith("_")}
    print(f"note: no label map at {p} — feeding short labels as-is")
    return {}


def normalize_attribute(a) -> dict:
    if isinstance(a, dict):
        return {"label": a["label"], "mentions": a.get("mentions") or []}
    label = str(a)
    mention = label.split(": ", 1)[1] if ": " in label else label
    return {"label": label, "mentions": [mention]}


# --------------------------------------------------------------------------- #
# Verbatim ground-truth matching
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u00ad", "")
    return " ".join(s.casefold().split())


def is_present(text_norm: str, mentions: list[str]) -> bool:
    return any(_norm(m) and _norm(m) in text_norm for m in mentions)


# --------------------------------------------------------------------------- #
# NER baseline — spaCy, no LLM tokens
# --------------------------------------------------------------------------- #

def _get_nlp():
    """Lazy-load spaCy en_core_web_sm. Falls back gracefully if not installed."""
    try:
        import spacy  # type: ignore
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("warning: spaCy model 'en_core_web_sm' not found. "
                  "Run: python -m spacy download en_core_web_sm", file=sys.stderr)
            return None
    except ImportError:
        print("warning: spaCy not installed — NER baseline skipped. "
              "Run: pip install spacy", file=sys.stderr)
        return None


# TAB entity type -> spaCy entity labels that cover it
_TAB_TO_SPACY = {
    "PERSON":   {"PERSON"},
    "ORG":      {"ORG"},
    "LOC":      {"GPE", "LOC", "FAC"},
    "DATETIME": {"DATE", "TIME"},
    "QUANTITY": {"MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"},
    "CODE":     set(),   # spaCy has no CODE type; regex could help but skip for fairness
    "DEM":      {"NORP", "LANGUAGE"},
    "MISC":     {"EVENT", "WORK_OF_ART", "LAW", "PRODUCT"},
}


def ner_mask_text(nlp, text: str, attrs: list[dict]) -> str:
    """Replace spaCy-detected entities with [LABEL] placeholders.

    Only masks entity types that overlap with the requested TAB labels,
    so we don't over-redact unrelated entity types.
    """
    if nlp is None:
        return text
    target_spacy = set()
    for a in attrs:
        target_spacy |= _TAB_TO_SPACY.get(a["label"], set())
    if not target_spacy:
        return text  # nothing mappable — return as-is (conservative baseline)

    doc = nlp(text)
    # Build replacement from end to start to preserve offsets
    replacements = []
    for ent in doc.ents:
        if ent.label_ in target_spacy:
            replacements.append((ent.start_char, ent.end_char, f"[{ent.label_}]"))
    result = list(text)
    for start, end, repl in sorted(replacements, reverse=True):
        result[start:end] = list(repl)
    return "".join(result)


def run_ner_baseline(nlp, rec: dict) -> dict:
    """Run spaCy NER masking on one record and score verbatim privacy."""
    attrs = [normalize_attribute(a) for a in rec.get("attributes_to_hide", [])]
    masked = ner_mask_text(nlp, rec["text"], attrs) if nlp else rec["text"]
    nd = _norm(masked)
    per_attr = []
    for a in attrs:
        per_attr.append({
            "label": a["label"],
            "present": is_present(nd, a["mentions"]),
        })
    leaked = [p for p in per_attr if p["present"]]
    return {
        "id": rec.get("id", rec.get("label", "?")),
        "masked_text": masked,
        "original_text": rec["text"],
        "n_attrs": len(attrs),
        "leaked_labels": [p["label"] for p in leaked],
        "clean": len(leaked) == 0,
        "per_attr": per_attr,
    }


def aggregate_ner(results: list[dict]) -> dict:
    leaked_by_label: dict[str, int] = {}
    total_by_label: dict[str, int] = {}
    clean = 0
    for r in results:
        for p in r["per_attr"]:
            lbl = p["label"]
            total_by_label[lbl] = total_by_label.get(lbl, 0) + 1
            if p["present"]:
                leaked_by_label[lbl] = leaked_by_label.get(lbl, 0) + 1
        if r["clean"]:
            clean += 1
    total_gold_leak = sum(leaked_by_label.values())
    total_gold = sum(total_by_label.values())
    return {
        "n_records": len(results),
        "gold_leak_rate": total_gold_leak / total_gold if total_gold else 0.0,
        "fully_clean_records": clean,
        "leak_rate_by_label": {
            lbl: leaked_by_label.get(lbl, 0) / n
            for lbl, n in sorted(total_by_label.items())
        },
    }


# --------------------------------------------------------------------------- #
# Run one record through the LLM graph
# --------------------------------------------------------------------------- #

def run_record(graph, config: dict, rec: dict, label_map: dict) -> dict:
    attrs = [normalize_attribute(a) for a in rec.get("attributes_to_hide", [])]
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
    judge_by_attr = {lk.get("attribute"): lk for lk in judge_result.get("leaks", [])}
    judge_leaked = {k: bool(v.get("leaked")) for k, v in judge_by_attr.items()}

    nd, nj = _norm(delivered), _norm(judged)
    per_attr = []
    for a, sys_label in zip(attrs, sys_labels):
        present_delivered = is_present(nd, a["mentions"])
        present_judged = is_present(nj, a["mentions"])
        flagged = judge_leaked.get(sys_label, False)
        if present_judged and flagged:
            cat = "TP"
        elif present_judged and not flagged:
            cat = "FN"
        elif (not present_judged) and flagged:
            cat = "FP_unverified"
        else:
            cat = "TN"
        lk = judge_by_attr.get(sys_label) or {}
        per_attr.append({
            "label": a["label"],
            "present_delivered": present_delivered,
            "present_judged": present_judged,
            "judge_flagged": flagged,
            "judge_matched": sys_label in judge_by_attr,
            "judge_rationale": lk.get("rationale", ""),
            "category": cat,
        })

    utility = {k: judge_result.get(k) for k in
               ("task_utility", "informational_completeness", "factual_consistency",
                "fluency", "format_preserved")
               if judge_result.get(k) is not None}

    # Attacker guesses with confidence
    raw_guesses = (final.get("attacker_result") or {}).get("guesses", [])
    attacker_guesses = [
        {
            "attribute": g.get("attribute"),
            "guess": g.get("guess"),
            "confidence": g.get("confidence"),
            "evidence_spans": g.get("evidence_spans", []),
        }
        for g in raw_guesses
    ]

    leaked_delivered = [p for p in per_attr if p["present_delivered"]]
    rounds = final.get("rounds", final.get("iteration", 0))

    return {
        "id": rec.get("id", rec.get("label", "?")),
        "verdict": final.get("verdict", "MAX_ITERS"),
        "rounds": rounds,
        "n_attrs": len(attrs),
        "per_attr": per_attr,
        "leaked_delivered": [p["label"] for p in leaked_delivered],
        "false_pass": final.get("verdict") == "PASS" and bool(leaked_delivered),
        "utility": utility,
        "judge_summary": judge_result.get("summary", ""),
        "judge_leaks": judge_result.get("leaks", []),
        "attacker_guesses": attacker_guesses,
        "judged_text": judged,
        "final_text": delivered,
        "original_text": rec["text"],   # saved for inspection / before-after
    }


# --------------------------------------------------------------------------- #
# Aggregate
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
    rounds_dist: dict[int, int] = {}

    # attacker confidence
    conf_all: list[float] = []
    conf_leaked: list[float] = []   # confidence on guesses where attr was actually leaked

    for r in results:
        # rounds distribution
        rnd = r.get("rounds", 0)
        rounds_dist[rnd] = rounds_dist.get(rnd, 0) + 1

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

        # attacker confidence
        leaked_set = set(r.get("leaked_delivered", []))
        for g in r.get("attacker_guesses", []):
            c = g.get("confidence")
            if c is None:
                continue
            conf_all.append(float(c))
            # map sys_label back to short label is tricky; use rough heuristic:
            # if the attribute name appears in leaked_delivered we count it
            attr = g.get("attribute", "")
            if any(ll.lower() in attr.lower() or attr.lower() in ll.lower()
                   for ll in leaked_set):
                conf_leaked.append(float(c))

    tp, fn, fp = cats["TP"], cats["FN"], cats["FP_unverified"]
    total_gold_leak = sum(leaked_by_label.values())
    total_gold = sum(total_by_label.values())

    return {
        "n_records": len(results),
        "judge": {
            "leak_recall": _rate(tp, tp + fn),
            "missed_leaks": fn,
            "caught_leaks": tp,
            "unverified_flags": fp,
            "confusion": cats,
        },
        "system_privacy": {
            "gold_leak_rate": _rate(total_gold_leak, total_gold),
            "leak_rate_by_label": {
                lbl: _rate(leaked_by_label.get(lbl, 0), n)
                for lbl, n in sorted(total_by_label.items())
            },
            "fully_clean_records": clean_records,
            "false_pass_records": false_pass,
        },
        "utility_mean": {k: v / util_n for k, v in util_sum.items()} if util_n else {},
        "utility_scored_records": util_n,
        "attacker_confidence": {
            "mean_all": sum(conf_all) / len(conf_all) if conf_all else None,
            "mean_on_leaked": sum(conf_leaked) / len(conf_leaked) if conf_leaked else None,
            "n_guesses": len(conf_all),
        },
        "rounds_distribution": dict(sorted(rounds_dist.items())),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def print_report(agg: dict, results: list[dict], ner_agg: dict | None = None) -> None:
    j, s = agg["judge"], agg["system_privacy"]
    print("\n" + "=" * 64)
    print(f"EVAL  —  {agg['n_records']} record(s)")
    print("=" * 64)

    print("\nJUDGE (vs verbatim ground truth)")
    print(f"  leak recall .............. {j['leak_recall']:.0%}  "
          f"(caught {j['caught_leaks']}, missed {j['missed_leaks']})")
    print(f"  MISSED leaks (FN) ........ {j['missed_leaks']}   <-- critical")
    print(f"  unverified flags ......... {j['unverified_flags']}   (may be semantic catches)")

    print("\nSYSTEM PRIVACY (delivered text)")
    print(f"  gold leak rate ........... {s['gold_leak_rate']:.0%}")
    for lbl, rate in s["leak_rate_by_label"].items():
        print(f"    {(lbl + ' ').ljust(20, '.')} {rate:.0%}")
    print(f"  fully-clean records ...... {s['fully_clean_records']}/{agg['n_records']}")
    print(f"  FALSE PASS records ....... {s['false_pass_records']}")

    if ner_agg:
        print("\nNER BASELINE (spaCy, no LLM)")
        print(f"  gold leak rate ........... {ner_agg['gold_leak_rate']:.0%}")
        for lbl, rate in ner_agg["leak_rate_by_label"].items():
            print(f"    {(lbl + ' ').ljust(20, '.')} {rate:.0%}")
        print(f"  fully-clean records ...... {ner_agg['fully_clean_records']}/{ner_agg['n_records']}")

    if agg["utility_mean"]:
        print(f"\nUTILITY (mean over {agg['utility_scored_records']} scored record(s))")
        for k, v in agg["utility_mean"].items():
            print(f"  {k:.<26} {v:.2f}")

    ac = agg["attacker_confidence"]
    if ac["mean_all"] is not None:
        print(f"\nATTACKER CONFIDENCE")
        print(f"  mean (all guesses) ....... {ac['mean_all']:.2f}  (n={ac['n_guesses']})")
        if ac["mean_on_leaked"] is not None:
            print(f"  mean (leaked attrs) ...... {ac['mean_on_leaked']:.2f}")

    rd = agg["rounds_distribution"]
    if rd:
        print(f"\nROUNDS DISTRIBUTION")
        for rnd, cnt in rd.items():
            bar = "█" * cnt
            print(f"  {rnd} round(s): {cnt:3d}  {bar}")

    print("\nPER-RECORD")
    for r in results:
        flag = "  ⚠ FALSE PASS" if r["false_pass"] else ""
        print(f"  {r['id']}: verdict={r['verdict']} rounds={r['rounds']} "
              f"leaked={len(r['leaked_delivered'])}/{r['n_attrs']}{flag}")
        for lbl in r["leaked_delivered"]:
            print(f"      leaked: {lbl}")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate the anonymizer (judge-focused) on TAB records")
    ap.add_argument("path", help="record file (.json/.jsonl) or directory")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate at most N records")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="flush results to --out after every N records (default: 10)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N records (for manual resume)")
    ap.add_argument("--resume", action="store_true",
                    help="auto-resume: read --out and skip already-evaluated record IDs")
    ap.add_argument("--label-map",
                    help="type->description JSON (default: type_labels.json next to records)")
    ap.add_argument("--out", help="write results + aggregate as JSON here (flushed per batch)")
    ap.add_argument("--texts-out", default=None,
                    help="write original / LLM-anonymized / NER-anonymized texts here "
                         "(default: <out>_texts.json if --out is given)")
    ap.add_argument("--ner-only", action="store_true",
                    help="run ONLY the NER baseline (no LLM calls, free)")
    ap.add_argument("--no-ner", action="store_true",
                    help="skip NER baseline entirely")
    args = ap.parse_args()

    records = load_records(Path(args.path))
    if args.limit:
        records = records[: args.limit]
    if not records:
        sys.exit(f"no records found at {args.path}")

    label_map = load_label_map(Path(args.path), args.label_map)
    if label_map:
        print(f"label map: {len(label_map)} type(s) loaded")

    # ---- NER baseline (always free) ----------------------------------------
    nlp = None
    ner_results: list[dict] = []
    if not args.no_ner:
        nlp = _get_nlp()
        if nlp:
            print(f"running NER baseline on {len(records)} record(s)...")
            ner_results = [run_ner_baseline(nlp, rec) for rec in records]
            ner_agg = aggregate_ner(ner_results)
        else:
            ner_agg = None
    else:
        ner_agg = None

    if args.ner_only:
        if ner_agg:
            print("\n" + "=" * 64)
            print(f"NER BASELINE  —  {ner_agg['n_records']} record(s)")
            print("=" * 64)
            print(f"  gold leak rate ........... {ner_agg['gold_leak_rate']:.0%}")
            for lbl, rate in ner_agg["leak_rate_by_label"].items():
                print(f"    {(lbl + ' ').ljust(20, '.')} {rate:.0%}")
            print(f"  fully-clean records ...... {ner_agg['fully_clean_records']}/{ner_agg['n_records']}")
            if args.out:
                Path(args.out).write_text(
                    json.dumps({"ner_baseline": ner_agg, "records": ner_results},
                               indent=2, ensure_ascii=False),
                    encoding="utf-8")
                print(f"wrote {args.out}")
        return

    # Derive texts output path
    texts_path: str | None = args.texts_out
    if texts_path is None and args.out:
        p = Path(args.out)
        texts_path = str(p.with_name(p.stem + "_texts" + p.suffix))

    # ---- Resume: skip already-done record IDs ------------------------------
    existing_results: list[dict] = []
    done_ids: set[str] = set()
    if args.resume and args.out and Path(args.out).exists():
        try:
            saved = json.loads(Path(args.out).read_text(encoding="utf-8"))
            existing_results = saved.get("results", [])
            done_ids = {r["id"] for r in existing_results if r.get("verdict") != "ERROR"}
            print(f"resuming: {len(done_ids)} record(s) already done, skipping them")
        except Exception as e:
            print(f"warning: could not read existing --out file: {e}", file=sys.stderr)

    # Manual skip
    if args.skip:
        records = records[args.skip:]

    # Filter already-done
    todo = [r for r in records if r.get("id", r.get("label", "?")) not in done_ids]
    print(f"records to process: {len(todo)}")

    # ---- LLM graph ---------------------------------------------------------
    config = load_config()
    from .graph import build_graph
    graph = build_graph()

    results: list[dict] = list(existing_results)
    batch_size = max(1, args.batch_size)

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(todo) + batch_size - 1) // batch_size
        print(f"\n── batch {batch_num}/{total_batches} "
              f"(records {batch_start + 1}–{batch_start + len(batch)}) ──")

        for i, rec in enumerate(batch, 1):
            rid = rec.get("id", rec.get("label", f"#{batch_start + i}"))
            print(f"  [{batch_start + i}/{len(todo)}] {rid} ...", flush=True)
            try:
                results.append(run_record(graph, config, rec, label_map))
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                results.append({
                    "id": rid, "error": f"{type(e).__name__}: {e}",
                    "verdict": "ERROR", "rounds": 0, "n_attrs": 0,
                    "per_attr": [], "leaked_delivered": [], "false_pass": False,
                    "utility": {}, "final_text": "", "original_text": rec.get("text", ""),
                    "attacker_guesses": [],
                })

        # Flush after each batch
        if args.out:
            ok_so_far = [r for r in results if r.get("verdict") != "ERROR"]
            agg_so_far = aggregate(ok_so_far)
            _flush(args.out, agg_so_far, results, ner_agg, ner_results)
            if texts_path:
                _flush_texts(texts_path, results, ner_results)
            print(f"  → flushed {len(results)} record(s) to {args.out}")

    ok = [r for r in results if r.get("verdict") != "ERROR"]
    agg = aggregate(ok)
    print_report(agg, ok, ner_agg)

    if args.out:
        _flush(args.out, agg, results, ner_agg, ner_results)
        if texts_path:
            _flush_texts(texts_path, results, ner_results)
            print(f"wrote {texts_path}")
        print(f"wrote {args.out}")


def _flush(path: str, agg: dict, results: list[dict],
           ner_agg: dict | None, ner_results: list[dict]) -> None:
    # results.json — metrics + per-record data, NO raw texts (they go to texts.json)
    clean_results = []
    for r in results:
        r2 = {k: v for k, v in r.items() if k not in ("original_text",)}
        clean_results.append(r2)
    out: dict = {"aggregate": agg, "results": clean_results}
    if ner_agg:
        out["ner_baseline"] = ner_agg
    Path(path).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def _flush_texts(path: str, results: list[dict], ner_results: list[dict]) -> None:
    """texts.json — one entry per record with all three text versions side by side."""
    ner_by_id = {r["id"]: r.get("masked_text", "") for r in ner_results}
    entries = []
    for r in results:
        entries.append({
            "id": r["id"],
            "original": r.get("original_text", ""),
            "anonymized_llm": r.get("final_text", ""),
            "anonymized_ner": ner_by_id.get(r["id"], ""),
        })
    Path(path).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
