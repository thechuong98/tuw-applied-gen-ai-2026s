"""FastAPI app: streams the adversarial run as newline-delimited JSON (one event per graph step)."""
import json
import os
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MAX_BATCH_SIZE = 10

from .config import load_config
from .graph import build_graph

CONFIG = load_config()
GRAPH = build_graph()

app = FastAPI(title="Semantic Anonymizer", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class AnonRequest(BaseModel):
    text: str
    attributes_to_hide: list[str] = Field(default_factory=list)
    utility_to_preserve: list[str] = Field(default_factory=list)
    channel: str = "text"
    ground_truth: dict = Field(default_factory=dict)  # optional: {attr: true_value} for eval mode


class BatchAnonRequest(BaseModel):
    texts: list[str]
    attributes_to_hide: list[str] = Field(default_factory=list)
    utility_to_preserve: list[str] = Field(default_factory=list)
    channel: str = "text"
    ground_truth: list[dict] = Field(default_factory=list)  # parallel to texts: ground_truth[i] for texts[i]


def _ollama_status(config: dict):
    """Probe Ollama when an ollama: model is configured. Never raises. None if not used."""
    specs = [s for s in config["models"].values()
             if isinstance(s, str) and s.startswith("ollama:")]
    if not specs:
        return None
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    wanted = {s.split(":", 1)[1] for s in specs}
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
            data = json.loads(resp.read())
        available = {m.get("name", "") for m in data.get("models", [])}
        models = {}
        for w in wanted:
            present = any(a == w or a.split(":", 1)[0] == w for a in available)
            models[w] = "present" if present else "missing"
        return {"reachable": True, "base_url": base_url, "models": models}
    except Exception:
        return {"reachable": False, "base_url": base_url,
                "models": {w: "unknown" for w in wanted}}


@app.get("/api/health")
def health():
    resp = {"status": "ok", "models": CONFIG["models"],
            "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))}
    ollama = _ollama_status(CONFIG)
    if ollama is not None:
        resp["ollama"] = ollama
    return resp


@app.get("/api/config")
def config():
    return {"defaults": CONFIG.get("defaults", {}), "loop": CONFIG["loop"], "models": CONFIG["models"]}


def _node_event(node: str, payload: dict, round_no: int) -> dict | None:
    """Translate a LangGraph node update into a UI-friendly event."""
    if node == "defender":
        return {"type": "node", "node": "defender", "round": round_no,
                "rewritten_text": payload.get("current_text", ""),
                "reasoning": payload.get("defender_reasoning", ""),
                "strategy_log": payload.get("strategy_log", {})}
    if node == "attacker":
        return {"type": "node", "node": "attacker", "round": round_no,
                "guesses": (payload.get("attacker_result") or {}).get("guesses", [])}
    if node == "judge":
        j = payload.get("judge_result", {})
        leaked = payload.get("leaked_attrs", [])
        ev = {"type": "node", "node": "judge", "round": round_no,
              "leaked": bool(leaked), "leaked_attrs": leaked,
              "leaks": j.get("leaks", []), "summary": j.get("summary", ""), "scores": j}
        if j.get("ground_truth_validation"):
            ev["ground_truth_validation"] = j["ground_truth_validation"]
        return ev
    return None


@app.post("/api/anonymize")
def anonymize(req: AnonRequest):
    init = {
        "original_text": req.text,
        "attributes_to_hide": req.attributes_to_hide or CONFIG["defaults"]["attributes_to_hide"],
        "utility_to_preserve": req.utility_to_preserve,
        "channel": req.channel,
        "ground_truth": req.ground_truth or {},
        "config": CONFIG,
        "iteration": 0,
        "history": [],
    }

    def gen():
        yield json.dumps({"type": "start", "attributes_to_hide": init["attributes_to_hide"]},
                         ensure_ascii=False) + "\n"
        round_no = 0
        verdict = None
        try:
            for update in GRAPH.stream(init, stream_mode="updates", config={"recursion_limit": 60}):
                for node, payload in update.items():
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("verdict"):
                        verdict = payload["verdict"]
                    if node == "defender":
                        round_no += 1
                    if node == "finalize":
                        yield json.dumps({
                            "type": "done",
                            "verdict": verdict or payload.get("verdict", "MAX_ITERS"),
                            "final_text": payload.get("final_text", ""),
                            "rounds": payload.get("rounds", round_no),
                        }, ensure_ascii=False) + "\n"
                        continue
                    ev = _node_event(node, payload, round_no)
                    if ev:
                        yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # surface errors to the UI instead of a dead stream
            yield json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/anonymize_batch")
def anonymize_batch(req: BatchAnonRequest):
    """Process multiple texts sequentially through the adversarial loop."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    if len(req.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"texts list exceeds max batch size of {MAX_BATCH_SIZE}")

    attrs = req.attributes_to_hide or CONFIG["defaults"]["attributes_to_hide"]
    results = []
    success_count = 0
    error_count = 0

    for i, text in enumerate(req.texts):
        gt = req.ground_truth[i] if i < len(req.ground_truth) else {}
        init = {
            "original_text": text,
            "attributes_to_hide": attrs,
            "utility_to_preserve": req.utility_to_preserve,
            "channel": req.channel,
            "ground_truth": gt,
            "config": CONFIG,
            "iteration": 0,
            "history": [],
        }
        try:
            final_state = GRAPH.invoke(init, config={"recursion_limit": 60})
            judge_result = final_state.get("judge_result") or {}
            gt_val = judge_result.get("ground_truth_validation")
            results.append({
                "index": i,
                "status": "success",
                "original_text": text,
                "final_text": final_state.get("final_text", ""),
                "verdict": final_state.get("verdict", "MAX_ITERS"),
                "rounds": final_state.get("rounds", final_state.get("iteration", 0)),
                "leaked_attrs": final_state.get("leaked_attrs", []),
                "ground_truth_validation": gt_val,
            })
            success_count += 1
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "original_text": text,
                "error": f"{type(e).__name__}: {e}",
            })
            error_count += 1

    return {
        "batch_size": len(req.texts),
        "success_count": success_count,
        "error_count": error_count,
        "results": results,
    }
