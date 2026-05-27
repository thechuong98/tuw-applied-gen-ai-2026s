"""FastAPI app: streams the adversarial run as newline-delimited JSON (one event per graph step)."""
import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


@app.get("/api/health")
def health():
    return {"status": "ok", "models": CONFIG["models"], "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))}


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
        return {"type": "node", "node": "judge", "round": round_no,
                "leaked": bool(leaked), "leaked_attrs": leaked,
                "leaks": j.get("leaks", []), "summary": j.get("summary", ""), "scores": j}
    return None


@app.post("/api/anonymize")
def anonymize(req: AnonRequest):
    init = {
        "original_text": req.text,
        "attributes_to_hide": req.attributes_to_hide or CONFIG["defaults"]["attributes_to_hide"],
        "utility_to_preserve": req.utility_to_preserve,
        "channel": req.channel,
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
