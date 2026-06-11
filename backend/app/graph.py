"""Build & compile the per-record adversarial graph (Tier B)."""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import AnonState


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(AnonState)
    g.add_node("defender", nodes.defender)
    g.add_node("attacker", nodes.attacker)
    g.add_node("judge", nodes.judge)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "defender")
    g.add_edge("defender", "attacker")
    g.add_edge("attacker", "judge")
    g.add_conditional_edges("judge", nodes.route_after_judge,
                            {"retry": "defender", "finalize": "finalize"})
    g.add_edge("finalize", END)

    return g.compile()
