"""
LangGraph builder — wires all agents into the multi-agent pipeline.
"""

from langgraph.graph import StateGraph, END
from rich.console import Console

from core.state import ScraperState
from agents.planner import planner_agent
from agents.search import search_agent
from agents.extractor import extractor_agent
from agents.validator import validator_agent
from agents.prompt_engineer import prompt_engineer_agent
from agents.merger import merger_agent
from agents.output import output_agent

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

def route_after_validation(state: ScraperState) -> str:
    """Decide whether to retry extraction or proceed to merging."""
    if state["validation_score"] < 0.5 and state.get("retry_count", 0) < 2:
        console.print(
            f"  [yellow]Score {state['validation_score']:.2f} < 0.5 -> rewriting prompt"
        )
        return "prompt_engineer"
    return "merger"


# ─────────────────────────────────────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """Compile the multi-agent LangGraph pipeline."""
    g = StateGraph(ScraperState)

    g.add_node("planner",         planner_agent)
    g.add_node("search",          search_agent)
    g.add_node("extractor",       extractor_agent)
    g.add_node("validator",       validator_agent)
    g.add_node("prompt_engineer", prompt_engineer_agent)
    g.add_node("merger",          merger_agent)
    g.add_node("output",          output_agent)

    g.set_entry_point("planner")
    g.add_edge("planner",  "search")
    g.add_edge("search",   "extractor")
    g.add_edge("extractor", "validator")
    g.add_conditional_edges(
        "validator",
        route_after_validation,
        {"prompt_engineer": "prompt_engineer", "merger": "merger"},
    )
    g.add_edge("prompt_engineer", "extractor")
    g.add_edge("merger",          "output")
    g.add_edge("output",          END)

    return g.compile()
