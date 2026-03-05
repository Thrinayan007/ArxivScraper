"""
Agent 1 — Planner
Generates search queries from the user prompt using an LLM.
"""

from langchain_core.prompts import PromptTemplate
from rich.console import Console

from core.state import ScraperState
from core.gpu import get_llm
from core.helpers import safe_json_obj

console = Console()


def planner_agent(state: ScraperState) -> ScraperState:
    console.rule("[bold blue]🧠 Planner Agent")

    llm    = get_llm(state["model"])
    prompt = PromptTemplate.from_template(
        """You are a research query expert.

User goal: {user_prompt}
Fields to extract: {fields}

Generate 5 short (3-6 word) search queries optimized for academic paper search.
The queries should cover different angles of the topic to maximize paper coverage.
Respond ONLY in this exact JSON (no markdown):
{{
  "queries": ["query1", "query2", "query3", "query4", "query5"]
}}"""
    )
    chain = prompt | llm

    try:
        resp    = chain.invoke({
            "user_prompt": state["user_prompt"],
            "fields":      ", ".join(state["fields"]),
        })
        data    = safe_json_obj(resp)
        queries = data.get("queries") or []
    except Exception as e:
        console.print(f"  [yellow]Planner LLM error: {e}")
        queries = []

    # Fallback: generate queries from the user prompt words
    if not queries:
        words   = [w for w in state["user_prompt"].split() if len(w) > 3][:6]
        queries = [
            " ".join(words[:4]),
            " ".join(words[:3]) + " recent",
            " ".join(words[1:5]),
        ]

    # Build the extraction prompt
    ep = (
        f"Extract the following fields from the academic paper text below: "
        f"{', '.join(state['fields'])}.\n"
        f"Return a JSON array where each element is a JSON object with exactly those field names.\n"
        f"Use null for any field not found in the text.\n"
        f"Return [] if the text contains nothing relevant.\n"
        f"No markdown fences. No explanation. Only valid JSON."
    )

    # Always search both arXiv and bioRxiv for maximum coverage
    use_biorxiv = True
    console.print("  [cyan]bioRxiv + medRxiv enabled (always-on for max coverage)")

    console.print(f"  Search queries: {queries}")

    state["search_queries"]            = queries
    state["extraction_prompt"]         = ep
    state["config"]["use_biorxiv"]     = use_biorxiv
    return state
