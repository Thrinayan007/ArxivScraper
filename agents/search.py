"""
Agent 2 — Search
Queries arXiv + bioRxiv APIs for paper metadata.
"""

from typing import Dict, List

from rich.console import Console

from core.state import ScraperState
from search_engines.arxiv import arxiv_search
from search_engines.biorxiv import biorxiv_search

console = Console()


def search_agent(state: ScraperState) -> ScraperState:
    console.rule("[bold blue]🔍 Search Agent")
    cfg        = state["config"]
    max_papers = cfg.get("max_pages", 150)
    all_papers: List[Dict] = []
    seen_titles: set = set()

    def _add_unique(papers: List[Dict]):
        added = 0
        for p in papers:
            title_key = p.get("_meta", {}).get("title", "").lower().strip()
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                all_papers.append(p)
                added += 1
        return added

    # ── arXiv search ─────────────────────────────────────────────────────────
    per_query = max(max_papers // max(len(state["search_queries"]), 1), 50)
    for query in state["search_queries"]:
        papers = arxiv_search(query, max_results=per_query)
        new = _add_unique(papers)
        console.print(f"  arXiv query done -> {new} new unique papers added")

    console.print(f"  arXiv total (deduped): [bold]{len(all_papers)}")

    # ── bioRxiv + medRxiv search (if bio topic or forced) ────────────────────
    use_biorxiv = (
        cfg.get("use_biorxiv", False)
        or cfg.get("force_biorxiv", False)
    ) and not cfg.get("no_biorxiv", False)

    if use_biorxiv:
        papers = biorxiv_search(
            state["user_prompt"],
            max_results=max(per_query, 50),
            include_medrxiv=True,
        )
        new = _add_unique(papers)
        bio_count = len([
            p for p in all_papers
            if p.get("_meta", {}).get("source") in ("bioRxiv", "medRxiv")
        ])
        console.print(f"  bioRxiv/medRxiv: [bold]{bio_count} matched")

    # Trim to max
    all_papers = all_papers[:max_papers]
    console.print(f"  Total papers collected: [bold green]{len(all_papers)}")

    state["raw_papers"] = all_papers
    return state
