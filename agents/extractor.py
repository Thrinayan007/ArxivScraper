"""
Agent 3 — Extractor
Uses local LLM (Ollama) to extract structured fields from papers.
Falls back to metadata-only extraction when Ollama is unavailable.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, track,
)

from core.state import ScraperState
from core.gpu import get_llm
from core.helpers import safe_json_array

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_ollama_model(model_name: str) -> bool:
    """Check if an Ollama model is available (Ollama running + model pulled)."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        available = [m.get("name", "") for m in models]
        return any(
            model_name == m or model_name == m.split(":")[0]
            for m in available
        )
    except Exception:
        return False


def _extract_from_metadata(paper: Dict) -> Dict:
    """Extract a structured record directly from API metadata (no LLM needed)."""
    meta = paper.get("_meta", {})
    record = {}

    field_mapping = {
        "title":          "title",
        "authors":        "authors",
        "published_date": "published_date",
        "doi":            "doi",
        "url":            "url",
        "pdf_url":        "pdf_url",
        "categories":     "categories",
        "source":         "source",
    }

    for meta_key, field_name in field_mapping.items():
        if meta.get(meta_key):
            record[field_name] = meta[meta_key]

    # Extract abstract from the text block
    text = paper.get("text", "")
    abstract_match = re.search(r'ABSTRACT:\s*(.+)', text, re.DOTALL)
    if abstract_match:
        record["abstract"] = abstract_match.group(1).strip()

    # Extract journal if present
    journal_match = re.search(r'JOURNAL:\s*(.+)', text)
    if journal_match:
        record["journal"] = journal_match.group(1).strip()

    record["_source_url"] = paper.get("url", "")
    record["_source"]     = meta.get("source", "unknown")
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def extractor_agent(state: ScraperState) -> ScraperState:
    console.rule("[bold blue]⚙️  Extractor Agent")

    if not state["raw_papers"]:
        console.print("  [red]No papers to extract from!")
        state["extracted_data"] = []
        return state

    model_name  = state["model"]
    num_workers = state["config"].get("extract_workers", 4)

    # ── Check if Ollama is available ─────────────────────────────────────────
    llm_available = _check_ollama_model(model_name)

    if llm_available:
        console.print(
            f"  [green]Ollama model '{model_name}' available — using LLM extraction"
        )
    else:
        console.print(
            f"  [yellow]Ollama model '{model_name}' not found — "
            f"using direct metadata extraction (faster, no LLM needed)"
        )
        console.print(
            "  [dim]To enable LLM extraction: ollama pull " + model_name
        )

    papers = state["raw_papers"]

    # ── FAST PATH: metadata-only extraction (no LLM) ────────────────────────
    if not llm_available:
        results: List[Dict] = []
        for paper in track(
            papers, description="Extracting metadata...", console=console
        ):
            record = _extract_from_metadata(paper)
            if record:
                results.append(record)

        console.print(f"  Records extracted: [bold]{len(results)} (metadata mode)")
        state["extracted_data"] = results
        return state

    # ── LLM PATH: parallel Ollama extraction ─────────────────────────────────
    fields_str  = ", ".join(state["fields"])
    ep          = state["extraction_prompt"]

    console.print(
        f"  Workers: [cyan]{num_workers}  |  "
        f"Papers: [cyan]{len(papers)}"
    )

    prompt_template = PromptTemplate.from_template(
        """{extraction_prompt}

Required output fields: {fields}

--- TEXT START ---
{text}
--- TEXT END ---"""
    )

    def _extract_one(paper: Dict) -> List[Dict]:
        """Extract structured data from a single paper using the LLM."""
        meta = paper.get("_meta", {})
        llm  = get_llm(model_name)
        chain = prompt_template | llm
        out: List[Dict] = []

        try:
            response = chain.invoke({
                "extraction_prompt": ep,
                "fields":            fields_str,
                "text":              paper["text"][:4000],
            })
            items = safe_json_array(response)

            if items:
                for item in items:
                    if isinstance(item, dict):
                        for k, v in meta.items():
                            nk = k.lower().replace(" ", "_")
                            if not item.get(nk) and not item.get(k):
                                item[nk] = v
                        item["_source_url"] = paper["url"]
                        item["_source"]     = meta.get("source", "unknown")
                        out.append(item)

            if not out and meta:
                record = {
                    k.lower().replace(" ", "_"): v for k, v in meta.items()
                }
                record["_source_url"] = paper["url"]
                record["_source"]     = meta.get("source", "unknown")
                out.append(record)

        except Exception as e:
            state["errors"].append(f"Extract {paper['url'][:40]}: {e}")
            # Fallback to metadata on error
            record = _extract_from_metadata(paper)
            if record:
                out.append(record)
        return out

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(papers))

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_extract_one, p): p for p in papers}
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    state["errors"].append(f"Thread error: {e}")
                progress.advance(task)

    console.print(f"  Records extracted: [bold]{len(results)}")
    state["extracted_data"] = results
    return state
