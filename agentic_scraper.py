#!/usr/bin/env python3
"""
agentic_scraper.py — Agentic AI Research Paper Scraper (CLI)
=============================================================
Multi-agent pipeline orchestrated with LangGraph.
Searches arXiv and bioRxiv for academic papers on any topic.

Agents:
  1. Planner      — Generates search queries from the user prompt
  2. Search       — Queries arXiv + bioRxiv APIs for paper metadata
  3. Extractor    — Uses local LLM (Ollama) to extract structured fields
  4. Validator    — Scores extraction quality, triggers re-extraction if poor
  5. Merger       — Deduplicates results (semantic or exact)
  6. Output       — Writes CSV / JSON / TXT

Usage:
  python agentic_scraper.py \
    --prompt "transformer architectures for medical image segmentation" \
    --fields title,authors,abstract,journal,doi,published_date \
    --output papers.csv --format csv
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TypedDict

import requests
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, track,
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# GPU — auto-detection and manual force
# ─────────────────────────────────────────────────────────────────────────────

def force_gpu(device_id: int = 0) -> str:
    """
    Force CUDA GPU usage. Sets env vars for Ollama and sentence-transformers.
    Returns 'cuda:N' or 'cpu'.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["OLLAMA_NUM_GPU"]       = "99"
    os.environ["OLLAMA_GPU_LAYERS"]    = "99"

    try:
        import torch
        if not torch.cuda.is_available():
            console.print(
                "[red bold]CUDA not available to torch.\n"
                "[yellow]Check: nvidia-smi | nvcc --version | "
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
            return "cpu"

        device = f"cuda:{device_id}"
        try:
            _ = torch.zeros(1, device=device)
        except RuntimeError as e:
            console.print(f"[red]GPU warm-up failed on {device}: {e}")
            return "cpu"

        name    = torch.cuda.get_device_name(device_id)
        mem     = torch.cuda.get_device_properties(device_id).total_mem // (1024 ** 2)
        compute = torch.cuda.get_device_capability(device_id)
        console.print(
            f"  [green bold]GPU FORCED: {name}  |  {mem} MB VRAM  |  "
            f"Compute {compute[0]}.{compute[1]}  |  device={device} ✓"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        return device

    except ImportError:
        console.print(
            "[red]torch not installed.\n"
            "[yellow]Run: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        return "cpu"


DEVICE: str = "cpu"


def get_llm(model: str) -> OllamaLLM:
    """Return an Ollama LLM instance, offloading to GPU when available."""
    on_gpu = DEVICE.startswith("cuda") or DEVICE == "mps"
    return OllamaLLM(model=model, temperature=0, num_gpu=-1 if on_gpu else 0)


# ─────────────────────────────────────────────────────────────────────────────
# API Constants
# ─────────────────────────────────────────────────────────────────────────────

ARXIV_API    = "http://export.arxiv.org/api/query"
BIORXIV_API  = "https://api.biorxiv.org/details/biorxiv"
MEDRXIV_API  = "https://api.biorxiv.org/details/medrxiv"

# Stop words to filter out when building arXiv queries
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "not", "no", "nor", "so",
    "as", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "about", "above", "after", "again", "all", "also", "any",
    "because", "before", "between", "both", "each", "few", "more",
    "most", "other", "over", "same", "some", "such", "through", "under",
    "up", "very", "what", "when", "where", "which", "while", "who",
    "how", "using", "based", "via", "into", "during", "upon", "their",
    "our", "your", "we", "they", "papers", "paper", "recent", "new",
    "find", "search", "get", "show", "use",
}

# bioRxiv subject areas that signal life-science queries
BIO_KEYWORDS = {
    "biology", "biomedical", "genomic", "protein", "gene", "cancer",
    "clinical", "drug", "molecule", "cell", "neuro", "brain", "disease",
    "pathology", "immunology", "virology", "microbiology", "ecology",
    "biochem", "bioinformatics", "protac", "crispr", "rna", "dna",
    "enzyme", "receptor", "antibody", "vaccine", "epidemiology",
    "pharmacol", "metabol", "transcript", "sequencing", "organism",
    "tissue", "morpholog", "phenotype", "genotype", "mutation",
    "medical", "health", "diagnosis", "therapy", "treatment",
}


def _is_bio_topic(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in BIO_KEYWORDS)


def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from user prompt, filtering stop words."""
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return keywords


# ─────────────────────────────────────────────────────────────────────────────
# Shared State
# ─────────────────────────────────────────────────────────────────────────────

class ScraperState(TypedDict):
    user_prompt:        str
    fields:             List[str]
    output_format:      str
    output_file:        str
    model:              str
    config:             Dict[str, Any]
    search_queries:     List[str]
    raw_papers:         List[Dict]        # papers from APIs
    extracted_data:     List[Dict]        # LLM-extracted records
    merged_data:        List[Dict]        # deduplicated final records
    extraction_prompt:  str
    validation_score:   float
    validation_issues:  List[str]
    retry_count:        int
    errors:             List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_json_array(text: str) -> List[Dict]:
    """Extract a JSON array from possibly noisy LLM output."""
    text = re.sub(r'```(?:json)?', '', text).strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def safe_json_obj(text: str) -> Dict:
    text = re.sub(r'```(?:json)?', '', text).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# arXiv API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_arxiv_queries(query: str) -> List[str]:
    """
    Build multiple arXiv API query strings from a natural-language query.
    Returns several strategies: AND (strict), OR (broad), title-only.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return [f'all:"{query}"']

    queries = []

    # Strategy 1: AND all keywords across title + abstract (most relevant)
    and_parts = [f'(ti:{kw} OR abs:{kw})' for kw in keywords[:5]]
    queries.append(' AND '.join(and_parts))

    # Strategy 2: AND on top 3 keywords only (broader)
    if len(keywords) >= 3:
        and_parts_short = [f'(ti:{kw} OR abs:{kw})' for kw in keywords[:3]]
        queries.append(' AND '.join(and_parts_short))

    # Strategy 3: OR all keywords (broadest, catches tangential papers)
    or_parts = [f'abs:{kw}' for kw in keywords[:6]]
    queries.append(' OR '.join(or_parts))

    # Strategy 4: Title-only search with top 3 keywords
    ti_parts = [f'ti:{kw}' for kw in keywords[:3]]
    queries.append(' AND '.join(ti_parts))

    return queries


def _arxiv_fetch_page(
    search_query: str, start: int, max_results: int
) -> List[Dict]:
    """Fetch a single page of arXiv results."""
    params = {
        "search_query": search_query,
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "relevance",
        "sortOrder":    "descending",
    }
    for attempt in range(3):
        try:
            resp = requests.get(ARXIV_API, params=params, timeout=30)
            resp.raise_for_status()
            return _parse_arxiv_xml(resp.text)
        except Exception as e:
            wait = 3 * (2 ** attempt)
            console.print(f"    [yellow]arXiv page attempt {attempt+1} failed: {e}")
            time.sleep(wait)
    return []


def arxiv_search(query: str, max_results: int = 80) -> List[Dict]:
    """
    Search arXiv using multiple query strategies + pagination.
    Returns list of paper dicts with text + metadata.
    """
    arxiv_queries = _build_arxiv_queries(query)
    all_papers: List[Dict] = []
    seen_ids: set = set()

    page_size = min(max_results, 100)  # arXiv max per request is ~100

    for aq in arxiv_queries:
        if len(all_papers) >= max_results:
            break

        # Paginate through results for this query
        for start in range(0, max_results, page_size):
            if len(all_papers) >= max_results:
                break

            papers = _arxiv_fetch_page(aq, start, page_size)
            if not papers:
                break  # No more results for this query

            new_count = 0
            for p in papers:
                pid = p.get("url", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)
                    new_count += 1

            console.print(
                f"  arXiv [{aq[:55]}...] page {start//page_size + 1} -> "
                f"{len(papers)} fetched, {new_count} new"
            )

            # If we got fewer results than page_size, no more pages
            if len(papers) < page_size:
                break

            # arXiv rate limit: 1 request per 3 seconds
            time.sleep(3.0)

        time.sleep(3.0)  # delay between different query strategies

    console.print(f"  arXiv total unique: [bold green]{len(all_papers)}")
    return all_papers[:max_results]


def _parse_arxiv_xml(xml_data: str) -> List[Dict]:
    """Parse arXiv Atom feed XML into paper dicts."""
    papers: List[Dict] = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return papers

    for entry in root.findall("atom:entry", ns):
        try:
            title     = (entry.findtext("atom:title", "", ns) or "").strip()
            abstract  = (entry.findtext("atom:summary", "", ns) or "").strip()
            published = (entry.findtext("atom:published", "", ns) or "").strip()
            arxiv_id  = (entry.findtext("atom:id", "", ns) or "").strip()

            # Clean up whitespace in title and abstract
            title    = " ".join(title.split())
            abstract = " ".join(abstract.split())

            if not title:
                continue

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.findtext("atom:name", "", ns)
                if name:
                    authors.append(name.strip())

            # Extract DOI from arxiv links if available
            doi = ""
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                href  = link.get("href", "")
                ltype = link.get("type", "")
                ltitle = link.get("title", "")
                if ltitle == "pdf" or "pdf" in href:
                    pdf_url = href
                if "doi.org" in href:
                    doi = href

            # Extract categories
            categories = []
            for cat in entry.findall("{http://arxiv.org/schemas/atom}primary_category"):
                term = cat.get("term", "")
                if term:
                    categories.append(term)
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term", "")
                if term and term not in categories:
                    categories.append(term)

            # Published date as YYYY-MM-DD
            pub_date = published[:10] if len(published) >= 10 else published

            papers.append({
                "url":  arxiv_id,
                "text": (
                    f"TITLE: {title}\n"
                    f"AUTHORS: {', '.join(authors)}\n"
                    f"DATE: {pub_date}\n"
                    f"DOI: {doi}\n"
                    f"CATEGORIES: {', '.join(categories)}\n"
                    f"ABSTRACT: {abstract}"
                ),
                "_meta": {
                    "title":          title,
                    "authors":        ", ".join(authors),
                    "published_date": pub_date,
                    "doi":            doi or arxiv_id,
                    "url":            arxiv_id,
                    "pdf_url":        pdf_url,
                    "categories":     ", ".join(categories),
                    "source":         "arXiv",
                },
            })
        except Exception:
            continue

    return papers


# ─────────────────────────────────────────────────────────────────────────────
# bioRxiv / medRxiv API helpers
# ─────────────────────────────────────────────────────────────────────────────

def biorxiv_search(query: str, max_results: int = 80, include_medrxiv: bool = True) -> List[Dict]:
    """
    Search bioRxiv (and optionally medRxiv) using the content-detail API.
    The bioRxiv API is date-range based, so we fetch recent papers and
    filter client-side by keyword matching.
    """
    from datetime import datetime, timedelta

    papers: List[Dict] = []

    # Build date ranges — try multiple windows for better coverage
    now = datetime.now()
    date_ranges = [
        # Last 30 days first (most recent)
        ((now - timedelta(days=30)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")),
        # 30-90 days ago
        ((now - timedelta(days=90)).strftime("%Y-%m-%d"), (now - timedelta(days=30)).strftime("%Y-%m-%d")),
        # 90-180 days ago (extended range for more coverage)
        ((now - timedelta(days=180)).strftime("%Y-%m-%d"), (now - timedelta(days=90)).strftime("%Y-%m-%d")),
    ]

    apis = [("bioRxiv", BIORXIV_API)]
    if include_medrxiv:
        apis.append(("medRxiv", MEDRXIV_API))

    # Use extracted keywords, not raw words
    query_keywords = _extract_keywords(query)
    if not query_keywords:
        query_keywords = [w.lower() for w in query.split() if len(w) > 2]

    console.print(f"  bioRxiv filter keywords: {query_keywords[:8]}")

    for source_name, api_base in apis:
        source_count = 0

        for start_date, end_date in date_ranges:
            if len(papers) >= max_results:
                break

            cursor = 0
            pages_fetched = 0
            max_pages = 8  # Limit API pages per date range to avoid excessive requests

            while pages_fetched < max_pages:
                url = f"{api_base}/{start_date}/{end_date}/{cursor}"
                try:
                    resp = requests.get(url, timeout=30, headers={
                        "User-Agent": "ResearchScraper/1.0 (academic research tool)"
                    })

                    if resp.status_code == 404:
                        console.print(f"    [dim]{source_name} no data for {start_date} to {end_date}")
                        break

                    resp.raise_for_status()
                    data = resp.json()

                    collection = data.get("collection", [])
                    if not collection:
                        break

                    batch_matches = 0
                    for item in collection:
                        title    = item.get("title", "").strip()
                        abstract = item.get("abstract", "").strip()
                        combined = (title + " " + abstract).lower()

                        # Require at least 1 keyword for broader coverage
                        min_match = min(1, len(query_keywords))
                        matches = sum(1 for w in query_keywords if w in combined)
                        if matches < min_match:
                            continue

                        authors  = item.get("authors", "").strip()
                        doi      = item.get("doi", "")
                        pub_date = item.get("date", "")
                        category = item.get("category", "")
                        version  = item.get("version", "1")

                        paper_url = f"https://doi.org/{doi}" if doi else ""

                        papers.append({
                            "url":  paper_url,
                            "text": (
                                f"TITLE: {title}\n"
                                f"AUTHORS: {authors}\n"
                                f"DATE: {pub_date}\n"
                                f"DOI: {doi}\n"
                                f"CATEGORY: {category}\n"
                                f"ABSTRACT: {abstract}"
                            ),
                            "_meta": {
                                "title":          title,
                                "authors":        authors,
                                "published_date": pub_date,
                                "doi":            doi,
                                "url":            paper_url,
                                "pdf_url":        f"https://www.biorxiv.org/content/{doi}v{version}.full.pdf" if doi else "",
                                "categories":     category,
                                "source":         source_name,
                            },
                        })
                        batch_matches += 1
                        source_count += 1

                        if len(papers) >= max_results:
                            break

                    cursor += len(collection)
                    pages_fetched += 1

                    console.print(
                        f"    {source_name} {start_date}..{end_date} page {pages_fetched}: "
                        f"{len(collection)} scanned, {batch_matches} matched"
                    )

                    if len(papers) >= max_results:
                        break

                    # Respect rate limits
                    time.sleep(1.5)

                except Exception as e:
                    console.print(f"    [yellow]{source_name} API error: {str(e)[:80]}")
                    break

        console.print(f"  {source_name} total matched: [bold]{source_count}")

    return papers[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Planner
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Search
# ─────────────────────────────────────────────────────────────────────────────

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
    # Each query already generates multiple sub-strategies internally
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
        # Use original prompt for keyword matching (richer than sub-queries)
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


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 — Extractor
# ─────────────────────────────────────────────────────────────────────────────

def _check_ollama_model(model_name: str) -> bool:
    """Check if an Ollama model is available (Ollama running + model pulled)."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        available = [m.get("name", "") for m in models]
        # Check exact match or match without tag
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

    # Map known metadata keys to common field names
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
        console.print(f"  [green]Ollama model '{model_name}' available — using LLM extraction")
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
        for paper in track(papers, description="Extracting metadata...", console=console):
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
                record = {k.lower().replace(" ", "_"): v for k, v in meta.items()}
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


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — Validator
# ─────────────────────────────────────────────────────────────────────────────

def validator_agent(state: ScraperState) -> ScraperState:
    console.rule("[bold blue]✅ Validator Agent")
    data   = state["extracted_data"]
    issues: List[str] = []
    score  = 1.0

    if not data:
        state["validation_score"]  = 0.0
        state["validation_issues"] = ["No records extracted"]
        console.print("  [red]Score: 0.0 — no data")
        return state

    norm_fields  = [f.lower().replace(" ", "_") for f in state["fields"]]
    total_cells  = len(data) * len(norm_fields)
    filled_cells = sum(
        1 for item in data for f in norm_fields
        if item.get(f) and str(item[f]).strip() not in ("", "null", "None", "none")
    )
    coverage = filled_cells / total_cells if total_cells else 0.0

    if coverage < 0.3:
        issues.append(f"Overall field coverage low ({coverage:.0%})")
        score *= 0.5

    # Check primary field (first field)
    key_field = norm_fields[0]
    key_hits  = sum(1 for item in data if item.get(key_field))
    if len(data) > 0 and key_hits / len(data) < 0.4:
        issues.append(f"Primary field '{key_field}' sparse ({key_hits}/{len(data)})")
        score *= 0.7

    console.print(
        f"  Records: {len(data)} | Coverage: {coverage:.0%} | "
        f"Score: {score:.2f} | Issues: {issues or ['none']}"
    )
    state["validation_score"]  = score
    state["validation_issues"] = issues
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4b — Prompt Engineer (only triggered on low validation)
# ─────────────────────────────────────────────────────────────────────────────

def prompt_engineer_agent(state: ScraperState) -> ScraperState:
    console.rule("[bold blue]🔧 Prompt Engineer Agent")
    llm = get_llm(state["model"])

    example = {f: f"<{f}_value>" for f in state["fields"]}
    prompt  = PromptTemplate.from_template(
        """You are a prompt engineering expert for structured data extraction.

Validation issues:
{issues}

Current (weak) prompt:
{current_prompt}

Required fields: {fields}
Target JSON format: {example}

Rewrite the extraction prompt to be clearer and fix each issue.
Extract ONLY what appears in the paper text. Do NOT invent data.
Return ONLY the improved prompt text."""
    )
    chain = prompt | llm
    try:
        improved = chain.invoke({
            "issues":         "; ".join(state["validation_issues"]) or "low quality",
            "current_prompt": state["extraction_prompt"],
            "fields":         ", ".join(state["fields"]),
            "example":        json.dumps([example]),
        }).strip()
        if improved and len(improved) > 40:
            state["extraction_prompt"] = improved
            console.print(f"  Prompt updated ({len(improved)} chars)")
    except Exception as e:
        console.print(f"  [red]Prompt engineer error: {e}")

    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 — Merger (normalize + semantic dedup)
# ─────────────────────────────────────────────────────────────────────────────

def merger_agent(state: ScraperState) -> ScraperState:
    console.rule("[bold blue]🔀 Merger Agent")
    data = state["extracted_data"]

    if not data:
        state["merged_data"] = []
        return state

    # Normalize all keys
    normalized: List[Dict] = []
    for item in data:
        clean: Dict = {}
        for k, v in item.items():
            key = k.strip().lower().replace(" ", "_").replace("-", "_")
            if isinstance(v, list):
                val = ", ".join(str(x) for x in v)
            elif v is None:
                val = ""
            else:
                val = str(v).strip()
            clean[key] = val
        normalized.append(clean)

    threshold = state["config"].get("dedup_threshold", 0.92)
    merged    = _semantic_dedup(normalized, state["fields"], threshold)

    console.print(f"  {len(data)} raw -> {len(merged)} after dedup")
    state["merged_data"] = merged
    return state


def _semantic_dedup(
    data: List[Dict], fields: List[str], threshold: float = 0.92
) -> List[Dict]:
    if len(data) <= 1:
        return data
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        console.print(f"  Semantic dedup on: [cyan]{DEVICE.upper()}")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        norm_f = [f.lower().replace(" ", "_") for f in fields[:3]]
        fingerprints = [
            " ".join(str(item.get(f, "")) for f in norm_f) or str(item)
            for item in data
        ]
        embeddings = model.encode(
            fingerprints,
            batch_size=128 if DEVICE != "cpu" else 32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        sims = cosine_similarity(embeddings)

        keep = [True] * len(data)
        for i in range(len(data)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(data)):
                if keep[j] and sims[i][j] > threshold:
                    keep[j] = False

        result = [item for item, k in zip(data, keep) if k]
        return result

    except Exception as e:
        console.print(f"  [yellow]Semantic dedup unavailable ({e}) — using exact dedup")
        norm_f = [f.lower().replace(" ", "_") for f in fields[:3]]
        seen: set = set()
        result: List[Dict] = []
        for item in data:
            key = tuple(item.get(f, "") for f in norm_f)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6 — Output
# ─────────────────────────────────────────────────────────────────────────────

def output_agent(state: ScraperState) -> ScraperState:
    console.rule("[bold blue]📄 Output Agent")
    data = state["merged_data"]
    fmt  = state["output_format"].lower()
    path = state["output_file"]

    if not data:
        console.print("  [red]No data to write.")
        return state

    # Determine column order: user-requested fields first, then extras
    user_fields = [f.lower().replace(" ", "_") for f in state["fields"]]
    all_keys    = list({k for item in data for k in item.keys() if not k.startswith("_")})
    ordered     = [f for f in user_fields if f in all_keys]
    ordered    += [k for k in sorted(all_keys) if k not in ordered]

    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
            writer.writeheader()
            for item in data:
                row = {k: v for k, v in item.items() if not k.startswith("_")}
                writer.writerow(row)

    elif fmt == "json":
        clean = [{k: v for k, v in item.items() if not k.startswith("_")} for item in data]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)

    elif fmt in ("txt", "text"):
        with open(path, "w", encoding="utf-8") as f:
            for i, item in enumerate(data, 1):
                f.write(f"{'─' * 60}\nRecord {i}\n{'─' * 60}\n")
                for k in ordered:
                    v = item.get(k, "")
                    if not str(k).startswith("_"):
                        f.write(f"  {k:<26}: {v}\n")
                f.write("\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    console.print(f"  [green bold]Saved {len(data)} records -> {path}")

    # Terminal preview table
    preview_fields = [
        f for f in user_fields
        if f in all_keys
    ][:5]
    if preview_fields:
        table = Table(
            title=f"Preview — {min(10, len(data))} of {len(data)} records",
            show_lines=True,
        )
        for f in preview_fields:
            table.add_column(f, max_width=40, overflow="fold")
        for row in data[:10]:
            table.add_row(*[str(row.get(f, ""))[:38] for f in preview_fields])
        console.print(table)

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

def route_after_validation(state: ScraperState) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agentic_scraper",
        description="Agentic AI Research Paper Scraper — arXiv + bioRxiv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU usage:
  --device cuda        auto-detect GPU 0
  --device cuda:1      use second GPU
  --device cpu         force CPU

Examples:
  python agentic_scraper.py \\
      --prompt "transformer architectures for image segmentation" \\
      --fields title,authors,abstract,doi,published_date \\
      --output papers.csv --format csv

  python agentic_scraper.py \\
      --prompt "CRISPR gene editing delivery methods 2024" \\
      --fields title,authors,journal,doi,published_date \\
      --output gene_editing.json --format json --device cuda
        """,
    )
    p.add_argument("--prompt",             required=True,
                   help="Natural language research query")
    p.add_argument("--fields",             required=True,
                   help="Comma-separated fields to extract (e.g. title,authors,abstract,doi)")
    p.add_argument("--output",             default="output.csv",
                   help="Output file path (default: output.csv)")
    p.add_argument("--format",             default="csv", choices=["csv", "json", "txt"],
                   help="Output format (default: csv)")
    p.add_argument("--max-papers",         type=int, default=150,
                   help="Max papers to fetch (default: 150)")
    p.add_argument("--model",              default="llama3.2",
                   help="Ollama model name (default: llama3.2)")
    p.add_argument("--dedup-threshold",    type=float, default=0.92,
                   help="Semantic dedup cosine threshold (default: 0.92)")
    p.add_argument("--device",             default="cuda",
                   help="Device: 'cuda', 'cuda:1', or 'cpu' (default: cuda)")
    p.add_argument("--workers",            type=int, default=4,
                   help="Parallel extraction workers (default: 4)")
    p.add_argument("--no-biorxiv",         action="store_true",
                   help="Disable bioRxiv/medRxiv search even for bio topics")
    p.add_argument("--force-biorxiv",      action="store_true",
                   help="Force bioRxiv/medRxiv search even for non-bio topics")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── GPU setup ────────────────────────────────────────────────────────────
    global DEVICE
    if args.device.startswith("cuda"):
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
        DEVICE    = force_gpu(device_id)
    else:
        DEVICE = "cpu"
        console.print("  [yellow]Using CPU (pass --device cuda to use GPU)")

    console.rule("[bold green]🚀 Agentic AI Research Paper Scraper")
    console.print(f"  Prompt  : [cyan]{args.prompt}")
    console.print(f"  Fields  : [cyan]{args.fields}")
    console.print(f"  Output  : [cyan]{args.output}  ({args.format})")
    console.print(f"  Model   : [cyan]{args.model}")
    console.print(f"  Device  : [cyan]{DEVICE.upper()}")
    console.print(f"  Workers : [cyan]{args.workers}")
    console.print(f"  Max     : [cyan]{args.max_papers} papers")
    console.print()

    state: ScraperState = {
        "user_prompt":       args.prompt,
        "fields":            [f.strip() for f in args.fields.split(",")],
        "output_format":     args.format,
        "output_file":       args.output,
        "model":             args.model,
        "config": {
            "max_pages":       args.max_papers,
            "dedup_threshold": args.dedup_threshold,
            "extract_workers": args.workers,
            "no_biorxiv":      args.no_biorxiv,
            "force_biorxiv":   args.force_biorxiv,
        },
        "search_queries":    [],
        "raw_papers":        [],
        "extracted_data":    [],
        "merged_data":       [],
        "extraction_prompt": "",
        "validation_score":  1.0,
        "validation_issues": [],
        "retry_count":       0,
        "errors":            [],
    }

    graph = build_graph()
    try:
        final = graph.invoke(state)
        console.rule("[bold green]✅ Done")
        console.print(f"  Records : [bold]{len(final['merged_data'])}")
        console.print(f"  Papers  : {len(final['raw_papers'])}")
        console.print(f"  Retries : {final['retry_count']}")
        console.print(f"  Errors  : {len(final['errors'])}")
        if final["errors"]:
            console.print("  [yellow]Errors (first 5):")
            for e in final["errors"][:5]:
                console.print(f"    • {e}")
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red bold]Fatal: {e}")
        raise


if __name__ == "__main__":
    main()
