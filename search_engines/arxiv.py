"""
arXiv API search — multiple query strategies + pagination.
"""

import time
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests
from rich.console import Console

from core.constants import ARXIV_API, extract_keywords

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Query builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_arxiv_queries(query: str) -> List[str]:
    """
    Build multiple arXiv API query strings from a natural-language query.
    Returns several strategies: AND (strict), OR (broad), title-only.
    """
    keywords = extract_keywords(query)
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


# ─────────────────────────────────────────────────────────────────────────────
# XML parser
# ─────────────────────────────────────────────────────────────────────────────

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
                href   = link.get("href", "")
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
# Fetch + search
# ─────────────────────────────────────────────────────────────────────────────

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
