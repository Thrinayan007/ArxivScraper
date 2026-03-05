"""
bioRxiv and medRxiv API search — date-range based with client-side keyword
filtering.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List

import requests
from rich.console import Console

from core.constants import BIORXIV_API, MEDRXIV_API, extract_keywords

console = Console()


def biorxiv_search(
    query: str, max_results: int = 80, include_medrxiv: bool = True
) -> List[Dict]:
    """
    Search bioRxiv (and optionally medRxiv) using the content-detail API.
    The bioRxiv API is date-range based, so we fetch recent papers and
    filter client-side by keyword matching.
    """
    papers: List[Dict] = []

    # Build date ranges — try multiple windows for better coverage
    now = datetime.now()
    date_ranges = [
        # Last 30 days first (most recent)
        (
            (now - timedelta(days=30)).strftime("%Y-%m-%d"),
            now.strftime("%Y-%m-%d"),
        ),
        # 30-90 days ago
        (
            (now - timedelta(days=90)).strftime("%Y-%m-%d"),
            (now - timedelta(days=30)).strftime("%Y-%m-%d"),
        ),
        # 90-180 days ago (extended range for more coverage)
        (
            (now - timedelta(days=180)).strftime("%Y-%m-%d"),
            (now - timedelta(days=90)).strftime("%Y-%m-%d"),
        ),
    ]

    apis = [("bioRxiv", BIORXIV_API)]
    if include_medrxiv:
        apis.append(("medRxiv", MEDRXIV_API))

    # Use extracted keywords, not raw words
    query_keywords = extract_keywords(query)
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
            max_pages = 8  # Limit API pages per date range

            while pages_fetched < max_pages:
                url = f"{api_base}/{start_date}/{end_date}/{cursor}"
                try:
                    resp = requests.get(url, timeout=30, headers={
                        "User-Agent": "ResearchScraper/1.0 (academic research tool)"
                    })

                    if resp.status_code == 404:
                        console.print(
                            f"    [dim]{source_name} no data for "
                            f"{start_date} to {end_date}"
                        )
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
                        matches = sum(
                            1 for w in query_keywords if w in combined
                        )
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
                                "pdf_url": (
                                    f"https://www.biorxiv.org/content/"
                                    f"{doi}v{version}.full.pdf"
                                    if doi else ""
                                ),
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
                        f"    {source_name} {start_date}..{end_date} "
                        f"page {pages_fetched}: "
                        f"{len(collection)} scanned, {batch_matches} matched"
                    )

                    if len(papers) >= max_results:
                        break

                    # Respect rate limits
                    time.sleep(1.5)

                except Exception as e:
                    console.print(
                        f"    [yellow]{source_name} API error: {str(e)[:80]}"
                    )
                    break

        console.print(f"  {source_name} total matched: [bold]{source_count}")

    return papers[:max_results]
