#!/usr/bin/env python3
"""
main.py — Agentic AI Research Paper Scraper (CLI Entry Point)
=============================================================
Multi-agent pipeline orchestrated with LangGraph.
Searches arXiv and bioRxiv for academic papers on any topic.

Usage:
  python main.py \
    --prompt "transformer architectures for medical image segmentation" \
    --fields title,authors,abstract,journal,doi,published_date \
    --output papers.csv --format csv
"""

import argparse
import sys

from rich.console import Console

from core.state import ScraperState
from core.gpu import force_gpu, set_device
from core.graph import build_graph

console = Console()


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
  python main.py \\
      --prompt "transformer architectures for image segmentation" \\
      --fields title,authors,abstract,doi,published_date \\
      --output papers.csv --format csv

  python main.py \\
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
    if args.device.startswith("cuda"):
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
        device    = force_gpu(device_id)
    else:
        device = "cpu"
        console.print("  [yellow]Using CPU (pass --device cuda to use GPU)")

    set_device(device)

    console.rule("[bold green]🚀 Agentic AI Research Paper Scraper")
    console.print(f"  Prompt  : [cyan]{args.prompt}")
    console.print(f"  Fields  : [cyan]{args.fields}")
    console.print(f"  Output  : [cyan]{args.output}  ({args.format})")
    console.print(f"  Model   : [cyan]{args.model}")
    console.print(f"  Device  : [cyan]{device.upper()}")
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
