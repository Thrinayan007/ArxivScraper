"""
Agent 6 — Output
Writes final records to CSV, JSON, or TXT, and displays a terminal preview.
"""

import csv
import json

from rich.console import Console
from rich.table import Table

from core.state import ScraperState

console = Console()


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
    all_keys    = list(
        {k for item in data for k in item.keys() if not k.startswith("_")}
    )
    ordered     = [f for f in user_fields if f in all_keys]
    ordered    += [k for k in sorted(all_keys) if k not in ordered]

    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=ordered, extrasaction="ignore"
            )
            writer.writeheader()
            for item in data:
                row = {k: v for k, v in item.items() if not k.startswith("_")}
                writer.writerow(row)

    elif fmt == "json":
        clean = [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in data
        ]
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
        f for f in user_fields if f in all_keys
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
