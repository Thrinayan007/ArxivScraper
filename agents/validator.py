"""
Agent 4 — Validator
Scores extraction quality based on field coverage.
"""

from typing import List

from rich.console import Console

from core.state import ScraperState

console = Console()


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
