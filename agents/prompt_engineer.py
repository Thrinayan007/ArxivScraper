"""
Agent 4b — Prompt Engineer
Rewrites the extraction prompt when validation scores are low.
"""

import json

from langchain_core.prompts import PromptTemplate
from rich.console import Console

from core.state import ScraperState
from core.gpu import get_llm

console = Console()


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
