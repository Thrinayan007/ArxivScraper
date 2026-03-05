"""
Agent 5 — Merger
Normalizes keys and performs semantic (or exact) deduplication.
"""

from typing import Dict, List

from rich.console import Console

from core.state import ScraperState
from core.gpu import get_device

console = Console()


def _semantic_dedup(
    data: List[Dict], fields: List[str], threshold: float = 0.92
) -> List[Dict]:
    """Deduplicate records using sentence embeddings or exact match fallback."""
    if len(data) <= 1:
        return data
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        device = get_device()
        console.print(f"  Semantic dedup on: [cyan]{device.upper()}")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        norm_f = [f.lower().replace(" ", "_") for f in fields[:3]]
        fingerprints = [
            " ".join(str(item.get(f, "")) for f in norm_f) or str(item)
            for item in data
        ]
        embeddings = model.encode(
            fingerprints,
            batch_size=128 if device != "cpu" else 32,
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
        console.print(
            f"  [yellow]Semantic dedup unavailable ({e}) — using exact dedup"
        )
        norm_f = [f.lower().replace(" ", "_") for f in fields[:3]]
        seen: set = set()
        result: List[Dict] = []
        for item in data:
            key = tuple(item.get(f, "") for f in norm_f)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


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
