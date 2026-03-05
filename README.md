# 🤖 Agentic AI Research Paper Scraper

A multi-agent pipeline built with **LangGraph** that autonomously searches, extracts, validates, and exports structured data from academic papers across **arXiv** and **bioRxiv/medRxiv**.

---

## ✨ Features

- **Multi-Agent Architecture** — 6 specialized agents orchestrated via LangGraph
- **Dual Source Search** — arXiv (Atom API) + bioRxiv/medRxiv (REST API) with multi-strategy queries
- **LLM-Powered Extraction** — Structured field extraction using local Ollama models (Mistral, LLaMA, etc.)
- **Smart Fallback** — Automatic metadata-only extraction when Ollama is unavailable
- **Self-Healing Pipeline** — Validator agent triggers prompt re-engineering on low-quality extractions
- **Semantic Deduplication** — Cosine similarity with sentence-transformers (falls back to exact dedup)
- **GPU Acceleration** — Auto-detects CUDA GPUs for faster LLM inference and embedding
- **Multi-Format Output** — CSV, JSON, or TXT with terminal preview

---

## 📁 Project Structure

```
agentic-scraper/
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .gitignore
├── README.md
│
├── agents/                    # One file per agent
│   ├── __init__.py
│   ├── planner.py             # Agent 1 — Generates search queries via LLM
│   ├── search.py              # Agent 2 — Orchestrates arXiv + bioRxiv search
│   ├── extractor.py           # Agent 3 — LLM or metadata extraction
│   ├── validator.py           # Agent 4 — Scores extraction quality
│   ├── prompt_engineer.py     # Agent 4b — Rewrites prompts on low scores
│   ├── merger.py              # Agent 5 — Normalizes + deduplicates records
│   └── output.py              # Agent 6 — Writes CSV/JSON/TXT output
│
├── core/                      # Shared infrastructure
│   ├── __init__.py
│   ├── state.py               # ScraperState TypedDict (shared state)
│   ├── gpu.py                 # GPU detection + Ollama LLM factory
│   ├── constants.py           # API URLs, stop words, bio keywords
│   ├── helpers.py             # JSON parsing utilities
│   └── graph.py               # LangGraph pipeline builder + router
│
└── search_engines/            # API wrappers
    ├── __init__.py
    ├── arxiv.py               # arXiv Atom API (multi-strategy + pagination)
    └── biorxiv.py             # bioRxiv / medRxiv REST API
```

---

## 🏗️ Agent Pipeline

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌───────────┐
│ Planner  │───▶│  Search  │───▶│ Extractor │───▶│ Validator │
└──────────┘    └──────────┘    └───────────┘    └─────┬─────┘
                                      ▲                │
                                      │          Score < 0.5?
                                      │           ┌────┴────┐
                                      │     Yes   │         │  No
                                      │   ┌───────▼──────┐  │
                                      └───│   Prompt     │  │
                                          │  Engineer    │  │
                                          └──────────────┘  │
                                                            ▼
                                                     ┌──────────┐    ┌────────┐
                                                     │  Merger  │───▶│ Output │
                                                     └──────────┘    └────────┘
```

| Agent | Role |
|-------|------|
| **Planner** | Generates 5 diverse search queries from the user prompt using the LLM |
| **Search** | Runs multi-strategy arXiv queries + bioRxiv/medRxiv keyword searches |
| **Extractor** | Extracts structured fields via Ollama LLM (or metadata fallback) |
| **Validator** | Computes field coverage score; triggers re-extraction if < 0.5 |
| **Prompt Engineer** | Rewrites the extraction prompt to improve quality (max 2 retries) |
| **Merger** | Normalizes keys + deduplicates using sentence-transformer embeddings |
| **Output** | Writes CSV / JSON / TXT and shows a Rich terminal preview table |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/agentic-scraper.git
cd agentic-scraper
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install & start Ollama (optional, for LLM extraction)

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2      # or mistral, gemma2, etc.
ollama serve              # starts on port 11434
```

> **Note:** The scraper works without Ollama — it will fall back to direct metadata extraction from the APIs (faster, no GPU needed).

### 4. Run the scraper

```bash
python main.py \
  --prompt "transformer architectures for medical image segmentation" \
  --fields title,authors,abstract,doi,published_date \
  --output papers.csv --format csv
```

---

## ⚙️ CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | *required* | Natural language research query |
| `--fields` | *required* | Comma-separated fields to extract |
| `--output` | `output.csv` | Output file path |
| `--format` | `csv` | Output format: `csv`, `json`, `txt` |
| `--max-papers` | `150` | Maximum papers to fetch |
| `--model` | `llama3.2` | Ollama model name |
| `--device` | `cuda` | `cuda`, `cuda:1`, or `cpu` |
| `--workers` | `4` | Parallel extraction threads |
| `--dedup-threshold` | `0.92` | Cosine similarity threshold for dedup |
| `--no-biorxiv` | `false` | Disable bioRxiv/medRxiv search |
| `--force-biorxiv` | `false` | Force bioRxiv even for non-bio topics |

---

## 📋 Examples

### Search for AI papers (arXiv focused)

```bash
python main.py \
  --prompt "large language models for code generation" \
  --fields title,authors,abstract,doi,published_date,categories \
  --output llm_code.csv --format csv
```

### Search for biology papers (arXiv + bioRxiv)

```bash
python main.py \
  --prompt "CRISPR gene editing delivery methods 2024" \
  --fields title,authors,journal,doi,published_date \
  --output gene_editing.json --format json --device cuda
```

### CPU-only with metadata extraction (no Ollama needed)

```bash
python main.py \
  --prompt "quantum computing error correction" \
  --fields title,authors,abstract,doi \
  --output quantum.csv --device cpu
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Pipeline Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM Inference | [Ollama](https://ollama.ai) (local) via LangChain |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) |
| Terminal UI | [Rich](https://github.com/Textualize/rich) |
| APIs | arXiv Atom API, bioRxiv/medRxiv REST API |

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
