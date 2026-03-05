"""
API endpoints, stop words, and bio-science keyword sets.
"""

import re
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# API Constants
# ─────────────────────────────────────────────────────────────────────────────

ARXIV_API   = "http://export.arxiv.org/api/query"
BIORXIV_API = "https://api.biorxiv.org/details/biorxiv"
MEDRXIV_API = "https://api.biorxiv.org/details/medrxiv"

# ─────────────────────────────────────────────────────────────────────────────
# Stop words to filter out when building arXiv queries
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# bioRxiv subject areas that signal life-science queries
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def is_bio_topic(text: str) -> bool:
    """Check if a query is related to biology / life sciences."""
    lower = text.lower()
    return any(kw in lower for kw in BIO_KEYWORDS)


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from user prompt, filtering stop words."""
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return keywords
