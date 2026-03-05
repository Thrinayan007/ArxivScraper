"""
Shared pipeline state definition.
All agents read from and write to this TypedDict.
"""

from typing import Any, Dict, List, TypedDict


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
