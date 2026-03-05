"""
JSON parsing helpers for noisy LLM output.
"""

import json
import re
from typing import Dict, List


def safe_json_array(text: str) -> List[Dict]:
    """Extract a JSON array from possibly noisy LLM output."""
    text = re.sub(r'```(?:json)?', '', text).strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def safe_json_obj(text: str) -> Dict:
    """Extract a JSON object from possibly noisy LLM output."""
    text = re.sub(r'```(?:json)?', '', text).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}
