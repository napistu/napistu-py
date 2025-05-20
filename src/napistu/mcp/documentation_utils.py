"""
Utilities for loading and processing documentation.
"""

import httpx
import asyncio
from typing import Dict, List, Optional, Any, Set


async def load_readme_content(readme_url: str) -> str:
    if readme_url.startswith("http://") or readme_url.startswith("https://"):
        async with httpx.AsyncClient() as client:
            response = await client.get(readme_url)
            response.raise_for_status()
            return response.text
    else:
        raise ValueError(f"Only HTTP(S) URLs are supported for documentation paths: {readme_url}")


def get_snippet(text: str, query: str, context: int = 100) -> str:
    """
    Get a text snippet around a search term.
    
    Args:
        text: Text to search in
        query: Search term
        context: Number of characters to include before and after the match
    
    Returns:
        Text snippet
    """
    query = query.lower()
    text_lower = text.lower()
    
    if query not in text_lower:
        return ""
    
    start_pos = text_lower.find(query)
    start = max(0, start_pos - context)
    end = min(len(text), start_pos + len(query) + context)
    
    snippet = text[start:end]
    
    # Add ellipsis if we're not at the beginning or end
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet

