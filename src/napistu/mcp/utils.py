import httpx


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


async def load_html_page(url: str) -> str:
    """
    Fetch the HTML content of a page from a URL.
    Returns the HTML as a string.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def _clean_signature_text(text: str) -> str:
    """
    Remove trailing Unicode headerlink icons and extra whitespace from text.
    """
    if text:
        return text.replace("\uf0c1", "").strip()
    return text
