import httpx

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
