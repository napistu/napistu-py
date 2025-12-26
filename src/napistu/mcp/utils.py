"""Utilities for the MCP server"""

import asyncio
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


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


async def load_html_page(
    url: str, max_retries: int = 5, base_delay: float = 2.0, initial_delay: float = 1.0
) -> str:
    """
    Fetch the HTML content of a page from a URL with retry logic for rate limiting.

    Parameters
    ----------
    url : str
        The URL to fetch
    max_retries : int, optional
        Maximum number of retry attempts (default: 5)
    base_delay : float, optional
        Base delay in seconds for exponential backoff (default: 2.0)
    initial_delay : float, optional
        Initial delay before first request in seconds (default: 1.0)

    Returns
    -------
    str
        The HTML content as a string

    Raises
    ------
    httpx.HTTPStatusError
        If the request fails after all retries
    httpx.RequestError
        If a network error occurs after all retries
    """
    # Add initial delay before first request to avoid immediate rate limiting
    if initial_delay > 0:
        await asyncio.sleep(initial_delay)

    headers = {
        "User-Agent": "Napistu-MCP-Server/1.0 (https://github.com/napistu/napistu)"
    }

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.text

            except httpx.HTTPStatusError as e:
                # Handle rate limiting (429) with exponential backoff
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        # Calculate exponential backoff delay
                        delay = base_delay * (2**attempt)

                        # Check for Retry-After header
                        retry_after_header = e.response.headers.get("Retry-After")
                        if retry_after_header:
                            parsed_delay = _parse_retry_after(retry_after_header)
                            if parsed_delay is not None:
                                delay = max(parsed_delay, base_delay)
                                logger.info(
                                    f"Retry-After header specifies {delay:.1f} seconds"
                                )
                            else:
                                logger.warning(
                                    f"Could not parse Retry-After header '{retry_after_header}'. "
                                    f"Using exponential backoff."
                                )

                        logger.warning(
                            f"Rate limited (429) for {url}. "
                            f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Final attempt failed
                        retry_after_header = e.response.headers.get("Retry-After")
                        logger.error(
                            f"Rate limited (429) for {url} after {max_retries} attempts. "
                            f"Retry-After header: {retry_after_header}"
                        )
                        raise
                else:
                    # For other HTTP errors, raise immediately
                    raise

            except httpx.RequestError as e:
                # For network errors, retry with exponential backoff
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Network error for {url}: {e}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Network error for {url} after {max_retries} attempts: {e}"
                    )
                    raise

    # This should never be reached, but included for type safety
    raise httpx.RequestError("Failed to fetch URL after all retries")


# private functions


def _clean_signature_text(text: str) -> str:
    """
    Remove trailing Unicode headerlink icons and extra whitespace from text.
    """
    if text:
        return text.replace("\uf0c1", "").strip()
    return text


def _parse_retry_after(retry_after: str) -> Optional[float]:
    """
    Parse Retry-After header value.

    The Retry-After header can be either:
    - An integer (seconds to wait)
    - An HTTP date (RFC 7231 format)

    Parameters
    ----------
    retry_after : str
        The Retry-After header value

    Returns
    -------
    Optional[float]
        Number of seconds to wait, or None if parsing failed
    """
    if not retry_after:
        return None

    # Try parsing as integer (seconds)
    try:
        return float(retry_after)
    except ValueError:
        pass

    # Try parsing as HTTP date
    try:
        retry_date = parsedate_to_datetime(retry_after)
        # Ensure retry_date is timezone-aware
        if retry_date.tzinfo is None:
            retry_date = retry_date.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delay = (retry_date - now).total_seconds()
        return max(delay, 0.0)  # Don't return negative delays
    except (ValueError, TypeError, OverflowError):
        return None
