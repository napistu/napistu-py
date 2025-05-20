"""
Utilities for loading and processing documentation.
"""

import httpx
import asyncio


async def load_readme_content(readme_url: str) -> str:
    if readme_url.startswith("http://") or readme_url.startswith("https://"):
        async with httpx.AsyncClient() as client:
            response = await client.get(readme_url)
            response.raise_for_status()
            return response.text
    else:
        raise ValueError(f"Only HTTP(S) URLs are supported for documentation paths: {readme_url}")

