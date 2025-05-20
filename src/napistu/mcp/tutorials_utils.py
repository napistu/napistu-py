"""
Utilities for loading and processing tutorials.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import httpx

from napistu.gcs.utils import _initialize_data_dir

from napistu.mcp.constants import TUTORIAL_URLS
from napistu.mcp.constants import TUTORIALS_CACHE_DIR

# Import optional dependencies with error handling
try:
    import nbformat
except ImportError:
    raise ImportError(
        "Tutorial utilities require additional dependencies. Install with 'pip install napistu[mcp]'"
    )

async def get_tutorial_markdown(tutorial_id: str, tutorial_urls: Dict[str, str] = TUTORIAL_URLS, cache_dir: Path = TUTORIALS_CACHE_DIR) -> str:
    """
    Download/cache the notebook if needed, load it, and return the markdown.
    Args:
        tutorial_id: ID of the tutorial
        tutorial_urls: Dict of tutorial_id to GitHub raw URL
        cache_dir: Directory to cache notebooks
    Returns:
        Markdown content as a string
    """
    try:
        path = await _ensure_notebook_cached(tutorial_id, tutorial_urls, cache_dir)
        with open(path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        return notebook_to_markdown(notebook)
    except Exception as e:
        print(f"Error getting tutorial content for {tutorial_id}: {e}")
        return f"Error loading tutorial: {e}"


async def fetch_notebook_from_github(tutorial_id: str, url: str, cache_dir: Path = TUTORIALS_CACHE_DIR) -> Path:
    """
    Fetch a notebook from GitHub and cache it locally.
    Returns the path to the cached file.
    """
    # create the cache directory if it doesn't exist
    _initialize_data_dir(cache_dir)
    cache_path = _get_cached_notebook_path(tutorial_id, cache_dir)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    return cache_path


def notebook_to_markdown(notebook: nbformat.NotebookNode) -> str:
    """
    Convert a Jupyter notebook to Markdown.
    Args:
        notebook: nbformat notebook object
    Returns:
        Markdown representation of the notebook
    """
    markdown = []
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            markdown.append(cell.source)
        elif cell.cell_type == 'code':
            markdown.append("```python")
            markdown.append(cell.source)
            markdown.append("```")
            if cell.outputs:
                markdown.append("\nOutput:")
                for output in cell.outputs:
                    if 'text' in output:
                        markdown.append("```")
                        markdown.append(output['text'])
                        markdown.append("```")
                    elif 'data' in output:
                        if 'text/plain' in output['data']:
                            markdown.append("```")
                            markdown.append(output['data']['text/plain'])
                            markdown.append("```")
        markdown.append("\n---\n")
    return "\n".join(markdown)


async def _ensure_notebook_cached(tutorial_id: str, tutorial_urls: Dict[str, str] = TUTORIAL_URLS, cache_dir: Path = TUTORIALS_CACHE_DIR) -> Path:
    """
    Ensure the notebook is cached locally, fetching from GitHub if needed.
    Returns the path to the cached file.
    """
    # create the cache directory if it doesn't exist
    cache_path = _get_cached_notebook_path(tutorial_id, cache_dir)
    if not cache_path.exists():
        url = tutorial_urls[tutorial_id]
        if not url:
            raise FileNotFoundError(f"No GitHub URL found for tutorial ID: {tutorial_id}")
        await fetch_notebook_from_github(tutorial_id, url, cache_dir)
    return cache_path


def _get_cached_notebook_path(tutorial_id: str, cache_dir: Path = TUTORIALS_CACHE_DIR) -> Path:
    """
    Get the local cache path for a tutorial notebook.
    """
    return os.path.join(cache_dir, f"{tutorial_id}.ipynb")
