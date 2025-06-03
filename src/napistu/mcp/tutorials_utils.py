"""
Utilities for loading and processing tutorials.
"""

import os
from pathlib import Path
from typing import Dict
import httpx
import logging

from napistu.gcs.utils import _initialize_data_dir

from napistu.mcp.constants import TUTORIAL_URLS
from napistu.mcp.constants import TUTORIALS_CACHE_DIR

logger = logging.getLogger(__name__)

# Import optional dependencies with error handling
try:
    import nbformat
except ImportError:
    raise ImportError(
        "Tutorial utilities require additional dependencies. Install with 'pip install napistu[mcp]'"
    )

# Configure logger for this module
logger = logging.getLogger(__name__)


async def get_tutorial_markdown(
    tutorial_id: str,
    tutorial_urls: Dict[str, str] = TUTORIAL_URLS,
    cache_dir: Path = TUTORIALS_CACHE_DIR,
) -> str:
    """
    Download/cache the notebook if needed, load it, and return the markdown.

    Parameters
    ----------
    tutorial_id : str
        The ID of the tutorial (key in tutorial_urls).
    tutorial_urls : dict, optional
        Mapping of tutorial IDs to GitHub raw URLs. Defaults to TUTORIAL_URLS.
    cache_dir : Path, optional
        Directory to cache downloaded notebooks. Defaults to TUTORIALS_CACHE_DIR.

    Returns
    -------
    str
        Markdown content of the notebook as a string.

    Raises
    ------
    Exception
        If the notebook cannot be downloaded, loaded, or parsed.

    Examples
    --------
    >>> markdown = await get_tutorial_markdown('my_tutorial')
    >>> print(markdown)
    """
    try:
        path = await _ensure_notebook_cached(tutorial_id, tutorial_urls, cache_dir)
        logger.debug(f"Loading notebook for tutorial '{tutorial_id}' from '{path}'")
        with open(path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
        logger.debug(f"Parsing notebook for tutorial '{tutorial_id}' to markdown")
        return notebook_to_markdown(notebook)
    except Exception as e:
        logger.error(
            f"Error getting tutorial content for tutorial_id='{tutorial_id}': {e}"
        )
        raise


async def fetch_notebook_from_github(
    tutorial_id: str, url: str, cache_dir: Path = TUTORIALS_CACHE_DIR
) -> Path:
    """
    Fetch a notebook from GitHub and cache it locally.

    Parameters
    ----------
    tutorial_id : str
        The ID of the tutorial.
    url : str
        The raw GitHub URL to the notebook file.
    cache_dir : Path, optional
        Directory to cache the notebook. Defaults to TUTORIALS_CACHE_DIR.

    Returns
    -------
    Path
        Path to the cached notebook file.

    Raises
    ------
    httpx.HTTPError
        If the download fails.

    Examples
    --------
    >>> await fetch_notebook_from_github('my_tutorial', 'https://github.com/.../my_tutorial.ipynb')
    """
    _initialize_data_dir(cache_dir)
    cache_path = _get_cached_notebook_path(tutorial_id, cache_dir)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    logger.info(
        f"Downloaded and cached notebook for tutorial '{tutorial_id}' at '{cache_path}'"
    )
    return cache_path


def notebook_to_markdown(notebook: "nbformat.NotebookNode") -> str:
    """
    Convert a Jupyter notebook to Markdown.

    Parameters
    ----------
    notebook : nbformat.NotebookNode
        The loaded notebook object (as returned by nbformat.read).

    Returns
    -------
    str
        Markdown representation of the notebook, including code cells and outputs.

    Examples
    --------
    >>> import nbformat
    >>> with open('notebook.ipynb') as f:
    ...     nb = nbformat.read(f, as_version=4)
    >>> md = notebook_to_markdown(nb)
    >>> print(md)
    """
    markdown = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            markdown.append(cell.source)
        elif cell.cell_type == "code":
            markdown.append("```python")
            markdown.append(cell.source)
            markdown.append("```")
            if cell.outputs:
                markdown.append("\nOutput:")
                for output in cell.outputs:
                    if "text" in output:
                        markdown.append("```")
                        markdown.append(output["text"])
                        markdown.append("```")
                    elif "data" in output:
                        if "text/plain" in output["data"]:
                            markdown.append("```")
                            markdown.append(output["data"]["text/plain"])
                            markdown.append("```")
        markdown.append("\n---\n")
    return "\n".join(markdown)


async def _ensure_notebook_cached(
    tutorial_id: str,
    tutorial_urls: Dict[str, str] = TUTORIAL_URLS,
    cache_dir: Path = TUTORIALS_CACHE_DIR,
) -> Path:
    """
    Ensure the notebook is cached locally, fetching from GitHub if needed.

    Parameters
    ----------
    tutorial_id : str
        The ID of the tutorial.
    tutorial_urls : dict, optional
        Mapping of tutorial IDs to GitHub raw URLs. Defaults to TUTORIAL_URLS.
    cache_dir : Path, optional
        Directory to cache notebooks. Defaults to TUTORIALS_CACHE_DIR.

    Returns
    -------
    Path
        Path to the cached notebook file.

    Raises
    ------
    FileNotFoundError
        If the tutorial ID is not found in tutorial_urls.
    httpx.HTTPError
        If the download fails.

    Examples
    --------
    >>> await _ensure_notebook_cached('my_tutorial')
    """
    cache_path = _get_cached_notebook_path(tutorial_id, cache_dir)
    if not os.path.isfile(cache_path):
        url = tutorial_urls[tutorial_id]
        if not url:
            raise FileNotFoundError(
                f"No GitHub URL found for tutorial ID: {tutorial_id}"
            )
        await fetch_notebook_from_github(tutorial_id, url, cache_dir)
    return cache_path


def _get_cached_notebook_path(
    tutorial_id: str, cache_dir: Path = TUTORIALS_CACHE_DIR
) -> Path:
    """
    Get the local cache path for a tutorial notebook.

    Parameters
    ----------
    tutorial_id : str
        The ID of the tutorial.
    cache_dir : Path, optional
        Directory to cache notebooks. Defaults to TUTORIALS_CACHE_DIR.

    Returns
    -------
    Path
        Path to the cached notebook file.

    Examples
    --------
    >>> _get_cached_notebook_path('my_tutorial')
    PosixPath('.../my_tutorial.ipynb')
    """
    return Path(cache_dir) / f"{tutorial_id}.ipynb"
