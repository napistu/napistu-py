"""
Utilities for loading and processing documentation.
"""

import base64
import httpx
import os
from napistu.constants import PACKAGE_DEFS
from napistu.mcp.constants import (
    DEFAULT_GITHUB_API,
    GITHUB_ISSUES_INDEXED,
    GITHUB_PRS_INDEXED,
)


async def load_readme_content(readme_url: str) -> str:
    if readme_url.startswith("http://") or readme_url.startswith("https://"):
        async with httpx.AsyncClient() as client:
            response = await client.get(readme_url)
            response.raise_for_status()
            return response.text
    else:
        raise ValueError(
            f"Only HTTP(S) URLs are supported for documentation paths: {readme_url}"
        )


async def list_wiki_pages(
    repo: str,
    owner: str = PACKAGE_DEFS.GITHUB_OWNER,
    github_api: str = DEFAULT_GITHUB_API,
) -> list:
    """
    List all Markdown (.md) page names in a GitHub wiki using the GitHub API.

    Parameters
    ----------
    repo : str, optional
        The repository name.
    owner : str, optional
        The GitHub username or organization (default is 'napistu').
    github_api : str, optional
        The GitHub API base URL (default is 'https://api.github.com').

    Returns
    -------
    list of str
        List of page names (e.g., ['Home.md', 'Some-Page.md'])
    """
    wiki_repo = f"{repo}.wiki"
    url = f"{github_api}/repos/{owner}/{wiki_repo}/contents/"
    async with httpx.AsyncClient(headers=_get_github_headers()) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return [item["name"] for item in resp.json() if item["name"].endswith(".md")]


async def fetch_wiki_page(
    page_name: str,
    repo: str,
    owner: str = PACKAGE_DEFS.GITHUB_OWNER,
    github_api: str = DEFAULT_GITHUB_API,
) -> str:
    """
    Fetch the decoded Markdown content of a given wiki page from a GitHub wiki.

    Parameters
    ----------
    page_name : str
        The name of the page file (e.g., 'Home.md').
    repo : str, optional
        The repository name.
    owner : str, optional
        The GitHub username or organization (default is 'napistu').
    github_api : str, optional
        The GitHub API base URL (default is 'https://api.github.com').

    Returns
    -------
    str
        The Markdown content as a string.
    """
    wiki_repo = f"{repo}.wiki"
    url = f"{github_api}/repos/{owner}/{wiki_repo}/contents/{page_name}"
    async with httpx.AsyncClient(headers=_get_github_headers()) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content


async def list_issues(
    repo: str,
    owner: str = PACKAGE_DEFS.GITHUB_OWNER,
    github_api: str = DEFAULT_GITHUB_API,
    state: str = GITHUB_ISSUES_INDEXED,
    include_prs: bool = False,
) -> list:
    """
    List issues (and optionally PRs) for a given GitHub repository using the GitHub API.

    Parameters
    ----------
    repo : str, optional
        The repository name.
    owner : str, optional
        The GitHub username or organization (default is 'napistu').
    github_api : str, optional
        The GitHub API base URL (default is 'https://api.github.com').
    state : str, optional
        The state of the issues to return. Can be 'open', 'closed', or 'all'. Default is 'open'.
    include_prs : bool, optional
        If True, include pull requests in the results. Default is False.

    Returns
    -------
    list of dict
        Each dict contains: number, title, state, url, and a truncated body (max 500 chars).
    """
    url = f"{github_api}/repos/{owner}/{repo}/issues?state={state}"
    filter_func = (
        (lambda item: True)
        if include_prs
        else (lambda item: "pull_request" not in item)
    )
    return await _fetch_github_items(url, filter_func=filter_func)


async def list_pull_requests(
    repo: str,
    owner: str = PACKAGE_DEFS.GITHUB_OWNER,
    github_api: str = DEFAULT_GITHUB_API,
    state: str = GITHUB_PRS_INDEXED,
) -> list:
    """
    List pull requests for a given GitHub repository using the GitHub API.

    Parameters
    ----------
    repo : str, optional
        The repository name.
    owner : str, optional
        The GitHub username or organization (default is 'napistu').
    github_api : str, optional
        The GitHub API base URL (default is 'https://api.github.com').
    state : str, optional
        The state of the PRs to return. Can be 'open', 'closed', or 'all'. Default is 'open'.

    Returns
    -------
    list of dict
        Each dict contains: number, title, state, url, and a truncated body (max 500 chars).
    """
    url = f"{github_api}/repos/{owner}/{repo}/pulls?state={state}"
    return await _fetch_github_items(url)


async def get_issue(
    repo: str,
    number: int,
    owner: str = PACKAGE_DEFS.GITHUB_OWNER,
    github_api: str = DEFAULT_GITHUB_API,
) -> dict:
    """
    Get a single issue (or PR) by number from a GitHub repository.

    Parameters
    ----------
    repo : str
        The repository name.
    number : int
        The issue or PR number.
    owner : str, optional
        The GitHub username or organization (default is 'napistu').
    github_api : str, optional
        The GitHub API base URL (default is 'https://api.github.com').

    Returns
    -------
    dict
        The issue or PR details as a dictionary.
    """
    url = f"{github_api}/repos/{owner}/{repo}/issues/{number}"
    async with httpx.AsyncClient(headers=_get_github_headers()) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        item = resp.json()
        return _format_github_issue(item)


def _get_github_headers():
    """
    Return headers for GitHub API requests, including Authorization if GITHUB_TOKEN is set.
    """
    headers = {}

    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    if GITHUB_TOKEN:
        print("Using token from environment variable GITHUB_TOKEN")
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def _format_github_issue(item):
    """
    Format a GitHub issue or PR item into a standard dict.
    """
    return {
        "number": item["number"],
        "title": item["title"],
        "state": item["state"],
        "url": item["html_url"],
        "body": (
            (item["body"][:500] + "...")
            if item.get("body") and len(item["body"]) > 500
            else item.get("body")
        ),
        "is_pr": "pull_request" in item or "merged_at" in item,
    }


async def _fetch_github_items(url, filter_func=None):
    """
    Fetch and format a list of GitHub issues or PRs from a given API endpoint.
    """
    async with httpx.AsyncClient(headers=_get_github_headers()) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        items = []
        for item in resp.json():
            if filter_func and not filter_func(item):
                continue
            items.append(_format_github_issue(item))
        return items
