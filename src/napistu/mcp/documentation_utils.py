"""
Utilities for loading and processing documentation.
"""

import os
from typing import Any, Dict

import httpx

from napistu.constants import PACKAGE_DEFS
from napistu.mcp import utils as mcp_utils
from napistu.mcp.constants import (
    DEFAULT_GITHUB_API,
    DOCUMENTATION,
    GITHUB_DEFS,
    GITHUB_ISSUES_INDEXED,
    GITHUB_PRS_INDEXED,
    SEARCH_RESULT_DEFS,
    SEARCH_TYPES,
)


async def fetch_wiki_page(
    page_name: str,
    repo: str = PACKAGE_DEFS.GITHUB_PROJECT_REPO,
    owner: str = PACKAGE_DEFS.GITHUB_OWNER,
) -> str:
    """
    Fetch wiki page content using raw GitHub URLs.

    Parameters
    ----------
    page_name : str
        The name of the page (without .md extension for wiki pages)
    repo : str
        The repository name
    owner : str
        The GitHub username or organization

    Returns
    -------
    str
        The raw Markdown content
    """
    # Use raw.githubusercontent.com for wiki pages
    url = f"https://raw.githubusercontent.com/wiki/{owner}/{repo}/{page_name}.md"

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


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
        else (lambda item: GITHUB_DEFS.PULL_REQUEST not in item)
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


# private utils


def _exact_search_documentation(
    query: str, docs_cache: Dict[str, Dict[str, Any]], max_exact_results: int = 20
) -> Dict[str, Any]:
    """
    Perform exact text search across documentation content.

    Searches for the query string in README files, wiki pages, GitHub issues,
    and pull requests. Returns results organized by type, or an error if
    too many matches are found.

    Parameters
    ----------
    query : str
        Search query string to match against documentation content
    docs_cache : Dict[str, Dict[str, Any]]
        Documentation cache dictionary with keys:
        - DOCUMENTATION.README: Dictionary of README content
        - DOCUMENTATION.WIKI: Dictionary of wiki page content
        - DOCUMENTATION.ISSUES: Dictionary mapping repos to lists of issues
        - DOCUMENTATION.PRS: Dictionary mapping repos to lists of PRs
    max_exact_results : int, optional
        Maximum number of total results allowed before returning an error.
        Default is 20.

    Returns
    -------
    Dict[str, Any]
        Search results dictionary containing:
        - query : str
            Original search query
        - search_type : str
            Always "exact"
        - results : Dict[str, List]
            Dictionary with keys README, WIKI, ISSUES, PRS, each containing
            a list of matching items with name, snippet, title, and url
        - tip : str
            Helpful guidance for improving search results
        OR (if too many results):
        - error : str
            Error message indicating too many results
        - suggestion : str
            Suggestions for refining the search
        - result_counts : Dict[str, int]
            Counts of matches by type
    """
    results = {
        DOCUMENTATION.README: [],
        DOCUMENTATION.WIKI: [],
        DOCUMENTATION.ISSUES: [],
        DOCUMENTATION.PRS: [],
    }

    # Search README files
    for readme_name, content in docs_cache[DOCUMENTATION.README].items():
        if query.lower() in content.lower():
            results[DOCUMENTATION.README].append(
                {
                    SEARCH_RESULT_DEFS.NAME: readme_name,
                    SEARCH_RESULT_DEFS.SNIPPET: mcp_utils.get_snippet(content, query),
                }
            )

    # Search wiki pages
    for page_name, content in docs_cache[DOCUMENTATION.WIKI].items():
        if query.lower() in content.lower():
            results[DOCUMENTATION.WIKI].append(
                {
                    SEARCH_RESULT_DEFS.NAME: page_name,
                    SEARCH_RESULT_DEFS.SNIPPET: mcp_utils.get_snippet(content, query),
                }
            )

    # Search issues
    for repo, issues in docs_cache[DOCUMENTATION.ISSUES].items():
        for issue in issues:
            issue_text = (
                f"{issue.get(GITHUB_DEFS.TITLE, '')} {issue.get(GITHUB_DEFS.BODY, '')}"
            )
            if query.lower() in issue_text.lower():
                results[DOCUMENTATION.ISSUES].append(
                    {
                        SEARCH_RESULT_DEFS.NAME: f"{repo}#{issue.get(GITHUB_DEFS.NUMBER)}",
                        SEARCH_RESULT_DEFS.TITLE: issue.get(GITHUB_DEFS.TITLE),
                        SEARCH_RESULT_DEFS.URL: issue.get(GITHUB_DEFS.URL),
                        SEARCH_RESULT_DEFS.SNIPPET: mcp_utils.get_snippet(
                            issue_text, query
                        ),
                    }
                )

    # Search PRs
    for repo, prs in docs_cache[DOCUMENTATION.PRS].items():
        for pr in prs:
            pr_text = f"{pr.get(GITHUB_DEFS.TITLE, '')} {pr.get(GITHUB_DEFS.BODY, '')}"
            if query.lower() in pr_text.lower():
                results[DOCUMENTATION.PRS].append(
                    {
                        SEARCH_RESULT_DEFS.NAME: f"{repo}#{pr.get(GITHUB_DEFS.NUMBER)}",
                        SEARCH_RESULT_DEFS.TITLE: pr.get(GITHUB_DEFS.TITLE),
                        SEARCH_RESULT_DEFS.URL: pr.get(GITHUB_DEFS.URL),
                        SEARCH_RESULT_DEFS.SNIPPET: mcp_utils.get_snippet(
                            pr_text, query
                        ),
                    }
                )

    # Count total results
    total_results = (
        len(results[DOCUMENTATION.README])
        + len(results[DOCUMENTATION.WIKI])
        + len(results[DOCUMENTATION.ISSUES])
        + len(results[DOCUMENTATION.PRS])
    )

    # If too many results, return error suggesting semantic search or more precise query
    if total_results > max_exact_results:
        return {
            SEARCH_RESULT_DEFS.QUERY: query,
            SEARCH_RESULT_DEFS.SEARCH_TYPE: SEARCH_TYPES.EXACT,
            "error": f"Too many results found ({total_results} matches). Exact search returned too many matches.",
            "suggestion": (
                f"Try one of the following:\n"
                f"1. Use search_type='semantic' for better relevance ranking and fewer, more relevant results\n"
                f"2. Use a more specific query (e.g., '{query} installation' or '{query} troubleshooting')\n"
                f"3. Search specific content types: filter by README, wiki, issues, or PRs"
            ),
            "result_counts": {
                DOCUMENTATION.README: len(results[DOCUMENTATION.README]),
                DOCUMENTATION.WIKI: len(results[DOCUMENTATION.WIKI]),
                DOCUMENTATION.ISSUES: len(results[DOCUMENTATION.ISSUES]),
                DOCUMENTATION.PRS: len(results[DOCUMENTATION.PRS]),
            },
        }

    return {
        SEARCH_RESULT_DEFS.QUERY: query,
        SEARCH_RESULT_DEFS.SEARCH_TYPE: SEARCH_TYPES.EXACT,
        SEARCH_RESULT_DEFS.RESULTS: results,
        SEARCH_RESULT_DEFS.TIP: "Use search_type='semantic' for natural language queries",
    }


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
        GITHUB_DEFS.NUMBER: item[GITHUB_DEFS.NUMBER],
        GITHUB_DEFS.TITLE: item[GITHUB_DEFS.TITLE],
        GITHUB_DEFS.STATE: item[GITHUB_DEFS.STATE],
        GITHUB_DEFS.URL: item[GITHUB_DEFS.HTML_URL],
        GITHUB_DEFS.BODY: (
            (item[GITHUB_DEFS.BODY][:500] + "...")
            if item.get(GITHUB_DEFS.BODY) and len(item[GITHUB_DEFS.BODY]) > 500
            else item.get(GITHUB_DEFS.BODY)
        ),
        GITHUB_DEFS.IS_PR: GITHUB_DEFS.PULL_REQUEST in item
        or GITHUB_DEFS.MERGED_AT in item,
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
