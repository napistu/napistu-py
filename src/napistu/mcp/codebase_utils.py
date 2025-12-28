"""
Utilities for scanning and analyzing the Napistu codebase.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Set

from napistu.mcp import utils as mcp_utils
from napistu.mcp.constants import (
    CODEBASE_DEFS,
    CODEBASE_INSPECT_DEFS,
    READTHEDOCS_TOC_CSS_SELECTOR,
    SEARCH_RESULT_DEFS,
    SEARCH_TYPES,
)

logger = logging.getLogger(__name__)

# Import optional dependencies with error handling
try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError(
        "Documentation utilities require additional dependencies. Install with 'pip install napistu[mcp]'"
    )


def add_stripped_names(functions: dict, classes: dict) -> None:
    """
    Add stripped names to all functions and classes for easier lookup.

    This function modifies the input dictionaries in-place by adding a 'stripped_name'
    attribute to each item. The stripped name is the last part of the fully qualified
    name (e.g., "NapistuGraph" from "napistu.network.ng_core.NapistuGraph").

    Parameters
    ----------
    functions : dict
        Dictionary of functions keyed by fully qualified name
    classes : dict
        Dictionary of classes keyed by fully qualified name

    Examples
    --------
    >>> functions = {"napistu.network.create_network": {...}}
    >>> classes = {"napistu.network.ng_core.NapistuGraph": {...}}
    >>> add_stripped_names(functions, classes)
    >>> print(functions["napistu.network.create_network"]["stripped_name"])
    'create_network'
    >>> print(classes["napistu.network.ng_core.NapistuGraph"]["stripped_name"])
    'NapistuGraph'
    """
    for item_type, items_dict in [
        (CODEBASE_DEFS.FUNCTIONS, functions),
        (CODEBASE_DEFS.CLASSES, classes),
    ]:
        for full_name, item_info in items_dict.items():
            # Extract the last part of the full name
            stripped_name = full_name.split(".")[-1]
            item_info["stripped_name"] = stripped_name


def extract_functions_and_classes_from_modules(modules: dict) -> tuple[dict, dict]:
    """
    Process the modules cache and return a tuple (functions_dict, classes_dict),
    where each is a dict keyed by fully qualified name (e.g., 'module.func', 'module.Class').
    Recursively processes submodules.

    Args:
        modules (dict): The modules cache as returned by read_read_the_docs.

    Returns:
        tuple: (functions_dict, classes_dict)
    """
    functions = {}
    classes = {}

    def _process_module(module_name: str, module_info: dict):
        # Functions
        for func_name, func_info in module_info.get(
            CODEBASE_DEFS.FUNCTIONS, {}
        ).items():
            fq_name = f"{module_name}.{func_name}"
            functions[fq_name] = func_info
        # Classes
        for class_name, class_info in module_info.get(
            CODEBASE_DEFS.CLASSES, {}
        ).items():
            fq_name = f"{module_name}.{class_name}"
            classes[fq_name] = class_info
        # Submodules (if present in the cache)
        for submod_name in module_info.get("submodules", {}):
            fq_submod_name = f"{module_name}.{submod_name}"
            if fq_submod_name in modules:
                _process_module(fq_submod_name, modules[fq_submod_name])

    for module_name, module_info in modules.items():
        _process_module(module_name, module_info)

    return functions, classes


def find_item_by_name(name: str, items_dict: dict) -> tuple[str, dict] | None:
    """
    Find an item by name using exact matching on both full path and stripped name.

    Parameters
    ----------
    name : str
        Name to search for (can be short name or full path)
    items_dict : dict
        Dictionary of items keyed by fully qualified name

    Returns
    -------
    tuple[str, dict] | None
        Tuple of (full_name, item_info) if found, None otherwise

    Examples
    --------
    >>> functions = {"napistu.network.create_network": {"stripped_name": "create_network", ...}}
    >>> result = find_item_by_name("create_network", functions)
    >>> if result:
    ...     full_name, func_info = result
    ...     print(f"Found: {full_name}")
    'Found: napistu.network.create_network'
    """
    # First try exact match on full path
    if name in items_dict:
        return name, items_dict[name]

    # Try exact match on stripped name
    for full_name, item_info in items_dict.items():
        if item_info.get("stripped_name") == name:
            return full_name, item_info

    return None


async def read_read_the_docs(package_toc_url: str, request_delay: float = 0.5) -> dict:
    """
    Recursively parse all modules and submodules starting from the package TOC.

    Parameters
    ----------
    package_toc_url : str
        URL of the ReadTheDocs package table of contents page
    request_delay : float, optional
        Delay in seconds between requests to avoid rate limiting (default: 0.5)

    Returns
    -------
    dict
        Dictionary of parsed module documentation

    Raises
    ------
    httpx.HTTPStatusError
        If any page fails to load after retries (fail-fast behavior)
    httpx.RequestError
        If any network error occurs after retries
    """
    # Step 1: Get all module URLs from the TOC
    # If this fails after retries, exception will propagate (fail-fast)
    packages_dict = await _process_rtd_package_toc(package_toc_url)
    docs_dict = {}
    visited = set()

    # Step 2: Recursively parse each module page
    # If any page fails after retries, exception will propagate (fail-fast)
    for package_name, module_url in packages_dict.items():
        if not module_url.startswith("http"):
            # Make absolute if needed
            base = package_toc_url.rsplit("/", 1)[0]
            module_url = base + "/" + module_url.lstrip("/")
        await _parse_rtd_module_recursive(
            module_url, visited, docs_dict, request_delay=request_delay
        )

    return docs_dict


# private utils


def _extract_module_docstring(soup, h1):
    """
    Extract module-level docstring from ReadTheDocs HTML.

    Module docstrings are typically in <p> tags between <h1> and the first
    rubric/class/function definition. This function finds all such paragraphs
    and combines them into a single docstring.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML soup of the module page
    h1 : Tag
        The <h1> tag containing the module name

    Returns
    -------
    str or None
        The extracted module docstring, or None if no docstring found
    """
    # Find stop points (rubric, class, or function definitions)
    first_rubric = soup.select_one("p.rubric")
    first_function = soup.select_one("dl.py.function")
    first_class = soup.select_one("dl.py.class")

    # Get all <p> tags in the document
    all_paragraphs = soup.find_all("p")

    # Find paragraphs that come after h1 but before stop points
    h1_index = _get_element_index(h1, soup)
    stop_indices = []
    if first_rubric:
        stop_indices.append(_get_element_index(first_rubric, soup))
    if first_function:
        stop_indices.append(_get_element_index(first_function, soup))
    if first_class:
        stop_indices.append(_get_element_index(first_class, soup))

    min_stop_index = min(stop_indices) if stop_indices else float("inf")

    # Collect paragraphs between h1 and the first stop point
    doc_paragraphs = []
    for p in all_paragraphs:
        p_index = _get_element_index(p, soup)
        # Must be after h1 and before any stop point
        if h1_index < p_index < min_stop_index:
            # Skip rubric paragraphs
            if "rubric" not in p.get("class", []):
                text = p.get_text(" ", strip=True)
                if text:
                    doc_paragraphs.append(text)

    if doc_paragraphs:
        return " ".join(doc_paragraphs).strip()
    return None


def _format_attribute(attr_dl) -> Dict[str, Any]:
    """
    Format a class attribute's signature and documentation into a dictionary.

    Args:
        attr_dl: The <dl> tag for the attribute, containing <dt> and <dd>.

    Returns:
        dict: A dictionary with keys 'name', 'signature', 'id', and 'doc'.
    """
    sig = attr_dl.find("dt")
    doc = attr_dl.find("dd")
    name = sig.find("span", class_="sig-name").get_text(strip=True) if sig else None
    signature = sig.get_text(strip=True) if sig else None
    return {
        "name": mcp_utils._clean_signature_text(name),
        "signature": mcp_utils._clean_signature_text(signature),
        "id": sig.get("id") if sig else None,
        "doc": doc.get_text(" ", strip=True) if doc else None,
    }


def _format_class(class_dl) -> Dict[str, Any]:
    """
    Format a class definition, including its methods and attributes, into a dictionary.

    Args:
        class_dl: The <dl> tag for the class, containing <dt> and <dd>.

    Returns:
        dict: A dictionary with keys 'name', 'signature', 'id', 'doc', 'methods', and 'attributes'.
              'methods' and 'attributes' are themselves dicts keyed by name.
    """
    sig = class_dl.find("dt")
    doc = class_dl.find("dd")
    class_name = (
        sig.find("span", class_="sig-name").get_text(strip=True) if sig else None
    )
    methods = {}
    attributes = {}
    if doc:
        for meth_dl in doc.find_all("dl", class_="py method"):
            meth = _format_function(meth_dl.find("dt"), meth_dl.find("dd"))
            if meth["name"]:
                methods[meth["name"]] = meth
        for attr_dl in doc.find_all("dl", class_="py attribute"):
            attr = _format_attribute(attr_dl)
            if attr["name"]:
                attributes[attr["name"]] = attr
    return {
        "name": mcp_utils._clean_signature_text(class_name),
        "signature": mcp_utils._clean_signature_text(
            sig.get_text(strip=True) if sig else None
        ),
        "id": sig.get("id") if sig else None,
        "doc": doc.get_text(" ", strip=True) if doc else None,
        "methods": methods,
        "attributes": attributes,
    }


def _format_function(sig_dt, doc_dd) -> Dict[str, Any]:
    """
    Format a function or method signature and its documentation into a dictionary.

    Args:
        sig_dt: The <dt> tag containing the function/method signature.
        doc_dd: The <dd> tag containing the function/method docstring.

    Returns:
        dict: A dictionary with keys 'name', 'signature', 'id', and 'doc'.
    """
    name = (
        sig_dt.find("span", class_="sig-name").get_text(strip=True) if sig_dt else None
    )
    signature = sig_dt.get_text(strip=True) if sig_dt else None
    return {
        "name": mcp_utils._clean_signature_text(name),
        "signature": mcp_utils._clean_signature_text(signature),
        "id": sig_dt.get("id") if sig_dt else None,
        "doc": doc_dd.get_text(" ", strip=True) if doc_dd else None,
    }


def _format_submodules(soup) -> dict:
    """
    Extract submodules from a ReadTheDocs module page soup object.
    Looks for a 'Modules' rubric and parses the following table or list for submodule names, URLs, and descriptions.

    Args:
        soup (BeautifulSoup): Parsed HTML soup of the module page.

    Returns:
        dict: {submodule_name: {"url": str, "description": str}}
    """
    submodules = {}
    for rubric in soup.find_all("p", class_="rubric"):
        if rubric.get_text(strip=True).lower() == "modules":
            sib = rubric.find_next_sibling()
            if sib and sib.name in ("table", "ul"):
                for a in sib.find_all("a", href=True):
                    submod_name = a.get_text(strip=True)
                    submod_url = a["href"]
                    desc = ""
                    td = a.find_parent("td")
                    if td and td.find_next_sibling("td"):
                        desc = td.find_next_sibling("td").get_text(strip=True)
                    elif a.parent.name == "li":
                        next_p = a.find_next_sibling("p")
                        if next_p:
                            desc = next_p.get_text(strip=True)
                    submodules[submod_name] = {"url": submod_url, "description": desc}
    return submodules


def _get_element_index(element, soup):
    """Get the index position of an element in the document."""
    try:
        return list(soup.descendants).index(element)
    except ValueError:
        return float("inf")


def _parse_rtd_module_page(html: str, url: Optional[str] = None) -> dict:
    """
    Parse a ReadTheDocs module HTML page and extract functions, classes, methods, attributes, and submodules.
    Returns a dict suitable for MCP server use, with functions, classes, and methods keyed by name.

    Args:
        html (str): The HTML content of the module page.
        url (Optional[str]): The URL of the page (for reference).

    Returns:
        dict: {
            'module': str,
            'url': str,
            'doc': str (module-level docstring if available),
            CODEBASE_DEFS.FUNCTIONS: Dict[str, dict],
            CODEBASE_DEFS.CLASSES: Dict[str, dict],
            'submodules': Dict[str, dict]
        }
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "module": None,
        "url": url,
        "doc": None,
        CODEBASE_DEFS.FUNCTIONS: {},
        CODEBASE_DEFS.CLASSES: {},
        "submodules": _format_submodules(soup),
    }
    # Get module name from <h1>
    h1 = soup.find("h1")
    if h1:
        module_name = h1.get_text(strip=True).replace("\uf0c1", "").strip()
        result["module"] = module_name

        # Extract module-level docstring
        module_doc = _extract_module_docstring(soup, h1)
        if module_doc:
            result["doc"] = module_doc

    # Functions
    for func_dl in soup.find_all("dl", class_="py function"):
        func = _format_function(func_dl.find("dt"), func_dl.find("dd"))
        if func["name"]:
            result[CODEBASE_DEFS.FUNCTIONS][func["name"]] = func
    # Classes
    for class_dl in soup.find_all("dl", class_="py class"):
        cls = _format_class(class_dl)
        if cls["name"]:
            result[CODEBASE_DEFS.CLASSES][cls["name"]] = cls
    return result


async def _process_rtd_package_toc(
    url: str, css_selector: str = READTHEDOCS_TOC_CSS_SELECTOR
) -> dict:
    """
    Parse the ReadTheDocs package TOC and return a dict of {name: url}.
    """
    page_html = await mcp_utils.load_html_page(url)
    soup = BeautifulSoup(page_html, "html.parser")
    selected = soup.select(css_selector)
    return _parse_module_tags(selected)


def _parse_module_tags(td_list: list, base_url: str = "") -> dict:
    """
    Parse a list of <td> elements containing module links and return a dict of {name: url}.
    Optionally prepends base_url to relative hrefs.
    """
    result = {}
    for td in td_list:
        a = td.find("a", class_="reference internal")
        if a:
            # Get the module name from the <span class="pre"> tag
            span = a.find("span", class_="pre")
            if span:
                name = span.text.strip()
                href = a.get("href")
                # Prepend base_url if href is relative
                if href and not href.startswith("http"):
                    href = base_url.rstrip("/") + "/" + href.lstrip("/")
                result[name] = href
    return result


async def _parse_rtd_module_recursive(
    module_url: str,
    visited: Optional[Set[str]] = None,
    docs_dict: Optional[Dict[str, Any]] = None,
    request_delay: float = 0.5,
) -> Dict[str, Any]:
    """
    Recursively parse a module page and all its submodules.

    Parameters
    ----------
    module_url : str
        URL of the module page to parse
    visited : Optional[Set[str]], optional
        Set of already visited URLs to avoid duplicates
    docs_dict : Optional[Dict[str, Any]], optional
        Dictionary to accumulate parsed documentation
    request_delay : float, optional
        Delay in seconds between requests to avoid rate limiting (default: 0.5)

    Raises
    ------
    httpx.HTTPStatusError
        If the page fails to load after retries (fail-fast behavior)
    httpx.RequestError
        If a network error occurs after retries
    """

    if visited is None:
        visited = set()
    if docs_dict is None:
        docs_dict = {}

    if module_url in visited:
        return docs_dict
    visited.add(module_url)

    # Add delay before request to avoid rate limiting
    if request_delay > 0:
        await asyncio.sleep(request_delay)

    page_html = await mcp_utils.load_html_page(module_url)
    module_doc = _parse_rtd_module_page(page_html, module_url)
    module_name = module_doc.get("module") or module_url
    docs_dict[module_name] = module_doc

    # Recursively parse submodules
    for submod_name, submod_info in module_doc.get("submodules", {}).items():
        submod_url = submod_info["url"]
        if not submod_url.startswith("http"):
            base = module_url.rsplit("/", 1)[0]
            submod_url = base + "/" + submod_url.lstrip("/")
        await _parse_rtd_module_recursive(
            submod_url, visited, docs_dict, request_delay=request_delay
        )

    return docs_dict


def _resolve_name_from_cache(name: str, items_dict: dict, package_name: str) -> str:
    """
    Resolve a name to its full import path using the scraped documentation cache.

    If the name contains a path (has "."), returns it as-is. Otherwise, searches
    the cache for a matching stripped_name and returns the path relative to package_name.

    Parameters
    ----------
    name : str
        Name to resolve (e.g., "SBML_dfs" or "sbml_dfs_core.SBML_dfs")
    items_dict : dict
        Dictionary of items from scraped documentation cache
    package_name : str
        Package name to use for relative path extraction

    Returns
    -------
    str
        Resolved name suitable for import_object (e.g., "sbml_dfs_core.SBML_dfs")
        If not found in cache and cache is empty, returns original name.
        If not found in cache but cache has items, returns original name (will be tried by import_object).

    Examples
    --------
    >>> classes = {"napistu.sbml_dfs_core.SBML_dfs": {"stripped_name": "SBML_dfs", ...}}
    >>> _resolve_name_from_cache("SBML_dfs", classes, "napistu")
    'sbml_dfs_core.SBML_dfs'
    >>> _resolve_name_from_cache("sbml_dfs_core.SBML_dfs", classes, "napistu")
    'sbml_dfs_core.SBML_dfs'
    """
    # If name already contains a path, return as-is
    if "." in name:
        logger.info(
            f"_resolve_name_from_cache: '{name}' already contains path, returning as-is"
        )
        return name

    # Skip cache lookup if cache is empty
    if not items_dict:
        logger.info(
            f"_resolve_name_from_cache: cache is empty for '{name}', returning original name"
        )
        return name

    logger.info(
        f"_resolve_name_from_cache: looking up short name '{name}' in cache (cache has {len(items_dict)} items)"
    )

    # Try to find in cache
    result = find_item_by_name(name, items_dict)
    if result:
        full_name, _ = result
        logger.info(
            f"_resolve_name_from_cache: found '{name}' in cache as '{full_name}'"
        )
        # Extract the path relative to package_name
        # e.g., "napistu.sbml_dfs_core.SBML_dfs" -> "sbml_dfs_core.SBML_dfs"
        if full_name.startswith(package_name + "."):
            resolved = full_name[len(package_name) + 1 :]
            logger.info(f"_resolve_name_from_cache: resolved '{name}' -> '{resolved}'")
            return resolved
        else:
            # If it's from a different package, use the full path
            logger.info(
                f"_resolve_name_from_cache: resolved '{name}' -> '{full_name}' (different package)"
            )
            return full_name

    # Not found in cache, return original name
    logger.info(
        f"_resolve_name_from_cache: '{name}' not found in cache, returning original name"
    )
    # import_object will handle this by trying to prepend package_name
    return name


def _exact_search_codebase(
    query: str, codebase_cache: Dict[str, Dict[str, Any]], max_exact_results: int = 20
) -> Dict[str, Any]:
    """
    Perform exact text search across codebase modules, classes, and functions.

    Searches for the query string in module names, class names, function names,
    and their documentation. Returns results organized by type, or an error if
    too many matches are found.

    Parameters
    ----------
    query : str
        Search query string to match against codebase items
    codebase_cache : Dict[str, Dict[str, Any]]
        Codebase cache dictionary with keys:
        - CODEBASE_DEFS.MODULES: Dictionary of module information
        - CODEBASE_DEFS.CLASSES: Dictionary of class information
        - CODEBASE_DEFS.FUNCTIONS: Dictionary of function information
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
            Dictionary with keys MODULES, CLASSES, FUNCTIONS, each containing
            a list of matching items with name, description, snippet, and signature
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
        CODEBASE_DEFS.MODULES: [],
        CODEBASE_DEFS.CLASSES: [],
        CODEBASE_DEFS.FUNCTIONS: [],
    }

    # Search modules - match on module name or module-level docstring
    for module_name, info in codebase_cache[CODEBASE_DEFS.MODULES].items():
        # Get module-level docstring
        doc = (
            info.get(CODEBASE_INSPECT_DEFS.DOC)
            or info.get(SEARCH_RESULT_DEFS.DESCRIPTION)
            or ""
        )

        # Check if query matches module name or module docstring
        module_name_matches = query.lower() in module_name.lower()
        module_doc_matches = doc and query.lower() in doc.lower()

        if module_name_matches or module_doc_matches:
            if doc:
                snippet = mcp_utils.get_snippet(doc, query)
            elif module_name_matches:
                snippet = f"Module name matches '{query}'"
            else:
                snippet = ""

            results[CODEBASE_DEFS.MODULES].append(
                {
                    SEARCH_RESULT_DEFS.NAME: module_name,
                    SEARCH_RESULT_DEFS.DESCRIPTION: doc or f"Module: {module_name}",
                    SEARCH_RESULT_DEFS.SNIPPET: snippet,
                }
            )

    # Search classes
    for class_name, info in codebase_cache[CODEBASE_DEFS.CLASSES].items():
        doc = (
            info.get(CODEBASE_INSPECT_DEFS.DOC)
            or info.get(SEARCH_RESULT_DEFS.DESCRIPTION)
            or ""
        )
        class_text = json.dumps(info)
        if query.lower() in class_text.lower():
            snippet = mcp_utils.get_snippet(doc, query)
            results[CODEBASE_DEFS.CLASSES].append(
                {
                    SEARCH_RESULT_DEFS.NAME: class_name,
                    SEARCH_RESULT_DEFS.DESCRIPTION: doc,
                    SEARCH_RESULT_DEFS.SNIPPET: snippet,
                }
            )

    # Search functions
    for func_name, info in codebase_cache[CODEBASE_DEFS.FUNCTIONS].items():
        doc = (
            info.get(CODEBASE_INSPECT_DEFS.DOC)
            or info.get(SEARCH_RESULT_DEFS.DESCRIPTION)
            or ""
        )
        func_text = json.dumps(info)
        if query.lower() in func_text.lower():
            snippet = mcp_utils.get_snippet(doc, query)
            results[CODEBASE_DEFS.FUNCTIONS].append(
                {
                    SEARCH_RESULT_DEFS.NAME: func_name,
                    SEARCH_RESULT_DEFS.DESCRIPTION: doc,
                    SEARCH_RESULT_DEFS.SIGNATURE: info.get(
                        SEARCH_RESULT_DEFS.SIGNATURE, ""
                    ),
                    SEARCH_RESULT_DEFS.SNIPPET: snippet,
                }
            )

    # Count total results
    total_results = (
        len(results[CODEBASE_DEFS.MODULES])
        + len(results[CODEBASE_DEFS.CLASSES])
        + len(results[CODEBASE_DEFS.FUNCTIONS])
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
                f"2. Use a more specific query with full module path (e.g., 'napistu.sbml_dfs_core.SBML_dfs' instead of '{query}')\n"
                f"3. Search for a specific type directly: use get_class_documentation('{query}') or get_function_documentation('{query}')"
            ),
            "result_counts": {
                CODEBASE_DEFS.MODULES: len(results[CODEBASE_DEFS.MODULES]),
                CODEBASE_DEFS.CLASSES: len(results[CODEBASE_DEFS.CLASSES]),
                CODEBASE_DEFS.FUNCTIONS: len(results[CODEBASE_DEFS.FUNCTIONS]),
            },
        }

    return {
        SEARCH_RESULT_DEFS.QUERY: query,
        SEARCH_RESULT_DEFS.SEARCH_TYPE: SEARCH_TYPES.EXACT,
        SEARCH_RESULT_DEFS.RESULTS: results,
        SEARCH_RESULT_DEFS.TIP: "Use search_type='semantic' for natural language queries about Napistu API",
    }
