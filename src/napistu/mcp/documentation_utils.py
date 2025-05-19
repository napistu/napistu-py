"""
Utilities for loading and processing documentation.
"""

import os
import re
import httpx
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

from napistu.mcp.constants import READTHEDOCS_TOC_CSS_SELECTOR

# Import optional dependencies with error handling
try:
    from bs4 import BeautifulSoup
    import markdown
except ImportError:
    raise ImportError(
        "Documentation utilities require additional dependencies. Install with 'pip install napistu[mcp]'"
    )

async def load_readme_content(readme_url: str) -> str:
    if readme_url.startswith("http://") or readme_url.startswith("https://"):
        async with httpx.AsyncClient() as client:
            response = await client.get(readme_url)
            response.raise_for_status()
            return response.text
    else:
        raise ValueError(f"Only HTTP(S) URLs are supported for documentation paths: {readme_url}")

async def load_readthedocs_content(url: str) -> Dict[str, str]:
    """
    Load content from a Read the Docs site.
    
    Args:
        url: Base URL of the Read the Docs site
    
    Returns:
        Dictionary mapping section names to content
    """
    sections = {}
    
    try:
        # Fetch the main page
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the table of contents
            toc = soup.select('.toctree-wrapper .toctree-l1 a')
            
            # Fetch each section
            for link in toc:
                section_name = link.text.strip()
                section_url = url + link['href'] if not link['href'].startswith('http') else link['href']
                
                # Fetch the section content
                section_response = await client.get(section_url)
                section_response.raise_for_status()
                
                section_soup = BeautifulSoup(section_response.text, 'html.parser')
                content_div = section_soup.select_one('.section')
                
                if content_div:
                    sections[section_name] = content_div.prettify()
                else:
                    sections[section_name] = "Content not found"
    
    except Exception as e:
        print(f"Error loading Read the Docs content from {url}: {e}")
        sections["error"] = f"Error loading Read the Docs content: {e}"
    
    return sections


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


async def _process_rtd_package_toc(url: str, css_selector: str = READTHEDOCS_TOC_CSS_SELECTOR) -> dict:
    
    page_html = await load_html_page(url)
    soup = BeautifulSoup(page_html, 'html.parser')
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


def extract_submodules_from_rtd_page(soup) -> dict:
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
                    submodules[submod_name] = {
                        "url": submod_url,
                        "description": desc
                    }
    return submodules


def parse_rtd_module_page(html: str, url: Optional[str] = None) -> dict:
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
            'functions': Dict[str, dict],
            'classes': Dict[str, dict],
            'submodules': Dict[str, dict]
        }
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "module": None,
        "url": url,
        "functions": {},
        "classes": {},
        "submodules": extract_submodules_from_rtd_page(soup)
    }
    # Get module name from <h1>
    h1 = soup.find("h1")
    if h1:
        module_name = h1.get_text(strip=True).replace("\uf0c1", "").strip()
        result["module"] = module_name
    # Functions
    for func_dl in soup.find_all("dl", class_="py function"):
        func = _format_function(func_dl.find("dt"), func_dl.find("dd"))
        if func["name"]:
            result["functions"][func["name"]] = func
    # Classes
    for class_dl in soup.find_all("dl", class_="py class"):
        cls = _format_class(class_dl)
        if cls["name"]:
            result["classes"][cls["name"]] = cls
    return result


def _format_function(sig_dt, doc_dd) -> Dict[str, Any]:
    """
    Format a function or method signature and its documentation into a dictionary.

    Args:
        sig_dt: The <dt> tag containing the function/method signature.
        doc_dd: The <dd> tag containing the function/method docstring.

    Returns:
        dict: A dictionary with keys 'name', 'signature', 'id', and 'doc'.
    """
    return {
        "name": sig_dt.find("span", class_="sig-name").get_text(strip=True) if sig_dt else None,
        "signature": sig_dt.get_text(strip=True) if sig_dt else None,
        "id": sig_dt.get("id") if sig_dt else None,
        "doc": doc_dd.get_text(" ", strip=True) if doc_dd else None
    }


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
    return {
        "name": sig.find("span", class_="sig-name").get_text(strip=True) if sig else None,
        "signature": sig.get_text(strip=True) if sig else None,
        "id": sig.get("id") if sig else None,
        "doc": doc.get_text(" ", strip=True) if doc else None
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
    class_name = sig.find("span", class_="sig-name").get_text(strip=True) if sig else None
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
        "name": class_name,
        "signature": sig.get_text(strip=True) if sig else None,
        "id": sig.get("id") if sig else None,
        "doc": doc.get_text(" ", strip=True) if doc else None,
        "methods": methods,
        "attributes": attributes
    }
