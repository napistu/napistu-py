"""
Utilities for scanning and analyzing the Napistu codebase.
"""

from typing import Dict, Optional, Any, Set

from napistu.mcp import utils as mcp_utils
from napistu.mcp.constants import READTHEDOCS_TOC_CSS_SELECTOR

# Import optional dependencies with error handling
try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError(
        "Documentation utilities require additional dependencies. Install with 'pip install napistu[mcp]'"
    )


async def read_read_the_docs(package_toc_url: str) -> dict:
    """
    Recursively parse all modules and submodules starting from the package TOC.
    """
    # Step 1: Get all module URLs from the TOC
    packages_dict = await _process_rtd_package_toc(package_toc_url)
    docs_dict = {}
    visited = set()

    # Step 2: Recursively parse each module page
    for package_name, module_url in packages_dict.items():
        if not module_url.startswith("http"):
            # Make absolute if needed
            base = package_toc_url.rsplit("/", 1)[0]
            module_url = base + "/" + module_url.lstrip("/")
        await _parse_rtd_module_recursive(module_url, visited, docs_dict)

    return docs_dict


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
        for func_name, func_info in module_info.get("functions", {}).items():
            fq_name = f"{module_name}.{func_name}"
            functions[fq_name] = func_info
        # Classes
        for class_name, class_info in module_info.get("classes", {}).items():
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
        "submodules": _format_submodules(soup),
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


async def _parse_rtd_module_recursive(
    module_url: str,
    visited: Optional[Set[str]] = None,
    docs_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Recursively parse a module page and all its submodules.
    """

    if visited is None:
        visited = set()
    if docs_dict is None:
        docs_dict = {}

    if module_url in visited:
        return docs_dict
    visited.add(module_url)

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
        await _parse_rtd_module_recursive(submod_url, visited, docs_dict)

    return docs_dict
