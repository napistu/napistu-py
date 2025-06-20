import os
from types import SimpleNamespace

from napistu.constants import PACKAGE_DEFS

MCP_COMPONENTS = SimpleNamespace(
    CODEBASE="codebase",
    DOCUMENTATION="documentation",
    EXECUTION="execution",
    TUTORIALS="tutorials",
)

DOCUMENTATION = SimpleNamespace(
    README="readme",
    WIKI="wiki",
    ISSUES="issues",
    PRS="prs",
    PACKAGEDOWN="packagedown",
)

EXECUTION = SimpleNamespace(
    NOTEBOOKS="notebooks",
)

TUTORIALS = SimpleNamespace(
    TUTORIALS="tutorials",
)

TOOL_VARS = SimpleNamespace(
    NAME="name",
    SNIPPET="snippet",
)

# MCP Server Configuration Constants
MCP_DEFAULTS = SimpleNamespace(
    # Local development defaults
    LOCAL_HOST="127.0.0.1",
    LOCAL_PORT=8765,
    # Production defaults
    PRODUCTION_HOST="0.0.0.0",
    PRODUCTION_PORT=8080,
    # Server names
    LOCAL_SERVER_NAME="napistu-local",
    PRODUCTION_SERVER_NAME="napistu-production",
    FULL_SERVER_NAME="napistu-full",
    # Transport configuration
    TRANSPORT="streamable-http",
    MCP_PATH="/mcp",
    # Standard protocol ports
    HTTP_PORT=80,
    HTTPS_PORT=443,
)

# Production server URL
MCP_PRODUCTION_URL = "https://napistu-mcp-server-844820030839.us-west1.run.app"

# Profile names (component configurations)
MCP_PROFILES = SimpleNamespace(
    EXECUTION="execution",  # execution only
    DOCS="docs",  # docs + codebase + tutorials
    FULL="full",  # all components
)

READMES = {
    "napistu": "https://raw.githubusercontent.com/napistu/napistu/main/README.md",
    "napistu-py": "https://raw.githubusercontent.com/napistu/napistu-py/main/README.md",
    "napistu-r": "https://raw.githubusercontent.com/napistu/napistu-r/main/README.md",
    "napistu/tutorials": "https://raw.githubusercontent.com/napistu/napistu/main/tutorials/README.md",
}

WIKI_PAGES = {
    "Consensus",
    "Data-Sources",
    "Napistu-Graphs",
    "Dev-Zone",
    "Environment-Setup",
    "Exploring-Molecular-Relationships-as-Networks",
    "GitHub-Actions-napistu‐py",
    "History",
    "Model-Context-Protocol-(MCP)-server",
    "Precomputed-distances",
    "SBML",
    "SBML-DFs",
}

NAPISTU_PY_READTHEDOCS = "https://napistu.readthedocs.io/en/latest"
NAPISTU_PY_READTHEDOCS_API = NAPISTU_PY_READTHEDOCS + "/api.html"
READTHEDOCS_TOC_CSS_SELECTOR = "td"

DEFAULT_GITHUB_API = "https://api.github.com"

REPOS_WITH_ISSUES = [
    PACKAGE_DEFS.GITHUB_PROJECT_REPO,
    PACKAGE_DEFS.GITHUB_NAPISTU_PY,
    PACKAGE_DEFS.GITHUB_NAPISTU_R,
]

GITHUB_ISSUES_INDEXED = "all"
GITHUB_PRS_INDEXED = "all"

REPOS_WITH_WIKI = [PACKAGE_DEFS.GITHUB_PROJECT_REPO]

# Example mapping: tutorial_id -> raw GitHub URL
TUTORIAL_URLS = {
    "adding_data_to_graphs": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/adding_data_to_graphs.ipynb",
    "downloading_pathway_data": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/downloading_pathway_data.ipynb",
    "creating_a_napistu_graph": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/creating_a_napistu_graph.ipynb",
    "merging_models_into_a_consensus": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/merging_models_into_a_consensus.ipynb",
    "r_based_network_visualization": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/r_based_network_visualization.ipynb",
    "suggesting_mechanisms_with_networks": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/suggesting_mechanisms_with_networks.ipynb",
    "understanding_sbml_dfs": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/understanding_sbml_dfs.ipynb",
    "working_with_genome_scale_networks": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/working_with_genome_scale_networks.ipynb",
}

TUTORIALS_CACHE_DIR = os.path.join(PACKAGE_DEFS.CACHE_DIR, TUTORIALS.TUTORIALS)
