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

READMES = {
    "napistu": "https://raw.githubusercontent.com/napistu/napistu/main/README.md",
    "napistu-py": "https://raw.githubusercontent.com/napistu/napistu-py/main/README.md",
    "napistu-r": "https://raw.githubusercontent.com/napistu/napistu-r/main/README.md",
    "napistu/tutorials": "https://raw.githubusercontent.com/napistu/napistu/main/tutorials/README.md",
}

WIKI_ROOT = "https://raw.githubusercontent.com/napistu/napistu/main/docs/wiki"

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
    "formatting_sbml_dfs_as_cpr_graphs": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/formatting_sbml_dfs_as_cpr_graphs.ipynb",
    "merging_models_into_a_consensus": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/merging_models_into_a_consensus.ipynb",
    "r_based_network_visualization": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/r_based_network_visualization.ipynb",
    "suggesting_mechanisms_with_networks": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/suggesting_mechanisms_with_networks.ipynb",
    "understanding_sbml_dfs": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/understanding_sbml_dfs.ipynb",
    "working_with_genome_scale_networks": "https://raw.githubusercontent.com/napistu/napistu/refs/heads/main/tutorials/working_with_genome_scale_networks.ipynb",
}

TUTORIALS_CACHE_DIR = os.path.join(PACKAGE_DEFS.CACHE_DIR, TUTORIALS.TUTORIALS)
