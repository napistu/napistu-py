import os
from types import SimpleNamespace

from napistu.constants import PACKAGE_DEFS

MCP_COMPONENTS = SimpleNamespace(
    CHAT="chat",
    CODEBASE="codebase",
    DOCUMENTATION="documentation",
    EXECUTION="execution",
    HEALTH="health",
    TUTORIALS="tutorials",
)

# Searchable components (subset of MCP_COMPONENTS)
SEARCH_COMPONENTS = {
    MCP_COMPONENTS.CODEBASE,
    MCP_COMPONENTS.DOCUMENTATION,
    MCP_COMPONENTS.TUTORIALS,
}

# Search type constants
SEARCH_TYPES = SimpleNamespace(
    SEMANTIC="semantic",
    EXACT="exact",
)

VALID_SEARCH_TYPES = SEARCH_TYPES.__dict__.values()

# Search result field names
SEARCH_RESULT_DEFS = SimpleNamespace(
    QUERY="query",
    SEARCH_TYPE="search_type",
    RESULTS="results",
    TIP="tip",
    NAME="name",
    SNIPPET="snippet",
    TITLE="title",
    URL="url",
    CONTENT="content",
    METADATA="metadata",
    SOURCE="source",
    SIMILARITY_SCORE="similarity_score",
    ID="id",
    DESCRIPTION="description",
    SIGNATURE="signature",
)

# Codebase categories (used in both inspect and readthedocs approaches)
CODEBASE_DEFS = SimpleNamespace(
    MODULES="modules",
    CLASSES="classes",
    FUNCTIONS="functions",
    METHODS="methods",
)

# Codebase inspection field names (specific to runtime inspection)
CODEBASE_INSPECT_DEFS = SimpleNamespace(
    DOC="doc",
    DOCSTRING="docstring",
    SOURCE="source",
    FILE_PATH="file_path",
    LINE_NUMBER="line_number",
    MODULE="module",
    INIT_SIGNATURE="init_signature",
    INIT_SOURCE="init_source",
    METHODS="methods",
    METHOD_NAME="method_name",
    CLASS_NAME="class_name",
    ERROR="error",
)

# Codebase ReadTheDocs field names (specific to scraped ReadTheDocs HTML)
CODEBASE_RTD_DEFS = SimpleNamespace(
    DOC="doc",
    SIGNATURE="signature",
    NAME="name",
    ID="id",
    METHODS="methods",
    ATTRIBUTES="attributes",
)

# CLI command field name constants
CLICK_COMMAND_DEFS = SimpleNamespace(
    NAME="name",
    TYPE="type",
    REQUIRED="required",
    DEFAULT="default",
    HELP="help",
    FLAGS="flags",
)

DOCUMENTATION = SimpleNamespace(
    README="readme",
    WIKI="wiki",
    ISSUES="issues",
    PRS="prs",
    PACKAGEDOWN="packagedown",
)

# Documentation summary category names
DOCUMENTATION_SUMMARY_DEFS = SimpleNamespace(
    README_FILES="readme_files",
    ISSUES="issues",
    PRS="prs",
    WIKI_PAGES="wiki_pages",
    PACKAGEDOWN_SECTIONS="packagedown_sections",
    SEMANTIC_SEARCH="semantic_search",
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

# Preset configuration names
PRESET_NAMES = SimpleNamespace(
    LOCAL="local",
    PRODUCTION="production",
)

PROFILE_DEFS = SimpleNamespace(
    ENABLE_CHAT="enable_chat",
    ENABLE_CODEBASE="enable_codebase",
    ENABLE_DOCUMENTATION="enable_documentation",
    ENABLE_EXECUTION="enable_execution",
    ENABLE_TUTORIALS="enable_tutorials",
)

# Health check status constants
HEALTH_CHECK_DEFS = SimpleNamespace(
    STATUS="status",
    ERROR="error",
    INITIALIZING="initializing",
    UNAVAILABLE="unavailable",
    INACTIVE="inactive",
    HEALTHY="healthy",
    DEGRADED="degraded",
    UNHEALTHY="unhealthy",
    UNKNOWN="unknown",
)

# Semantic search metadata field names (for ChromaDB indexing)
SEMANTIC_SEARCH_METADATA_DEFS = SimpleNamespace(
    TYPE="type",
    NAME="name",
    SOURCE="source",
    CHUNK="chunk",
    IS_CHUNKED="is_chunked",
    TOTAL_CHUNKS="total_chunks",
    CLASS_NAME="class_name",
)

# Semantic search configuration constants
SEMANTIC_SEARCH_DEFS = SimpleNamespace(
    # Content length thresholds
    MIN_CONTENT_LENGTH_SHORT=20,  # For issues/PRs and codebase items
    MIN_CONTENT_LENGTH_LONG=50,  # For regular content
    # Chunking defaults
    CHUNK_THRESHOLD=1200,  # Content length threshold for chunking
    MAX_CHUNK_SIZE=1000,  # Maximum size per chunk
    # String patterns
    CHUNK_PART_PREFIX=" (part ",
    CHUNK_PART_SUFFIX=")",
    METHOD_SOURCE_PREFIX="method: ",
)

# Health summary field names
HEALTH_SUMMARIES = SimpleNamespace(
    # Health check response fields
    STATUS="status",
    COMPONENTS="components",
    TIMESTAMP="timestamp",
    VERSION="version",
    FAILED_COMPONENTS="failed_components",
    LAST_CHECK="last_check",
    MESSAGE="message",
    COLLECTIONS="collections",
    TOTAL_COLLECTIONS="total_collections",
    # Codebase component
    MODULES_COUNT="modules_count",
    CLASSES_COUNT="classes_count",
    FUNCTIONS_COUNT="functions_count",
    TOTAL_ITEMS="total_items",
    # Documentation component
    README_COUNT="readme_count",
    WIKI_PAGES="wiki_pages",
    ISSUES_REPOS="issues_repos",
    PRS_REPOS="prs_repos",
    TOTAL_SECTIONS="total_sections",
    # Tutorials component
    TUTORIAL_COUNT="tutorial_count",
    TUTORIAL_IDS="tutorial_ids",
    # Execution component
    SESSION_CONTEXT_ITEMS="session_context_items",
    REGISTERED_OBJECTS="registered_objects",
    CONTEXT_KEYS="context_keys",
    OBJECT_NAMES="object_names",
)

READMES = {
    "napistu": "https://raw.githubusercontent.com/napistu/napistu/main/README.md",
    "napistu-py": "https://raw.githubusercontent.com/napistu/napistu-py/main/README.md",
    "napistu-r": "https://raw.githubusercontent.com/napistu/napistu-r/main/README.md",
    "napistu-torch": "https://raw.githubusercontent.com/napistu/napistu-torch/main/README.md",
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
NAPISTU_TORCH_READTHEDOCS = "https://napistu-torch.readthedocs.io/en/latest"
NAPISTU_TORCH_READTHEDOCS_API = NAPISTU_TORCH_READTHEDOCS + "/api.html"
READTHEDOCS_TOC_CSS_SELECTOR = "td"

DEFAULT_GITHUB_API = "https://api.github.com"

REPOS_WITH_ISSUES = [
    PACKAGE_DEFS.GITHUB_PROJECT_REPO,
    PACKAGE_DEFS.GITHUB_NAPISTU_PY,
    PACKAGE_DEFS.GITHUB_NAPISTU_R,
    PACKAGE_DEFS.GITHUB_NAPISTU_TORCH,
]

GITHUB_ISSUES_INDEXED = "all"
GITHUB_PRS_INDEXED = "all"
# GitHub API field names
GITHUB_DEFS = SimpleNamespace(
    NUMBER="number",
    TITLE="title",
    STATE="state",
    HTML_URL="html_url",
    URL="url",
    BODY="body",
    IS_PR="is_pr",
    PULL_REQUEST="pull_request",
    MERGED_AT="merged_at",
)

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

# chat

DEFAULT_ALLOWED_ORIGINS = [
    "https://napistu.com",
    "https://www.napistu.com",
    "http://localhost:4321",
    "http://127.0.0.1:4321",
]

API_ENDPOINTS = SimpleNamespace(
    API="/api",
    CHAT="chat",
    STATS="stats",
    HEALTH="health",
)

CHAT_DEFAULTS = SimpleNamespace(
    RATE_LIMIT_PER_HOUR=5,
    RATE_LIMIT_PER_DAY=15,
    DAILY_BUDGET=5.0,
    MAX_TOKENS=2000,
    MAX_MESSAGE_LENGTH=2000,
    CLAUDE_MODEL="claude-sonnet-4-20250514",
)

CHAT_ENV_VARS = SimpleNamespace(
    RATE_LIMIT_PER_HOUR="CHAT_RATE_LIMIT_HOUR",
    RATE_LIMIT_PER_DAY="CHAT_RATE_LIMIT_DAY",
    DAILY_BUDGET="CHAT_DAILY_BUDGET",
    MAX_TOKENS="CHAT_MAX_TOKENS",
    MAX_MESSAGE_LENGTH="CHAT_MAX_MESSAGE_LENGTH",
    ANTHROPIC_API_KEY="ANTHROPIC_API_KEY",
    MCP_SERVER_URL="MCP_SERVER_URL",
    CLAUDE_MODEL="CLAUDE_MODEL",
)
