# src/napistu/utils/__init__.py
"""
Napistu utilities package.

This package provides utilities organized into logical submodules:
- io_utils: File I/O and download operations
- path_utils: Path and URI operations
- pd_utils: Pandas DataFrame operations
- string_utils: String and text processing
- ig_utils: igraph operations
- display_utils: Display and formatting utilities
- optional: Optional dependency handling and lazy loading
"""

# Import display utilities from display_utils
from napistu.utils.display_utils import (
    # Private/helper functions
    _create_left_align_formatters,
    _in_jupyter_environment,
    _show_as_string,
    # Display utilities
    show,
)

# Import igraph utilities from ig_utils
from napistu.utils.ig_utils import (
    find_weakly_connected_subgraphs,
)

# Import I/O functions from io_utils
from napistu.utils.io_utils import (
    download_and_extract,
    download_ftp,
    download_wget,
    extract,
    gunzip,
    load_json,
    load_parquet,
    load_pickle,
    pickle_cache,
    read_pickle,
    requests_retry_session,
    save_json,
    save_parquet,
    save_pickle,
    write_file_contents_to_path,
    write_pickle,
)

# Import optional dependency utilities from optional
from napistu.utils.optional import (
    create_package_importer,
    import_anndata,
    import_gseapy,
    import_mudata,
    import_omnipath,
    import_omnipath_interactions,
    import_package,
    require_anndata,
    require_gseapy,
    require_mudata,
    require_omnipath,
    require_package,
)

# Import path utilities from path_utils
from napistu.utils.path_utils import (
    copy_uri,
    get_extn_from_url,
    get_source_base_and_path,
    get_target_base_and_path,
    initialize_dir,
    path_exists,
)

# Import pandas utilities from pd_utils
from napistu.utils.pd_utils import (
    _merge_and_log_overwrites,
    check_unique_index,
    drop_extra_cols,
    ensure_pd_df,
    format_identifiers_as_edgelist,
    infer_entity_type,
    match_pd_vars,
    matrix_to_edgelist,
    style_df,
    update_pathological_names,
)

# Import string utilities from string_utils
from napistu.utils.string_utils import (
    _add_nameness_score,
    _add_nameness_score_wrapper,
    extract_regex_match,
    extract_regex_search,
    match_regex_dict,
    safe_capitalize,
    safe_fill,
    safe_join_set,
    safe_series_tolist,
    score_nameness,
)

# Public API - excludes private functions by convention
__all__ = [
    # File I/O and downloads
    "download_and_extract",
    "download_ftp",
    "download_wget",
    "extract",
    "gunzip",
    "load_json",
    "load_parquet",
    "load_pickle",
    "pickle_cache",
    "read_pickle",
    "requests_retry_session",
    "save_json",
    "save_parquet",
    "save_pickle",
    "write_file_contents_to_path",
    "write_pickle",
    # Path utilities
    "copy_uri",
    "get_extn_from_url",
    "get_source_base_and_path",
    "get_target_base_and_path",
    "initialize_dir",
    "path_exists",
    # Pandas utilities
    "check_unique_index",
    "drop_extra_cols",
    "ensure_pd_df",
    "format_identifiers_as_edgelist",
    "infer_entity_type",
    "match_pd_vars",
    "matrix_to_edgelist",
    "style_df",
    "update_pathological_names",
    # String utilities
    "extract_regex_match",
    "extract_regex_search",
    "match_regex_dict",
    "safe_capitalize",
    "safe_fill",
    "safe_join_set",
    "safe_series_tolist",
    "score_nameness",
    # Graph utilities
    "find_weakly_connected_subgraphs",
    # Display utilities
    "show",
    # Optional dependency utilities
    "create_package_importer",
    "import_omnipath",
    "import_omnipath_interactions",
    "import_package",
    "import_anndata",
    "import_gseapy",
    "import_mudata",
    "require_anndata",
    "require_gseapy",
    "require_mudata",
    "require_omnipath",
    "require_package",
    # Private/helper functions (included for backwards compatibility)
    "_add_nameness_score",
    "_add_nameness_score_wrapper",
    "_create_left_align_formatters",
    "_in_jupyter_environment",
    "_merge_and_log_overwrites",
    "_show_as_string",
]
