# src/napistu/utils/__init__.py
"""
Napistu utilities package.

Currently re-exports everything from the legacy _legacy_utils.py module.
This package will be gradually refactored into logical submodules.
"""

# Import everything from the legacy utils module
from napistu.utils._legacy_utils import (
    # Private/helper functions
    _add_nameness_score,
    _add_nameness_score_wrapper,
    _create_left_align_formatters,
    _in_jupyter_environment,
    _merge_and_log_overwrites,
    _show_as_string,
    # Pandas utilities
    check_unique_index,
    # Path utilities
    copy_uri,
    # File I/O and downloads
    download_and_extract,
    download_ftp,
    download_wget,
    drop_extra_cols,
    ensure_pd_df,
    extract,
    # String utilities
    extract_regex_match,
    extract_regex_search,
    # Graph utilities
    find_weakly_connected_subgraphs,
    format_identifiers_as_edgelist,
    get_extn_from_url,
    get_source_base_and_path,
    get_target_base_and_path,
    gunzip,
    infer_entity_type,
    initialize_dir,
    load_json,
    load_parquet,
    load_pickle,
    match_pd_vars,
    match_regex_dict,
    matrix_to_edgelist,
    path_exists,
    pickle_cache,
    read_pickle,
    requests_retry_session,
    safe_capitalize,
    safe_fill,
    safe_join_set,
    safe_series_tolist,
    save_json,
    save_parquet,
    save_pickle,
    score_nameness,
    # Display utilities
    show,
    style_df,
    update_pathological_names,
    write_file_contents_to_path,
    write_pickle,
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
    # Private/helper functions (included for backwards compatibility)
    "_add_nameness_score",
    "_add_nameness_score_wrapper",
    "_create_left_align_formatters",
    "_in_jupyter_environment",
    "_merge_and_log_overwrites",
    "_show_as_string",
]
