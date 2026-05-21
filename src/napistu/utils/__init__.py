# src/napistu/utils/__init__.py
"""
Napistu utilities package.

Submodules provide focused helpers (``io_utils``, ``path_utils``, ``pd_utils``, etc.).
Optional dependencies are imported via ``napistu.utils.optional``, not re-exported here.
"""

# Import display utilities from display_utils
from napistu.utils.display_utils import show

# Import igraph utilities from ig_utils
from napistu.utils.ig_utils import find_weakly_connected_subgraphs

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
    requests_retry_session,
    save_json,
    save_parquet,
    save_pickle,
    write_file_contents_to_path,
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
    check_unique_index,
    downcast_float_dataframe,
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
    extract_regex_match,
    extract_regex_search,
    match_regex_dict,
    safe_capitalize,
    safe_fill,
    safe_join_set,
    score_nameness,
)

# Public names for ``from napistu import utils`` (legacy barrel); prefer submodule imports.
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
    "requests_retry_session",
    "save_json",
    "save_parquet",
    "save_pickle",
    "write_file_contents_to_path",
    # Path utilities
    "copy_uri",
    "get_extn_from_url",
    "get_source_base_and_path",
    "get_target_base_and_path",
    "initialize_dir",
    "path_exists",
    # Pandas utilities
    "check_unique_index",
    "downcast_float_dataframe",
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
    "score_nameness",
    # Graph utilities
    "find_weakly_connected_subgraphs",
    # Display utilities
    "show",
]
