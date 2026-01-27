"""
Napistu genomics module.

This module provides utilities for interfacing with genomics data formats and methods.

Modules
--------
gsea:
    Utilities for applying gene set enrichment analysis (GSEA) to genomics data.
scverse_loading:
    Utilities for loading and working with scverse data - i.e., anndata and mudata objects.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from napistu.utils.optional import import_mudata

# Configure mudata to use new behavior and suppress warnings
# Lazy load mudata to avoid import errors if genomics extra is not installed
try:
    mudata = import_mudata()
    mudata.set_options(pull_on_update=False)
except ImportError:
    pass

try:
    __version__ = version("napistu")
except PackageNotFoundError:
    # package is not installed
    pass
