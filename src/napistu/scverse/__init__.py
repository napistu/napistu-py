from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

import mudata

# Configure mudata to use new behavior and suppress warnings
mudata.set_options(pull_on_update=False)

try:
    __version__ = version("napistu")
except PackageNotFoundError:
    # package is not installed
    pass
