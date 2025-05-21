from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

__version__ = "0.2.2"

try:
    __version__ = version("calicolabs-cpr")
except PackageNotFoundError:
    # package is not installed
    pass
