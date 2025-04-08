from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("calicolabs-cpr")
except PackageNotFoundError:
    # package is not installed
    pass
