"""Constants for the utils module."""

from types import SimpleNamespace
from typing import Dict

# io utils

DOWNLOAD_METHODS = SimpleNamespace(
    WGET="wget",
    FTP="ftp",
)

VALID_DOWNLOAD_METHODS = list(DOWNLOAD_METHODS.__dict__.values())

# docker utils

DOCKER_REGISTRY_NAMES = SimpleNamespace(
    DOCKER_HUB="docker.io",
    GOOGLE_CONTAINER_REGISTRY="gcr.io",
    GITHUB_CONTAINER_REGISTRY="ghcr.io",
    LOCAL="local",
)

# optional dependencies

IMPORTABLE_PACKAGES = SimpleNamespace(
    ANNDATA="anndata",
    GSEAPY="gseapy",
    MUDATA="mudata",
    OMNIPATH="omnipath",
)

NAPISTU_EXTRAS = SimpleNamespace(
    GENOMICS="genomics",
    INGESTION="ingestion",
)

# Mapping of package names to their extras (if any)
PACKAGE_TO_EXTRA: Dict[str, str] = {
    IMPORTABLE_PACKAGES.ANNDATA: NAPISTU_EXTRAS.GENOMICS,
    IMPORTABLE_PACKAGES.GSEAPY: NAPISTU_EXTRAS.GENOMICS,
    IMPORTABLE_PACKAGES.MUDATA: NAPISTU_EXTRAS.GENOMICS,
    IMPORTABLE_PACKAGES.OMNIPATH: NAPISTU_EXTRAS.INGESTION,
}

CRITICAL_LOGGING_ONLY_PACKAGES = [
    IMPORTABLE_PACKAGES.OMNIPATH,
    IMPORTABLE_PACKAGES.MUDATA,
]
