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
    STATSMODELS="statsmodels",
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
    IMPORTABLE_PACKAGES.STATSMODELS: NAPISTU_EXTRAS.GENOMICS,
}

CRITICAL_LOGGING_ONLY_PACKAGES = [
    IMPORTABLE_PACKAGES.OMNIPATH,
    IMPORTABLE_PACKAGES.MUDATA,
]

# pandas

MERGE_RELATIONSHIP_TYPES = SimpleNamespace(
    ONE_TO_ONE="1:1",
    ONE_TO_MANY="1:m",
    MANY_TO_ONE="m:1",
    MANY_TO_MANY="m:m",
    ONE_TO_ZERO="1:0",
    ZERO_TO_ONE="0:1",
)

VALID_MERGE_RELATIONSHIP_TYPES = list(MERGE_RELATIONSHIP_TYPES.__dict__.values())
