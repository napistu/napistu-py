"""Constants for the utils module."""

from types import SimpleNamespace

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
