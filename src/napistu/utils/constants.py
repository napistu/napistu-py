"""Constants for the utils module."""

from types import SimpleNamespace

DOCKER_REGISTRY_NAMES = SimpleNamespace(
    DOCKER_HUB="docker.io",
    GOOGLE_CONTAINER_REGISTRY="gcr.io",
    GITHUB_CONTAINER_REGISTRY="ghcr.io",
    LOCAL="local",
)
