"""
Base Docker image management for Napistu workflows.

Provides image validation, availability checking, and container execution.

Classes
-------
ImageInfo
    Docker image information.


"""

import logging
import subprocess
from abc import ABC
from typing import Dict, List, Optional

from pydantic import BaseModel

from napistu.utils.constants import DOCKER_REGISTRY_NAMES

logger = logging.getLogger(__name__)


class ImageInfo(BaseModel):
    """
    Docker image information with registry configuration.

    Attributes
    ----------
    name : str
        Image name without registry prefix (e.g., "project/repo/image")
    tag : str
        Image tag
    registry : str, optional
        Registry URL. Valid values:
        - "local": Pre-built local image, no pulling
        - "docker.io": Docker Hub
        - "ghcr.io": GitHub Container Registry
        - Regional Artifact Registry: "us-west1-docker.pkg.dev", "us-central1-docker.pkg.dev", etc.
        - Custom registry URLs
    platform : Optional[str]
        Target platform (e.g., "linux/arm64", "linux/amd64")

    Properties
    ----------
    is_local : bool
        Check if this is a local-only image
    full_name : str
        Get full image name with registry and tag
    pull_name : Optional[str]
        Get the name to use for pulling

    Examples
    --------
    Local-only image:
    >>> info = ImageInfo(name="diffdock-arm", tag="latest", registry="local")
    >>> info.full_name
    'diffdock-arm:latest'

    Google Artifact Registry (public):
    >>> info = ImageInfo(
    ...     name="shackett/napistu-images/napistu-base",
    ...     tag="latest",
    ...     registry="us-west1-docker.pkg.dev"
    ... )
    >>> info.full_name
    'us-west1-docker.pkg.dev/shackett/napistu-images/napistu-base:latest'
    """

    name: str
    tag: str
    registry: str
    platform: Optional[str] = None

    @property
    def is_local(self) -> bool:
        """Check if this is a local-only image."""
        return self.registry == DOCKER_REGISTRY_NAMES.LOCAL

    @property
    def full_name(self) -> str:
        """
        Get full image name with registry and tag.

        Returns
        -------
        str
            Full image reference:
            - Local: "name:tag"
            - Remote: "registry/name:tag"
        """
        if self.is_local:
            return f"{self.name}:{self.tag}"
        else:
            return f"{self.registry}/{self.name}:{self.tag}"

    @property
    def pull_name(self) -> Optional[str]:
        """
        Get the name to use for pulling.

        Returns
        -------
        str or None
            Full pull reference, or None if local-only
        """
        if self.is_local:
            return None
        return self.full_name


NAPISTU_IMAGE = ImageInfo(
    name="shackett/napistu-images/napistu-base",
    tag="latest",
    registry="us-west1-docker.pkg.dev",
)


class DockerImageManager(ABC):
    """
    Base class for managing Docker images.

    Provides image validation, availability checking, and container execution.
    Subclasses implement specific image requirements and entrypoint access.

    Attributes
    ----------
    image_info : ImageInfo
        Image specification including registry
    auto_pull : bool
        Automatically pull image if not available locally (ignored for local images)

    Properties
    ----------
    full_image_name : str
        Get full image name with tag

    Public Methods
    --------------
    ensure_available()
        Ensure image is available, pulling if necessary
    is_available()
        Check if Docker image exists locally
    pull()
        Pull image from registry
    run_command(cmd: List[str], volumes: Optional[Dict[str, str]] = None, environment: Optional[Dict[str, str]] = None, **kwargs) -> subprocess.CompletedProcess
        Run a command in the Docker container

    Examples
    --------
    Remote image from Docker Hub:
    >>> info = ImageInfo(name="username/diffdock", tag="latest")
    >>> manager = SomeDockerImage(info, auto_pull=True)

    Local-only image:
    >>> info = ImageInfo(name="diffdock-arm", tag="latest", registry="local")
    >>> manager = DiffDockImageARM(info)

    GCR image:
    >>> info = ImageInfo(name="project/diffdock", tag="v1.0", registry="gcr.io")
    >>> manager = SomeDockerImage(info, auto_pull=True)
    """

    def __init__(self, image_info: ImageInfo, auto_pull: bool = False):
        self.image_info = image_info
        self.auto_pull = auto_pull

        # Verify that Docker is available
        _verify_docker_available()

        # Ensure that the image is available
        self.ensure_available()

    @property
    def full_image_name(self) -> str:
        """Get full image name with tag."""
        return self.image_info.full_name

    def ensure_available(self) -> bool:
        """
        Ensure image is available, pulling if necessary.

        Returns
        -------
        bool
            True if image is available (or was successfully pulled)

        Raises
        ------
        RuntimeError
            If image cannot be made available
        """
        if self.is_available():
            logger.debug(f"✓ Image available: {self.full_image_name}")
            return True

        logger.info(f"Image not found locally: {self.full_image_name}")

        # For local-only images, don't try to pull
        if self.image_info.is_local:
            raise RuntimeError(
                f"Local-only Docker image '{self.full_image_name}' not found."
            )

        if not self.auto_pull:
            raise RuntimeError(
                f"Remote Docker image '{self.full_image_name}' not found locally and auto-pull is disabled."
            )

        # Try to pull remote image
        if self.pull():
            return True

        raise RuntimeError(f"Failed to pull Docker image '{self.full_image_name}'.")

    def is_available(self) -> bool:
        """
        Check if Docker image exists locally.

        For local images, checks the local name.
        For remote images, checks the full registry path.

        Returns
        -------
        bool
            True if image is available locally
        """
        result = subprocess.run(
            ["docker", "images", "-q", self.full_image_name],
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    def pull(self) -> bool:
        """
        Pull image from registry.

        Returns
        -------
        bool
            True if pull succeeded, False if failed or local-only image
        """
        if self.image_info.is_local:
            logger.warning(
                f"Cannot pull {self.full_image_name}: registry='local'. "
                "Image must be built locally."
            )
            return False

        pull_name = self.image_info.pull_name
        logger.info(f"Pulling Docker image: {pull_name}")

        cmd = ["docker", "pull"]
        if self.image_info.platform:
            cmd.extend(["--platform", self.image_info.platform])
        cmd.append(pull_name)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"✓ Successfully pulled {pull_name}")
                return True
            else:
                logger.error(f"Failed to pull image: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Image pull timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"Error pulling image: {e}")
            return False

    def run_command(
        self,
        cmd: List[str],
        volumes: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """
        Run a command in the Docker container.

        Parameters
        ----------
        cmd : List[str]
            Command to run inside container
        volumes : Dict[str, str], optional
            Volume mounts {host_path: container_path} or {host_path: "container_path:mode"}
            Mode can be 'ro' (read-only) or 'rw' (read-write, default)
        environment : Dict[str, str], optional
            Environment variables
        **kwargs
            Additional arguments passed to subprocess.run

        Returns
        -------
        subprocess.CompletedProcess
            Result of the command execution
        """
        docker_cmd = ["docker", "run", "--rm"]

        # Add platform if specified
        if self.image_info.platform:
            docker_cmd.extend(["--platform", self.image_info.platform])

        # Add volumes
        if volumes:
            for host_path, container_spec in volumes.items():
                # Handle both "container_path" and "container_path:mode" formats
                docker_cmd.extend(["-v", f"{host_path}:{container_spec}"])

        # Add environment variables
        if environment:
            for key, value in environment.items():
                docker_cmd.extend(["-e", f"{key}={value}"])

        # Add image name and command
        docker_cmd.append(self.full_image_name)
        docker_cmd.extend(cmd)

        logger.debug(f"Running: {' '.join(docker_cmd)}")

        return subprocess.run(docker_cmd, **kwargs)


def _verify_docker_available() -> None:
    """
    Verify Docker is installed and running.

    Raises
    ------
    RuntimeError
        If Docker is not available or not running
    """
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Docker daemon is not running. Please start Docker Desktop."
            )
    except FileNotFoundError:
        raise RuntimeError(
            "Docker is not installed. Please install Docker Desktop:\n"
            "  https://www.docker.com/products/docker-desktop"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Docker daemon is not responding. Please restart Docker Desktop."
        )
