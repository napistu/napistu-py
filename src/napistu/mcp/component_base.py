"""
Base classes for MCP server components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


class ComponentState(ABC):
    """
    Base class for component state management.

    Provides standard interface for health checking and initialization tracking.
    """

    def __init__(self):
        self.initialized = False
        self.initialization_error = None

    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if component has successfully loaded data and is functioning.

        Returns
        -------
        bool
            True if component is healthy (has data and no errors)
        """
        pass

    def is_available(self) -> bool:
        """
        Check if component is available (initialized without critical errors).

        Returns
        -------
        bool
            True if component initialized successfully, regardless of data status
        """
        return self.initialized and self.initialization_error is None

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get standardized health status for health checks.

        Returns
        -------
        Dict[str, Any]
            Health status dictionary with standard format:
            - status: 'initializing', 'unavailable', 'inactive', or 'healthy'
            - error: error message if unavailable
            - additional component-specific details if healthy
        """
        if not self.initialized:
            return {"status": "initializing"}
        elif self.initialization_error:
            return {"status": "unavailable", "error": str(self.initialization_error)}
        elif not self.is_healthy():
            return {"status": "inactive"}
        else:
            return {"status": "healthy", **self.get_health_details()}

    def get_health_details(self) -> Dict[str, Any]:
        """
        Get component-specific health details.

        Override in subclasses to provide additional health information.

        Returns
        -------
        Dict[str, Any]
            Component-specific health details
        """
        return {}


class MCPComponent(ABC):
    """
    Base class for MCP server components.

    Provides standard interface for initialization and registration.
    """

    def __init__(self):
        self.state = self._create_state()

    @abstractmethod
    def _create_state(self) -> ComponentState:
        """
        Create the component state object.

        Returns
        -------
        ComponentState
            Component-specific state instance
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the component asynchronously.

        Should populate component state and handle any external data loading.

        Returns
        -------
        bool
            True if initialization successful
        """
        pass

    @abstractmethod
    def register(self, mcp: FastMCP) -> None:
        """
        Register component resources and tools with the MCP server.

        Parameters
        ----------
        mcp : FastMCP
            FastMCP server instance to register with
        """
        pass

    def get_state(self) -> ComponentState:
        """
        Get the component state for health checks and testing.

        Returns
        -------
        ComponentState
            Current component state
        """
        return self.state

    async def safe_initialize(self) -> bool:
        """
        Initialize with error handling and state tracking.

        Returns
        -------
        bool
            True if initialization successful
        """
        try:
            logger.info(f"Initializing {self.__class__.__name__}...")

            result = await self.initialize()

            self.state.initialized = True
            self.state.initialization_error = None

            if result:
                logger.info(f"✅ {self.__class__.__name__} initialized successfully")
            else:
                logger.warning(f"⚠️ {self.__class__.__name__} initialized with issues")

            return result

        except Exception as e:
            logger.error(f"❌ {self.__class__.__name__} failed to initialize: {e}")
            self.state.initialized = True
            self.state.initialization_error = e
            return False
