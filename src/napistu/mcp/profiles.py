from typing import Any, Dict

from napistu.mcp.constants import MCP_PROFILES


class ServerProfile:
    """Base profile for MCP server configuration."""

    def __init__(self, **kwargs):
        self.config = {
            # Default configuration
            "server_name": "napistu-mcp",
            "enable_documentation": False,
            "enable_execution": False,
            "enable_codebase": False,
            "enable_tutorials": False,
            "enable_chat": False,
            "session_context": None,
            "object_registry": None,
            "tutorials_path": None,
        }
        # Override with provided kwargs
        self.config.update(kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary."""
        return self.config.copy()

    def update(self, **kwargs) -> "ServerProfile":
        """Update profile with additional configuration."""
        new_profile = ServerProfile(**self.config)
        new_profile.config.update(kwargs)
        return new_profile


# Pre-defined profiles
EXECUTION_PROFILE = ServerProfile(
    server_name="napistu-execution", enable_execution=True
)

DOCS_PROFILE = ServerProfile(
    server_name="napistu-docs",
    enable_documentation=True,
    enable_codebase=True,
    enable_tutorials=True,
    enable_chat=True,
)

FULL_PROFILE = ServerProfile(
    server_name="napistu-full",
    enable_documentation=True,
    enable_codebase=True,
    enable_execution=True,
    enable_tutorials=True,
    enable_chat=True,
)


def get_profile(profile_name: str, **overrides) -> ServerProfile:
    """
    Get a predefined profile with optional overrides.

    Args:
        profile_name: Name of the profile ('execution', 'docs', or 'full')
        **overrides: Configuration overrides

    Returns:
        ServerProfile instance
    """
    profiles = {
        MCP_PROFILES.EXECUTION: EXECUTION_PROFILE,
        MCP_PROFILES.DOCS: DOCS_PROFILE,
        MCP_PROFILES.FULL: FULL_PROFILE,
    }

    if profile_name not in profiles:
        raise ValueError(f"Unknown profile: {profile_name}")

    # Return a copy of the profile with overrides
    return profiles[profile_name].update(**overrides)
