from typing import Dict, Any


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
LOCAL_PROFILE = ServerProfile(server_name="napistu-local", enable_execution=True)

REMOTE_PROFILE = ServerProfile(
    server_name="napistu-docs",
    enable_documentation=True,
    enable_codebase=True,
    enable_tutorials=True,
)

FULL_PROFILE = ServerProfile(
    server_name="napistu-full",
    enable_documentation=True,
    enable_codebase=True,
    enable_execution=True,
    enable_tutorials=True,
)


def get_profile(profile_name: str, **overrides) -> ServerProfile:
    """
    Get a predefined profile with optional overrides.

    Args:
        profile_name: Name of the profile ('local', 'remote', or 'full')
        **overrides: Configuration overrides

    Returns:
        ServerProfile instance
    """
    profiles = {
        "local": LOCAL_PROFILE,
        "remote": REMOTE_PROFILE,
        "full": FULL_PROFILE,
    }

    if profile_name not in profiles:
        raise ValueError(f"Unknown profile: {profile_name}")

    # Return a copy of the profile with overrides
    return profiles[profile_name].update(**overrides)
