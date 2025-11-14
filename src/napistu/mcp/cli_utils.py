"""Utilities for CLI inspection and documentation generation."""

import inspect
import json
from typing import Any, Dict, List, Optional

import click
from click.types import ParamType
from pydantic import BaseModel, Field

from napistu.mcp.constants import CLICK_COMMAND_DEFS


class CLICommandInfo(BaseModel):
    """Information about a CLI command."""

    name: str = Field(description="Command name")
    path: str = Field(
        description="Full command path (e.g., 'napistu ingestion reactome')"
    )
    help_text: Optional[str] = Field(None, description="Command help text")
    arguments: List[Dict[str, Any]] = Field(default_factory=list)
    options: List[Dict[str, Any]] = Field(default_factory=list)
    subcommands: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(
        None, description="Full source code of the command function"
    )
    file_path: Optional[str] = Field(
        None, description="Path to file containing the command function"
    )
    line_number: Optional[int] = Field(
        None, description="Starting line number of the command function"
    )

    @classmethod
    def from_click_command(
        cls, cmd: click.Command, path: str = "", use_command_name: bool = True
    ) -> "CLICommandInfo":
        """Extract info from a Click command.

        Parameters
        ----------
        cmd : click.Command
            The Click command to extract info from.
        path : str, optional
            The base path for this command (e.g., "napistu ingestion").
        use_command_name : bool, optional
            If True, append the command name to the path. If False, use the path as-is.
            Default is True.
        """
        if use_command_name:
            full_path = f"{path} {cmd.name}".strip()
        else:
            full_path = path

        # Extract arguments
        arguments = []
        options = []
        for param in cmd.params:
            # Get default value, handling callables, sentinels, and non-serializable values
            default_value = None
            if hasattr(param, "default"):
                default_value = _serialize_default_value(param.default)

            param_info = {
                CLICK_COMMAND_DEFS.NAME: str(getattr(param, CLICK_COMMAND_DEFS.NAME)),
                CLICK_COMMAND_DEFS.TYPE: str(getattr(param, CLICK_COMMAND_DEFS.TYPE)),
                CLICK_COMMAND_DEFS.REQUIRED: bool(
                    getattr(param, CLICK_COMMAND_DEFS.REQUIRED)
                ),
                CLICK_COMMAND_DEFS.DEFAULT: default_value,
                CLICK_COMMAND_DEFS.HELP: getattr(
                    param, CLICK_COMMAND_DEFS.HELP, None
                ),  # Arguments don't have help, only Options do
            }

            if isinstance(param, click.Argument):
                arguments.append(param_info)
            elif isinstance(param, click.Option):
                param_info[CLICK_COMMAND_DEFS.FLAGS] = param.opts
                options.append(param_info)

        # Get subcommands if it's a group
        subcommands = []
        if isinstance(cmd, click.Group):
            subcommands = list(cmd.commands.keys())

        # Get function source code if callback exists
        source = None
        file_path = None
        line_number = None

        if hasattr(cmd, "callback") and cmd.callback is not None:
            func = cmd.callback
            try:
                source = inspect.getsource(func)
            except (OSError, TypeError):
                source = None

            try:
                file_path = inspect.getfile(func)
                line_number = inspect.getsourcelines(func)[1]
            except (OSError, TypeError):
                file_path = None
                line_number = None

        return cls(
            name=cmd.name,
            path=full_path,
            help_text=cmd.help or cmd.short_help,
            arguments=arguments,
            options=options,
            subcommands=subcommands,
            source=source,
            file_path=file_path,
            line_number=line_number,
        )


class CLIStructure(BaseModel):
    """Complete CLI structure."""

    commands: Dict[str, CLICommandInfo] = Field(default_factory=dict)

    @classmethod
    def from_cli_group(cls, group: click.Group, prefix: str = "") -> "CLIStructure":
        """Recursively extract entire CLI structure."""
        structure = cls()

        def walk_commands(cmd: click.Command, path: str, is_root: bool = False):
            # For the root group, use the prefix directly without the group name
            if is_root and prefix:
                current_path = prefix
                # Add root group to commands dict with prefix as path (skip command name)
                cmd_info = CLICommandInfo.from_click_command(
                    cmd, prefix, use_command_name=False
                )
                structure.commands[prefix] = cmd_info
            else:
                # Add this command
                cmd_info = CLICommandInfo.from_click_command(cmd, path)
                current_path = cmd_info.path
                structure.commands[current_path] = cmd_info

            # Recurse into subcommands
            if isinstance(cmd, click.Group):
                for subcmd_name, subcmd in cmd.commands.items():
                    walk_commands(subcmd, current_path, is_root=False)

        walk_commands(group, prefix, is_root=True)
        return structure


def _serialize_default_value(default: Any) -> Any:
    """Convert a Click parameter default value to a JSON-serializable format.

    Click uses various types for defaults:
    - Callables (default factories) - cannot be evaluated at inspection time
    - Sentinels (ParamType.empty) - indicates no default
    - Values - may or may not be JSON-serializable

    Parameters
    ----------
    default : Any
        The default value from a Click parameter

    Returns
    -------
    Any
        JSON-serializable value, or None if default cannot be determined
    """
    if default is None:
        return None

    # Callables are default factories - can't evaluate at inspection time
    if callable(default):
        return None

    # Check if it's a sentinel value (indicates no default)
    try:
        sentinel = getattr(ParamType, "empty", None)
        if sentinel is not None and default == sentinel:
            return None
    except (AttributeError, TypeError, ValueError):
        pass

    # Try to serialize the value - if it fails, convert to string
    try:
        json.dumps(default)
        return default
    except (TypeError, ValueError):
        # Not JSON-serializable, convert to string representation
        return str(default)
