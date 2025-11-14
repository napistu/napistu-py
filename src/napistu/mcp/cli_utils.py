"""Utilities for CLI inspection and documentation generation."""

from typing import Any, Dict, List, Optional

import click
from click.types import ParamType
from pydantic import BaseModel, Field


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
            # Handle default value - Click uses callables for default factories
            # and sentinels for "no default". For Options, default can be None (no default)
            # or a value. For Arguments, default is typically None.
            default_value = None
            if hasattr(param, "default"):
                if callable(param.default):
                    # It's a default factory, we can't easily get the value at inspection time
                    default_value = None
                elif param.default is not None:
                    # Try to use the default value
                    # Check if it's a sentinel by comparing to ParamType.empty if available
                    try:
                        sentinel = getattr(ParamType, "empty", None)
                        if sentinel is not None and param.default == sentinel:
                            default_value = None
                        else:
                            default_value = param.default
                    except (AttributeError, TypeError, ValueError):
                        # If comparison fails, assume it's a valid default value
                        default_value = param.default

            param_info = {
                "name": param.name,
                "type": str(param.type),
                "required": param.required,
                "default": default_value,
                "help": getattr(
                    param, "help", None
                ),  # Arguments don't have help, only Options do
            }

            if isinstance(param, click.Argument):
                arguments.append(param_info)
            elif isinstance(param, click.Option):
                param_info["flags"] = param.opts
                options.append(param_info)

        # Get subcommands if it's a group
        subcommands = []
        if isinstance(cmd, click.Group):
            subcommands = list(cmd.commands.keys())

        return cls(
            name=cmd.name,
            path=full_path,
            help_text=cmd.help or cmd.short_help,
            arguments=arguments,
            options=options,
            subcommands=subcommands,
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
