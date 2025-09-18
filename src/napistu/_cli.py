"""Shared CLI utilities for Napistu CLIs"""

from __future__ import annotations

import logging
import re
from typing import Callable

import click
from rich.console import Console
from rich.logging import RichHandler

import napistu
from napistu.constants import NAPISTU_CLI


def setup_logging() -> tuple[logging.Logger, Console]:
    """
    Set up the standardized logging configuration for Napistu CLIs.

    Returns:
        tuple: (logger, console) - The configured logger and Rich console
    """
    # Minimal early logging setup - silence problematic loggers
    logging.getLogger().setLevel(logging.CRITICAL + 1)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("requests").setLevel(logging.CRITICAL)

    # Configure the main logger
    logger = logging.getLogger(napistu.__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Rich console and handler setup
    console = Console(width=120)
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,
        markup=True,
        log_time_format="[%m/%d %H:%M]",
    )

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(handler)

    return logger, console


def click_str_to_list(string: str) -> list[str]:
    """Convert a string-based representation of a list inputted from the CLI into a list of strings."""

    var_extract_regex = re.compile("\\'?([a-zA-Z_]+)\\'?")

    re_search = re.search("^\\[(.*)\\]$", string)
    if re_search:
        return var_extract_regex.findall(re_search.group(0))
    else:
        raise ValueError(
            f"The provided string, {string}, could not be reformatted as a list. An example string which can be formatted is: \"['weight', 'upstream_weight']\""
        )


# click formatting options


def verbosity_option(f: Callable) -> Callable:
    """
    Decorator that adds --verbosity option for napistu logging.

    This should be applied to CLI commands that need verbosity control.
    Must be used after setup_logging() has been called.
    """

    def configure_logging_callback(ctx, param, value):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level_map.get(value.upper(), logging.INFO)

        # Get the logger that was configured by setup_logging
        logger = logging.getLogger(napistu.__name__)
        logger.setLevel(level)
        return value

    return click.option(
        "--verbosity",
        "-v",
        type=click.Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
        ),
        default="INFO",
        callback=configure_logging_callback,
        expose_value=False,
        is_eager=True,
        help="Set the logging verbosity level for napistu.",
    )(f)


def get_logger() -> logging.Logger:
    """
    Get the configured Napistu logger.

    This should be called after setup_logging() has been run.
    """
    return logging.getLogger(napistu.__name__)


def overwrite_option(f: Callable) -> Callable:
    """
    Decorator that adds a standardized --overwrite option.

    Common pattern for CLI commands that create files/outputs.
    """
    return click.option(
        "--overwrite",
        is_flag=True,
        default=False,
        help="Overwrite existing files.",
    )(f)


def target_uri_output(f: Callable) -> Callable:
    """
    Decorator that adds a standardized target_uri argument.
    """
    return click.argument(NAPISTU_CLI.TARGET_URI, type=str)(f)


def sbml_dfs_input(f: Callable) -> Callable:
    """
    Decorator that adds a standardized model_uri argument.

    Common pattern for CLI commands that take a model file as input.
    """
    return click.argument(NAPISTU_CLI.SBML_DFS_URI, type=str)(f)


def sbml_dfs_output(f: Callable) -> Callable:
    """
    Decorator that adds a standardized output_model_uri argument.

    Common pattern for CLI commands that create a new model file as output.
    """
    return click.argument(NAPISTU_CLI.OUTPUT_MODEL_URI, type=str)(f)


def sbml_dfs_io(f: Callable) -> Callable:
    """
    Decorator that adds both model_uri and output_model_uri arguments.

    Common pattern for CLI commands that read a model and write a new one.
    """
    return click.argument(NAPISTU_CLI.SBML_DFS_URI, type=str)(
        click.argument(NAPISTU_CLI.OUTPUT_MODEL_URI, type=str)(f)
    )


def organismal_species_argument(f: Callable) -> Callable:
    """
    Decorator that adds a standardized organismal_species argument.

    Common pattern for CLI commands that work with specific species.
    """
    return click.argument("organismal_species", type=str)(f)


def nondogmatic_option(f: Callable) -> Callable:
    """
    Decorator that adds a standardized --nondogmatic option.

    Common pattern for CLI commands that can run in non-dogmatic mode.
    """
    return click.option(
        "--nondogmatic",
        "-n",
        is_flag=True,
        default=False,
        help="Run in non-dogmatic mode (trying to merge genes and proteins)?",
    )(f)


def genodexito_options(f: Callable) -> Callable:
    """
    Decorator that adds standardized Genodexito method options.

    Common pattern for CLI commands that use Genodexito for identifier mapping.
    """
    return click.option(
        "--preferred-method",
        "-p",
        default="bioconductor",
        type=str,
        help="Preferred method to use for identifier expansion",
    )(
        click.option(
            "--allow-fallback",
            "-a",
            default=True,
            type=bool,
            help="Allow fallback to other methods if preferred method fails",
        )(f)
    )
