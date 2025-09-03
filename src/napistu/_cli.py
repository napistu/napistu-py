"""Shared CLI utilities for Napistu CLIs"""

from __future__ import annotations

import logging
from typing import Callable

import click
from rich.console import Console
from rich.logging import RichHandler

import napistu


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
