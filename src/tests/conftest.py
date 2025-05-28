from __future__ import annotations

import os
import sys

from napistu import consensus
from napistu import indices
from napistu import sbml_dfs_core
from napistu.ingestion import sbml
from pytest import fixture
from pytest import skip


@fixture
def sbml_path():
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    sbml_path = os.path.join(test_path, "test_data", "R-HSA-1237044.sbml")

    if not os.path.isfile(sbml_path):
        raise ValueError(f"{sbml_path} not found")
    return sbml_path


@fixture
def sbml_model(sbml_path):
    sbml_model = sbml.SBML(sbml_path)
    return sbml_model


@fixture
def sbml_dfs(sbml_model):
    sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)
    return sbml_dfs


@fixture
def sbml_dfs_metabolism():
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    test_data = os.path.join(test_path, "test_data")

    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index_metabolism.tsv"))
    sbml_dfs_dict = consensus.construct_sbml_dfs_dict(pw_index)
    sbml_dfs = consensus.construct_consensus_model(sbml_dfs_dict, pw_index)

    return sbml_dfs


@fixture
def sbml_dfs_glucose_metabolism():
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    test_data = os.path.join(test_path, "test_data")
    sbml_path = os.path.join(test_data, "reactome_glucose_metabolism.sbml")

    sbml_model = sbml.SBML(sbml_path).model
    sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)

    return sbml_dfs


# Define custom markers for platforms
def pytest_configure(config):
    config.addinivalue_line("markers", "skip_on_windows: mark test to skip on Windows")
    config.addinivalue_line("markers", "skip_on_macos: mark test to skip on macOS")
    config.addinivalue_line(
        "markers", "unix_only: mark test to run only on Unix/Linux systems"
    )


# Define platform conditions
is_windows = sys.platform == "win32"
is_macos = sys.platform == "darwin"
is_unix = not (is_windows or is_macos)


# Apply skipping based on platform
def pytest_runtest_setup(item):
    # Skip tests marked to be skipped on Windows
    if is_windows and any(
        mark.name == "skip_on_windows" for mark in item.iter_markers()
    ):
        skip("Test skipped on Windows")

    # Skip tests marked to be skipped on macOS
    if is_macos and any(mark.name == "skip_on_macos" for mark in item.iter_markers()):
        skip("Test skipped on macOS")

    # Skip tests that should run only on Unix
    if not is_unix and any(mark.name == "unix_only" for mark in item.iter_markers()):
        skip("Test runs only on Unix systems")
