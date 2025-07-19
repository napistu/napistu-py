from __future__ import annotations

import functools
import os
import sys
import threading

import pandas as pd
import pytest
from pytest import fixture
from pytest import skip

from napistu import consensus
from napistu import indices
from napistu.identifiers import Identifiers
from napistu.sbml_dfs_core import SBML_dfs
from napistu.source import Source
from napistu.ingestion.sbml import SBML
from napistu.network.net_create import process_napistu_graph
from napistu.constants import SBML_DFS, MINI_SBO_FROM_NAME, SBOTERM_NAMES


@fixture
def sbml_path():
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    sbml_path = os.path.join(test_path, "test_data", "R-HSA-1237044.sbml")

    if not os.path.isfile(sbml_path):
        raise ValueError(f"{sbml_path} not found")
    return sbml_path


@fixture
def sbml_model(sbml_path):
    sbml_model = SBML(sbml_path)
    return sbml_model


@fixture
def sbml_dfs(sbml_model):
    sbml_dfs = SBML_dfs(sbml_model)
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

    sbml_model = SBML(sbml_path)
    sbml_dfs = SBML_dfs(sbml_model)

    return sbml_dfs


@pytest.fixture
def minimal_valid_sbml_dfs():
    """Create a minimal valid SBML_dfs object for testing."""
    blank_id = Identifiers([])
    source = Source(init=True)

    sbml_dict = {
        SBML_DFS.COMPARTMENTS: pd.DataFrame(
            {
                SBML_DFS.C_NAME: ["cytosol"],
                SBML_DFS.C_IDENTIFIERS: [blank_id],
                SBML_DFS.C_SOURCE: [source],
            },
            index=pd.Index(["C00001"], name=SBML_DFS.C_ID),
        ),
        SBML_DFS.SPECIES: pd.DataFrame(
            {
                SBML_DFS.S_NAME: ["ATP"],
                SBML_DFS.S_IDENTIFIERS: [blank_id],
                SBML_DFS.S_SOURCE: [source],
            },
            index=pd.Index(["S00001"], name=SBML_DFS.S_ID),
        ),
        SBML_DFS.COMPARTMENTALIZED_SPECIES: pd.DataFrame(
            {
                SBML_DFS.SC_NAME: ["ATP [cytosol]"],
                SBML_DFS.S_ID: ["S00001"],
                SBML_DFS.C_ID: ["C00001"],
                SBML_DFS.SC_SOURCE: [source],
            },
            index=pd.Index(["SC00001"], name=SBML_DFS.SC_ID),
        ),
        SBML_DFS.REACTIONS: pd.DataFrame(
            {
                SBML_DFS.R_NAME: ["test_reaction"],
                SBML_DFS.R_IDENTIFIERS: [blank_id],
                SBML_DFS.R_SOURCE: [source],
                SBML_DFS.R_ISREVERSIBLE: [False],
            },
            index=pd.Index(["R00001"], name=SBML_DFS.R_ID),
        ),
        SBML_DFS.REACTION_SPECIES: pd.DataFrame(
            {
                SBML_DFS.R_ID: ["R00001"],
                SBML_DFS.SC_ID: ["SC00001"],
                SBML_DFS.STOICHIOMETRY: [1.0],
                SBML_DFS.SBO_TERM: ["SBO:0000011"],
            },
            index=pd.Index(["RSC00001"], name=SBML_DFS.RSC_ID),
        ),
    }

    return SBML_dfs(sbml_dict)


@fixture
def napistu_graph(sbml_dfs):
    """
    Pytest fixture to create a NapistuGraph from sbml_dfs with directed=True and topology weighting.
    """
    return process_napistu_graph(sbml_dfs, directed=True, weighting_strategy="topology")


@fixture
def napistu_graph_undirected(sbml_dfs):
    """
    Pytest fixture to create a NapistuGraph from sbml_dfs with directed=False and topology weighting.
    """
    return process_napistu_graph(
        sbml_dfs, directed=False, weighting_strategy="topology"
    )


@fixture
def napistu_graph_metabolism(sbml_dfs_metabolism):
    """
    Pytest fixture to create a NapistuGraph from sbml_dfs_glucose_metabolism with directed=True and topology weighting.
    """
    return process_napistu_graph(
        sbml_dfs_metabolism, directed=True, weighting_strategy="topology"
    )


@pytest.fixture
def reaction_species_examples():
    """
    Pytest fixture providing a dictionary of example reaction species DataFrames for various test cases.
    """

    d = dict()
    d["valid_interactor"] = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR],
            ],
            SBML_DFS.SC_ID: ["sc1", "sc2"],
            SBML_DFS.STOICHIOMETRY: [0, 0],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    d["invalid_interactor"] = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
            ],
            SBML_DFS.SC_ID: ["sc1", "sc2"],
            SBML_DFS.STOICHIOMETRY: [0, 0],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    d["sub_and_prod"] = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
            ],
            SBML_DFS.SC_ID: ["sub", "prod"],
            SBML_DFS.STOICHIOMETRY: [-1, 1],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    d["stimulator"] = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.STIMULATOR],
            ],
            SBML_DFS.SC_ID: ["sub", "prod", "stim"],
            SBML_DFS.STOICHIOMETRY: [-1, 1, 0],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    d["all_entities"] = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.STIMULATOR],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.CATALYST],
            ],
            SBML_DFS.SC_ID: ["sub", "prod", "stim", "cat"],
            SBML_DFS.STOICHIOMETRY: [-1, 1, 0, 0],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    d["no_substrate"] = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.STIMULATOR],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.STIMULATOR],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INHIBITOR],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.CATALYST],
            ],
            SBML_DFS.SC_ID: ["prod", "stim1", "stim2", "inh", "cat"],
            SBML_DFS.STOICHIOMETRY: [1, 0, 0, 0, 0],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    d["single_species"] = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT]],
            SBML_DFS.SC_ID: ["lone_prod"],
            SBML_DFS.STOICHIOMETRY: [1],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    d["activator_and_inhibitor_only"] = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.STIMULATOR],  # activator
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INHIBITOR],  # inhibitor
            ],
            SBML_DFS.SC_ID: ["act", "inh"],
            SBML_DFS.STOICHIOMETRY: [0, 0],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    return d


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


def skip_on_timeout(timeout_seconds):
    """Cross-platform decorator that skips a test if it takes longer than timeout_seconds"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            finished = [False]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                    finished[0] = True
                except Exception as e:
                    exception[0] = e
                    finished[0] = True

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if not finished[0]:
                # Thread is still running, timeout occurred
                pytest.skip(f"Test skipped due to timeout ({timeout_seconds}s)")

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


pytest.skip_on_timeout = skip_on_timeout
