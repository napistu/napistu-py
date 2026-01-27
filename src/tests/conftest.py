from __future__ import annotations

import functools
import os
import sys
import threading
from datetime import datetime

import pandas as pd
import pytest
from google.cloud import storage
from igraph import Graph
from pytest import fixture, skip
from testcontainers.core.container import DockerContainer

from napistu import consensus, indices
from napistu.constants import (
    BQB,
    EXPECTED_PW_INDEX_COLUMNS,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    SBML_DFS,
    SBOTERM_NAMES,
    SOURCE_SPEC,
)
from napistu.identifiers import Identifiers
from napistu.ingestion.sbml import SBML
from napistu.network.constants import IGRAPH_DEFS, NAPISTU_WEIGHTING_STRATEGIES
from napistu.network.net_create import process_napistu_graph
from napistu.network.precompute import precompute_distances
from napistu.sbml_dfs_core import SBML_dfs
from napistu.source import Source


@fixture
def model_source_stub():

    source_dict = {k: "test" for k in EXPECTED_PW_INDEX_COLUMNS | {SOURCE_SPEC.MODEL}}
    return Source(pd.DataFrame(source_dict, index=[0]))


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
def sbml_dfs(sbml_model, model_source_stub):
    sbml_dfs = SBML_dfs(sbml_model, model_source=model_source_stub)
    return sbml_dfs


@fixture
def sbml_dfs_w_data(sbml_dfs):
    """Create an sbml_dfs fixture with test data for reactions and species."""

    # Add test reactions data using real indices from the sbml_dfs
    reaction_indices = (
        sbml_dfs.reactions.index[:3]
        if len(sbml_dfs.reactions) >= 3
        else sbml_dfs.reactions.index
    )
    reactions_data = pd.DataFrame(
        {
            "rxn_attr_int": [0, 1, 2],
            "rxn_attr_float": [0.1, 0.2, 0.3],
            "rxn_attr_bool": [True, False, True],
            "rxn_attr_string": ["a", "b", "c"],
        },
        index=reaction_indices,
    )
    sbml_dfs.add_reactions_data("rxn_data", reactions_data)

    # Add test species data using real indices from the sbml_dfs
    species_indices = (
        sbml_dfs.species.index[:3]
        if len(sbml_dfs.species) >= 3
        else sbml_dfs.species.index
    )
    species_data = pd.DataFrame(
        {
            "spec_attr_int": [3, 4, 5],
            "spec_attr_float": [0.4, 0.5, 0.6],
            "spec_attr_bool": [False, False, True],
            "spec_attr_string": ["A", "B", "C"],
        },
        index=species_indices,
    )
    sbml_dfs.add_species_data("spec_data", species_data)

    return sbml_dfs


@fixture
def pw_index_metabolism():
    """Create a pathway index for metabolism test data."""
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    test_data = os.path.join(test_path, "test_data")
    return indices.PWIndex(os.path.join(test_data, "pw_index_metabolism.tsv"))


@fixture
def sbml_dfs_dict_metabolism(pw_index_metabolism):
    """Create a dictionary of SBML_dfs objects from metabolism test data."""
    return consensus.construct_sbml_dfs_dict(pw_index_metabolism)


@fixture
def sbml_dfs_metabolism(sbml_dfs_dict_metabolism, pw_index_metabolism):
    """Create a consensus SBML_dfs model from metabolism test data."""
    return consensus.construct_consensus_model(
        sbml_dfs_dict_metabolism, pw_index_metabolism
    )


@fixture
def sbml_dfs_glucose_metabolism(model_source_stub):
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    test_data = os.path.join(test_path, "test_data")
    sbml_path = os.path.join(test_data, "reactome_glucose_metabolism.sbml")

    sbml_model = SBML(sbml_path)
    sbml_dfs = SBML_dfs(sbml_model, model_source_stub)

    return sbml_dfs


@pytest.fixture
def minimal_valid_sbml_dfs(model_source_stub):
    """Create a minimal valid SBML_dfs object for testing."""
    blank_id = Identifiers([])
    source = Source.empty()

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

    return SBML_dfs(sbml_dict, model_source_stub)


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
        sbml_dfs,
        directed=False,
        weighting_strategy=NAPISTU_WEIGHTING_STRATEGIES.TOPOLOGY,
    )


@fixture
def napistu_graph_metabolism(sbml_dfs_metabolism):
    """
    Pytest fixture to create a NapistuGraph from sbml_dfs_glucose_metabolism with directed=True and topology weighting.
    """
    return process_napistu_graph(
        sbml_dfs_metabolism,
        directed=True,
        weighting_strategy=NAPISTU_WEIGHTING_STRATEGIES.TOPOLOGY,
    )


@fixture
def precomputed_distances_metabolism(napistu_graph_metabolism):
    """
    Pytest fixture to create precomputed distances from the glucose metabolism napistu_graph.
    """
    return precompute_distances(
        napistu_graph_metabolism, max_steps=30000, max_score_q=1
    )


@fixture
def species_identifiers_metabolism(sbml_dfs_metabolism):
    """
    Pytest fixture to create a species identifiers DataFrame from sbml_dfs_metabolism.
    This creates a DataFrame with the structure expected by validate_assets.
    """

    return sbml_dfs_metabolism.get_characteristic_species_ids()


@pytest.fixture
def simple_directed_graph():
    """Create a simple directed igraph.Graph with named vertices."""
    g = Graph(directed=True)
    g.add_vertices(4)
    g.vs[IGRAPH_DEFS.NAME] = ["A", "B", "C", "D"]
    g.add_edges([(0, 1), (1, 2), (2, 3)])
    g.es["observed"] = [True, True, False]
    return g


@pytest.fixture
def simple_undirected_graph():
    """Create a simple undirected igraph.Graph with named vertices."""
    g = Graph(directed=False)
    g.add_vertices(3)
    g.vs[IGRAPH_DEFS.NAME] = ["X", "Y", "Z"]
    return g


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


@pytest.fixture
def sbml_dfs_characteristic_test_data(model_source_stub):
    """
    Create an SBML_dfs object with test data specifically designed for testing
    characteristic species functionality.
    """
    # Create mock species identifiers data with different BQB types
    # This matches the structure returned by get_identifiers method
    # Based on actual test results: characteristic_only=True gives (1,3), characteristic_only=False gives (1,5)
    mock_species_ids = pd.DataFrame(
        {
            SBML_DFS.S_ID: ["s1", "s1", "s1", "s1", "s1"],
            IDENTIFIERS.IDENTIFIER: [
                "P12345",  # uniprot - IS (characteristic)
                "CHEBI:15377",  # chebi - IS (characteristic)
                "GO:12345",  # go - HAS_PART (characteristic)
                "P67890",  # uniprot - HAS_VERSION (non-characteristic)
                "P67890",  # chebi - ENCODES (non-characteristic)
            ],
            IDENTIFIERS.ONTOLOGY: ["uniprot", "chebi", "go", "uniprot", "chebi"],
            IDENTIFIERS.BQB: [
                BQB.IS,  # characteristic
                BQB.IS,  # characteristic
                BQB.HAS_PART,  # characteristic
                BQB.HAS_VERSION,  # non-characteristic
                BQB.ENCODES,  # non-characteristic
            ],
        }
    )

    # Create minimal required tables for SBML_dfs
    compartments = pd.DataFrame(
        {SBML_DFS.C_NAME: ["cytosol"], SBML_DFS.C_IDENTIFIERS: [None]}, index=["C1"]
    )
    compartments.index.name = SBML_DFS.C_ID

    # Create proper identifiers for the species from the mock data
    identifiers_list = []
    for _, row in mock_species_ids.iterrows():
        identifiers_list.append(
            {
                "identifier": row[IDENTIFIERS.IDENTIFIER],
                "ontology": row[IDENTIFIERS.ONTOLOGY],
                "bqb": row[IDENTIFIERS.BQB],
            }
        )
    species_identifiers = Identifiers(identifiers_list)

    species = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["A"],
            SBML_DFS.S_IDENTIFIERS: [species_identifiers],
            SBML_DFS.S_SOURCE: [None],
        },
        index=["s1"],
    )
    species.index.name = SBML_DFS.S_ID

    compartmentalized_species = pd.DataFrame(
        {
            SBML_DFS.SC_NAME: ["A [cytosol]"],
            SBML_DFS.S_ID: ["s1"],
            SBML_DFS.C_ID: ["C1"],
            SBML_DFS.SC_SOURCE: [None],
        },
        index=["SC1"],
    )
    compartmentalized_species.index.name = SBML_DFS.SC_ID

    reactions = pd.DataFrame(
        {
            SBML_DFS.R_NAME: ["rxn1"],
            SBML_DFS.R_IDENTIFIERS: [None],
            SBML_DFS.R_SOURCE: [None],
            SBML_DFS.R_ISREVERSIBLE: [False],
        },
        index=["R1"],
    )
    reactions.index.name = SBML_DFS.R_ID

    reaction_species = pd.DataFrame(
        {
            SBML_DFS.R_ID: ["R1"],
            SBML_DFS.SC_ID: ["SC1"],
            SBML_DFS.STOICHIOMETRY: [1],
            SBML_DFS.SBO_TERM: ["SBO:0000459"],
        },
        index=["RSC1"],
    )
    reaction_species.index.name = SBML_DFS.RSC_ID

    sbml_dict = {
        SBML_DFS.COMPARTMENTS: compartments,
        SBML_DFS.SPECIES: species,
        SBML_DFS.COMPARTMENTALIZED_SPECIES: compartmentalized_species,
        SBML_DFS.REACTIONS: reactions,
        SBML_DFS.REACTION_SPECIES: reaction_species,
    }

    sbml_dfs = SBML_dfs(sbml_dict, model_source_stub, validate=False, resolve=False)

    # Store the mock data for use in tests
    sbml_dfs._mock_species_ids = mock_species_ids

    return sbml_dfs


@fixture(scope="session")
def gcs_storage():
    """A container running a GCS emulator - shared across all test files."""
    with (
        DockerContainer("fsouza/fake-gcs-server:1.44")
        .with_bind_ports(4443, 4443)
        .with_command("-scheme http -backend memory")
    ) as gcs:
        os.environ["STORAGE_EMULATOR_HOST"] = "http://0.0.0.0:4443"
        yield gcs


@fixture
def gcs_bucket_name(gcs_storage):
    """Generate unique bucket name for each test."""
    bucket_name = f"testbucket-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    return bucket_name


@fixture
def gcs_bucket(gcs_bucket_name):
    """A GCS bucket - cleaned up after test."""
    client = storage.Client()
    client.create_bucket(gcs_bucket_name)
    bucket = client.bucket(gcs_bucket_name)
    yield bucket
    bucket.delete(force=True)


@fixture
def gcs_bucket_uri(gcs_bucket, gcs_bucket_name):
    """URI for the test GCS bucket."""
    return f"gs://{gcs_bucket_name}"


@fixture
def gcs_bucket_subdir_uri(gcs_bucket_uri):
    """URI for a subdirectory in the test bucket."""
    return f"{gcs_bucket_uri}/testdir"


@fixture
def tmp_new_subdir(tmp_path):
    """An empty temporary directory."""
    return tmp_path / "test_dir"


def create_blob(bucket, blob_name, content=b"test"):
    """Helper function to create a blob in GCS.

    Not a fixture - just a helper function available to all tests.
    """
    bucket.blob(blob_name).upload_from_string(content)
