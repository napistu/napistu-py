from __future__ import annotations

import os
import pytest

import numpy as np
import pandas as pd

from napistu import sbml_dfs_core
from napistu.ingestion import sbml
from napistu.network import net_create
from napistu.network import ng_utils
from napistu.constants import MINI_SBO_FROM_NAME
from napistu.constants import SBML_DFS
from napistu.network.constants import DEFAULT_WT_TRANS
from napistu.network.constants import WEIGHTING_SPEC


test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")

sbml_path = os.path.join(test_data, "R-HSA-1237044.sbml")
sbml_model = sbml.SBML(sbml_path)
sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)


@pytest.fixture
def reaction_species_examples(sbml_dfs):
    """
    Pytest fixture providing a dictionary of example reaction species DataFrames for various test cases.
    """
    r_id = sbml_dfs.reactions.index[0]
    d = dict()
    d["valid_interactor"] = pd.DataFrame(
        {
            "r_id": [r_id, r_id],
            "sbo_term": [
                MINI_SBO_FROM_NAME["interactor"],
                MINI_SBO_FROM_NAME["interactor"],
            ],
            "sc_id": ["sc1", "sc2"],
            "stoichiometry": [0, 0],
        }
    ).set_index(["r_id", "sbo_term"])
    d["invalid_interactor"] = pd.DataFrame(
        {
            "r_id": [r_id, r_id],
            "sbo_term": [
                MINI_SBO_FROM_NAME["interactor"],
                MINI_SBO_FROM_NAME["product"],
            ],
            "sc_id": ["sc1", "sc2"],
            "stoichiometry": [0, 0],
        }
    ).set_index(["r_id", "sbo_term"])
    d["sub_and_prod"] = pd.DataFrame(
        {
            "r_id": [r_id, r_id],
            "sbo_term": [MINI_SBO_FROM_NAME["reactant"], MINI_SBO_FROM_NAME["product"]],
            "sc_id": ["sub", "prod"],
            "stoichiometry": [-1, 1],
        }
    ).set_index(["r_id", "sbo_term"])
    d["stimulator"] = pd.DataFrame(
        {
            "r_id": [r_id, r_id, r_id],
            "sbo_term": [
                MINI_SBO_FROM_NAME["reactant"],
                MINI_SBO_FROM_NAME["product"],
                MINI_SBO_FROM_NAME["stimulator"],
            ],
            "sc_id": ["sub", "prod", "stim"],
            "stoichiometry": [-1, 1, 0],
        }
    ).set_index(["r_id", "sbo_term"])
    d["all_entities"] = pd.DataFrame(
        {
            "r_id": [r_id, r_id, r_id, r_id],
            "sbo_term": [
                MINI_SBO_FROM_NAME["reactant"],
                MINI_SBO_FROM_NAME["product"],
                MINI_SBO_FROM_NAME["stimulator"],
                MINI_SBO_FROM_NAME["catalyst"],
            ],
            "sc_id": ["sub", "prod", "stim", "cat"],
            "stoichiometry": [-1, 1, 0, 0],
        }
    ).set_index(["r_id", "sbo_term"])
    d["no_substrate"] = pd.DataFrame(
        {
            "r_id": [r_id, r_id, r_id, r_id, r_id],
            "sbo_term": [
                MINI_SBO_FROM_NAME["product"],
                MINI_SBO_FROM_NAME["stimulator"],
                MINI_SBO_FROM_NAME["stimulator"],
                MINI_SBO_FROM_NAME["inhibitor"],
                MINI_SBO_FROM_NAME["catalyst"],
            ],
            "sc_id": ["prod", "stim1", "stim2", "inh", "cat"],
            "stoichiometry": [1, 0, 0, 0, 0],
        }
    ).set_index(["r_id", "sbo_term"])

    return r_id, d


def test_create_napistu_graph():
    _ = net_create.create_napistu_graph(sbml_dfs, graph_type="bipartite")
    _ = net_create.create_napistu_graph(sbml_dfs, graph_type="regulatory")
    _ = net_create.create_napistu_graph(sbml_dfs, graph_type="surrogate")


def test_create_napistu_graph_edge_reversed():
    """Test that edge_reversed=True properly reverses edges in the graph for all graph types."""
    # Test each graph type
    for graph_type in ["bipartite", "regulatory", "surrogate"]:
        # Create graphs with and without edge reversal
        normal_graph = net_create.create_napistu_graph(
            sbml_dfs, graph_type=graph_type, directed=True, edge_reversed=False
        )
        reversed_graph = net_create.create_napistu_graph(
            sbml_dfs, graph_type=graph_type, directed=True, edge_reversed=True
        )

        # Get edge dataframes for comparison
        normal_edges = normal_graph.get_edge_dataframe()
        reversed_edges = reversed_graph.get_edge_dataframe()

        # Verify we have edges to test
        assert len(normal_edges) > 0, f"No edges found in {graph_type} graph"
        assert len(normal_edges) == len(
            reversed_edges
        ), f"Edge count mismatch in {graph_type} graph"

        # Test edge reversal
        # Check a few edges to verify from/to are swapped
        for i in range(min(5, len(normal_edges))):
            # Check from/to are swapped
            assert (
                normal_edges.iloc[i]["from"] == reversed_edges.iloc[i]["to"]
            ), f"From/to not properly swapped in {graph_type} graph"
            assert (
                normal_edges.iloc[i]["to"] == reversed_edges.iloc[i]["from"]
            ), f"From/to not properly swapped in {graph_type} graph"

            # Check stoichiometry is negated
            assert (
                normal_edges.iloc[i]["stoichiometry"]
                == -reversed_edges.iloc[i]["stoichiometry"]
            ), f"Stoichiometry not properly negated in {graph_type} graph"

            # Check direction attributes are properly swapped
            if normal_edges.iloc[i]["direction"] == "forward":
                assert (
                    reversed_edges.iloc[i]["direction"] == "reverse"
                ), f"Direction not properly reversed (forward->reverse) in {graph_type} graph"
            elif normal_edges.iloc[i]["direction"] == "reverse":
                assert (
                    reversed_edges.iloc[i]["direction"] == "forward"
                ), f"Direction not properly reversed (reverse->forward) in {graph_type} graph"

            # Check parents/children are swapped
            assert (
                normal_edges.iloc[i]["sc_parents"]
                == reversed_edges.iloc[i]["sc_children"]
            ), f"Parents/children not properly swapped in {graph_type} graph"
            assert (
                normal_edges.iloc[i]["sc_children"]
                == reversed_edges.iloc[i]["sc_parents"]
            ), f"Parents/children not properly swapped in {graph_type} graph"


def test_create_napistu_graph_none_attrs():
    # Should not raise when reaction_graph_attrs is None
    _ = net_create.create_napistu_graph(
        sbml_dfs, reaction_graph_attrs=None, graph_type="bipartite"
    )


def test_process_napistu_graph_none_attrs():
    # Should not raise when reaction_graph_attrs is None
    _ = net_create.process_napistu_graph(sbml_dfs, reaction_graph_attrs=None)


@pytest.mark.skip_on_windows
def test_igraph_loading():
    # test read/write of an igraph network
    directeds = [True, False]
    graph_types = ["bipartite", "regulatory"]

    ng_utils.export_networks(
        sbml_dfs,
        model_prefix="tmp",
        outdir="/tmp",
        directeds=directeds,
        graph_types=graph_types,
    )

    for graph_type in graph_types:
        for directed in directeds:
            import_pkl_path = ng_utils._create_network_save_string(
                model_prefix="tmp",
                outdir="/tmp",
                directed=directed,
                graph_type=graph_type,
            )
            network_graph = ng_utils.read_network_pkl(
                model_prefix="tmp",
                network_dir="/tmp",
                directed=directed,
                graph_type=graph_type,
            )

            assert network_graph.is_directed() == directed
            # cleanup
            os.unlink(import_pkl_path)


def test_format_interactors(reaction_species_examples):
    r_id, reaction_species_examples_dict = reaction_species_examples
    # interactions are formatted

    graph_hierarchy_df = net_create._create_graph_hierarchy_df("regulatory")

    assert (
        net_create._format_tiered_reaction_species(
            r_id,
            reaction_species_examples_dict["valid_interactor"],
            sbml_dfs,
            graph_hierarchy_df,
        ).shape[0]
        == 1
    )

    print("Re-enable test once Issue #102 is solved")

    # catch error from invalid interactor specification
    # with pytest.raises(ValueError) as excinfo:
    #    net_create._format_tiered_reaction_species(
    #        r_id, reaction_species_examples_dict["invalid_interactor"], sbml_dfs
    #    )
    # assert str(excinfo.value).startswith("Invalid combinations of SBO_terms")

    # simple reaction with just substrates and products
    assert (
        net_create._format_tiered_reaction_species(
            r_id,
            reaction_species_examples_dict["sub_and_prod"],
            sbml_dfs,
            graph_hierarchy_df,
        ).shape[0]
        == 2
    )

    # add a stimulator (activator)
    rxn_edges = net_create._format_tiered_reaction_species(
        r_id, reaction_species_examples_dict["stimulator"], sbml_dfs, graph_hierarchy_df
    )

    assert rxn_edges.shape[0] == 3
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim", "sub"]

    # add catalyst + stimulator
    rxn_edges = net_create._format_tiered_reaction_species(
        r_id,
        reaction_species_examples_dict["all_entities"],
        sbml_dfs,
        graph_hierarchy_df,
    )

    assert rxn_edges.shape[0] == 4
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim", "cat"]
    assert rxn_edges.iloc[1][["from", "to"]].tolist() == ["cat", "sub"]

    # no substrate
    rxn_edges = net_create._format_tiered_reaction_species(
        r_id,
        reaction_species_examples_dict["no_substrate"],
        sbml_dfs,
        graph_hierarchy_df,
    )

    assert rxn_edges.shape[0] == 5
    # stimulator -> reactant
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim1", "cat"]
    assert rxn_edges.iloc[1][["from", "to"]].tolist() == ["stim2", "cat"]
    assert rxn_edges.iloc[2][["from", "to"]].tolist() == ["inh", "cat"]

    # use the surrogate model tiered layout also

    graph_hierarchy_df = net_create._create_graph_hierarchy_df("surrogate")

    rxn_edges = net_create._format_tiered_reaction_species(
        r_id,
        reaction_species_examples_dict["all_entities"],
        sbml_dfs,
        graph_hierarchy_df,
    )

    assert rxn_edges.shape[0] == 4
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim", "sub"]
    assert rxn_edges.iloc[1][["from", "to"]].tolist() == ["sub", "cat"]


def test_reverse_network_edges(reaction_species_examples):
    r_id, reaction_species_examples_dict = reaction_species_examples

    graph_hierarchy_df = net_create._create_graph_hierarchy_df("regulatory")

    rxn_edges = net_create._format_tiered_reaction_species(
        r_id,
        reaction_species_examples_dict["all_entities"],
        sbml_dfs,
        graph_hierarchy_df,
    )
    augmented_network_edges = rxn_edges.assign(r_isreversible=True)
    augmented_network_edges["sc_parents"] = range(0, augmented_network_edges.shape[0])
    augmented_network_edges["sc_children"] = range(
        augmented_network_edges.shape[0], 0, -1
    )

    assert net_create._reverse_network_edges(augmented_network_edges).shape[0] == 2


def test_entity_validation():
    # Test basic validation
    entity_attrs = {"table": "reactions", "variable": "foo"}
    assert net_create._EntityAttrValidator(**entity_attrs).model_dump() == {
        **entity_attrs,
        **{"trans": DEFAULT_WT_TRANS},
    }

    # Test validation with custom transformations
    custom_transformations = {
        "nlog10": lambda x: -np.log10(x),
        "square": lambda x: x**2,
    }

    # Test valid custom transformation
    entity_attrs_custom = {
        "attr1": {
            WEIGHTING_SPEC.TABLE: "reactions",
            WEIGHTING_SPEC.VARIABLE: "foo",
            WEIGHTING_SPEC.TRANSFORMATION: "nlog10",
        },
        "attr2": {
            WEIGHTING_SPEC.TABLE: "species",
            WEIGHTING_SPEC.VARIABLE: "bar",
            WEIGHTING_SPEC.TRANSFORMATION: "square",
        },
    }
    # Should not raise any errors
    net_create._validate_entity_attrs(
        entity_attrs_custom, custom_transformations=custom_transformations
    )

    # Test invalid transformation
    entity_attrs_invalid = {
        "attr1": {
            WEIGHTING_SPEC.TABLE: "reactions",
            WEIGHTING_SPEC.VARIABLE: "foo",
            WEIGHTING_SPEC.TRANSFORMATION: "invalid_trans",
        }
    }
    with pytest.raises(ValueError) as excinfo:
        net_create._validate_entity_attrs(
            entity_attrs_invalid, custom_transformations=custom_transformations
        )
    assert "transformation 'invalid_trans' was not defined" in str(excinfo.value)

    # Test with validate_transformations=False
    # Should not raise any errors even with invalid transformation
    net_create._validate_entity_attrs(
        entity_attrs_invalid, validate_transformations=False
    )

    # Test with non-dict input
    with pytest.raises(AssertionError) as excinfo:
        net_create._validate_entity_attrs(["not", "a", "dict"])
    assert "entity_attrs must be a dictionary" in str(excinfo.value)


def test_pluck_entity_data_species_identity(sbml_dfs):
    # Take first 10 species IDs
    species_ids = sbml_dfs.species.index[:10]
    # Create mock data with explicit dtype to ensure cross-platform consistency
    # Fix for issue-42: Use explicit dtypes to avoid platform-specific dtype differences
    # between Windows (int32) and macOS/Linux (int64)
    mock_df = pd.DataFrame(
        {
            "string_col": [f"str_{i}" for i in range(10)],
            "mixed_col": np.arange(-5, 5, dtype=np.int64),  # Explicitly use int64
            "ones_col": np.ones(10, dtype=np.float64),  # Explicitly use float64
            "squared_col": np.arange(10, dtype=np.int64),  # Explicitly use int64
        },
        index=species_ids,
    )
    # Assign to species_data
    sbml_dfs.species_data["mock_table"] = mock_df

    # Custom transformation: square
    def square(x):
        return x**2

    custom_transformations = {"square": square}
    # Create graph_attrs for species
    graph_attrs = {
        "species": {
            "string_col": {
                WEIGHTING_SPEC.TABLE: "mock_table",
                WEIGHTING_SPEC.VARIABLE: "string_col",
                WEIGHTING_SPEC.TRANSFORMATION: "identity",
            },
            "mixed_col": {
                WEIGHTING_SPEC.TABLE: "mock_table",
                WEIGHTING_SPEC.VARIABLE: "mixed_col",
                WEIGHTING_SPEC.TRANSFORMATION: "identity",
            },
            "ones_col": {
                WEIGHTING_SPEC.TABLE: "mock_table",
                WEIGHTING_SPEC.VARIABLE: "ones_col",
                WEIGHTING_SPEC.TRANSFORMATION: "identity",
            },
            "squared_col": {
                WEIGHTING_SPEC.TABLE: "mock_table",
                WEIGHTING_SPEC.VARIABLE: "squared_col",
                WEIGHTING_SPEC.TRANSFORMATION: "square",
            },
        }
    }
    # Call pluck_entity_data with custom transformation
    result = net_create.pluck_entity_data(
        sbml_dfs, graph_attrs, "species", custom_transformations=custom_transformations
    )
    # Check output
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"string_col", "mixed_col", "ones_col", "squared_col"}
    assert list(result.index) == list(species_ids)
    # Check values
    pd.testing.assert_series_equal(result["string_col"], mock_df["string_col"])
    pd.testing.assert_series_equal(result["mixed_col"], mock_df["mixed_col"])
    pd.testing.assert_series_equal(result["ones_col"], mock_df["ones_col"])
    pd.testing.assert_series_equal(
        result["squared_col"], mock_df["squared_col"].apply(square)
    )


def test_pluck_entity_data_missing_species_key(sbml_dfs):
    # graph_attrs does not contain 'species' key
    graph_attrs = {}
    result = net_create.pluck_entity_data(sbml_dfs, graph_attrs, SBML_DFS.SPECIES)
    assert result is None


def test_pluck_entity_data_empty_species_dict(sbml_dfs):
    # graph_attrs contains 'species' key but value is empty dict
    graph_attrs = {SBML_DFS.SPECIES: {}}
    result = net_create.pluck_entity_data(sbml_dfs, graph_attrs, SBML_DFS.SPECIES)
    assert result is None
