import pandas as pd
import pytest

from napistu.constants import SBML_DFS
from napistu.network import neighborhoods, ng_utils
from napistu.network.constants import (
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
    NEIGHBORHOOD_DICT_KEYS,
    NEIGHBORHOOD_NETWORK_TYPES,
)


def test_neighborhood(sbml_dfs, napistu_graph):
    species = sbml_dfs.species
    source_species = species[species[SBML_DFS.S_NAME] == "NADH"].index.tolist()

    query_sc_species = ng_utils.compartmentalize_species(sbml_dfs, source_species)
    compartmentalized_species = query_sc_species[SBML_DFS.SC_ID].tolist()

    neighborhood = neighborhoods.find_neighborhoods(
        sbml_dfs,
        napistu_graph,
        compartmentalized_species=compartmentalized_species,
        order=3,
    )

    assert neighborhood["species_73473"][NEIGHBORHOOD_DICT_KEYS.VERTICES].shape[0] == 6


def test_find_and_prune_neighborhoods_with_source_counts(
    sbml_dfs_metabolism, napistu_graph_metabolism
):
    """
    Test find_and_prune_neighborhoods function with source_total_counts parameter.

    This test verifies that the function works correctly when source_total_counts
    is provided, which enables source-based reaction assignment in neighborhoods.
    """
    # Create source_total_counts using the source module
    source_total_counts = sbml_dfs_metabolism.get_source_total_counts(
        SBML_DFS.REACTIONS
    )

    # Verify source_total_counts is created correctly
    assert isinstance(source_total_counts, pd.Series)
    assert len(source_total_counts) > 0
    assert source_total_counts.name == "total_counts"
    assert all(source_total_counts > 0)

    # Get a test species to create neighborhood around
    species = sbml_dfs_metabolism.species
    source_species = species[species[SBML_DFS.S_NAME] == "NADH"].index.tolist()

    query_sc_species = ng_utils.compartmentalize_species(
        sbml_dfs_metabolism, source_species
    )
    compartmentalized_species = query_sc_species[SBML_DFS.SC_ID].tolist()

    # Test find_and_prune_neighborhoods with source_total_counts
    neighborhoods_result = neighborhoods.find_and_prune_neighborhoods(
        sbml_dfs=sbml_dfs_metabolism,
        napistu_graph=napistu_graph_metabolism,
        compartmentalized_species=compartmentalized_species,
        min_pw_size=1,
        source_total_counts=source_total_counts,
        network_type=NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
        order=3,
        verbose=False,
        top_n=10,
    )

    # Verify the result structure
    assert isinstance(neighborhoods_result, dict)
    assert len(neighborhoods_result) > 0

    # Check each neighborhood has the expected structure
    for sc_id, neighborhood in neighborhoods_result.items():
        assert isinstance(neighborhood, dict)
        assert NEIGHBORHOOD_DICT_KEYS.GRAPH in neighborhood
        assert NEIGHBORHOOD_DICT_KEYS.VERTICES in neighborhood
        assert NEIGHBORHOOD_DICT_KEYS.EDGES in neighborhood
        assert NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES in neighborhood

        # Verify reaction_sources is populated when source_total_counts is provided
        # (this is the key difference when source_total_counts is passed)
        if neighborhood[NEIGHBORHOOD_DICT_KEYS.EDGES].shape[0] > 0:
            # If there are edges, reaction_sources should be populated
            assert neighborhood[NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES] is not None
            assert isinstance(
                neighborhood[NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES], pd.DataFrame
            )
            assert neighborhood[NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES].shape[0] > 0

            # Check reaction_sources has expected columns
            expected_columns = [SBML_DFS.R_ID, "pathway_id", "name"]
            for col in expected_columns:
                assert (
                    col in neighborhood[NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES].columns
                )

        # Verify vertices structure
        vertices = neighborhood[NEIGHBORHOOD_DICT_KEYS.VERTICES]
        assert isinstance(vertices, pd.DataFrame)
        assert vertices.shape[0] > 0

        # Verify edges structure
        edges = neighborhood[NEIGHBORHOOD_DICT_KEYS.EDGES]
        assert isinstance(edges, pd.DataFrame)

        # Verify graph structure
        graph = neighborhood[NEIGHBORHOOD_DICT_KEYS.GRAPH]
        assert hasattr(graph, "vcount")
        assert hasattr(graph, "ecount")

    # Test without source_total_counts for comparison
    neighborhoods_result_no_source = neighborhoods.find_and_prune_neighborhoods(
        sbml_dfs=sbml_dfs_metabolism,
        napistu_graph=napistu_graph_metabolism,
        compartmentalized_species=compartmentalized_species,
        source_total_counts=None,  # No source counts
        min_pw_size=1,
        network_type=NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
        order=3,
        verbose=False,
        top_n=10,
    )

    # Verify both results have the same basic structure
    assert len(neighborhoods_result) == len(neighborhoods_result_no_source)

    # The main difference should be in reaction_sources handling
    for sc_id in neighborhoods_result:
        with_source = neighborhoods_result[sc_id][
            NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES
        ]
        without_source = neighborhoods_result_no_source[sc_id][
            NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES
        ]

        # Both should either be None or DataFrames, but the content may differ
        assert (with_source is None) == (without_source is None)
        if with_source is not None and without_source is not None:
            assert isinstance(with_source, pd.DataFrame)
            assert isinstance(without_source, pd.DataFrame)

    # Test error handling for invalid parameters
    # Test invalid network_type
    with pytest.raises(ValueError):
        neighborhoods.find_and_prune_neighborhoods(
            sbml_dfs=sbml_dfs_metabolism,
            napistu_graph=napistu_graph_metabolism,
            compartmentalized_species=compartmentalized_species,
            source_total_counts=source_total_counts,
            min_pw_size=1,
            network_type="invalid_network_type",
            order=3,
            verbose=False,
            top_n=10,
        )

    # Test invalid order (negative)
    with pytest.raises(ValueError):
        neighborhoods.find_and_prune_neighborhoods(
            sbml_dfs=sbml_dfs_metabolism,
            napistu_graph=napistu_graph_metabolism,
            compartmentalized_species=compartmentalized_species,
            source_total_counts=source_total_counts,
            min_pw_size=1,
            network_type=NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
            order=-1,
            verbose=False,
            top_n=10,
        )


def test_add_vertices_uri_urls(sbml_dfs):
    """
    Test add_vertices_uri_urls function.

    This test verifies that the function correctly adds URI URLs to vertices
    DataFrame for both species and reactions.
    """

    # Get real species and reaction names from the sbml_dfs fixture
    real_species = sbml_dfs.compartmentalized_species.index[0]  # Get first species
    real_reaction = sbml_dfs.reactions.index[0]  # Get first reaction

    # Create a test vertices DataFrame with real species and reactions
    test_vertices = pd.DataFrame(
        {
            NAPISTU_GRAPH_VERTICES.NAME: [real_species, real_reaction],
            NAPISTU_GRAPH_VERTICES.NODE_TYPE: [
                NAPISTU_GRAPH_NODE_TYPES.SPECIES,
                NAPISTU_GRAPH_NODE_TYPES.REACTION,
            ],
        }
    )

    # Test basic functionality
    result = neighborhoods.add_vertices_uri_urls(test_vertices, sbml_dfs)

    # Verify basic structure
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == test_vertices.shape[0]  # Same number of rows
    assert result.shape[1] >= test_vertices.shape[1]  # At least same number of columns

    # Verify original columns are preserved
    for col in test_vertices.columns:
        assert col in result.columns
        assert all(result[col] == test_vertices[col])

    # Verify species vertices have s_id column
    species_vertices = result[
        result[NAPISTU_GRAPH_VERTICES.NODE_TYPE] == NAPISTU_GRAPH_NODE_TYPES.SPECIES
    ]
    assert SBML_DFS.S_ID in species_vertices.columns

    # Test error handling
    import pytest

    # Test with empty DataFrame
    empty_vertices = pd.DataFrame(
        columns=[NAPISTU_GRAPH_VERTICES.NAME, NAPISTU_GRAPH_VERTICES.NODE_TYPE]
    )
    with pytest.raises(ValueError, match="vertices must have at least one row"):
        neighborhoods.add_vertices_uri_urls(empty_vertices, sbml_dfs)
