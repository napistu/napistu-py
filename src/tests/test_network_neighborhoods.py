import pandas as pd

from napistu.network import ng_utils
from napistu.network import neighborhoods
from napistu import source

from napistu.constants import SBML_DFS
from napistu.network.constants import NEIGHBORHOOD_NETWORK_TYPES


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

    assert neighborhood["species_73473"]["vertices"].shape[0] == 6


def test_find_and_prune_neighborhoods_with_source_counts(
    sbml_dfs_metabolism, napistu_graph_metabolism
):
    """
    Test find_and_prune_neighborhoods function with source_total_counts parameter.

    This test verifies that the function works correctly when source_total_counts
    is provided, which enables source-based edge assignment in neighborhoods.
    """
    # Create source_total_counts using the source module
    source_total_counts = source.get_source_total_counts(
        sbml_dfs_metabolism, SBML_DFS.REACTIONS
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
        assert "graph" in neighborhood
        assert "vertices" in neighborhood
        assert "edges" in neighborhood
        assert "edge_sources" in neighborhood

        # Verify edge_sources is populated when source_total_counts is provided
        # (this is the key difference when source_total_counts is passed)
        if neighborhood["edges"].shape[0] > 0:
            # If there are edges, edge_sources should be populated
            assert neighborhood["edge_sources"] is not None
            assert isinstance(neighborhood["edge_sources"], pd.DataFrame)

            # Check edge_sources has expected columns
            expected_columns = [SBML_DFS.R_ID, "pathway_id", "name"]
            for col in expected_columns:
                assert col in neighborhood["edge_sources"].columns

        # Verify vertices structure
        vertices = neighborhood["vertices"]
        assert isinstance(vertices, pd.DataFrame)
        assert vertices.shape[0] > 0

        # Verify edges structure
        edges = neighborhood["edges"]
        assert isinstance(edges, pd.DataFrame)

        # Verify graph structure
        graph = neighborhood["graph"]
        assert hasattr(graph, "vcount")
        assert hasattr(graph, "ecount")

    # Test without source_total_counts for comparison
    neighborhoods_result_no_source = neighborhoods.find_and_prune_neighborhoods(
        sbml_dfs=sbml_dfs_metabolism,
        napistu_graph=napistu_graph_metabolism,
        compartmentalized_species=compartmentalized_species,
        source_total_counts=None,  # No source counts
        network_type=NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
        order=3,
        verbose=False,
        top_n=10,
    )

    # Verify both results have the same basic structure
    assert len(neighborhoods_result) == len(neighborhoods_result_no_source)

    # The main difference should be in edge_sources handling
    for sc_id in neighborhoods_result:
        with_source = neighborhoods_result[sc_id]["edge_sources"]
        without_source = neighborhoods_result_no_source[sc_id]["edge_sources"]

        # Both should either be None or DataFrames, but the content may differ
        assert (with_source is None) == (without_source is None)
        if with_source is not None and without_source is not None:
            assert isinstance(with_source, pd.DataFrame)
            assert isinstance(without_source, pd.DataFrame)
