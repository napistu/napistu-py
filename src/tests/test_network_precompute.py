from __future__ import annotations

import os
import tempfile

import igraph as ig
import numpy as np
import pandas as pd
import pytest

from napistu import utils
from napistu.network import neighborhoods, paths, precompute
from napistu.network.constants import (
    DISTANCES,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_VERTICES,
    NEIGHBORHOOD_DICT_KEYS,
    NEIGHBORHOOD_NETWORK_TYPES,
    SBML_DFS,
)

# number of species to include when finding all x all paths
N_SPECIES = 12

# setting for neighborhoods
NETWORK_TYPE = "hourglass"
ORDER = 20
TOP_N = 20


def test_precomputed_distances(precomputed_distances_metabolism):
    assert precomputed_distances_metabolism.shape == (10243, 5)


def test_precomputed_distances_shortest_paths(
    sbml_dfs_metabolism, napistu_graph_metabolism, precomputed_distances_metabolism
):

    sbml_dfs = sbml_dfs_metabolism
    napistu_graph = napistu_graph_metabolism
    precomputed_distances = precomputed_distances_metabolism

    cspecies_subset = sbml_dfs.compartmentalized_species.index.tolist()[0:N_SPECIES]

    # we should get the same answer for shortest paths whether or not we use pre-computed distances
    all_species_pairs = pd.DataFrame(
        np.array([(x, y) for x in cspecies_subset for y in cspecies_subset]),
        columns=[DISTANCES.SC_ID_ORIGIN, DISTANCES.SC_ID_DEST],
    )

    (
        path_vertices,
        _,
        _,
        _,
    ) = paths.find_all_shortest_reaction_paths(
        napistu_graph, sbml_dfs, all_species_pairs
    )

    shortest_path_weights = (
        path_vertices.groupby(["origin", "dest", "path"])[NAPISTU_GRAPH_EDGES.WEIGHT]
        .sum()
        .reset_index()
        .sort_values(NAPISTU_GRAPH_EDGES.WEIGHT)
        .groupby(["origin", "dest"])
        .first()
        .reset_index()
    )

    precomputed_distance_subset_mask = [
        True if x and y else False
        for x, y in zip(
            precomputed_distances[DISTANCES.SC_ID_ORIGIN]
            .isin(cspecies_subset)
            .tolist(),
            precomputed_distances[DISTANCES.SC_ID_DEST].isin(cspecies_subset).tolist(),
        )
    ]
    precomputed_distance_subset = precomputed_distances[
        precomputed_distance_subset_mask
    ]

    path_method_comparison_full_merge = shortest_path_weights.merge(
        precomputed_distance_subset,
        left_on=["origin", "dest"],
        right_on=[DISTANCES.SC_ID_ORIGIN, DISTANCES.SC_ID_DEST],
        how="outer",
    )

    # tables have identical pairs with a valid path
    assert (
        path_method_comparison_full_merge.shape[0]
        == precomputed_distance_subset.shape[0]
    )
    assert path_method_comparison_full_merge.shape[0] == shortest_path_weights.shape[0]
    assert all(
        abs(
            path_method_comparison_full_merge[NAPISTU_GRAPH_EDGES.WEIGHT]
            - path_method_comparison_full_merge[DISTANCES.PATH_WEIGHT]
        )
        < 1e-13
    )

    # using the precomputed distances generates the same result as excluding it
    precompute_path_vertices, _, _, _ = paths.find_all_shortest_reaction_paths(
        napistu_graph,
        sbml_dfs,
        all_species_pairs,
        precomputed_distances=precomputed_distances,
    )

    precompute_shortest_path_weights = (
        precompute_path_vertices.groupby(["origin", "dest", "path"])[
            NAPISTU_GRAPH_EDGES.WEIGHT
        ]
        .sum()
        .reset_index()
        .sort_values(NAPISTU_GRAPH_EDGES.WEIGHT)
        .groupby(["origin", "dest"])
        .first()
        .reset_index()
    )

    precompute_full_merge = shortest_path_weights.merge(
        precompute_shortest_path_weights,
        left_on=["origin", "dest", "path"],
        right_on=["origin", "dest", "path"],
        how="outer",
    )

    assert precompute_full_merge.shape[0] == precompute_shortest_path_weights.shape[0]
    assert precompute_full_merge.shape[0] == shortest_path_weights.shape[0]
    assert all(
        abs(precompute_full_merge["weight_x"] - precompute_full_merge["weight_y"])
        < 1e-13
    )


def test_precomputed_distances_neighborhoods(
    sbml_dfs_metabolism, napistu_graph_metabolism, precomputed_distances_metabolism
):

    sbml_dfs = sbml_dfs_metabolism
    napistu_graph = napistu_graph_metabolism
    precomputed_distances = precomputed_distances_metabolism

    compartmentalized_species = sbml_dfs.compartmentalized_species[
        sbml_dfs.compartmentalized_species[SBML_DFS.S_ID] == "S00000000"
    ].index.tolist()

    pruned_neighborhoods_precomputed = neighborhoods.find_and_prune_neighborhoods(
        sbml_dfs,
        napistu_graph,
        compartmentalized_species,
        precomputed_distances=precomputed_distances,
        network_type=NETWORK_TYPE,
        order=ORDER,
        verbose=True,
        top_n=TOP_N,
    )

    pruned_neighborhoods_otf = neighborhoods.find_and_prune_neighborhoods(
        sbml_dfs,
        napistu_graph,
        compartmentalized_species,
        precomputed_distances=None,
        network_type=NETWORK_TYPE,
        order=ORDER,
        verbose=True,
        top_n=TOP_N,
    )

    comparison_l = list()
    for key in pruned_neighborhoods_precomputed.keys():
        pruned_vert_otf = pruned_neighborhoods_otf[key][NEIGHBORHOOD_DICT_KEYS.VERTICES]
        pruned_vert_precomp = pruned_neighborhoods_precomputed[key][
            NEIGHBORHOOD_DICT_KEYS.VERTICES
        ]

        join_key = [
            NAPISTU_GRAPH_VERTICES.NAME,
            NAPISTU_GRAPH_VERTICES.NODE_NAME,
            "node_orientation",
        ]
        join_key_w_vars = [*join_key, *[DISTANCES.PATH_WEIGHT, DISTANCES.PATH_LENGTH]]
        neighbor_comparison = (
            pruned_vert_precomp[join_key_w_vars]
            .assign(in_precompute=True)
            .merge(
                pruned_vert_otf[join_key_w_vars].assign(in_otf=True),
                left_on=join_key,
                right_on=join_key,
                how="outer",
            )
        )
        for col in ["in_precompute", "in_otf"]:
            neighbor_comparison[col] = (
                neighbor_comparison[col].astype("boolean").fillna(False)
            )
        comparison_l.append(neighbor_comparison.assign(focal_sc_id=key))

    comparison_df = pd.concat(comparison_l)
    comparison_df_disagreements = comparison_df.query("in_precompute != in_otf")

    # pruned neighborhoods are identical with and without using precalculated neighbors
    assert comparison_df_disagreements.shape[0] == 0

    # compare shortest paths calculated through neighborhoods with precomputed distances
    # which should be the same if we are pre-selecting the correct neighbors
    # as part of _precompute_neighbors()
    downstream_disagreement_w_precompute = (
        comparison_df[
            comparison_df["node_orientation"] == NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM
        ]
        .merge(
            precomputed_distances,
            left_on=["focal_sc_id", NAPISTU_GRAPH_VERTICES.NAME],
            right_on=[DISTANCES.SC_ID_ORIGIN, DISTANCES.SC_ID_DEST],
        )
        .query("abs(path_weight_x - path_weight) > 1e-13")
    )

    upstream_disagreement_w_precompute = (
        comparison_df[
            comparison_df["node_orientation"] == NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM
        ]
        .merge(
            precomputed_distances,
            left_on=["focal_sc_id", NAPISTU_GRAPH_VERTICES.NAME],
            right_on=[DISTANCES.SC_ID_DEST, DISTANCES.SC_ID_ORIGIN],
        )
        .query("abs(path_weight_x - path_weight_upstream) > 1e-13")
    )

    assert downstream_disagreement_w_precompute.shape[0] == 0
    assert upstream_disagreement_w_precompute.shape[0] == 0


@pytest.mark.skip_on_windows
def test_precomputed_distances_serialization():
    """
    Test that validates the serialization -> deserialization approach works correctly.

    Notes
    -----
    This function creates a sample DataFrame with the structure of precomputed
    distances data, saves it to a temporary JSON file, loads it back, and
    validates that all data is preserved correctly through the serialization
    round-trip.
    """
    # Create a sample DataFrame that mimics the precomputed distances structure
    sample_data = {
        DISTANCES.SC_ID_ORIGIN: {
            1: "SC00000000",
            3: "SC00000003",
            4: "SC00000004",
            5: "SC00000005",
            6: "SC00000011",
        },
        DISTANCES.SC_ID_DEST: {
            1: "SC00000001",
            3: "SC00000001",
            4: "SC00000001",
            5: "SC00000001",
            6: "SC00000001",
        },
        DISTANCES.PATH_LENGTH: {1: 1.0, 3: 4.0, 4: 6.0, 5: 6.0, 6: 1.0},
        DISTANCES.PATH_WEIGHT_UPSTREAM: {1: 1.0, 3: 4.0, 4: 6.0, 5: 6.0, 6: 1.0},
        DISTANCES.PATH_WEIGHT: {1: 1.0, 3: 4.0, 4: 6.0, 5: 6.0, 6: 1.0},
    }

    # Create original DataFrame
    original_df = pd.DataFrame(sample_data)

    # Create a temporary file path
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        temp_path = tmp_file.name

    try:
        # Test serialization
        utils.save_parquet(original_df, temp_path)

        # Test deserialization
        loaded_df = utils.load_parquet(temp_path)

        # Validate that the loaded DataFrame is identical to the original
        pd.testing.assert_frame_equal(original_df, loaded_df, check_like=True)

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_filter_precomputed_distances_top_n_subset(precomputed_distances_metabolism):

    precomputed_distances = precomputed_distances_metabolism
    # Use a small top_n for a quick test
    top_n = 5
    filtered = precompute.filter_precomputed_distances_top_n(
        precomputed_distances, top_n=top_n
    )
    # Check that the filtered DataFrame is a subset of the original
    merged = filtered.merge(
        precomputed_distances,
        on=[
            precompute.NAPISTU_EDGELIST.SC_ID_ORIGIN,
            precompute.NAPISTU_EDGELIST.SC_ID_DEST,
        ],
        how="left",
        indicator=True,
    )
    assert (
        merged["_merge"] == "both"
    ).all(), "Filtered rows must be present in the original DataFrame"
    # Check that columns are preserved
    assert set(
        [
            precompute.NAPISTU_EDGELIST.SC_ID_ORIGIN,
            precompute.NAPISTU_EDGELIST.SC_ID_DEST,
        ]
    ).issubset(filtered.columns)
    # Optionally, check that the number of rows is less than or equal to the input
    assert filtered.shape[0] <= precomputed_distances.shape[0]


def test_find_unique_weight_vars():
    """Test the _find_unique_weight_vars function with different scenarios."""

    # Create a simple test graph
    g = ig.Graph(directed=True)
    g.add_vertices(4)
    g.add_edges([(0, 1), (1, 2), (2, 3), (0, 3)])

    # Test Case 1: All weight variables are different
    g.es[NAPISTU_GRAPH_EDGES.WEIGHT] = [1.0, 2.0, 3.0, 4.0]
    g.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] = [1.1, 2.1, 3.1, 4.1]
    g.es["custom_weight"] = [1.5, 2.5, 3.5, 4.5]

    weight_vars = [
        NAPISTU_GRAPH_EDGES.WEIGHT,
        NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
        "custom_weight",
    ]
    unique_map, representatives = precompute._find_unique_weight_vars(g, weight_vars)

    assert len(representatives) == 3
    assert unique_map == {
        NAPISTU_GRAPH_EDGES.WEIGHT: NAPISTU_GRAPH_EDGES.WEIGHT,
        NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM: NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
        "custom_weight": "custom_weight",
    }

    # Test Case 2: weight and upstream_weight are identical
    g.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] = [1.0, 2.0, 3.0, 4.0]  # Same as weight

    unique_map, representatives = precompute._find_unique_weight_vars(
        g, [NAPISTU_GRAPH_EDGES.WEIGHT, NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM]
    )

    assert len(representatives) == 1
    assert (
        unique_map[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] == NAPISTU_GRAPH_EDGES.WEIGHT
    )  # upstream_weight should map to weight
    assert NAPISTU_GRAPH_EDGES.WEIGHT in representatives
    assert set(representatives[NAPISTU_GRAPH_EDGES.WEIGHT]) == {
        NAPISTU_GRAPH_EDGES.WEIGHT,
        NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
    }

    # Test Case 3: All three weights are identical
    g.es["custom_weight"] = [1.0, 2.0, 3.0, 4.0]  # Same as weight

    unique_map, representatives = precompute._find_unique_weight_vars(
        g,
        [
            NAPISTU_GRAPH_EDGES.WEIGHT,
            NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
            "custom_weight",
        ],
    )

    assert len(representatives) == 1
    assert NAPISTU_GRAPH_EDGES.WEIGHT in representatives
    assert set(representatives[NAPISTU_GRAPH_EDGES.WEIGHT]) == {
        NAPISTU_GRAPH_EDGES.WEIGHT,
        NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
        "custom_weight",
    }

    # Test Case 4: Single weight variable
    unique_map, representatives = precompute._find_unique_weight_vars(
        g, [NAPISTU_GRAPH_EDGES.WEIGHT]
    )

    assert unique_map == {NAPISTU_GRAPH_EDGES.WEIGHT: NAPISTU_GRAPH_EDGES.WEIGHT}
    assert representatives == {NAPISTU_GRAPH_EDGES.WEIGHT: [NAPISTU_GRAPH_EDGES.WEIGHT]}

    # Test Case 5: Empty list should raise an exception
    with pytest.raises(ValueError, match="weight_vars cannot be empty"):
        precompute._find_unique_weight_vars(g, [])


def test_precompute_distances_max_score_q_filtering(napistu_graph_metabolism):
    """Test that max_score_q properly filters distances by quantile."""

    distances_full = precompute.precompute_distances(
        napistu_graph_metabolism, max_steps=100, max_score_q=1.0
    )
    distances_half = precompute.precompute_distances(
        napistu_graph_metabolism, max_steps=100, max_score_q=0.5
    )

    # More aggressive filtering should result in fewer rows
    assert distances_half.shape[0] <= distances_full.shape[0]

    # Max weights should be lower with more aggressive filtering
    if distances_half.shape[0] > 0:
        max_weight_full = distances_full[DISTANCES.PATH_WEIGHT].max()
        max_weight_half = distances_half[DISTANCES.PATH_WEIGHT].max()
        assert max_weight_half <= max_weight_full


def test_precompute_distances_max_score_q_edge_cases(napistu_graph_metabolism):
    """Test that max_score_q validates input range."""

    with pytest.raises(ValueError, match="max_score_q must be between 0 and 1"):
        precompute.precompute_distances(
            napistu_graph_metabolism, max_steps=100, max_score_q=0.0
        )

    with pytest.raises(ValueError, match="max_score_q must be between 0 and 1"):
        precompute.precompute_distances(
            napistu_graph_metabolism, max_steps=100, max_score_q=1.1
        )


def test_filter_precomputed_distances_masking_logic():
    """Test that _filter_precomputed_distances uses proper masking instead of NaN values."""

    sample_data = pd.DataFrame(
        {
            DISTANCES.SC_ID_ORIGIN: ["A", "B", "C", "D", "E"],
            DISTANCES.SC_ID_DEST: ["X", "Y", "Z", "W", "V"],
            DISTANCES.PATH_LENGTH: [1, 2, 3, 4, 5],
            DISTANCES.PATH_WEIGHT: [1.0, 2.0, 3.0, 4.0, 5.0],
            DISTANCES.PATH_WEIGHT_UPSTREAM: [1.1, 2.1, 3.1, 4.1, 5.1],
        }
    )

    # Test filtering removes rows instead of setting values to NaN
    filtered = precompute._filter_precomputed_distances(
        sample_data,
        max_score_q=0.6,
        path_weight_vars=[DISTANCES.PATH_WEIGHT, DISTANCES.PATH_WEIGHT_UPSTREAM],
    )

    assert filtered.shape[0] <= sample_data.shape[0]
    assert not filtered.isna().any().any()  # No NaN values in output

    # Test max_score_q=1.0 keeps all rows
    filtered_all = precompute._filter_precomputed_distances(
        sample_data,
        max_score_q=1.0,
        path_weight_vars=[DISTANCES.PATH_WEIGHT, DISTANCES.PATH_WEIGHT_UPSTREAM],
    )
    assert filtered_all.shape[0] == sample_data.shape[0]
