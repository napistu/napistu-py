from __future__ import annotations

import pytest
import os
import tempfile

import numpy as np
import pandas as pd
from napistu import sbml_dfs_core
from napistu import utils
from napistu.ingestion import sbml
from napistu.network import neighborhoods
from napistu.network import net_create
from napistu.network import paths
from napistu.network import precompute

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
sbml_path = os.path.join(test_path, "test_data", "reactome_glucose_metabolism.sbml")
if not os.path.isfile(sbml_path):
    raise ValueError(f"{sbml_path} not found")

sbml_model = sbml.SBML(sbml_path)
sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)
sbml_dfs.validate()

napistu_graph = net_create.process_napistu_graph(
    sbml_dfs, wiring_approach="bipartite", directed=True, weighting_strategy="topology"
)

# number of species to include when finding all x all paths
N_SPECIES = 12

# setting for neighborhoods
NETWORK_TYPE = "hourglass"
ORDER = 20
TOP_N = 20

precomputed_distances = precompute.precompute_distances(
    napistu_graph, max_steps=30000, max_score_q=1
)


def test_precomputed_distances():
    assert precomputed_distances.shape == (3934, 5)


def test_precomputed_distances_shortest_paths():
    cspecies_subset = sbml_dfs.compartmentalized_species.index.tolist()[0:N_SPECIES]

    # we should get the same answer for shortest paths whether or not we use pre-computed distances
    all_species_pairs = pd.DataFrame(
        np.array([(x, y) for x in cspecies_subset for y in cspecies_subset]),
        columns=["sc_id_origin", "sc_id_dest"],
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
        path_vertices.groupby(["origin", "dest", "path"])["weights"]
        .sum()
        .reset_index()
        .sort_values("weights")
        .groupby(["origin", "dest"])
        .first()
        .reset_index()
    )

    precomputed_distance_subset_mask = [
        True if x and y else False
        for x, y in zip(
            precomputed_distances["sc_id_origin"].isin(cspecies_subset).tolist(),
            precomputed_distances["sc_id_dest"].isin(cspecies_subset).tolist(),
        )
    ]
    precomputed_distance_subset = precomputed_distances[
        precomputed_distance_subset_mask
    ]

    path_method_comparison_full_merge = shortest_path_weights.merge(
        precomputed_distance_subset,
        left_on=["origin", "dest"],
        right_on=["sc_id_origin", "sc_id_dest"],
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
            path_method_comparison_full_merge["weights"]
            - path_method_comparison_full_merge["path_weights"]
        )
        < 1e-13
    )

    # using the precomputed distances generates the same result as excluding it
    (precompute_path_vertices, _, _, _) = paths.find_all_shortest_reaction_paths(
        napistu_graph,
        sbml_dfs,
        all_species_pairs,
        precomputed_distances=precomputed_distances,
    )

    precompute_shortest_path_weights = (
        precompute_path_vertices.groupby(["origin", "dest", "path"])["weights"]
        .sum()
        .reset_index()
        .sort_values("weights")
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
        abs(precompute_full_merge["weights_x"] - precompute_full_merge["weights_y"])
        < 1e-13
    )


def test_precomputed_distances_neighborhoods():
    compartmentalized_species = sbml_dfs.compartmentalized_species[
        sbml_dfs.compartmentalized_species["s_id"] == "S00000000"
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
        pruned_vert_otf = pruned_neighborhoods_otf[key]["vertices"]
        pruned_vert_precomp = pruned_neighborhoods_precomputed[key]["vertices"]

        join_key = ["name", "node_name", "node_orientation"]
        join_key_w_vars = [*join_key, *["path_weight", "path_length"]]
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
        comparison_df[comparison_df["node_orientation"] == "downstream"]
        .merge(
            precomputed_distances,
            left_on=["focal_sc_id", "name"],
            right_on=["sc_id_origin", "sc_id_dest"],
        )
        .query("abs(path_weight_x - path_weights) > 1e-13")
    )

    upstream_disagreement_w_precompute = (
        comparison_df[comparison_df["node_orientation"] == "upstream"]
        .merge(
            precomputed_distances,
            left_on=["focal_sc_id", "name"],
            right_on=["sc_id_dest", "sc_id_origin"],
        )
        .query("abs(path_weight_x - path_upstream_weights) > 1e-13")
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
        "sc_id_origin": {
            1: "SC00000000",
            3: "SC00000003",
            4: "SC00000004",
            5: "SC00000005",
            6: "SC00000011",
        },
        "sc_id_dest": {
            1: "SC00000001",
            3: "SC00000001",
            4: "SC00000001",
            5: "SC00000001",
            6: "SC00000001",
        },
        "path_length": {1: 1.0, 3: 4.0, 4: 6.0, 5: 6.0, 6: 1.0},
        "path_upstream_weights": {1: 1.0, 3: 4.0, 4: 6.0, 5: 6.0, 6: 1.0},
        "path_weights": {1: 1.0, 3: 4.0, 4: 6.0, 5: 6.0, 6: 1.0},
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
