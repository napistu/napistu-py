"""Test network propagation with alternative null strategies."""

import logging

import igraph as ig
import numpy as np
import pandas as pd
import pytest

from napistu.network.constants import (
    NAPISTU_GRAPH_VERTICES,
    NET_PROPAGATION_METRICS,
    NULL_STRATEGIES,
)
from napistu.network.net_propagation import (
    NULL_GENERATORS,
    _compute_log2_enrichment,
    _edge_permutation_null,
    _parametric_null,
    _uniform_null,
    _vertex_permutation_null,
    melt_propagation_results,
    net_propagate_attributes,
    network_propagation_with_null,
)


def test_melt_propagation_results():
    """Melt propagation results into tall format."""
    index = pd.Index(["A", "B", "C"], name="vertex_id")
    columns = pd.MultiIndex.from_tuples(
        [
            (NET_PROPAGATION_METRICS.OBSERVED, "attr1"),
            (NET_PROPAGATION_METRICS.LOG2_ENRICHMENT, "attr1"),
            (NET_PROPAGATION_METRICS.QUANTILE, "attr1"),
        ],
        names=[None, "attribute"],
    )
    data = np.array([[0.1, 1.2, 0.7], [0.2, 0.8, 0.4], [0.3, 1.0, 0.6]])
    propagation_results = pd.DataFrame(data, index=index, columns=columns)

    tall = melt_propagation_results(
        propagation_results, index_name="vertex_id", attribute_name="attribute"
    )

    assert tall.shape == (3, 5)
    assert list(tall.columns) == [
        "vertex_id",
        "attribute",
        NET_PROPAGATION_METRICS.OBSERVED,
        NET_PROPAGATION_METRICS.LOG2_ENRICHMENT,
        NET_PROPAGATION_METRICS.QUANTILE,
    ]
    assert list(tall["vertex_id"]) == ["A", "B", "C"]
    assert list(tall["attribute"]) == ["attr1", "attr1", "attr1"]
    np.testing.assert_array_almost_equal(
        tall[NET_PROPAGATION_METRICS.OBSERVED], [0.1, 0.2, 0.3]
    )
    np.testing.assert_array_almost_equal(
        tall[NET_PROPAGATION_METRICS.QUANTILE], [0.7, 0.4, 0.6]
    )


def test_network_propagation_with_null():
    """Test the main orchestrator function with different null strategies."""
    # Create test graph
    graph = ig.Graph(5)
    graph.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["A", "B", "C", "D", "E"]
    graph.vs["attr1"] = [1.0, 0.0, 2.0, 0.0, 1.5]
    graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4)])

    attributes = ["attr1"]

    def assert_multiindex_structure(result, attributes, expected_metrics):
        """Helper to validate MultiIndex column structure."""
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)
        assert list(result.columns.get_level_values(0).unique()) == expected_metrics
        for metric in expected_metrics:
            assert list(result[metric].columns) == attributes
        assert list(result.index) == ["A", "B", "C", "D", "E"]

    # Test 1: Uniform null — no quantile level
    result_uniform = network_propagation_with_null(
        graph, attributes, null_strategy=NULL_STRATEGIES.UNIFORM
    )

    assert_multiindex_structure(
        result_uniform,
        attributes,
        [NET_PROPAGATION_METRICS.OBSERVED, NET_PROPAGATION_METRICS.LOG2_ENRICHMENT],
    )
    assert (
        NET_PROPAGATION_METRICS.QUANTILE
        not in result_uniform.columns.get_level_values(0)
    )
    assert (result_uniform[NET_PROPAGATION_METRICS.OBSERVED].values > 0).all()
    assert (result_uniform[NET_PROPAGATION_METRICS.LOG2_ENRICHMENT].values > 0).any()

    # Test 2: Vertex permutation null — includes quantile level
    result_permutation = network_propagation_with_null(
        graph,
        attributes,
        null_strategy=NULL_STRATEGIES.VERTEX_PERMUTATION,
        n_samples=10,
    )

    assert_multiindex_structure(
        result_permutation,
        attributes,
        [
            NET_PROPAGATION_METRICS.OBSERVED,
            NET_PROPAGATION_METRICS.QUANTILE,
            NET_PROPAGATION_METRICS.LOG2_ENRICHMENT,
        ],
    )
    permutation_quantiles = result_permutation[NET_PROPAGATION_METRICS.QUANTILE].values
    permutation_quantiles = permutation_quantiles[~np.isnan(permutation_quantiles)]
    assert (permutation_quantiles >= 0).all()
    assert (permutation_quantiles <= 1).all()

    # Test 3: Edge permutation null — includes quantile level
    result_edge = network_propagation_with_null(
        graph,
        attributes,
        null_strategy=NULL_STRATEGIES.EDGE_PERMUTATION,
        n_samples=5,
        burn_in_ratio=2,
        sampling_ratio=0.2,
    )

    assert_multiindex_structure(
        result_edge,
        attributes,
        [
            NET_PROPAGATION_METRICS.OBSERVED,
            NET_PROPAGATION_METRICS.QUANTILE,
            NET_PROPAGATION_METRICS.LOG2_ENRICHMENT,
        ],
    )
    edge_quantiles = result_edge[NET_PROPAGATION_METRICS.QUANTILE].values
    edge_quantiles = edge_quantiles[~np.isnan(edge_quantiles)]
    assert (edge_quantiles >= 0).all()
    assert (edge_quantiles <= 1).all()

    # Test 4: Parametric null — includes quantile level
    result_parametric = network_propagation_with_null(
        graph, attributes, null_strategy=NULL_STRATEGIES.PARAMETRIC, n_samples=8
    )

    assert_multiindex_structure(
        result_parametric,
        attributes,
        [
            NET_PROPAGATION_METRICS.OBSERVED,
            NET_PROPAGATION_METRICS.QUANTILE,
            NET_PROPAGATION_METRICS.LOG2_ENRICHMENT,
        ],
    )
    parametric_quantiles = result_parametric[NET_PROPAGATION_METRICS.QUANTILE].values
    assert (parametric_quantiles >= 0).all()
    assert (parametric_quantiles <= 1).all()

    # Test 5: Custom propagation parameters — different observed scores
    result_custom = network_propagation_with_null(
        graph,
        attributes,
        null_strategy=NULL_STRATEGIES.UNIFORM,
        additional_propagation_args={"damping": 0.7},
    )

    assert not np.allclose(
        result_uniform[NET_PROPAGATION_METRICS.OBSERVED].values,
        result_custom[NET_PROPAGATION_METRICS.OBSERVED].values,
    ), "Different propagation parameters should give different results"

    # Test 6: Masked vertex permutation
    mask_array = np.array([True, False, True, False, True])
    result_masked = network_propagation_with_null(
        graph,
        attributes,
        null_strategy=NULL_STRATEGIES.VERTEX_PERMUTATION,
        n_samples=5,
        mask=mask_array,
    )

    assert_multiindex_structure(
        result_masked,
        attributes,
        [
            NET_PROPAGATION_METRICS.OBSERVED,
            NET_PROPAGATION_METRICS.QUANTILE,
            NET_PROPAGATION_METRICS.LOG2_ENRICHMENT,
        ],
    )

    # Test 7: Pooled vertex permutation — includes quantile level
    graph_pooled = ig.Graph(5)
    graph_pooled.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["A", "B", "C", "D", "E"]
    graph_pooled.vs["attr1"] = [1.0, 0.0, 2.0, 0.0, 1.5]
    graph_pooled.vs["attr2"] = [0.5, 0.0, 1.0, 0.0, 2.0]
    graph_pooled.add_edges([(0, 1), (1, 2), (2, 3), (3, 4)])
    shared_mask = np.array([True, False, True, False, True])
    result_pooled = network_propagation_with_null(
        graph_pooled,
        ["attr1", "attr2"],
        null_strategy=NULL_STRATEGIES.POOLED_VERTEX_PERMUTATION,
        n_samples=10,
        mask=shared_mask,
    )

    assert_multiindex_structure(
        result_pooled,
        ["attr1", "attr2"],
        [
            NET_PROPAGATION_METRICS.OBSERVED,
            NET_PROPAGATION_METRICS.QUANTILE,
            NET_PROPAGATION_METRICS.LOG2_ENRICHMENT,
        ],
    )
    pooled_quantiles = result_pooled[NET_PROPAGATION_METRICS.QUANTILE].values
    pooled_quantiles = pooled_quantiles[~np.isnan(pooled_quantiles)]
    assert (pooled_quantiles >= 0).all()
    assert (pooled_quantiles <= 1).all()

    # Test 8: Error handling - invalid null strategy
    with pytest.raises(ValueError, match="Unknown null strategy"):
        network_propagation_with_null(
            graph, attributes, null_strategy="invalid_strategy"
        )


def test_network_propagation_invalid_quantile_method():
    """quantile_method is validated before null generation."""
    graph = ig.Graph(3)
    graph.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["A", "B", "C"]
    graph.vs["attr1"] = [1.0, 0.0, 2.0]
    graph.add_edges([(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="quantile_method must be one of"):
        network_propagation_with_null(
            graph, ["attr1"], quantile_method="bad", n_samples=3
        )


def test_net_propagate_attributes():
    """Test net_propagate_attributes with multiple attributes and various scenarios."""
    # Create test graph with edges for realistic propagation
    graph = ig.Graph(4)
    graph.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["node1", "node2", "node3", "node4"]
    graph.vs["attr1"] = [1.0, 0.0, 2.0, 0.0]  # Non-negative, not all zero
    graph.vs["attr2"] = [0.5, 1.5, 0.0, 1.0]  # Non-negative, not all zero
    graph.add_edges([(0, 1), (1, 2), (2, 3), (0, 3)])  # Create connected graph

    # Test 1: Basic functionality with two attributes
    result = net_propagate_attributes(graph, ["attr1", "attr2"])

    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 2)
    assert list(result.index) == ["node1", "node2", "node3", "node4"]
    assert list(result.columns) == ["attr1", "attr2"]

    # Check that values are valid probabilities (PPR returns probabilities)
    assert np.all(result.values >= 0)
    assert np.all(result.values <= 1)
    # Each column should sum to approximately 1 (PPR property)
    assert np.allclose(result.sum(axis=0), [1.0, 1.0], atol=1e-10)

    # Test 2: Single attribute
    result_single = net_propagate_attributes(graph, ["attr1"])
    assert result_single.shape == (4, 1)
    assert list(result_single.columns) == ["attr1"]

    # Test 3: Graph without names (should use indices)
    graph_no_names = ig.Graph(3)
    graph_no_names.vs["attr1"] = [1.0, 2.0, 1.0]
    graph_no_names.add_edges([(0, 1), (1, 2)])

    result_no_names = net_propagate_attributes(graph_no_names, ["attr1"])
    assert list(result_no_names.index) == [0, 1, 2]  # Should use integer indices

    # Test 4: Invalid propagation method
    with pytest.raises(ValueError, match="Invalid propagation method"):
        net_propagate_attributes(graph, ["attr1"], propagation_method="invalid_method")

    # Test 5: Additional arguments (test damping parameter)
    result_default = net_propagate_attributes(graph, ["attr1"])
    result_damped = net_propagate_attributes(
        graph, ["attr1"], additional_propagation_args={"damping": 0.5}  # Lower damping
    )

    # Results should be different with different damping
    assert not np.allclose(result_default.values, result_damped.values)

    # Test 6: Invalid attribute (should be caught by internal validation)
    graph.vs["bad_attr"] = [-1.0, 1.0, 2.0, 0.0]  # Has negative values
    with pytest.raises(ValueError, match="contains negative values"):
        net_propagate_attributes(graph, ["bad_attr"])

    # Test 7: Zero attribute (should be caught by internal validation)
    graph.vs["zero_attr"] = [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="zero for all vertices"):
        net_propagate_attributes(graph, ["zero_attr"])


def test_all_null_generators_structure():
    """Test all null generators with default options and validate output structure."""
    # Create test graph with edges for realistic propagation
    graph = ig.Graph(5)
    graph.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["A", "B", "C", "D", "E"]
    graph.vs["attr1"] = [1.0, 0.0, 2.0, 0.0, 1.5]  # Non-negative, not all zero
    graph.vs["attr2"] = [0.5, 1.0, 0.0, 2.0, 0.0]  # Non-negative, not all zero
    graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4)])

    attributes = ["attr1", "attr2"]
    n_samples = 3  # Small for testing

    for generator_name, generator_func in NULL_GENERATORS.items():

        if generator_name in (
            NULL_STRATEGIES.POOLED_VERTEX_PERMUTATION,
            NULL_STRATEGIES.ATTR_POOLED_VERTEX_PERMUTATION,
        ):
            continue  # not supported: these require identical masks; attr1/attr2 differ

        print(f"Testing {generator_name}")

        if generator_name == NULL_STRATEGIES.UNIFORM:
            # Uniform null doesn't take n_samples
            result = generator_func(graph, attributes)
            expected_rows = 5  # One row per node
        elif generator_name == NULL_STRATEGIES.EDGE_PERMUTATION:
            # Edge permutation has different parameters
            result = generator_func(graph, attributes, n_samples=n_samples)
            expected_rows = n_samples * 5  # n_samples rows per node
        else:
            # Gaussian and vertex_permutation
            result = generator_func(graph, attributes, n_samples=n_samples)
            expected_rows = n_samples * 5  # n_samples rows per node

        # Validate structure
        assert isinstance(
            result, pd.DataFrame
        ), f"{generator_name} should return DataFrame"
        assert result.shape == (
            expected_rows,
            2,
        ), f"{generator_name} wrong shape: {result.shape}"
        assert list(result.columns) == attributes, f"{generator_name} wrong columns"

        # Validate index structure
        if generator_name == NULL_STRATEGIES.UNIFORM:
            assert list(result.index) == [
                "A",
                "B",
                "C",
                "D",
                "E",
            ], f"{generator_name} wrong index"
        else:
            expected_index = ["A", "B", "C", "D", "E"] * n_samples
            assert (
                list(result.index) == expected_index
            ), f"{generator_name} wrong repeated index"

        # Validate values are numeric and finite (propagated outputs should be valid probabilities)
        assert result.isna().sum().sum() == 0, f"{generator_name} contains NaN values"
        assert np.isfinite(
            result.values
        ).all(), f"{generator_name} contains infinite values"
        assert (result.values >= 0).all(), f"{generator_name} contains negative values"
        assert (
            result.values <= 1
        ).all(), f"{generator_name} should contain probabilities <= 1"

        # Each sample should sum to approximately 1 (PPR property)
        if generator_name == NULL_STRATEGIES.UNIFORM:
            assert np.allclose(
                result.sum(axis=0), [1.0, 1.0], atol=1e-10
            ), f"{generator_name} doesn't sum to 1"
        else:
            # For multiple samples, each individual sample should sum to 1
            for i in range(n_samples):
                start_idx = i * 5
                end_idx = (i + 1) * 5
                sample_data = result.iloc[start_idx:end_idx]
                assert np.allclose(
                    sample_data.sum(axis=0), [1.0, 1.0], atol=1e-10
                ), f"{generator_name} sample {i} doesn't sum to 1"


def test_mask_application():
    """Test that masks are correctly applied across all null generators."""
    # Create test graph
    graph = ig.Graph(6)
    graph.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["A", "B", "C", "D", "E", "F"]
    graph.vs["attr1"] = [1.0, 0.0, 2.0, 0.0, 1.5, 0.0]  # Nonzero at indices 0, 2, 4
    graph.vs["attr2"] = [0.0, 1.0, 0.0, 2.0, 0.0, 1.0]  # Nonzero at indices 1, 3, 5
    graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])

    attributes = ["attr1", "attr2"]

    # Test mask that includes nodes with nonzero values for both attributes
    # Use nodes 0, 1, 2, 3 which covers nonzero values for both attributes
    mask_array = np.array([True, True, True, True, False, False])  # Nodes 0, 1, 2, 3

    for generator_name, generator_func in NULL_GENERATORS.items():
        print(f"Testing mask application for {generator_name}")

        if generator_name in (
            NULL_STRATEGIES.POOLED_VERTEX_PERMUTATION,
            NULL_STRATEGIES.ATTR_POOLED_VERTEX_PERMUTATION,
        ):
            continue  # pooled methods are covered in test_pooled_null_methods

        if generator_name == NULL_STRATEGIES.UNIFORM:
            result = generator_func(graph, attributes, mask=mask_array)

            # For uniform null with mask, verify structure is correct
            assert result.shape == (6, 2), f"{generator_name} wrong shape with mask"
            # After propagation, all nodes will have some value due to network effect
            assert (
                result.values > 0
            ).all(), "All nodes should have positive values after propagation"

        elif generator_name == NULL_STRATEGIES.EDGE_PERMUTATION:
            # Edge permutation ignores mask, just test it doesn't crash
            result = generator_func(graph, attributes, n_samples=2)
            assert result.shape[0] == 12  # 2 samples * 6 nodes

        else:
            # Gaussian and vertex_permutation with mask
            result = generator_func(graph, attributes, mask=mask_array, n_samples=2)

            # Check that structure is maintained
            assert result.shape == (12, 2)  # 2 samples * 6 nodes


def test_edge_cases_and_errors():
    """Test edge cases and error conditions for null generators."""
    # Create minimal test graph
    graph = ig.Graph(3)
    graph.vs["attr1"] = [1.0, 2.0, 0.0]
    graph.vs["bad_attr"] = [0.0, 0.0, 0.0]  # All zeros
    graph.add_edges([(0, 1), (1, 2)])

    # Test 1: All zero attribute should raise error for all generators
    with pytest.raises(ValueError):
        _uniform_null(graph, ["bad_attr"])

    with pytest.raises(ValueError):
        _parametric_null(graph, ["bad_attr"])

    with pytest.raises(ValueError):
        _vertex_permutation_null(graph, ["bad_attr"])

    with pytest.raises(ValueError):
        _edge_permutation_null(graph, ["bad_attr"])

    # Test 2: Empty mask should raise error
    empty_mask = np.array([False, False, False])
    with pytest.raises(ValueError, match="No nodes in mask"):
        _uniform_null(graph, ["attr1"], mask=empty_mask)

    # Test 3: Single node mask (edge case)
    single_mask = np.array([True, False, False])
    result = _uniform_null(graph, ["attr1"], mask=single_mask)
    assert result.shape == (3, 1)  # Should work

    # Test 4: Replace parameter in node permutation
    result_no_replace = _vertex_permutation_null(
        graph, ["attr1"], replace=False, n_samples=2
    )
    result_replace = _vertex_permutation_null(
        graph, ["attr1"], replace=True, n_samples=2
    )

    # Both should have same structure
    assert result_no_replace.shape == result_replace.shape


def test_propagation_method_parameters():
    """Test that propagation method and additional arguments are properly passed through."""
    # Create test graph
    graph = ig.Graph(4)
    graph.vs["attr1"] = [1.0, 2.0, 0.0, 1.5]
    graph.add_edges([(0, 1), (1, 2), (2, 3)])

    # Test different damping parameters produce different results
    result_default = _uniform_null(graph, ["attr1"])
    result_damped = _uniform_null(
        graph, ["attr1"], additional_propagation_args={"damping": 0.5}
    )

    # Results should be different with different damping
    assert not np.allclose(
        result_default.values, result_damped.values
    ), "Different damping should produce different results"

    # Test that all generators accept method parameters
    for generator_name, generator_func in NULL_GENERATORS.items():

        if generator_name == NULL_STRATEGIES.UNIFORM:
            result = generator_func(
                graph, ["attr1"], additional_propagation_args={"damping": 0.8}
            )
        else:
            result = generator_func(
                graph,
                ["attr1"],
                additional_propagation_args={"damping": 0.8},
                n_samples=2,
            )

        # Should produce valid results
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


def test_compute_log2_enrichment():
    """Tests for _compute_log2_enrichment."""
    np.random.seed(42)
    features = ["A", "B", "C"]
    attributes = ["attr1", "attr2"]
    n_samples = 10

    observed = pd.DataFrame(
        np.abs(np.random.randn(len(features), len(attributes))),
        index=features,
        columns=attributes,
    )

    # Null is stacked: n_samples rows per feature
    null_index = np.repeat(features, n_samples)
    null_data = np.abs(np.random.randn(len(features) * n_samples, len(attributes)))
    null_df = pd.DataFrame(null_data, index=null_index, columns=attributes)

    result = _compute_log2_enrichment(observed, null_df)

    # Output shape and structure
    assert result.shape == observed.shape
    assert list(result.index) == features
    assert list(result.columns) == attributes
    assert not result.isna().any().any()

    # Validate values manually for one feature
    null_mean_A = null_df.loc["A"].mean()
    expected_A = np.log2(observed.loc["A"] / (null_mean_A + 1e-10))
    pd.testing.assert_series_equal(result.loc["A"], expected_A.rename("A"))

    # Observed == null mean should give log2(1) == 0
    flat_null_index = np.repeat(features, n_samples)
    flat_null_data = np.vstack(
        [np.tile(observed.loc[f].values, (n_samples, 1)) for f in features]
    )
    flat_null_df = pd.DataFrame(
        flat_null_data, index=flat_null_index, columns=attributes
    )
    flat_result = _compute_log2_enrichment(observed, flat_null_df)
    np.testing.assert_allclose(flat_result.values, 0.0, atol=1e-6)

    # Observed 2x null mean should give log2(2) == 1
    double_null_data = np.vstack(
        [np.tile(observed.loc[f].values / 2, (n_samples, 1)) for f in features]
    )
    double_null_df = pd.DataFrame(
        double_null_data, index=flat_null_index, columns=attributes
    )
    double_result = _compute_log2_enrichment(observed, double_null_df)
    np.testing.assert_allclose(double_result.values, 1.0, atol=1e-6)

    # Shuffled null index order should give same result as unshuffled
    shuffled_null_df = null_df.sample(frac=1, random_state=42)
    shuffled_result = _compute_log2_enrichment(observed, shuffled_null_df)
    pd.testing.assert_frame_equal(result, shuffled_result)

    # Mismatched columns raises
    with pytest.raises(ValueError, match="Column names must match"):
        _compute_log2_enrichment(observed, null_df.rename(columns={"attr1": "wrong"}))

    # Missing features in null raises
    with pytest.raises(ValueError, match="Missing features"):
        _compute_log2_enrichment(observed, null_df.drop("A"))

    # NaN in observed raises
    observed_with_nan = observed.copy()
    observed_with_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN values found in observed"):
        _compute_log2_enrichment(observed_with_nan, null_df)

    # NaN in null raises
    null_with_nan = null_df.copy()
    null_with_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN values found in null"):
        _compute_log2_enrichment(observed, null_with_nan)


def test_log2_enrichment_reflects_signal_concentration():
    """Vertices receiving concentrated signal should have higher log2_enrichment."""
    graph = ig.Graph(6, directed=True)
    graph.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["A", "B", "C", "D", "E", "F"]
    graph.add_edges(
        [(0, 5), (1, 5), (2, 5), (3, 4), (4, 5)]
    )  # E is a hub, F receives everything

    # All signal on A, B, C, D — E is a pure conduit, F is the convergence point
    graph.vs["attr1"] = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    result = network_propagation_with_null(
        graph,
        ["attr1"],
        null_strategy=NULL_STRATEGIES.VERTEX_PERMUTATION,
        n_samples=100,
    )

    # F receives signal from all paths — should have highest log2_enrichment
    enrichment = result[NET_PROPAGATION_METRICS.LOG2_ENRICHMENT]["attr1"]
    assert (
        enrichment["F"] > enrichment["E"]
    ), "Convergence point should be more enriched than conduit"


def test_observed_scores_invariant_to_null_strategy():
    """Observed scores should be identical regardless of null strategy."""
    graph = ig.Graph(5)
    graph.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["A", "B", "C", "D", "E"]
    graph.vs["attr1"] = [1.0, 0.0, 2.0, 0.0, 1.5]
    graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4)])

    result_uniform = network_propagation_with_null(
        graph, ["attr1"], null_strategy=NULL_STRATEGIES.UNIFORM
    )
    result_perm = network_propagation_with_null(
        graph,
        ["attr1"],
        null_strategy=NULL_STRATEGIES.VERTEX_PERMUTATION,
        n_samples=10,
    )

    np.testing.assert_array_equal(
        result_uniform[NET_PROPAGATION_METRICS.OBSERVED].values,
        result_perm[NET_PROPAGATION_METRICS.OBSERVED].values,
        err_msg="Observed scores should be identical across null strategies",
    )


def test_pooled_null_methods(caplog):
    """Coverage for pooled_vertex_permutation and attr_pooled_vertex_permutation."""
    graph = ig.Graph(6)
    graph.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["A", "B", "C", "D", "E", "F"]
    # Three attributes with different sparsity, all sharing the same mask
    graph.vs["sparse"] = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    graph.vs["medium"] = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    graph.vs["dense"] = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])

    attrs = ["sparse", "medium", "dense"]
    shared_mask = np.array([True, True, True, True, False, False])

    # Both methods produce expected shape with shared mask
    for strategy in (
        NULL_STRATEGIES.POOLED_VERTEX_PERMUTATION,
        NULL_STRATEGIES.ATTR_POOLED_VERTEX_PERMUTATION,
    ):
        result = network_propagation_with_null(
            graph, attrs, null_strategy=strategy, n_samples=12, mask=shared_mask
        )
        assert result.shape == (6, 9)  # 6 nodes, 3 metrics x 3 attrs
        # Exact n_samples allocation matters for p-value resolution
        null_rows = result[NET_PROPAGATION_METRICS.QUANTILE].notna().sum().sum()
        assert null_rows > 0

    # attr_pooled with n_samples not divisible by n_attributes — must hit n_samples exactly
    n_samples = 10  # 10 // 3 = 3 base, remainder 1, so [4, 3, 3]
    result_uneven = NULL_GENERATORS[NULL_STRATEGIES.ATTR_POOLED_VERTEX_PERMUTATION](
        graph, attrs, n_samples=n_samples, mask=shared_mask
    )
    assert result_uneven.shape == (n_samples * 6, 3)

    # attr_pooled warns and proceeds when n_samples < n_attributes
    with caplog.at_level(logging.WARNING):
        result_under = NULL_GENERATORS[NULL_STRATEGIES.ATTR_POOLED_VERTEX_PERMUTATION](
            graph, attrs, n_samples=2, mask=shared_mask
        )
    assert "less than n_attributes" in caplog.text
    assert result_under.shape == (2 * 6, 3)

    # Both methods reject non-identical masks
    for strategy in (
        NULL_STRATEGIES.POOLED_VERTEX_PERMUTATION,
        NULL_STRATEGIES.ATTR_POOLED_VERTEX_PERMUTATION,
    ):
        with pytest.raises(ValueError, match="different mask"):
            NULL_GENERATORS[strategy](graph, attrs, n_samples=4)
