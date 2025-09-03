import igraph as ig
import numpy as np
import pandas as pd
import pytest

from napistu.network.constants import (
    NAPISTU_GRAPH_VERTICES,
    NULL_STRATEGIES,
)
from napistu.network.net_propagation import (
    NULL_GENERATORS,
    edge_permutation_null,
    net_propagate_attributes,
    network_propagation_with_null,
    parametric_null,
    uniform_null,
    vertex_permutation_null,
)


def test_network_propagation_with_null():
    """Test the main orchestrator function with different null strategies."""
    # Create test graph
    graph = ig.Graph(5)
    graph.vs[NAPISTU_GRAPH_VERTICES.NAME] = ["A", "B", "C", "D", "E"]
    graph.vs["attr1"] = [1.0, 0.0, 2.0, 0.0, 1.5]  # Non-negative, not all zero
    graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4)])

    attributes = ["attr1"]

    # Test 1: Uniform null (should return ratios)
    result_uniform = network_propagation_with_null(
        graph, attributes, null_strategy=NULL_STRATEGIES.UNIFORM
    )

    # Check structure
    assert isinstance(result_uniform, pd.DataFrame)
    assert result_uniform.shape == (5, 1)
    assert list(result_uniform.columns) == attributes
    assert list(result_uniform.index) == ["A", "B", "C", "D", "E"]

    # Should be ratios (can be > 1)
    assert (result_uniform.values > 0).all(), "Ratios should be positive"
    # Some ratios should be > 1 since observed scores concentrate on fewer nodes
    assert (result_uniform.values > 1).any(), "Some ratios should be > 1"

    # Test 2: Node permutation null (should return quantiles)
    result_permutation = network_propagation_with_null(
        graph,
        attributes,
        null_strategy=NULL_STRATEGIES.VERTEX_PERMUTATION,
        n_samples=10,  # Small for testing
    )

    # Check structure
    assert isinstance(result_permutation, pd.DataFrame)
    assert result_permutation.shape == (5, 1)
    assert list(result_permutation.columns) == attributes

    # Should be quantiles (0 to 1)
    # Drop NaNs before checking value range. NaNs can be introduced if a
    # nodes PPR values with and without the null are all a constant which
    # can come up in a small sample # testing situation
    permutation_values = result_permutation.values[~np.isnan(result_permutation.values)]
    assert (permutation_values >= 0).all(), "Quantiles should be >= 0"
    assert (permutation_values <= 1).all(), "Quantiles should be <= 1"

    # Test 3: Edge permutation null
    result_edge = network_propagation_with_null(
        graph,
        attributes,
        null_strategy=NULL_STRATEGIES.EDGE_PERMUTATION,
        n_samples=5,
        burn_in_ratio=2,  # Small for testing
        sampling_ratio=0.2,
    )

    # Check structure
    assert isinstance(result_edge, pd.DataFrame)
    assert result_edge.shape == (5, 1)
    # Drop NaNs before checking value range. NaNs can be introduced if a
    # nodes PPR values with and without the null are all a constant which
    # can come up in a small sample # testing situation
    edge_values = result_edge.values[~np.isnan(result_edge.values)]
    assert (edge_values >= 0).all()
    assert (edge_values <= 1).all()

    # Test 4: Gaussian null
    result_parametric = network_propagation_with_null(
        graph, attributes, null_strategy=NULL_STRATEGIES.PARAMETRIC, n_samples=8
    )

    # Check structure
    assert isinstance(result_parametric, pd.DataFrame)
    assert result_parametric.shape == (5, 1)
    assert (result_parametric.values >= 0).all()
    assert (result_parametric.values <= 1).all()

    # Test 5: Custom propagation parameters
    result_custom = network_propagation_with_null(
        graph,
        attributes,
        null_strategy=NULL_STRATEGIES.UNIFORM,
        additional_propagation_args={"damping": 0.7},
    )

    # Should be different from default
    assert not np.allclose(
        result_uniform.values, result_custom.values
    ), "Different propagation parameters should give different results"

    # Test 6: Custom null parameters (mask)
    mask_array = np.array([True, False, True, False, True])
    result_masked = network_propagation_with_null(
        graph,
        attributes,
        null_strategy=NULL_STRATEGIES.VERTEX_PERMUTATION,
        n_samples=5,
        mask=mask_array,
    )

    # Should work without error
    assert isinstance(result_masked, pd.DataFrame)
    assert result_masked.shape == (5, 1)

    # Test 7: Error handling - invalid null strategy
    with pytest.raises(ValueError, match="Unknown null strategy"):
        network_propagation_with_null(
            graph, attributes, null_strategy="invalid_strategy"
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
        uniform_null(graph, ["bad_attr"])

    with pytest.raises(ValueError):
        parametric_null(graph, ["bad_attr"])

    with pytest.raises(ValueError):
        vertex_permutation_null(graph, ["bad_attr"])

    with pytest.raises(ValueError):
        edge_permutation_null(graph, ["bad_attr"])

    # Test 2: Empty mask should raise error
    empty_mask = np.array([False, False, False])
    with pytest.raises(ValueError, match="No nodes in mask"):
        uniform_null(graph, ["attr1"], mask=empty_mask)

    # Test 3: Single node mask (edge case)
    single_mask = np.array([True, False, False])
    result = uniform_null(graph, ["attr1"], mask=single_mask)
    assert result.shape == (3, 1)  # Should work

    # Test 4: Replace parameter in node permutation
    result_no_replace = vertex_permutation_null(
        graph, ["attr1"], replace=False, n_samples=2
    )
    result_replace = vertex_permutation_null(
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
    result_default = uniform_null(graph, ["attr1"])
    result_damped = uniform_null(
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
