import pytest

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
    
from napistu.scverse.loading import (
    _load_raw_table,
    _get_table_from_dict_attr,
    _select_results_attrs,
    _create_results_df
)
from napistu.scverse.constants import ADATA

@pytest.fixture
def minimal_adata():
    """Create a minimal AnnData object for testing."""
    # Create random data
    n_obs, n_vars = 10, 5
    X = np.random.randn(n_obs, n_vars)
    
    # Create observation and variable annotations
    obs = pd.DataFrame(
        {'cell_type': ['type_' + str(i) for i in range(n_obs)]},
        index=['cell_' + str(i) for i in range(n_obs)]
    )
    var = pd.DataFrame(
        {'gene_name': ['gene_' + str(i) for i in range(n_vars)]},
        index=['gene_' + str(i) for i in range(n_vars)]
    )
    
    # Create AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    
    # Add multiple layers to test table_name specification
    adata.layers['counts'] = np.random.randint(0, 100, size=(n_obs, n_vars))
    adata.layers['normalized'] = np.random.randn(n_obs, n_vars)
    
    # Add varm matrix (n_vars × n_features)
    n_features = 3
    varm_array = np.random.randn(n_vars, n_features)
    adata.varm['gene_scores'] = varm_array
    # Store column names separately since varm is a numpy array
    adata.uns['gene_scores_features'] = ['score1', 'score2', 'score3']

    # Add variable pairwise matrices (varp)
    # Dense correlation matrix (n_vars × n_vars)
    correlations = np.random.rand(n_vars, n_vars)
    # Make it symmetric for a correlation matrix
    correlations = (correlations + correlations.T) / 2
    adata.varp['correlations'] = correlations
    
    # Sparse adjacency matrix
    adjacency = sparse.random(n_vars, n_vars, density=0.2)
    # Make it symmetric
    adjacency = (adjacency + adjacency.T) / 2
    adata.varp['adjacency'] = adjacency
    
    return adata

def test_load_raw_table_success(minimal_adata):
    """Test successful loading of various table types."""
    # Test identity table (X)
    x_result = _load_raw_table(minimal_adata, "X")
    assert isinstance(x_result, np.ndarray)
    assert x_result.shape == minimal_adata.X.shape
    
    # Test dict-like table with name
    layer_result = _load_raw_table(minimal_adata, "layers", "counts")
    assert isinstance(layer_result, np.ndarray)
    assert layer_result.shape == minimal_adata.layers["counts"].shape

def test_load_raw_table_errors(minimal_adata):
    """Test error cases for loading tables."""
    # Test invalid table type
    with pytest.raises(ValueError, match="is not a valid AnnData attribute"):
        _load_raw_table(minimal_adata, "invalid_type")
    
    # Test missing table name when required
    with pytest.raises(ValueError, match="Multiple tables found.*and table_name is not specified"):
        _load_raw_table(minimal_adata, "layers")

def test_get_table_from_dict_attr_success(minimal_adata):
    """Test successful retrieval from dict-like attributes."""
    # Test getting specific table
    result = _get_table_from_dict_attr(minimal_adata, "varm", "gene_scores")
    assert isinstance(result, np.ndarray)
    assert result.shape == (minimal_adata.n_vars, 3)

def test_get_table_from_dict_attr_errors(minimal_adata):
    """Test error cases for dict-like attribute access."""
    # Test missing table name with multiple tables
    with pytest.raises(ValueError, match="Multiple tables found.*and table_name is not specified"):
        _get_table_from_dict_attr(minimal_adata, "layers")
    
    # Test nonexistent table name
    with pytest.raises(ValueError, match="table_name 'nonexistent' not found"):
        _get_table_from_dict_attr(minimal_adata, "layers", "nonexistent")

def test_select_results_attrs_success(minimal_adata):
    """Test successful selection of results attributes."""
    # Test numpy array selection - shape should be (n_obs x n_vars)
    array = np.random.randn(minimal_adata.n_obs, minimal_adata.n_vars)  # 10x5 to match minimal_adata
    array_results_attrs = minimal_adata.obs.index[:3].tolist()
    array_result = _select_results_attrs(minimal_adata, array, "X", array_results_attrs)
    assert isinstance(array_result, pd.DataFrame)
    assert array_result.shape[0] == minimal_adata.var.shape[0]
    assert len(array_result.columns) == len(array_results_attrs)
    # Check orientation - vars should be rows
    assert list(array_result.index) == minimal_adata.var.index.tolist()

    # Test varm selection
    varm_results_attrs = ['score1', 'score2']
    varm_features = minimal_adata.uns['gene_scores_features']
    # Get column indices for the requested features
    varm_col_indices = [varm_features.index(attr) for attr in varm_results_attrs]
    varm_result = _select_results_attrs(
        minimal_adata,
        minimal_adata.varm['gene_scores'],
        "varm",
        varm_results_attrs,
        table_colnames=varm_features
    )
    assert isinstance(varm_result, pd.DataFrame)
    assert varm_result.shape == (minimal_adata.n_vars, 2)  # All vars x selected features
    assert list(varm_result.columns) == varm_results_attrs
    assert list(varm_result.index) == minimal_adata.var.index.tolist()
    # Check values match original
    np.testing.assert_array_equal(
        varm_result.values,
        minimal_adata.varm['gene_scores'][:, varm_col_indices]
    )

    # Test varp selection with dense matrix
    varp_results_attrs = minimal_adata.var.index[:2].tolist()  # Select first two genes
    varp_result = _select_results_attrs(
        minimal_adata,
        minimal_adata.varp['correlations'],
        "varp",
        varp_results_attrs
    )
    assert isinstance(varp_result, pd.DataFrame)
    assert varp_result.shape == (minimal_adata.n_vars, 2)  # All vars x selected genes
    assert list(varp_result.columns) == varp_results_attrs
    assert list(varp_result.index) == minimal_adata.var.index.tolist()
    # Check values match original
    np.testing.assert_array_equal(
        varp_result.values,
        minimal_adata.varp['correlations'][:, :2]  # First two columns
    )

    # Test varp selection with sparse matrix
    sparse_result = _select_results_attrs(
        minimal_adata,
        minimal_adata.varp['adjacency'],
        "varp",
        varp_results_attrs
    )
    assert isinstance(sparse_result, pd.DataFrame)
    assert sparse_result.shape == (minimal_adata.n_vars, 2)
    assert list(sparse_result.columns) == varp_results_attrs
    assert list(sparse_result.index) == minimal_adata.var.index.tolist()

    # Test full table selection (results_attrs=None)
    full_varm_result = _select_results_attrs(
        minimal_adata,
        minimal_adata.varm['gene_scores'],
        "varm",
        table_colnames=minimal_adata.uns['gene_scores_features']
    )
    assert isinstance(full_varm_result, pd.DataFrame)
    assert full_varm_result.shape == (minimal_adata.n_vars, 3)
    assert list(full_varm_result.columns) == minimal_adata.uns['gene_scores_features']
    assert list(full_varm_result.index) == minimal_adata.var.index.tolist()

def test_select_results_attrs_errors(minimal_adata):
    """Test error cases for results attribute selection."""
    # Test invalid results attributes - shape should match minimal_adata
    array = np.random.randn(minimal_adata.n_obs, minimal_adata.n_vars)
    with pytest.raises(ValueError, match="The following results attributes are not valid"):
        _select_results_attrs(minimal_adata, array, "X", ['nonexistent_attr'])

    # Test invalid gene names for varp
    with pytest.raises(ValueError, match="The following results attributes are not valid"):
        _select_results_attrs(
            minimal_adata,
            minimal_adata.varp['correlations'],
            "varp",
            results_attrs=['nonexistent_gene']
        )

    # Test missing table_colnames for varm
    with pytest.raises(ValueError, match="table_colnames is required for varm tables"):
        _select_results_attrs(
            minimal_adata,
            minimal_adata.varm['gene_scores'],
            "varm",
            ['score1']
        )

    # Test DataFrame for array-type table
    with pytest.raises(ValueError, match="must be a numpy array, not a DataFrame"):
        _select_results_attrs(
            minimal_adata,
            pd.DataFrame(minimal_adata.varm['gene_scores']),
            "varm",
            ['score1'],
            table_colnames=minimal_adata.uns['gene_scores_features']
        )

def test_create_results_df(minimal_adata):
    """Test DataFrame creation from different AnnData table types."""
    # Test varm table
    varm_attrs = ['score1', 'score2']
    varm_features = minimal_adata.uns['gene_scores_features']
    # Get column indices for the requested features
    varm_col_indices = [varm_features.index(attr) for attr in varm_attrs]
    varm_array = minimal_adata.varm['gene_scores'][:, varm_col_indices]
    varm_result = _create_results_df(
        array=varm_array,
        attrs=varm_attrs,
        var_index=minimal_adata.var.index,
        table_type=ADATA.VARM
    )
    assert varm_result.shape == (minimal_adata.n_vars, len(varm_attrs))
    pd.testing.assert_index_equal(varm_result.index, minimal_adata.var.index)
    pd.testing.assert_index_equal(varm_result.columns, pd.Index(varm_attrs))
    np.testing.assert_array_equal(varm_result.values, varm_array)

    # Test varp table with dense correlations
    varp_attrs = minimal_adata.var.index[:2].tolist()  # First two genes
    varp_array = minimal_adata.varp['correlations'][:, :2]
    varp_result = _create_results_df(
        array=varp_array,
        attrs=varp_attrs,
        var_index=minimal_adata.var.index,
        table_type=ADATA.VARP
    )
    assert varp_result.shape == (minimal_adata.n_vars, len(varp_attrs))
    pd.testing.assert_index_equal(varp_result.index, minimal_adata.var.index)
    pd.testing.assert_index_equal(varp_result.columns, pd.Index(varp_attrs))
    np.testing.assert_array_equal(varp_result.values, varp_array)

    # Test X table
    obs_attrs = minimal_adata.obs.index[:3].tolist()  # First three observations
    x_array = minimal_adata.X[0:3, :]  # Select first three observations
    x_result = _create_results_df(
        array=x_array,
        attrs=obs_attrs,
        var_index=minimal_adata.var.index,
        table_type=ADATA.X
    )
    assert x_result.shape == (minimal_adata.n_vars, len(obs_attrs))
    pd.testing.assert_index_equal(x_result.index, minimal_adata.var.index)
    pd.testing.assert_index_equal(x_result.columns, pd.Index(obs_attrs))
    np.testing.assert_array_equal(x_result.values, x_array.T)

    # Test layers table
    layer_attrs = minimal_adata.obs.index[:2].tolist()  # First two observations
    layer_array = minimal_adata.layers['counts'][0:2, :]  # Select first two observations
    layer_result = _create_results_df(
        array=layer_array,
        attrs=layer_attrs,
        var_index=minimal_adata.var.index,
        table_type=ADATA.LAYERS
    )
    assert layer_result.shape == (minimal_adata.n_vars, len(layer_attrs))
    pd.testing.assert_index_equal(layer_result.index, minimal_adata.var.index)
    pd.testing.assert_index_equal(layer_result.columns, pd.Index(layer_attrs))
    np.testing.assert_array_equal(layer_result.values, layer_array.T) 
    