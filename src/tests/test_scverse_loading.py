import pytest

import anndata
import numpy as np
import pandas as pd
from scipy import sparse
    
from napistu.scverse.loading import (
    _load_raw_table,
    _get_table_from_dict_attr,
    _select_results_attrs
)

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
    
    # Add some additional matrices
    adata.varm['gene_scores'] = pd.DataFrame(
        np.random.randn(n_vars, 3),
        index=adata.var.index,
        columns=['score1', 'score2', 'score3']
    )

    # Add variable pairwise matrices (varp)
    # Dense correlation matrix
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
    assert isinstance(result, pd.DataFrame)
    assert result.shape == minimal_adata.varm["gene_scores"].shape
    assert all(col in result.columns for col in ['score1', 'score2', 'score3'])

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
    # Test DataFrame selection
    df = pd.DataFrame(
        np.random.randn(5, 5),
        index=['attr_' + str(i) for i in range(5)],
        columns=minimal_adata.var.index
    )
    df_results_attrs = ['attr_0', 'attr_2']
    df_result = _select_results_attrs(minimal_adata, df, "X", df_results_attrs)
    assert isinstance(df_result, pd.DataFrame)
    assert all(attr in df_result.index for attr in df_results_attrs)
    
    # Test numpy array selection - shape should be (n_obs x n_vars)
    array = np.random.randn(minimal_adata.n_obs, minimal_adata.n_vars)  # 10x5 to match minimal_adata
    array_results_attrs = minimal_adata.obs.index[:3].tolist()
    array_result = _select_results_attrs(minimal_adata, array, "X", array_results_attrs)
    assert isinstance(array_result, pd.DataFrame)
    assert array_result.shape[0] == minimal_adata.var.shape[0]
    assert len(array_result.columns) == len(array_results_attrs)

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