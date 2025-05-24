import pytest
import numpy as np
import pandas as pd
import anndata

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
    
    # Add some layers
    adata.layers['counts'] = np.random.randint(0, 100, size=(n_obs, n_vars))
    
    # Add some additional matrices
    adata.varm['gene_scores'] = pd.DataFrame(
        np.random.randn(n_vars, 3),
        index=adata.var.index,
        columns=['score1', 'score2', 'score3']
    )
    
    return adata

def test_load_raw_table_X(minimal_adata):
    """Test loading X matrix."""
    result = _load_raw_table(minimal_adata, "X")
    assert isinstance(result, np.ndarray)
    assert result.shape == minimal_adata.X.shape

def test_load_raw_table_obs(minimal_adata):
    """Test loading obs table."""
    result = _load_raw_table(minimal_adata, "obs")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == minimal_adata.obs.shape

def test_load_raw_table_layers(minimal_adata):
    """Test loading from layers."""
    result = _load_raw_table(minimal_adata, "layers", "counts")
    assert isinstance(result, np.ndarray)
    assert result.shape == minimal_adata.layers["counts"].shape

def test_load_raw_table_invalid_type(minimal_adata):
    """Test loading with invalid table type."""
    with pytest.raises(ValueError):
        _load_raw_table(minimal_adata, "invalid_type")

def test_get_table_from_dict_attr(minimal_adata):
    """Test getting table from dict-like attribute."""
    result = _get_table_from_dict_attr(minimal_adata, "varm", "gene_scores")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == minimal_adata.varm["gene_scores"].shape

def test_get_table_from_dict_attr_auto_select(minimal_adata):
    """Test auto-selection when only one table exists."""
    result = _get_table_from_dict_attr(minimal_adata, "layers")
    assert isinstance(result, np.ndarray)
    assert result.shape == minimal_adata.layers["counts"].shape

def test_get_table_from_dict_attr_invalid_name(minimal_adata):
    """Test error when invalid table name is provided."""
    with pytest.raises(ValueError):
        _get_table_from_dict_attr(minimal_adata, "varm", "invalid_table")

def test_select_results_attrs_dataframe(minimal_adata):
    """Test selecting results from DataFrame."""
    # Create a DataFrame with the same number of columns as minimal_adata.var
    df = pd.DataFrame(
        np.random.randn(5, 5), 
        index=['attr_' + str(i) for i in range(5)],
        columns=minimal_adata.var.index
    )
    results_attrs = ['attr_0', 'attr_2']
    result = _select_results_attrs(minimal_adata, df, results_attrs)
    assert isinstance(result, pd.DataFrame)
    assert all(attr in result.index for attr in results_attrs)

def test_select_results_attrs_numpy(minimal_adata):
    """Test selecting results from numpy array."""
    array = np.random.randn(10, 5)  # matches adata dimensions
    selected_obs = minimal_adata.obs.index[:3].tolist()
    result = _select_results_attrs(minimal_adata, array, selected_obs)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == minimal_adata.var.shape[0]  # transposed
    assert len(result.columns) == len(selected_obs) 