import anndata
import mudata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from napistu.genomics import scverse_loading
from napistu.genomics.constants import ADATA, SCVERSE_DEFS


@pytest.fixture
def minimal_adata():
    """Create a minimal AnnData object for testing."""
    # Create random data
    n_obs, n_vars = 10, 5
    X = np.random.randn(n_obs, n_vars)

    # Create observation and variable annotations
    obs = pd.DataFrame(
        {"cell_type": ["type_" + str(i) for i in range(n_obs)]},
        index=["cell_" + str(i) for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {
            "gene_name": ["gene_" + str(i) for i in range(n_vars)],
            "ensembl_transcript": [
                f"ENST{i:011d}" for i in range(n_vars)
            ],  # Add ensembl_transcript
        },
        index=["gene_" + str(i) for i in range(n_vars)],
    )

    # Create AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Add multiple layers to test table_name specification
    adata.layers["counts"] = np.random.randint(0, 100, size=(n_obs, n_vars))
    adata.layers["normalized"] = np.random.randn(n_obs, n_vars)

    # Add varm matrix (n_vars × n_features)
    n_features = 3
    varm_array = np.random.randn(n_vars, n_features)
    adata.varm["gene_scores"] = varm_array
    # Store column names separately since varm is a numpy array
    adata.uns["gene_scores_features"] = ["score1", "score2", "score3"]

    # Add variable pairwise matrices (varp)
    # Dense correlation matrix (n_vars × n_vars)
    correlations = np.random.rand(n_vars, n_vars)
    # Make it symmetric for a correlation matrix
    correlations = (correlations + correlations.T) / 2
    adata.varp["correlations"] = correlations

    # Sparse adjacency matrix
    adjacency = sparse.random(n_vars, n_vars, density=0.2)
    # Make it symmetric
    adjacency = (adjacency + adjacency.T) / 2
    adata.varp["adjacency"] = adjacency

    return adata


def test_load_raw_table_success(minimal_adata):
    """Test successful loading of various table types."""
    # Test identity table (X)
    x_result = scverse_loading._load_raw_table(minimal_adata, "X")
    assert isinstance(x_result, np.ndarray)
    assert x_result.shape == minimal_adata.X.shape

    # Test dict-like table with name
    layer_result = scverse_loading._load_raw_table(minimal_adata, "layers", "counts")
    assert isinstance(layer_result, np.ndarray)
    assert layer_result.shape == minimal_adata.layers["counts"].shape


def test_load_raw_table_errors(minimal_adata):
    """Test error cases for loading tables."""
    # Test invalid table type
    with pytest.raises(ValueError, match="is not a valid AnnData attribute"):
        scverse_loading._load_raw_table(minimal_adata, "invalid_type")

    # Test missing table name when required
    with pytest.raises(
        ValueError, match="Multiple tables found.*and table_name is not specified"
    ):
        scverse_loading._load_raw_table(minimal_adata, "layers")


def test_get_table_from_dict_attr_success(minimal_adata):
    """Test successful retrieval from dict-like attributes."""
    # Test getting specific table
    result = scverse_loading._get_table_from_dict_attr(
        minimal_adata, "varm", "gene_scores"
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (minimal_adata.n_vars, 3)


def test_get_table_from_dict_attr_errors(minimal_adata):
    """Test error cases for dict-like attribute access."""
    # Test missing table name with multiple tables
    with pytest.raises(
        ValueError, match="Multiple tables found.*and table_name is not specified"
    ):
        scverse_loading._get_table_from_dict_attr(minimal_adata, "layers")

    # Test nonexistent table name
    with pytest.raises(ValueError, match="table_name 'nonexistent' not found"):
        scverse_loading._get_table_from_dict_attr(
            minimal_adata, "layers", "nonexistent"
        )


def test_select_results_attrs_success(minimal_adata):
    """Test successful selection of results attributes."""
    # Test numpy array selection - shape should be (n_obs x n_vars)
    array = np.random.randn(
        minimal_adata.n_obs, minimal_adata.n_vars
    )  # 10x5 to match minimal_adata
    array_results_attrs = minimal_adata.obs.index[:3].tolist()
    array_result = scverse_loading._select_results_attrs(
        minimal_adata, array, "X", array_results_attrs
    )
    assert isinstance(array_result, pd.DataFrame)
    assert array_result.shape[0] == minimal_adata.var.shape[0]
    assert len(array_result.columns) == len(array_results_attrs)
    # Check orientation - vars should be rows
    assert list(array_result.index) == minimal_adata.var.index.tolist()

    # Test varm selection
    varm_results_attrs = ["score1", "score2"]
    varm_features = minimal_adata.uns["gene_scores_features"]
    # Get column indices for the requested features
    varm_col_indices = [varm_features.index(attr) for attr in varm_results_attrs]
    varm_result = scverse_loading._select_results_attrs(
        minimal_adata,
        minimal_adata.varm["gene_scores"],
        "varm",
        varm_results_attrs,
        table_colnames=varm_features,
    )
    assert isinstance(varm_result, pd.DataFrame)
    assert varm_result.shape == (
        minimal_adata.n_vars,
        2,
    )  # All vars x selected features
    assert list(varm_result.columns) == varm_results_attrs
    assert list(varm_result.index) == minimal_adata.var.index.tolist()
    # Check values match original
    np.testing.assert_array_equal(
        varm_result.values, minimal_adata.varm["gene_scores"][:, varm_col_indices]
    )

    # Test varp selection with dense matrix
    varp_results_attrs = minimal_adata.var.index[:2].tolist()  # Select first two genes
    varp_result = scverse_loading._select_results_attrs(
        minimal_adata, minimal_adata.varp["correlations"], "varp", varp_results_attrs
    )
    assert isinstance(varp_result, pd.DataFrame)
    assert varp_result.shape == (minimal_adata.n_vars, 2)  # All vars x selected genes
    assert list(varp_result.columns) == varp_results_attrs
    assert list(varp_result.index) == minimal_adata.var.index.tolist()
    # Check values match original
    np.testing.assert_array_equal(
        varp_result.values,
        minimal_adata.varp["correlations"][:, :2],  # First two columns
    )

    # Test varp selection with sparse matrix
    sparse_result = scverse_loading._select_results_attrs(
        minimal_adata, minimal_adata.varp["adjacency"], "varp", varp_results_attrs
    )
    assert isinstance(sparse_result, pd.DataFrame)
    assert sparse_result.shape == (minimal_adata.n_vars, 2)
    assert list(sparse_result.columns) == varp_results_attrs
    assert list(sparse_result.index) == minimal_adata.var.index.tolist()

    # Test full table selection (results_attrs=None)
    full_varm_result = scverse_loading._select_results_attrs(
        minimal_adata,
        minimal_adata.varm["gene_scores"],
        "varm",
        table_colnames=minimal_adata.uns["gene_scores_features"],
    )
    assert isinstance(full_varm_result, pd.DataFrame)
    assert full_varm_result.shape == (minimal_adata.n_vars, 3)
    assert list(full_varm_result.columns) == minimal_adata.uns["gene_scores_features"]
    assert list(full_varm_result.index) == minimal_adata.var.index.tolist()


def test_select_results_attrs_errors(minimal_adata):
    """Test error cases for results attribute selection."""
    # Test invalid results attributes - shape should match minimal_adata
    array = np.random.randn(minimal_adata.n_obs, minimal_adata.n_vars)
    with pytest.raises(
        ValueError, match="The following results attributes were not found"
    ):
        scverse_loading._select_results_attrs(
            minimal_adata, array, "X", ["nonexistent_attr"]
        )

    # Test invalid gene names for varp
    with pytest.raises(
        ValueError, match="The following results attributes were not found"
    ):
        scverse_loading._select_results_attrs(
            minimal_adata,
            minimal_adata.varp["correlations"],
            "varp",
            results_attrs=["nonexistent_gene"],
        )

    # Test missing table_colnames for varm
    with pytest.raises(ValueError, match="table_colnames is required for varm tables"):
        scverse_loading._select_results_attrs(
            minimal_adata, minimal_adata.varm["gene_scores"], "varm", ["score1"]
        )

    # Test DataFrame for array-type table
    with pytest.raises(ValueError, match="must be a numpy array, not a DataFrame"):
        scverse_loading._select_results_attrs(
            minimal_adata,
            pd.DataFrame(minimal_adata.varm["gene_scores"]),
            "varm",
            ["score1"],
            table_colnames=minimal_adata.uns["gene_scores_features"],
        )


def test_create_results_df(minimal_adata):
    """Test DataFrame creation from different AnnData table types."""
    # Test varm table
    varm_attrs = ["score1", "score2"]
    varm_features = minimal_adata.uns["gene_scores_features"]
    # Get column indices for the requested features
    varm_col_indices = [varm_features.index(attr) for attr in varm_attrs]
    varm_array = minimal_adata.varm["gene_scores"][:, varm_col_indices]
    varm_result = scverse_loading._create_results_df(
        array=varm_array,
        attrs=varm_attrs,
        var_index=minimal_adata.var.index,
        table_type=ADATA.VARM,
    )
    assert varm_result.shape == (minimal_adata.n_vars, len(varm_attrs))
    pd.testing.assert_index_equal(varm_result.index, minimal_adata.var.index)
    pd.testing.assert_index_equal(varm_result.columns, pd.Index(varm_attrs))
    np.testing.assert_array_equal(varm_result.values, varm_array)

    # Test varp table with dense correlations
    varp_attrs = minimal_adata.var.index[:2].tolist()  # First two genes
    varp_array = minimal_adata.varp["correlations"][:, :2]
    varp_result = scverse_loading._create_results_df(
        array=varp_array,
        attrs=varp_attrs,
        var_index=minimal_adata.var.index,
        table_type=ADATA.VARP,
    )
    assert varp_result.shape == (minimal_adata.n_vars, len(varp_attrs))
    pd.testing.assert_index_equal(varp_result.index, minimal_adata.var.index)
    pd.testing.assert_index_equal(varp_result.columns, pd.Index(varp_attrs))
    np.testing.assert_array_equal(varp_result.values, varp_array)

    # Test X table
    obs_attrs = minimal_adata.obs.index[:3].tolist()  # First three observations
    x_array = minimal_adata.X[0:3, :]  # Select first three observations
    x_result = scverse_loading._create_results_df(
        array=x_array,
        attrs=obs_attrs,
        var_index=minimal_adata.var.index,
        table_type=ADATA.X,
    )
    assert x_result.shape == (minimal_adata.n_vars, len(obs_attrs))
    pd.testing.assert_index_equal(x_result.index, minimal_adata.var.index)
    pd.testing.assert_index_equal(x_result.columns, pd.Index(obs_attrs))
    np.testing.assert_array_equal(x_result.values, x_array.T)

    # Test layers table
    layer_attrs = minimal_adata.obs.index[:2].tolist()  # First two observations
    layer_array = minimal_adata.layers["counts"][
        0:2, :
    ]  # Select first two observations
    layer_result = scverse_loading._create_results_df(
        array=layer_array,
        attrs=layer_attrs,
        var_index=minimal_adata.var.index,
        table_type=ADATA.LAYERS,
    )
    assert layer_result.shape == (minimal_adata.n_vars, len(layer_attrs))
    pd.testing.assert_index_equal(layer_result.index, minimal_adata.var.index)
    pd.testing.assert_index_equal(layer_result.columns, pd.Index(layer_attrs))
    np.testing.assert_array_equal(layer_result.values, layer_array.T)


@pytest.fixture
def minimal_mudata(minimal_adata):
    """Create a minimal MuData object for testing.

    Uses minimal_adata as the RNA modality and creates a simple protein modality.
    Focuses on testing MuData-level operations rather than modality-specific features.
    """
    # Create protein modality with minimal features
    n_vars_protein = 3
    adata_protein = anndata.AnnData(
        X=np.random.randn(minimal_adata.n_obs, n_vars_protein),
        obs=minimal_adata.obs,  # Share obs to ensure alignment
        var=pd.DataFrame(
            {
                "uniprot": [
                    f"P{i:05d}" for i in range(n_vars_protein)
                ]  # Valid ontology column
            },
            index=[f"protein_{i}" for i in range(n_vars_protein)],
        ),
    )

    # Create MuData with both modalities
    mdata = mudata.MuData({"rna": minimal_adata, "protein": adata_protein})

    # Add varm table at MuData level
    n_features = 3
    varm_array = np.random.randn(mdata.n_vars, n_features)
    mdata.varm["gene_scores"] = varm_array
    mdata.uns["gene_scores_features"] = ["score1", "score2", "score3"]

    return mdata


def test_prepare_anndata_results_df_anndata(minimal_adata):
    """Test prepare_anndata_results_df with AnnData input."""
    # Test var table
    var_results = scverse_loading.prepare_anndata_results_df(
        minimal_adata, table_type=ADATA.VAR
    )
    assert isinstance(var_results, pd.DataFrame)
    assert var_results.shape[0] == minimal_adata.n_vars
    assert "gene_name" in var_results.columns
    assert "ensembl_transcript" in var_results.columns

    # Test varm table
    varm_results = scverse_loading.prepare_anndata_results_df(
        minimal_adata,
        table_type=ADATA.VARM,
        table_name="gene_scores",
        results_attrs=minimal_adata.uns["gene_scores_features"],
        table_colnames=minimal_adata.uns["gene_scores_features"],  # Pass column names
    )
    assert isinstance(varm_results, pd.DataFrame)
    assert varm_results.shape[0] == minimal_adata.n_vars
    assert (
        varm_results.shape[1] == 5
    )  # score1, score2, score3 + gene_name + ensembl_transcript from var table

    # Check we have both the scores and systematic identifiers
    assert all(
        score in varm_results.columns
        for score in minimal_adata.uns["gene_scores_features"]
    )
    assert "gene_name" in varm_results.columns
    assert "ensembl_transcript" in varm_results.columns

    # Test with ontology extraction
    var_results_with_ontology = scverse_loading.prepare_anndata_results_df(
        minimal_adata,
        table_type=ADATA.VAR,
        index_which_ontology="ensembl_gene",  # Use a new ontology name
    )
    assert "ensembl_gene" in var_results_with_ontology.columns
    pd.testing.assert_series_equal(
        var_results_with_ontology["ensembl_gene"],
        pd.Series(
            minimal_adata.var.index, index=minimal_adata.var.index, name="ensembl_gene"
        ),
    )

    # Test error when trying to use existing column
    with pytest.raises(
        ValueError, match="Cannot use 'gene_name' as index_which_ontology"
    ):
        scverse_loading.prepare_anndata_results_df(
            minimal_adata,
            table_type=ADATA.VAR,
            index_which_ontology="gene_name",  # Should fail - already exists
        )


def test_split_mdata_results_by_modality(minimal_mudata):
    """Test splitting results table by modality."""
    # Create a combined results table with all vars
    all_results = pd.DataFrame(
        np.random.randn(len(minimal_mudata.var_names), 2),
        index=minimal_mudata.var_names,
        columns=["score1", "score2"],
    )

    # Split by modality
    modality_results = scverse_loading._split_mdata_results_by_modality(
        minimal_mudata, all_results
    )

    # Check we got both modalities
    assert set(modality_results.keys()) == {"rna", "protein"}

    # Check RNA results
    rna_results = modality_results["rna"]
    assert isinstance(rna_results, pd.DataFrame)
    assert rna_results.shape[0] == minimal_mudata.mod["rna"].n_vars
    assert list(rna_results.index) == list(minimal_mudata.mod["rna"].var.index)
    assert list(rna_results.columns) == ["score1", "score2"]

    # Check protein results
    protein_results = modality_results["protein"]
    assert isinstance(protein_results, pd.DataFrame)
    assert protein_results.shape[0] == minimal_mudata.mod["protein"].n_vars
    assert list(protein_results.index) == list(minimal_mudata.mod["protein"].var.index)
    assert list(protein_results.columns) == ["score1", "score2"]

    # Check that all original results are preserved
    assert len(all_results) == sum(len(df) for df in modality_results.values())
    for modality, results in modality_results.items():
        pd.testing.assert_frame_equal(
            results, all_results.loc[minimal_mudata.mod[modality].var_names]
        )


def test_split_mdata_results_by_modality_errors(minimal_mudata):
    """Test error cases for splitting results by modality."""
    # Create results with wrong index but matching length
    wrong_index_results = pd.DataFrame(
        np.random.randn(len(minimal_mudata.var_names), 2),
        # Create completely different indices that don't overlap with real ones
        index=[f"wrong_var_{i}" for i in range(len(minimal_mudata.var_names))],
        columns=["score1", "score2"],
    )

    # Should raise error due to index mismatch
    with pytest.raises(ValueError, match="Index mismatch in rna"):
        scverse_loading._split_mdata_results_by_modality(
            minimal_mudata, wrong_index_results
        )


def test_multimodality_ontology_config():
    """Test MultiModalityOntologyConfig creation and validation."""
    # Test successful creation with different ontology types
    config_dict = {
        "transcriptomics": {
            "ontologies": None,  # Auto-detect
            "index_which_ontology": None,
        },
        "proteomics": {
            "ontologies": {"uniprot", "pharos"},  # Set of columns
            "index_which_ontology": "uniprot",
        },
        "atac": {
            "ontologies": {"peak1": "peak_id"},  # Dict mapping
            "index_which_ontology": None,
        },
    }
    config = scverse_loading.MultiModalityOntologyConfig.from_dict(config_dict)

    # Test dictionary-like access
    assert len(config) == 3
    assert set(config) == {"transcriptomics", "proteomics", "atac"}

    # Test modality access and type preservation
    transcriptomics = config["transcriptomics"]
    assert transcriptomics.ontologies is None
    assert transcriptomics.index_which_ontology is None

    proteomics = config["proteomics"]
    assert isinstance(proteomics.ontologies, set)
    assert proteomics.ontologies == {"uniprot", "pharos"}
    assert proteomics.index_which_ontology == "uniprot"

    atac = config["atac"]
    assert isinstance(atac.ontologies, dict)
    assert atac.ontologies == {"peak1": "peak_id"}
    assert atac.index_which_ontology is None

    # Test items() method
    for modality, modality_config in config.items():
        assert modality in config_dict
        assert modality_config.ontologies == config_dict[modality]["ontologies"]
        assert (
            modality_config.index_which_ontology
            == config_dict[modality]["index_which_ontology"]
        )

    # Test validation - missing required field
    with pytest.raises(ValueError):
        scverse_loading.MultiModalityOntologyConfig.from_dict(
            {
                "transcriptomics": {
                    "index_which_ontology": "ensembl_gene"  # Missing ontologies field
                }
            }
        )

    # Test validation - wrong type for ontologies
    with pytest.raises(ValueError):
        scverse_loading.MultiModalityOntologyConfig.from_dict(
            {
                "transcriptomics": {
                    "ontologies": "ensembl_gene",  # Should be None, set, or dict
                    "index_which_ontology": "ensembl_gene",
                }
            }
        )

    # Test empty config
    empty_config = scverse_loading.MultiModalityOntologyConfig(root={})
    assert len(empty_config) == 0

    # Test optional index_which_ontology
    minimal_config = scverse_loading.MultiModalityOntologyConfig.from_dict(
        {"transcriptomics": {"ontologies": None}}  # No index_which_ontology
    )
    assert minimal_config["transcriptomics"].index_which_ontology is None


def test_prepare_mudata_results_df(minimal_mudata):
    """Test prepare_mudata_results_df with different ontology configurations.

    The function should:
    1. Use MuData's var/varm tables for the actual data
    2. Use each modality's var table for ontology information
    3. Return a dictionary of DataFrames, one per modality
    4. Each DataFrame should have the ontology columns and any requested data columns
    """
    # Arrange
    config = {
        "rna": {
            "ontologies": None,  # Auto-detect
            "index_which_ontology": "ensembl_gene",  # Rename index to this
        },
        "protein": {
            "ontologies": {
                "protein_id": "ensembl_protein",
                "uniprot": "uniprot",
            },  # Map source column to valid ontology name
            "index_which_ontology": None,
        },
    }

    # First add the source column to the protein modality
    minimal_mudata.mod["protein"].var["protein_id"] = minimal_mudata.mod["protein"].var[
        "uniprot"
    ]

    expected_rna_ontologies = {
        "ensembl_gene",  # From index
        "ensembl_transcript",  # From RNA modality var
        "gene_name",  # From RNA modality var
    }
    expected_protein_ontologies = {
        "uniprot",
        "ensembl_protein",
    }  # Both original and renamed columns

    # Act - Test var table extraction
    var_results = scverse_loading.prepare_mudata_results_df(
        minimal_mudata, mudata_ontologies=config, table_type=ADATA.VAR
    )

    # Assert - Basic structure
    assert set(var_results.keys()) == {"rna", "protein"}

    # Assert - RNA modality
    rna_results = var_results["rna"]
    assert isinstance(rna_results, pd.DataFrame)
    assert rna_results.shape[0] == minimal_mudata.mod["rna"].n_vars
    assert expected_rna_ontologies.issubset(
        set(rna_results.columns)
    ), f"Missing ontology columns: {expected_rna_ontologies - set(rna_results.columns)}"

    # Assert - Protein modality
    protein_results = var_results["protein"]
    assert isinstance(protein_results, pd.DataFrame)
    assert protein_results.shape[0] == minimal_mudata.mod["protein"].n_vars
    assert expected_protein_ontologies.issubset(
        set(protein_results.columns)
    ), f"Missing ontology columns: {expected_protein_ontologies - set(protein_results.columns)}"
    # Check that source column was correctly renamed
    assert "protein_id" not in protein_results.columns
    assert "ensembl_protein" in protein_results.columns
    pd.testing.assert_series_equal(
        protein_results["ensembl_protein"],
        minimal_mudata.mod["protein"].var["protein_id"],
        check_names=False,  # Ignore Series names in comparison
    )

    # Act - Test varm table extraction with explicit results_attrs
    varm_results = scverse_loading.prepare_mudata_results_df(
        minimal_mudata,
        mudata_ontologies=config,
        table_type=ADATA.VARM,
        table_name="gene_scores",
        results_attrs=["score1", "score2"],
        table_colnames=["score1", "score2", "score3"],
    )

    # Assert - RNA varm results
    rna_varm = varm_results["rna"]
    expected_rna_varm_cols = expected_rna_ontologies | {"score1", "score2"}
    assert isinstance(rna_varm, pd.DataFrame)
    assert rna_varm.shape[0] == minimal_mudata.mod["rna"].n_vars
    assert expected_rna_varm_cols.issubset(
        set(rna_varm.columns)
    ), f"Missing columns: {expected_rna_varm_cols - set(rna_varm.columns)}"

    # Assert - Protein varm results
    protein_varm = varm_results["protein"]
    expected_protein_varm_cols = expected_protein_ontologies | {"score1", "score2"}
    assert isinstance(protein_varm, pd.DataFrame)
    assert protein_varm.shape[0] == minimal_mudata.mod["protein"].n_vars
    assert expected_protein_varm_cols.issubset(
        set(protein_varm.columns)
    ), f"Missing columns: {expected_protein_varm_cols - set(protein_varm.columns)}"


def test_prepare_mudata_results_df_errors(minimal_mudata):
    """Test error cases for prepare_mudata_results_df."""
    # Test missing modality configuration
    with pytest.raises(
        ValueError, match="Missing ontology configurations for modalities"
    ):
        scverse_loading.prepare_mudata_results_df(
            minimal_mudata,
            mudata_ontologies={"rna": {"ontologies": None}},
            table_type=ADATA.VAR,
        )

    # Test invalid table type
    with pytest.raises(ValueError, match="table_type must be one of"):
        scverse_loading.prepare_mudata_results_df(
            minimal_mudata,
            mudata_ontologies={
                "rna": {"ontologies": None},
                "protein": {"ontologies": None},
            },
            table_type="invalid_type",
        )

    # Test missing table_colnames for varm
    # Add varm table to RNA modality first
    minimal_mudata.mod["rna"].varm["scores"] = np.random.randn(
        minimal_mudata.mod["rna"].n_vars, 3
    )
    minimal_mudata.mod["rna"].uns["scores_features"] = ["score1", "score2", "score3"]
    with pytest.raises(ValueError, match="table_name 'scores' not found in adata.varm"):
        scverse_loading.prepare_mudata_results_df(
            minimal_mudata,
            mudata_ontologies={
                "rna": {"ontologies": None},
                "protein": {"ontologies": None},
            },
            table_type=ADATA.VARM,
            table_name="scores",
            results_attrs=["score1", "score2"],  # Missing table_colnames
        )


def test_prepare_mudata_results_df_adata_level(minimal_mudata):
    """Test prepare_mudata_results_df with level='adata' to extract adata-specific attributes."""
    # Arrange - Add some adata-specific var attributes that don't exist at MuData level
    minimal_mudata.mod["rna"].var["rna_specific"] = [
        f"rna_val_{i}" for i in range(minimal_mudata.mod["rna"].n_vars)
    ]
    minimal_mudata.mod["protein"].var["protein_specific"] = [
        f"prot_val_{i}" for i in range(minimal_mudata.mod["protein"].n_vars)
    ]

    config = {
        "rna": {
            "ontologies": None,  # Auto-detect
            "index_which_ontology": None,
        },
        "protein": {
            "ontologies": {"uniprot"},
            "index_which_ontology": None,
        },
    }

    # Act - Test var table extraction at adata level
    var_results = scverse_loading.prepare_mudata_results_df(
        minimal_mudata,
        mudata_ontologies=config,
        table_type=ADATA.VAR,
        level=SCVERSE_DEFS.ADATA,
    )

    # Assert - Check that adata-specific attributes are included
    rna_results = var_results["rna"]
    protein_results = var_results["protein"]

    # RNA should have its specific attribute
    assert "rna_specific" in rna_results.columns
    assert "gene_name" in rna_results.columns  # From original RNA var
    assert "ensembl_transcript" in rna_results.columns  # From original RNA var

    # Protein should have its specific attribute
    assert "protein_specific" in protein_results.columns
    assert "uniprot" in protein_results.columns  # From original protein var

    # Check values are correct
    pd.testing.assert_series_equal(
        rna_results["rna_specific"],
        minimal_mudata.mod["rna"].var["rna_specific"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        protein_results["protein_specific"],
        minimal_mudata.mod["protein"].var["protein_specific"],
        check_names=False,
    )


def test_prepare_mudata_results_df_mdata_vs_adata_level(minimal_mudata):
    """Test that level='mdata' vs level='adata' produce different results when appropriate."""
    # Arrange - Add adata-specific varm tables
    rna_varm = np.random.randn(minimal_mudata.mod["rna"].n_vars, 2)
    protein_varm = np.random.randn(minimal_mudata.mod["protein"].n_vars, 2)

    minimal_mudata.mod["rna"].varm["modality_scores"] = rna_varm
    minimal_mudata.mod["protein"].varm["modality_scores"] = protein_varm

    config = {
        "rna": {"ontologies": None},
        "protein": {"ontologies": {"uniprot"}},
    }

    # Act - Extract using both levels
    mdata_results = scverse_loading.prepare_mudata_results_df(
        minimal_mudata,
        mudata_ontologies=config,
        table_type=ADATA.VARM,
        table_name="gene_scores",  # This exists at MuData level
        results_attrs=["score1"],
        table_colnames=["score1", "score2", "score3"],
        level=SCVERSE_DEFS.MDATA,
    )

    adata_results = scverse_loading.prepare_mudata_results_df(
        minimal_mudata,
        mudata_ontologies=config,
        table_type=ADATA.VARM,
        table_name="modality_scores",  # This exists at modality level
        results_attrs=[
            "0"
        ],  # Using column index as string since we don't have explicit names
        table_colnames=["0", "1"],
        level=SCVERSE_DEFS.ADATA,
    )

    # Assert - Both should succeed but access different data
    assert "score1" in mdata_results["rna"].columns
    assert "score1" in mdata_results["protein"].columns

    assert "0" in adata_results["rna"].columns
    assert "0" in adata_results["protein"].columns

    # The values should be different since they come from different varm tables
    # (We can't easily check exact values due to random generation, but structure should be correct)
    assert mdata_results["rna"].shape[0] == minimal_mudata.mod["rna"].n_vars
    assert adata_results["rna"].shape[0] == minimal_mudata.mod["rna"].n_vars


def test_prepare_mudata_results_df_level_validation(minimal_mudata):
    """Test that invalid level parameter raises appropriate error."""
    config = {
        "rna": {"ontologies": None},
        "protein": {"ontologies": {"uniprot"}},
    }

    with pytest.raises(
        ValueError,
        match=r"level must be one of \['adata', 'mdata'\], got invalid_level",
    ):
        scverse_loading.prepare_mudata_results_df(
            minimal_mudata,
            mudata_ontologies=config,
            table_type=ADATA.VAR,
            level="invalid_level",
        )
