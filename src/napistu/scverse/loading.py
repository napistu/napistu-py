import copy
import logging 
from typing import Optional, List, Union, Set, Dict

import anndata
import pandas as pd
import mudata
import numpy as np

from napistu import mechanism_matching
from napistu.scverse.constants import ADATA, ADATA_DICTLIKE_ATTRS, ADATA_IDENTITY_ATTRS, ADATA_FEATURELEVEL_ATTRS, ADATA_ARRAY_ATTRS

logger = logging.getLogger(__name__)

def prepare_scverse_results_df(
    adata: Union[anndata.AnnData, mudata.MuData],
    table_type: str = ADATA.VAR,
    table_name: Optional[str] = None,
    results_attrs: Optional[List[str]] = None,
    ontologies: Optional[Union[Set[str], Dict[str, str]]] = None,
    index_which_ontology: Optional[str] = None,
    table_colnames: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare a results table from an AnnData object for use in Napistu.

    This function extracts a table from an AnnData object and formats it for use in Napistu.
    The returned DataFrame will always include systematic identifiers from the var table,
    along with the requested results data.

    Parameters
    ----------
    adata : anndata.AnnData or mudata.MuData
        The AnnData or MuData object containing the results to be formatted.
    table_type : str, optional
        The type of table to extract from the AnnData object. Must be one of: "var", "varm", or "X".
    table_name : str, optional
        The name of the table to extract from the AnnData object.
    results_attrs : list of str, optional
        The attributes to extract from the table.
    index_which_ontology : str, optional
        The ontology to use for the systematic identifiers. This column will be pulled out of the
        index renamed to the ontology name, and added to the results table as a new column with
        the same name. Must not already exist in var table.
    ontologies : Optional[Union[Set[str], Dict[str, str]]], default=None
        Either:
        - Set of columns to treat as ontologies (these should be entries in ONTOLOGIES_LIST )
        - Dict mapping wide column names to ontology names in the ONTOLOGIES_LIST controlled vocabulary
        - None to automatically detect valid ontology columns based on ONTOLOGIES_LIST

        If index_which_ontology is defined, it should be represented in these ontologies.
    table_colnames : Optional[List[str]], optional
        Column names for varm tables. Required when table_type is "varm". Ignored otherwise.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the formatted results with systematic identifiers.
        The index will match the var_names of the AnnData object.

    Raises
    ------
    ValueError
        If table_type is not one of: "var", "varm", or "X"
        If index_which_ontology already exists in var table
    """
    
    if table_type not in ADATA_FEATURELEVEL_ATTRS:
        raise ValueError(f"table_type must be one of {ADATA_FEATURELEVEL_ATTRS}, got {table_type}")

    # pull out the table containing results
    raw_results_table = _load_raw_table(adata, table_type, table_name)

    # convert the raw results to a pd.DataFrame with rows corresponding to vars and columns
    # being attributes of interest
    results_data_table = _select_results_attrs(adata, raw_results_table, table_type, results_attrs, table_colnames)

    # Load var_table which contains systematic identifiers
    var_table = adata.var.copy()  # Make a copy to avoid modifying original

    # Extract index as ontology if requested
    if index_which_ontology is not None:
        if index_which_ontology in var_table.columns:
            raise ValueError(
                f"Cannot use '{index_which_ontology}' as index_which_ontology - "
                f"column already exists in var table"
            )
        # Add the column with index values
        var_table[index_which_ontology] = var_table.index

    # if ontologies is a dict, we actually want the keys but the previous _validate_wide_ontologies() will
    # still validate that these keys can be transformed into valid ontology names
    matching_ontologies = mechanism_matching._validate_wide_ontologies(var_table, ontologies)
    if isinstance(ontologies, dict):
        var_ontologies = var_table.loc[:, ontologies.keys()]
    else:
        var_ontologies = var_table.loc[:, list(matching_ontologies)]

    # Combine ontologies with results data
    # Both should have the same index (var_names)
    results_table = pd.concat([var_ontologies, results_data_table], axis=1)

    return results_table


def _load_raw_table(
    adata: Union[anndata.AnnData, mudata.MuData],
    table_type: str,
    table_name: Optional[str] = None
) -> Union[pd.DataFrame, np.ndarray]:
    
    """
    Load an AnnData table.
    
    This function loads an AnnData table and returns it as a pd.DataFrame.
    
    Parameters
    ----------
    adata : anndata.AnnData or mudata.MuData
        The AnnData or MuData object to load the table from.
    table_type : str
        The type of table to load.
    table_name : str, optional
        The name of the table to load.

    Returns
    -------
    pd.DataFrame or np.ndarray
        The loaded table.
    """
    
    valid_attrs = ADATA_DICTLIKE_ATTRS | ADATA_IDENTITY_ATTRS
    if table_type not in valid_attrs:
        raise ValueError(f"table_type {table_type} is not a valid AnnData attribute. Valid attributes are: {valid_attrs}")

    if table_type in ADATA_IDENTITY_ATTRS:
        if table_name is not None:
            logger.debug(f"table_name {table_name} is not None, but table_type is in IDENTITY_TABLES. "
                        f"table_name will be ignored.")
        return getattr(adata, table_type)

    # pull out a dict-like attribute
    return _get_table_from_dict_attr(
        adata,
        table_type,
        table_name
    )
    
    
def _get_table_from_dict_attr(
    adata: Union[anndata.AnnData, mudata.MuData],
    attr_name: str,
    table_name: Optional[str] = None
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Get a table from a dict-like AnnData attribute (varm, layers, etc.)
    
    Parameters
    ----------
    adata : anndata.AnnData or mudata.MuData
        The AnnData or MuData object to load the table from
    attr_name : str
        Name of the attribute ('varm', 'layers', etc.)
    table_name : str, optional
        Specific table name to retrieve. If None and only one table exists,
        that table will be returned. If None and multiple tables exist,
        raises ValueError
        
    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        The table data. For array-type attributes (varm, varp, X, layers),
        returns numpy array. For other attributes, returns DataFrame
        
    Raises
    ------
    ValueError
        If attr_name is not a valid dict-like attribute
        If no tables found in the attribute
        If multiple tables found and table_name not specified
        If specified table_name not found
    """

    if attr_name not in ADATA_DICTLIKE_ATTRS:
        raise ValueError(f"attr_name {attr_name} is not a dict-like AnnData attribute. Valid attributes are: {ADATA_DICTLIKE_ATTRS}")

    attr_dict = getattr(adata, attr_name)
    available_tables = list(attr_dict.keys())
    
    if len(available_tables) == 0:
        raise ValueError(f"No tables found in adata.{attr_name}")
    elif (len(available_tables) > 1) and (table_name is None):
        raise ValueError(f"Multiple tables found in adata.{attr_name} and table_name is not specified. "
                        f"Available: {available_tables}")
    elif (len(available_tables) == 1) and (table_name is None):
        return attr_dict[available_tables[0]]
    elif table_name not in available_tables:
        raise ValueError(f"table_name '{table_name}' not found in adata.{attr_name}. "
                        f"Available: {available_tables}")
    else:
        return attr_dict[table_name]
    

def _select_results_attrs(
    adata: anndata.AnnData,
    raw_results_table: Union[pd.DataFrame, np.ndarray],
    table_type: str,
    results_attrs: Optional[List[str]] = None,
    table_colnames: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Select results attributes from an AnnData object.

    This function selects results attributes from raw_results_table derived
    from an AnnData object and converts them if needed to a pd.DataFrame
    with appropriate indices.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the results to be formatted.
    raw_results_table : pd.DataFrame or np.ndarray
        The raw results table to be formatted.
    table_type: str,
        The type of table `raw_results_table` refers to.
    results_attrs : list of str, optional
        The attributes to extract from the raw_results_table.
    table_colnames: list of str, optional,
        If `table_type` is `varm`, this is the names of all columns (e.g., PC1, PC2, etc.). Ignored otherwise

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the formatted results.
    """
    logger.debug(f"_select_results_attrs called with table_type={table_type}, results_attrs={results_attrs}")

    # Validate that array-type tables are not passed as DataFrames
    if table_type in ADATA_ARRAY_ATTRS and isinstance(raw_results_table, pd.DataFrame):
        raise ValueError(f"Table type {table_type} must be a numpy array, not a DataFrame. Got {type(raw_results_table)}")

    if isinstance(raw_results_table, pd.DataFrame):
        if results_attrs is not None:
            results_table_data = raw_results_table.loc[results_attrs]
        else:
            results_table_data = raw_results_table
        return results_table_data

    # Convert sparse matrix to dense if needed
    if hasattr(raw_results_table, 'toarray'):
        raw_results_table = raw_results_table.toarray()

    valid_attrs = _get_valid_attrs_for_feature_level_array(
        adata,
        table_type,
        raw_results_table,
        table_colnames
    )

    if results_attrs is not None:
        invalid_results_attrs = [x for x in results_attrs if x not in valid_attrs]
        if len(invalid_results_attrs) > 0:
            raise ValueError(f"The following results attributes are not valid: {invalid_results_attrs}")

        # Get positions based on table type
        if table_type == ADATA.VARM:
            positions = [table_colnames.index(attr) for attr in results_attrs]
            selected_array = raw_results_table[:, positions]
        elif table_type == ADATA.VARP:
            positions = [adata.var.index.get_loc(attr) for attr in results_attrs]
            selected_array = raw_results_table[:, positions]
        else:  # X or layers
            positions = [adata.obs.index.get_loc(attr) for attr in results_attrs]
            selected_array = raw_results_table[positions, :]

        results_table_data = _create_results_df(selected_array, results_attrs, adata.var.index, table_type)
    else:
        results_table_data = _create_results_df(raw_results_table, valid_attrs, adata.var.index, table_type)

    return results_table_data

def _get_valid_attrs_for_feature_level_array(
    adata: anndata.AnnData,
    table_type: str,
    raw_results_table: np.ndarray,
    table_colnames: Optional[List[str]] = None
) -> list[str]:
    """
    Get valid attributes for a feature-level array.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object
    table_type : str
        The type of table
    raw_results_table : np.ndarray
        The raw results table for dimension validation
    table_colnames : Optional[List[str]]
        Column names for varm tables
        
    Returns
    -------
    list[str]
        List of valid attributes for this table type
        
    Raises
    ------
    ValueError
        If table_type is invalid or if table_colnames validation fails for varm tables
    """
    if table_type not in ADATA_ARRAY_ATTRS:
        raise ValueError(f"table_type {table_type} is not a valid AnnData array attribute. Valid attributes are: {ADATA_ARRAY_ATTRS}")

    if table_type in [ADATA.X, ADATA.LAYERS]:
        valid_attrs = adata.obs.index.tolist()
    elif table_type == ADATA.VARP:
        valid_attrs = adata.var.index.tolist()
    else:  # varm
        if table_colnames is None:
            raise ValueError("table_colnames is required for varm tables")
        if len(table_colnames) != raw_results_table.shape[1]:
            raise ValueError(f"table_colnames must have length {raw_results_table.shape[1]}")
        valid_attrs = table_colnames

    return valid_attrs


def _create_results_df(
    array: np.ndarray,
    attrs: List[str],
    var_index: pd.Index,
    table_type: str
) -> pd.DataFrame:
    """Create a DataFrame with the right orientation based on table type.
    
    For varm/varp tables:
        - rows are vars (var_index)
        - columns are attrs (features/selected vars)
    For X/layers:
        - rows are attrs (selected observations)
        - columns are vars (var_index)
        - then transpose to get vars as rows
    """
    if table_type in [ADATA.VARM, ADATA.VARP]:
        return pd.DataFrame(
            array,
            index=var_index,
            columns=attrs
        )
    else:
        return pd.DataFrame(
            array,
            index=attrs,
            columns=var_index
        ).T

def split_mdata_results_by_modality(
    mdata: mudata.MuData,
    results_data_table: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Split a results table by modality and verify compatibility with var tables.
    
    Parameters
    ----------
    mdata : mudata.MuData
        MuData object containing multiple modalities
    results_data_table : pd.DataFrame
        Results table with vars as rows, typically from prepare_anndata_results_df()
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with modality names as keys and DataFrames as values.
        Each DataFrame contains just the results for that modality.
        The index of each DataFrame is guaranteed to match the corresponding
        modality's var table for later merging.
        
    Raises
    ------
    ValueError
        If any modality's vars are not found in the results table
        If any modality's results have different indices than its var table
    """
    # Initialize results dictionary
    results: Dict[str, pd.DataFrame] = {}
    
    # Process each modality
    for modality in mdata.mod.keys():
        # Get the var_names for this modality
        mod_vars = mdata.mod[modality].var_names
        
        # Check if all modality vars exist in results
        missing_vars = set(mod_vars) - set(results_data_table.index)
        if missing_vars:
            raise ValueError(
                f"Index mismatch in {modality}: vars {missing_vars} not found in results table"
            )
        
        # Extract results for this modality
        mod_results = results_data_table.loc[mod_vars]
        
        # Verify index alignment with var table
        if not mod_results.index.equals(mdata.mod[modality].var.index):
            raise ValueError(
                f"Index mismatch in {modality}: var table and results subset have different indices"
            )
        
        # Store just the results
        results[modality] = mod_results
    
    return results

