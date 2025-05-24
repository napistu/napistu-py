import anndata
from typing import Optional, List, Union
import pandas as pd
import numpy as np

from napistu.scverse.constants import ADATA, ADATA_DICTLIKE_ATTRS, ADATA_IDENTITY_ATTRS

def _load_raw_table(
    adata: anndata.AnnData,
    table_type: str,
    table_name: Optional[str] = None
):
    
    """
    Load an AnnData table.
    
    This function loads an AnnData table and returns it as a pd.DataFrame.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to load the table from.
    table_type : str
        The type of table to load.
    table_name : str, optional
        The name of the table to load.

    Returns
    -------
    pd.DataFrame
        The loaded table.
    """
    
    if table_type not in [*ADATA_DICTLIKE_ATTRS, *ADATA_IDENTITY_ATTRS]:
        raise ValueError(f"table_type {table_type} is not a valid AnnData attribute. Valid attributes are: {ADATA_DICTLIKE_ATTRS + ADATA_IDENTITY_ATTRS}")

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
    adata: anndata.AnnData,
    attr_name: str,
    table_name: Optional[str] = None
):
    """
    Generic function to get a table from a dict-like AnnData attribute (varm, layers, etc.)
    
    Args:
        adata: AnnData object
        attr_name: Name of the attribute ('varm', 'layers', etc.)
        table_name: Specific table name to retrieve, or None for auto-selection
    """

    if attr_name not in ADATA_DICTLIKE_ATTRS:
        raise ValueError(f"attr_name {attr_name} is not a dict-like AnnData attribute. Valid attributes are: {VALID_ATTRS}")

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
    
#_load_raw_table(adata, "X")
#_load_raw_table(adata, "var", "ignored")
# _load_raw_table(adata, "layers")
# _load_raw_table(adata, "foo")

def _select_results_attrs(
    adata: anndata.AnnData,
    raw_results_table: Union[pd.DataFrame, np.ndarray],
    results_attrs: Optional[List[str]] = None
) -> pd.DataFrame:

    """
    Select results attributes from an AnnData object.

    This function selects results attributes from raw_results_table derived
    from an AnnData object and converts them if needed to a pd.DataFrame
    with appropriate indicies.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the results to be formatted.
    raw_results_table : pd.DataFrame or np.ndarray
        The raw results table to be formatted.
    results_attrs : list of str, optional
        The attributes to extract from the raw_results_table.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the formatted results.
    """
    if isinstance(raw_results_table, pd.DataFrame):
        if results_attrs is not None:
            results_table_data = raw_results_table.loc[results_attrs]
        else:
            results_table_data = raw_results_table
    else:
        if results_attrs is not None:
            # Check that results_attrs exist in adata.obs.index
            valid_obs = adata.obs.index.tolist()
            
            invalid_results_attrs = [x for x in results_attrs if x not in valid_obs]
            if len(invalid_results_attrs) > 0:
                raise ValueError(f"The following results attributes are not present in the AnnData object's obs index: {invalid_results_attrs}")

            # Find positions of desired rows in adata.obs.index
            row_positions = [adata.obs.index.get_loc(attr) for attr in results_attrs]
            
            # Select ROWS from numpy array using positions
            selected_array = raw_results_table[row_positions, :]
            
            # Convert to DataFrame and set row names to results_attrs
            results_table_data = pd.DataFrame(
                selected_array,
                index = results_attrs,
                columns = adata.var.index
                ).T
        else:
            # Convert entire array to DataFrame
            results_table_data = pd.DataFrame(
                raw_results_table,
                index = adata.obs.index,
                columns = adata.var.index
            ).T

    return results_table_data