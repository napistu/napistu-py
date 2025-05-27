import copy
import re
from typing import Union, List, Optional, Callable, Dict

import pandas as pd

from napistu import sbml_dfs_core
from napistu.constants import ENTITIES_W_DATA

def _select_sbml_dfs_data_table(sbml_dfs: sbml_dfs_core.SBML_dfs, table_name: Optional[str] = None, table_type: str = "species") -> pd.DataFrame:

    """
    Select an SBML_dfs data table by name and type.

    This function validates the table type and name and returns the table.

    Parameters
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        The sbml_dfs object containing the data tables.
    table_name: str, optional
        The name of the table to select. If not provided, the first table of the given type will be returned.
    table_type: str, optional
        The type of table to select. Must be one of {VALID_SBML_DFS_DATA_TYPES}.

    Returns
    -------
    data_table: pd.DataFrame
    """

    # validate table_type
    if table_type not in ENTITIES_W_DATA:
        raise ValueError(f"Invalid table_type: {table_type}. Must be one of {ENTITIES_W_DATA}")    
    table_type_data_attr = f"{table_type}_data"

    # validate table_name
    data_attr = getattr(sbml_dfs, table_type_data_attr)
    
    if len(data_attr) == 0:
        raise ValueError(f"No {table_type} data found in sbml_dfs")
    valid_table_names = list(data_attr.keys())

    if table_name is None:
        if len(data_attr) != 1:
            raise ValueError(f"Expected a single {table_type} data table but found {len(data_attr)}")
        table_name = valid_table_names[0]
    
    if table_name not in valid_table_names:
        raise ValueError(f"Invalid table_name: {table_name}. Must be one of {valid_table_names}")
    
    data_table = data_attr[table_name]

    return data_table

def _select_sbml_dfs_data_table_attrs(data_table: pd.DataFrame, attribute_names: Union[str, List[str], Dict[str, str]], table_type: Optional[str] = "species") -> pd.DataFrame:
    """
    Select attributes from an sbml_dfs data table.

    This function validates the attribute names and returns the selected attributes.

    Parameters
    ----------
    data_table: pd.DataFrame
        The data table to select attributes from.
    attribute_names: str or list of str, optional
        Either:
            - The name of the attribute to add to the graph. 
            - A list of attribute names to add to the graph.
            - A regular expression pattern to match attribute names.
            - A dictionary with attributes as names and re-named attributes as values.
            - If None, all attributes in the species_data table will be added.
    table_type: str, optional
        The type of table to use. Must be one of {VALID_SBML_DFS_DATA_TYPES}. (Only used for error messages).

    Returns
    -------
    selected_data_table: pd.DataFrame
        pd.DataFrame containing the selected attributes.
    """
    valid_data_table_columns = data_table.columns.tolist()
    
    # Initialize attribute_names_list
    attribute_names_list = []

    # select the attributes to add
    if attribute_names is None:
        attribute_names_list = valid_data_table_columns
    elif isinstance(attribute_names, str):
        # try to find an exact match
        if attribute_names in valid_data_table_columns:
            attribute_names_list = [attribute_names]
        else:
            # try to find a regex match
            attribute_names_list = [attr for attr in valid_data_table_columns if re.match(attribute_names, attr)]
            if len(attribute_names_list) == 0:
                raise ValueError(f"No attributes found matching {attribute_names} as a literal or regular expression. Valid attributes: {valid_data_table_columns}")
    elif isinstance(attribute_names, list):
        # Validate that all attributes exist
        invalid_attributes = [attr for attr in attribute_names if attr not in valid_data_table_columns]
        if len(invalid_attributes) > 0:
            raise ValueError(f"The following attributes were missing from the {table_type}_data table: {invalid_attributes}. Valid attributes: {valid_data_table_columns}")
        attribute_names_list = attribute_names
    elif isinstance(attribute_names, dict):
        # validate the keys exist in the table
        invalid_keys = [key for key in attribute_names.keys() if key not in valid_data_table_columns]
        if len(invalid_keys) > 0:
            raise ValueError(f"The following source columns were missing from the {table_type}_data table: {invalid_keys}. Valid columns: {valid_data_table_columns}")
        
        # validate that new column names don't conflict with existing ones
        # except when a column is being renamed to itself
        conflicting_names = [
            new_name for old_name, new_name in attribute_names.items()
            if new_name in valid_data_table_columns and new_name != old_name
        ]
        if conflicting_names:
            raise ValueError(f"The following new column names conflict with existing columns: {conflicting_names}")
        
        attribute_names_list = list(attribute_names.keys())
        if len(attribute_names_list) == 0:
            raise ValueError(f"No attributes found in the dictionary. Valid attributes: {valid_data_table_columns}")
    else:
        # shouldn't be reached - for clarity
        raise ValueError(f"Invalid type for attribute_names: {type(attribute_names)}. Must be str, list, dict, or None.")

    # return the selected attributes
    selected_data_table = data_table[attribute_names_list]
    
    # rename columns if a dictionary was provided
    if isinstance(attribute_names, dict):
        selected_data_table = selected_data_table.rename(columns=attribute_names)
    
    return selected_data_table