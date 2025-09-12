from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import pandas as pd

from napistu import sbml_dfs_core
from napistu.constants import ENTITIES_W_DATA, SBML_DFS
from napistu.network.constants import DEFAULT_WT_TRANS, NAPISTU_GRAPH, WEIGHTING_SPEC

if TYPE_CHECKING:
    from napistu.network.ng_core import NapistuGraph

logger = logging.getLogger(__name__)


def add_results_table_to_graph(
    napistu_graph: NapistuGraph,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    attribute_names: Optional[Union[str, List[str]]] = None,
    table_name: str = None,
    table_type: str = SBML_DFS.SPECIES,
    graph_attr_modified: str = NAPISTU_GRAPH.VERTICES,
    transformation: Optional[Callable] = None,
    custom_transformations: Optional[Dict[str, Callable]] = None,
    mode="fresh",
    overwrite=False,
    inplace: bool = True,
):
    """
    Add Results Table to Graph

    This function extracts one or more attributes from an sbml_dfs species_data table, applies an optional transformation, and adds the result as a vertex attributes to a Napistu graph.

    Parameters
    ----------
    napistu_graph: NapistuGraph
        The Napistu graph to which attributes will be added.
    sbml_dfs: sbml_dfs_core.SBML_dfs
        The sbml_dfs object containing the species_data table.
    attribute_names: str or list of str, optional
        Either:
            - The name of the attribute to add to the graph.
            - A list of attribute names to add to the graph.
            - A regular expression pattern to match attribute names.
            - If None, all attributes in the species_data table will be added.
    table_name: str, optional
        The name of the species_data table to use. If not provided, then a single table will be expected in species_data.
    table_type: str, optional
        The type of table to use (e.g., species for species_data, reactions for reaction_data). Currently, only species is supproted.
    graph_attr_modified: str, optional
        The type of graph attribute to modify: vertices or edges. Certain table_types can only modify vertices (species) while others can modify either vertices or edges (reactions). Currently, ignore.
    transformation: str or Callable, optional
        Either:
            - the name of a function in custom_transformations or the built-in transformations.
            - A function to apply to the attribute.
        If not provided, the attribute will not be transformed.
    custom_transformations: dict, optional
        A dictionary of custom transformations which could be applied to the attributes. The keys are the transformation names and the values are the transformation functions.
    mode: str, optional
        The mode to use for adding the attributes. Must be one of "fresh" or "extend".
    overwrite: bool, optional
        If True, the attributes will be overwritten if they already exist.
    inplace: bool, optional
        If True, the attribute will be added to the graph in place. If False, a new graph will be returned.

    Returns
    -------
    napistu_graph: NapistuGraph
        If inplace is False, the Napistu graph with attributes added.
    """

    if not inplace:
        napistu_graph = napistu_graph.copy()

    if table_type not in ENTITIES_W_DATA:
        raise ValueError(
            f"Invalid table_type: {table_type}. Must be one of {ENTITIES_W_DATA}"
        )
    if table_type == SBML_DFS.REACTIONS:
        raise NotImplementedError("Reactions are not yet supported")

    if graph_attr_modified != NAPISTU_GRAPH.VERTICES:
        raise NotImplementedError(
            f"graph_attr_modified must be {NAPISTU_GRAPH.VERTICES}"
        )

    # load the to-be-added table
    logger.debug(f"Loading table {table_name} from {table_type}_data")
    data_table = _select_sbml_dfs_data_table(sbml_dfs, table_name, table_type)

    # filter to attributes of interest
    logger.debug("Creating a mapping of attributes to add")
    attribute_mapping = _create_data_table_column_mapping(
        data_table, attribute_names, table_type
    )

    if transformation is None:
        transformation = DEFAULT_WT_TRANS

    # create the configuration dict which is used by lower-level functions
    species_attrs = _create_graph_attrs_config(
        column_mapping=attribute_mapping,
        data_type=table_type,
        table_name=table_name,
        transformation=transformation,
    )

    # add the attribute to the graph
    napistu_graph.set_graph_attrs(
        species_attrs,
        mode=mode,
        overwrite=overwrite,
        custom_transformations=custom_transformations,
    )

    # add the new attributes
    napistu_graph.add_vertex_data(sbml_dfs, mode=mode, overwrite=overwrite)
    napistu_graph.transform_vertices(custom_transformations=custom_transformations)

    return napistu_graph if not inplace else None


def _select_sbml_dfs_data_table(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    table_name: Optional[str] = None,
    table_type: str = SBML_DFS.SPECIES,
) -> pd.DataFrame:
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
    entity_data: pd.DataFrame
    """

    # validate table_type
    if table_type not in ENTITIES_W_DATA:
        raise ValueError(
            f"Invalid table_type: {table_type}. Must be one of {ENTITIES_W_DATA}"
        )
    table_type_data_attr = f"{table_type}_data"

    # validate table_name
    data_attr = getattr(sbml_dfs, table_type_data_attr)

    if len(data_attr) == 0:
        raise ValueError(f"No {table_type} data found in sbml_dfs")
    valid_table_names = list(data_attr.keys())

    if table_name is None:
        if len(data_attr) != 1:
            raise ValueError(
                f"Expected a single {table_type} data table but found {len(data_attr)}"
            )
        table_name = valid_table_names[0]

    if table_name not in valid_table_names:
        raise ValueError(
            f"Invalid table_name: {table_name}. Must be one of {valid_table_names}"
        )

    entity_data = data_attr[table_name]

    return entity_data


def _create_data_table_column_mapping(
    entity_data: pd.DataFrame,
    attribute_names: Union[str, List[str], Dict[str, str]],
    table_type: Optional[str] = SBML_DFS.SPECIES,
) -> Dict[str, str]:
    """
    Select attributes from an sbml_dfs data table.

    This function validates the attribute names and returns a mapping of original names to new names.

    Parameters
    ----------
    entity_data: pd.DataFrame
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
    Dict[str, str]
        A dictionary mapping original column names to their new names.
        For non-renamed columns, the mapping will be identity (original -> original).
    """
    valid_data_table_columns = entity_data.columns.tolist()

    # select the attributes to add
    if attribute_names is None:
        # For None, create identity mapping for all columns
        return {col: col for col in valid_data_table_columns}
    elif isinstance(attribute_names, str):
        # try to find an exact match
        if attribute_names in valid_data_table_columns:
            return {attribute_names: attribute_names}
        else:
            # try to find a regex match
            matching_attrs = [
                attr
                for attr in valid_data_table_columns
                if re.match(attribute_names, attr)
            ]
            if len(matching_attrs) == 0:
                raise ValueError(
                    f"No attributes found matching {attribute_names} as a literal or regular expression. Valid attributes: {valid_data_table_columns}"
                )
            return {attr: attr for attr in matching_attrs}
    elif isinstance(attribute_names, list):
        # Validate that all attributes exist
        invalid_attributes = [
            attr for attr in attribute_names if attr not in valid_data_table_columns
        ]
        if len(invalid_attributes) > 0:
            raise ValueError(
                f"The following attributes were missing from the {table_type}_data table: {invalid_attributes}. Valid attributes: {valid_data_table_columns}"
            )
        return {attr: attr for attr in attribute_names}
    elif isinstance(attribute_names, dict):
        # validate the keys exist in the table
        invalid_keys = [
            key for key in attribute_names.keys() if key not in valid_data_table_columns
        ]
        if len(invalid_keys) > 0:
            raise ValueError(
                f"The following source columns were missing from the {table_type}_data table: {invalid_keys}. Valid columns: {valid_data_table_columns}"
            )

        # validate that new column names don't conflict with existing ones
        # except when a column is being renamed to itself
        conflicting_names = [
            new_name
            for old_name, new_name in attribute_names.items()
            if new_name in valid_data_table_columns and new_name != old_name
        ]
        if conflicting_names:
            raise ValueError(
                f"The following new column names conflict with existing columns: {conflicting_names}"
            )

        if len(attribute_names) == 0:
            raise ValueError(
                f"No attributes found in the dictionary. Valid attributes: {valid_data_table_columns}"
            )

        return attribute_names
    else:
        # shouldn't be reached - for clarity
        raise ValueError(
            f"Invalid type for attribute_names: {type(attribute_names)}. Must be str, list, dict, or None."
        )


def _create_graph_attrs_config(
    column_mapping: Dict[str, str],
    data_type: Optional[str],
    table_name: str,
    transformation: str = DEFAULT_WT_TRANS,
) -> Union[Dict[str, Dict[str, Dict[str, str]]], Dict[str, Dict[str, str]]]:
    """
    Create a configuration dictionary for graph attributes.

    Parameters
    ----------
    column_mapping : Dict[str, str]
        A dictionary mapping original column names to their new names in the graph
    data_type : str or None
        The type of data (e.g. "species", "reactions"). If None, returns the inner dict directly.
    table_name : str
        The name of the table containing the data
    transformation : str, optional
        The transformation to apply to the data, by default "identity"

    Returns
    -------
    Dict[str, Dict[str, Dict[str, str]]] or Dict[str, Dict[str, str]]
        A nested dictionary containing the graph attributes configuration
        If data_type is None, returns the inner dict directly:
        {
            new_col_name: {
                "table": table_name,
                "variable": original_col_name,
                "trans": transformation
            }
        }
        Otherwise, returns the full nested structure:
        {
            data_type: {
                new_col_name: {
                    "table": table_name,
                    "variable": original_col_name,
                    "trans": transformation
                }
            }
        }
    """
    inner_dict = {}

    for original_col, new_col in column_mapping.items():
        inner_dict[new_col] = {
            WEIGHTING_SPEC.TABLE: table_name,
            WEIGHTING_SPEC.VARIABLE: original_col,
            WEIGHTING_SPEC.TRANSFORMATION: transformation,
        }

    if data_type is None:
        return inner_dict
    else:
        return {data_type: inner_dict}
