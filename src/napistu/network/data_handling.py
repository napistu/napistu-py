import logging
import re
from typing import Union, List, Optional, Callable, Dict

import pandas as pd

from napistu import sbml_dfs_core
from napistu.network import net_create
from napistu.network.ng_core import NapistuGraph

from napistu.constants import SBML_DFS, ENTITIES_W_DATA
from napistu.network.constants import NAPISTU_GRAPH, DEFAULT_WT_TRANS, WEIGHTING_SPEC

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
            - the name of a function in custom_transformations or DEFINED_WEIGHT_TRANSFORMATION.
            - A function to apply to the attribute.
        If not provided, the attribute will not be transformed.
    custom_transformations: dict, optional
        A dictionary of custom transformations which could be applied to the attributes. The keys are the transformation names and the values are the transformation functions.
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
    reaction_attrs = _create_graph_attrs_config(
        column_mapping=attribute_mapping,
        data_type=table_type,
        table_name=table_name,
        transformation=transformation,
    )

    # add the attribute to the graph
    napistu_graph = _add_graph_species_attribute(
        napistu_graph,
        sbml_dfs,
        species_graph_attrs=reaction_attrs,
        custom_transformations=custom_transformations,
    )

    return napistu_graph if not inplace else None


def _add_graph_species_attribute(
    napistu_graph: NapistuGraph,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_graph_attrs: dict,
    custom_transformations: Optional[dict] = None,
) -> NapistuGraph:
    """
    Add meta-data from species_data to existing igraph's vertices.

    This function augments the vertices of an igraph network with additional attributes
    derived from the species-level data in the provided SBML_dfs object. The attributes
    to add are specified in the species_graph_attrs dictionary, and can be transformed
    using either built-in or user-supplied transformation functions.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        The igraph network to augment.
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object containing species data.
    species_graph_attrs : dict
        Dictionary specifying which attributes to pull from species_data and how to transform them.
        The structure should be {attribute_name: {"table": ..., "variable": ..., "trans": ...}}.
    custom_transformations : dict, optional
        Dictionary mapping transformation names to functions. If provided, these will be checked
        before built-in transformations. Example: {"square": lambda x: x**2}

    Returns
    -------
    NapistuGraph
        The input igraph network with additional vertex attributes added from species_data.
    """
    if not isinstance(species_graph_attrs, dict):
        raise TypeError(
            f"species_graph_attrs must be a dict, but was {type(species_graph_attrs)}"
        )

    # fail fast if species_graph_attrs is not properly formatted
    # also flatten attribute list to be added to vertex nodes
    sp_graph_key_list = []
    sp_node_attr_list = []
    for k in species_graph_attrs.keys():
        net_create._validate_entity_attrs(
            species_graph_attrs[k], custom_transformations=custom_transformations
        )

        sp_graph_key_list.append(k)
        sp_node_attr_list.append(list(species_graph_attrs[k].keys()))

    # flatten sp_node_attr_list
    flat_sp_node_attr_list = [item for items in sp_node_attr_list for item in items]

    # Check for attribute collisions before proceeding
    existing_attrs = set(napistu_graph.vs.attributes())
    for attr in flat_sp_node_attr_list:
        if attr in existing_attrs:
            raise ValueError(f"Attribute '{attr}' already exists in graph vertices")

    logger.info("Adding meta-data from species_data")

    curr_network_nodes_df = napistu_graph.get_vertex_dataframe()

    # add species-level attributes to nodes dataframe
    augmented_network_nodes_df = net_create._augment_network_nodes(
        curr_network_nodes_df,
        sbml_dfs,
        species_graph_attrs,
        custom_transformations=custom_transformations,
    )

    # Add each attribute to the graph vertices
    for vs_attr in flat_sp_node_attr_list:
        logger.info(f"Adding new attribute {vs_attr} to vertices")
        napistu_graph.vs[vs_attr] = augmented_network_nodes_df[vs_attr].values

    return napistu_graph


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
    data_type: str,
    table_name: str,
    transformation: str = DEFAULT_WT_TRANS,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Create a configuration dictionary for graph attributes.

    Parameters
    ----------
    column_mapping : Dict[str, str]
        A dictionary mapping original column names to their new names in the graph
    data_type : str
        The type of data (e.g. "species", "reactions")
    table_name : str
        The name of the table containing the data
    transformation : str, optional
        The transformation to apply to the data, by default "identity"

    Returns
    -------
    Dict[str, Dict[str, Dict[str, str]]]
        A nested dictionary containing the graph attributes configuration
        Format:
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
    graph_attrs = {data_type: {}}

    for original_col, new_col in column_mapping.items():
        graph_attrs[data_type][new_col] = {
            WEIGHTING_SPEC.TABLE: table_name,
            WEIGHTING_SPEC.VARIABLE: original_col,
            WEIGHTING_SPEC.TRANSFORMATION: transformation,
        }

    return graph_attrs
