"""
Utilities specific to NapistuGraph objects and the wider Napistu ecosystem.

This module contains utilities that are specific to NapistuGraph subclasses
and require knowledge of the Napistu data model (SBML_dfs objects, etc.).
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import igraph as ig
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

from napistu import sbml_dfs_core, source
from napistu.identifiers import _validate_assets_sbml_ids

if TYPE_CHECKING:
    from napistu.network.ng_core import NapistuGraph
from napistu.constants import (
    ENTITIES_TO_ENTITY_DATA,
    ENTITIES_W_DATA,
    SBML_DFS,
    SBML_DFS_SCHEMA,
    SCHEMA_DEFS,
    SOURCE_SPEC,
)
from napistu.network.constants import (
    ADDING_ENTITY_DATA_DEFS,
    DEFAULT_WT_TRANS,
    DISTANCES,
    NAPISTU_GRAPH,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
    NODE_TYPES_TO_ENTITY_TABLES,
    SINGULAR_GRAPH_ENTITIES,
    VALID_ADDING_ENTITY_DATA_DEFS,
    VALID_NAPISTU_GRAPH_NODE_TYPES,
    VALID_VERTEX_SBML_DFS_SUMMARIES,
    VERTEX_SBML_DFS_SUMMARIES,
    WEIGHT_TRANSFORMATIONS,
    WEIGHTING_SPEC,
)

logger = logging.getLogger(__name__)


def apply_weight_transformations(
    edges_df: pd.DataFrame, reaction_attrs: dict, custom_transformations: dict = None
):
    """
    Apply Weight Transformations to edge attributes.

    Parameters
    ----------
    edges_df : pd.DataFrame
        A table of edges and their attributes extracted from a NapistuGraph.
    reaction_attrs : dict
        A dictionary of attributes identifying weighting attributes within
        an sbml_df's reaction_data, how they will be named in edges_df (the keys),
        and how they should be transformed (the "trans" aliases).
    custom_transformations : dict, optional
        A dictionary mapping transformation names to functions. If provided, these
        will be checked before built-in transformations.

    Returns
    -------
    pd.DataFrame
        edges_df with weight variables transformed.

    Raises
    ------
    ValueError
        If a weighting variable is missing or transformation is not found.
    """

    _validate_entity_attrs(
        reaction_attrs, custom_transformations=custom_transformations
    )

    transformed_edges_df = copy.deepcopy(edges_df)
    for k, v in reaction_attrs.items():
        if k not in transformed_edges_df.columns:
            raise ValueError(f"A weighting variable {k} was missing from edges_df")

        trans_name = v["trans"]

        # Look up transformation
        if custom_transformations:
            valid_transformations = {
                **DEFINED_WEIGHT_TRANSFORMATION,
                **custom_transformations,
            }
        else:
            valid_transformations = DEFINED_WEIGHT_TRANSFORMATION

        if trans_name in valid_transformations:
            trans_fxn = valid_transformations[trans_name]
        else:
            # This should never be hit if _validate_entity_attrs is called correctly.
            raise ValueError(
                f"Transformation '{trans_name}' not found in custom_transformations or DEFINED_WEIGHT_TRANSFORMATION."
            )

        # Handle NaN values properly
        original_series = transformed_edges_df[k]
        transformed_series = original_series.apply(
            lambda x: trans_fxn(x) if pd.notna(x) else x
        )
        transformed_edges_df[k] = transformed_series

    return transformed_edges_df


def compartmentalize_species(
    sbml_dfs: sbml_dfs_core.SBML_dfs, species: str | list[str]
) -> pd.DataFrame:
    """
    Compartmentalize Species

    Returns the compartmentalized species IDs (sc_ids) corresponding to a list of species (s_ids)

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        A model formed by aggregating pathways
    species : list
        Species IDs

    Returns
    -------
    pd.DataFrame containings the s_id and sc_id pairs
    """
    if isinstance(species, str):
        species = [species]
    if not isinstance(species, list):
        raise TypeError("species is not a str or list")

    return sbml_dfs.compartmentalized_species[
        sbml_dfs.compartmentalized_species[SBML_DFS.S_ID].isin(species)
    ].reset_index()[[SBML_DFS.S_ID, SBML_DFS.SC_ID]]


def compartmentalize_species_pairs(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    origin_species: str | list[str],
    dest_species: str | list[str],
) -> pd.DataFrame:
    """
    Compartmentalize Shortest Paths

    For a set of origin and destination species pairs, consider each species in every
    compartment it operates in, seperately.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        A model formed by aggregating pathways
    origin_species : list
        Species IDs as starting points
    dest_species : list
        Species IDs as ending points

    Returns
    -------
    pd.DataFrame containing pairs of origin and destination compartmentalized species
    """
    compartmentalized_origins = compartmentalize_species(
        sbml_dfs, origin_species
    ).rename(columns={SBML_DFS.SC_ID: "sc_id_origin", SBML_DFS.S_ID: "s_id_origin"})
    if isinstance(origin_species, str):
        origin_species = [origin_species]

    compartmentalized_dests = compartmentalize_species(sbml_dfs, dest_species).rename(
        columns={SBML_DFS.SC_ID: "sc_id_dest", SBML_DFS.S_ID: "s_id_dest"}
    )
    if isinstance(dest_species, str):
        dest_species = [dest_species]

    # create an all x all of origins and destinations
    target_species_paths = pd.DataFrame(
        [(x, y) for x in origin_species for y in dest_species]
    )
    target_species_paths.columns = ["s_id_origin", "s_id_dest"]

    target_species_paths = target_species_paths.merge(compartmentalized_origins).merge(
        compartmentalized_dests
    )

    if target_species_paths.shape[0] == 0:
        raise ValueError(
            "No compartmentalized paths exist, this is unexpected behavior"
        )

    return target_species_paths


def create_entity_attrs_from_data_tables(
    entity_data_dict: dict[str, pd.DataFrame],
    table_names: Optional[list[str]] = None,
    add_name_prefixes: bool = True,
) -> dict[str, dict[str, str]]:
    """
    Create entity_attrs configuration from data tables.

    This utility converts a dictionary of data tables into the entity_attrs
    format expected by NapistuGraph methods, automatically generating
    attribute configurations for all columns in the specified tables.

    Parameters
    ----------
    entity_data_dict : dict[str, pd.DataFrame]
        Dictionary mapping table names to DataFrames (e.g., sbml_dfs.species_data)
    table_names : Optional[list[str]], default=None
        Specific table names to include. If None, includes all available tables.
    add_name_prefixes : bool, default=True
        Whether to prefix attribute names with table name (e.g., "table_name_column_name")

    Returns
    -------
    dict[str, dict[str, str]]
        Entity attributes configuration dictionary in the format:
        {
            "attr_name": {
                "table": "table_name",
                "variable": "column_name"
            }
        }

    Raises
    ------
    ValueError
        If requested table names don't exist in entity_data_dict

    Examples
    --------
    Create attrs from all species data tables:
    >>> entity_attrs = create_entity_attrs_from_data_tables(sbml_dfs.species_data)

    Create attrs from specific tables:
    >>> entity_attrs = create_entity_attrs_from_data_tables(
    ...     sbml_dfs.reactions_data,
    ...     table_names=["kinetics", "literature"]
    ... )

    Create attrs without table name prefixes:
    >>> entity_attrs = create_entity_attrs_from_data_tables(
    ...     sbml_dfs.species_data,
    ...     add_name_prefixes=False
    ... )
    """

    if len(entity_data_dict) == 0:
        logger.warning("entity_data_dict is empty")
        return {}

    # Validate and filter table names
    if table_names is None:
        table_names = list(entity_data_dict.keys())
    else:
        invalid_tables = set(table_names) - set(entity_data_dict.keys())
        if invalid_tables:
            available_tables = list(entity_data_dict.keys())
            raise ValueError(
                f"Requested tables not found in entity_data_dict: {invalid_tables}. "
                f"Available tables: {available_tables}"
            )

    # Build entity_attrs configuration
    entity_attrs = {}

    for table_name in table_names:
        table_data = entity_data_dict[table_name]

        for column_name in table_data.columns:
            # Create attribute name
            if add_name_prefixes:
                attr_name = f"{table_name}_{column_name}"
            else:
                attr_name = column_name

            # Handle potential naming conflicts when add_name_prefixes=False
            if not add_name_prefixes and attr_name in entity_attrs:
                logger.warning(
                    f"Attribute name conflict: '{attr_name}' exists in multiple tables. "
                    f"Consider using add_name_prefixes=True to avoid conflicts."
                )
                # Auto-resolve by adding table prefix as fallback
                attr_name = f"{table_name}_{column_name}"

            entity_attrs[attr_name] = {
                WEIGHTING_SPEC.TABLE: table_name,
                WEIGHTING_SPEC.VARIABLE: column_name,
            }

    logger.debug(
        f"Created {len(entity_attrs)} entity attributes from "
        f"{len(table_names)} tables: {table_names}"
    )

    return entity_attrs


def format_napistu_graph_summary(data):
    """Format NapistuGraph summary data into a clean summary table for Jupyter display"""

    # Extract summary statistics
    total_vertices = data["n_vertices"]
    vertex_node_types = data["vertex_node_type_dict"]
    vertex_species_types = data["vertex_species_type_dict"]
    total_edges = data["n_edges"]
    sbo_name_counts = data["sbo_name_counts_dict"]
    vertex_attributes = data["vertex_attributes"]
    edge_attributes = data["edge_attributes"]

    # Build the summary data
    summary_data = [["Vertices", f"{total_vertices:,}"]]

    # Add vertex breakdown by node type, sorted by count (descending)
    for node_type, count in sorted(
        vertex_node_types.items(), key=lambda x: x[1], reverse=True
    ):
        pct = count / total_vertices * 100
        summary_data.append(
            [
                f"- {node_type.replace('_', ' ').title()}",
                f"{count:,} ({pct:.1f}%)",
            ]
        )

    # Add spacing and species type section
    summary_data.extend(
        [
            ["", ""],  # Empty row for spacing
            ["Species Types", ""],
        ]
    )

    # Calculate total species vertices
    total_species_vertices = vertex_node_types.get(NAPISTU_GRAPH_NODE_TYPES.SPECIES, 0)

    # Add species type breakdown sorted by count (descending)
    for species_type, count in sorted(
        vertex_species_types.items(), key=lambda x: x[1], reverse=True
    ):
        # Calculate percentage of species vertices (not total vertices)
        if total_species_vertices > 0:
            pct = count / total_species_vertices * 100
        else:
            pct = 0.0
        summary_data.append(
            [
                f"- {species_type.replace('_', ' ').title()}",
                f"{count:,} ({pct:.1f}%)",
            ]
        )

    # Add spacing and edges section
    summary_data.extend(
        [
            ["", ""],  # Empty row for spacing
            ["Edges", f"{total_edges:,}"],
        ]
    )

    # Add SBO term breakdown sorted by count (descending)
    for sbo_name, count in sorted(
        sbo_name_counts.items(), key=lambda x: x[1], reverse=True
    ):
        sbo_pct = count / total_edges * 100
        # Clean up SBO name for display (handle potential None values)
        display_name = sbo_name if sbo_name else "Unknown"
        summary_data.append([f"- {display_name}", f"{count:,} ({sbo_pct:.1f}%)"])

    # Add attributes sections
    summary_data.extend(
        [
            ["", ""],  # Empty row for spacing
            ["Vertex Attributes", ", ".join(vertex_attributes)],
            ["Edge Attributes", ", ".join(edge_attributes)],
        ]
    )

    # Create DataFrame and return
    df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

    return df


def get_minimal_sources_edges(
    vertices: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    min_pw_size: int = 3,
    source_total_counts: Optional[pd.Series | pd.DataFrame] = None,
    verbose: bool = False,
) -> pd.DataFrame | None:
    """
    Assign edges to a set of sources.

    Parameters
    ----------
    vertices: pd.DataFrame
        A table of vertices.
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A pathway model
    min_pw_size: int
        the minimum size of a pathway to be considered
    source_total_counts: pd.Series | pd.DataFrame
        A series of the total counts of each source or a pd.DataFrame with two columns:
        pathway_id and total_counts.
    verbose: bool
        Whether to print verbose output

    Returns
    -------
    reaction_sources: pd.DataFrame
        A table of reactions and the sources they are assigned to.
    """

    nodes = vertices["node"].tolist()
    present_reactions = sbml_dfs.reactions[sbml_dfs.reactions.index.isin(nodes)]

    if len(present_reactions) == 0:
        return None

    source_df = source.unnest_sources(present_reactions)

    if source_df is None:
        return None
    else:
        if source_total_counts is not None:

            source_total_counts = source._ensure_source_total_counts(
                source_total_counts, verbose=verbose
            )
            defined_source_totals = source_total_counts.index.tolist()

            source_mask = source_df[SOURCE_SPEC.PATHWAY_ID].isin(defined_source_totals)

            if sum(~source_mask) > 0:
                if verbose:
                    dropped_pathways = (
                        source_df[~source_mask][SOURCE_SPEC.PATHWAY_ID]
                        .unique()
                        .tolist()
                    )
                    logger.warning(
                        f"Some pathways in `source_df` are not present in `source_total_counts` ({sum(~source_mask)} entries). Dropping these pathways: {dropped_pathways}."
                    )
                source_df = source_df[source_mask]

            if source_df.shape[0] == 0:
                select_source_total_pathways = defined_source_totals[:5]
                if verbose:
                    logger.warning(
                        f"None of the pathways in `source_df` are present in `source_total_counts ({source_df[SOURCE_SPEC.PATHWAY_ID].unique().tolist()})`. Example pathways in `source_total_counts` are: {select_source_total_pathways}; returning None."
                    )
                return None

        reaction_sources = source.source_set_coverage(
            source_df,
            source_total_counts,
            sbml_dfs,
            min_pw_size=min_pw_size,
            verbose=verbose,
        )
        return reaction_sources.reset_index()[
            [SBML_DFS.R_ID, SOURCE_SPEC.PATHWAY_ID, SOURCE_SPEC.NAME]
        ]


def get_sbml_dfs_vertex_summaries(
    sbml_dfs,
    summary_types=VALID_VERTEX_SBML_DFS_SUMMARIES,
    priority_pathways=None,
    stratify_by_bqb=True,
    characteristic_only=False,
    dogmatic=False,
    add_name_prefixes=False,
    binarize=False,
    has_reactions=True,
) -> pd.DataFrame:
    """
    Prepare species and reaction ontology and/or source occurrence summaries which are ready to be merged with NapistuGraph vertices.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        A pathway model
    summary_types : list
        The summary types to get
    priority_pathways : list
        The priority pathways to get
    stratify_by_bqb : bool
        Whether to stratify by BQB
    characteristic_only : bool
        Whether to only get characteristic ontologies
    dogmatic : bool
        Whether to use dogmatic ontologies
    add_name_prefixes : bool, default False
        If True, add prefixes to column names: 'source_' for source data
        and 'ontology_' for ontology data
    binarize: bool, optional
        Whether to convert the summary to binary values (0 vs 1+). Default is False.
    has_reactions : bool, default True
        Whether the graph has reaction vertices. If False, reaction-specific summaries
        will be skipped.
    """

    if len(summary_types) == 0:
        raise ValueError(
            "No summary types provided, valid summary types are: {VALID_VERTEX_SBML_DFS_SUMMARIES}"
        )

    invalid_summary_types = set(summary_types) - set(VALID_VERTEX_SBML_DFS_SUMMARIES)
    if len(invalid_summary_types) > 0:
        raise ValueError(
            f"Invalid summary types: {invalid_summary_types}. valid summary types are: {VALID_VERTEX_SBML_DFS_SUMMARIES}"
        )

    summaries = list()

    if VERTEX_SBML_DFS_SUMMARIES.SOURCES in summary_types:

        # Only include reaction sources if graph has reaction vertices
        node_types_to_summarize = (
            VALID_NAPISTU_GRAPH_NODE_TYPES
            if has_reactions
            else [NAPISTU_GRAPH_NODE_TYPES.SPECIES]
        )

        entity_tables = {
            NODE_TYPES_TO_ENTITY_TABLES[x] for x in node_types_to_summarize
        }

        source_dfs = list()
        for entity_table in entity_tables:
            logger.info(f"Getting source occurrence for {entity_table}")
            df = sbml_dfs.get_source_occurrence(
                entity_table, priority_pathways, binarize=binarize
            )
            df.columns.name = None
            source_dfs.append(df.rename_axis(NAPISTU_GRAPH_VERTICES.NAME))

        logger.debug("Concatenating source occurrences")
        source_summary = pd.concat(source_dfs).fillna(int(0))

        if add_name_prefixes:
            source_summary.columns = ["source_" + col for col in source_summary.columns]

        summaries.append(source_summary)

    if VERTEX_SBML_DFS_SUMMARIES.ONTOLOGIES in summary_types:

        ontology_dfs = []

        # get reaction ontologies directly (since these are vertex names) - only if graph has reactions
        if has_reactions:
            logger.info(f"Getting ontology occurrence for {SBML_DFS.REACTIONS}")
            df = sbml_dfs.get_ontology_occurrence(
                SBML_DFS.REACTIONS,
                stratify_by_bqb=stratify_by_bqb,
                characteristic_only=characteristic_only,
                dogmatic=dogmatic,
                include_missing=True,
                binarize=binarize,
            )
            df.columns.name = None
            reaction_ontologies = df.rename_axis(NAPISTU_GRAPH_VERTICES.NAME)
            ontology_dfs.append(reaction_ontologies)
        else:
            logger.debug(
                "Skipping reaction ontology summaries - graph has no reaction vertices"
            )

        # get species ontologies then map them to compartmentalized species (since the cspecies are the vertex names)
        logger.info(f"Getting ontology occurrence for {SBML_DFS.SPECIES}")
        df = sbml_dfs.get_ontology_occurrence(
            SBML_DFS.SPECIES,
            stratify_by_bqb=stratify_by_bqb,
            characteristic_only=characteristic_only,
            dogmatic=dogmatic,
            include_missing=True,
            binarize=binarize,
        )
        df.columns.name = None

        species_ontologies = (
            sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID]]
            .merge(df, left_on=SBML_DFS.S_ID, right_index=True)
            .drop(columns=[SBML_DFS.S_ID])
        )
        ontology_dfs.append(species_ontologies)

        logger.debug("Concatenating ontology occurrences")
        ontology_summary = pd.concat(ontology_dfs).fillna(int(0))

        if add_name_prefixes:
            ontology_summary.columns = [
                "ontology_" + col for col in ontology_summary.columns
            ]

        summaries.append(ontology_summary)

    logger.debug("Concatenating all summaries")
    out = pd.concat(summaries, axis=1).astype(int)
    out.index.name = NAPISTU_GRAPH_VERTICES.NAME

    return out


def pluck_data(
    data_tables: dict[str, pd.DataFrame],
    entity_attrs: dict[str, dict],
) -> pd.DataFrame | None:
    """
    Pluck data from a dictionary of DataFrames based on specified attributes.

    Parameters
    ----------
    data_tables : dict[str, pd.DataFrame]
        A dictionary mapping table names to pandas DataFrames.
    entity_attrs : dict[str, dict]
        A dictionary containing the attributes to pull out. Of the form:
        {
            "to_be_created_column_name": {
                "table": "table name in data_tables",
                "variable": "column name in the specified table"
            }
        }

    Returns
    -------
    pd.DataFrame or None
        A table where all extracted attributes are merged based on a common index or None
        if no attributes were extracted. If the attribute dict is empty, returns None.

    Raises
    ------
    ValueError
        If requested tables/variables are missing.
    """

    # Use existing validation logic (without transformation validation)
    _validate_entity_attrs(entity_attrs, validate_transformations=False)

    if not isinstance(data_tables, dict):
        raise ValueError("data_tables must be a dictionary")

    if len(data_tables) == 0:
        raise ValueError("data_tables must be a non-empty dictionary")
    for table_name in data_tables.keys():
        if not isinstance(data_tables[table_name], pd.DataFrame):
            raise ValueError(f"data_tables[{table_name}] must be a pandas DataFrame")

    if len(entity_attrs) == 0:
        logger.warning("No attributes were provided in entity_attrs; returning None")
        return None

    data_list = []

    for column_name, attr_dict in entity_attrs.items():
        table_name = attr_dict[WEIGHTING_SPEC.TABLE]
        variable_name = attr_dict[WEIGHTING_SPEC.VARIABLE]

        # Check if table exists
        if table_name not in data_tables.keys():
            raise ValueError(
                f"'{table_name}' was defined as a table in entity_attrs but "
                f"it is not present in the provided data_tables"
            )

        # Check if variable exists in the table
        if variable_name not in data_tables[table_name].columns.tolist():
            raise ValueError(
                f"'{variable_name}' was defined as a variable in entity_attrs but "
                f"it is not present in the '{table_name}' table"
            )

        # Extract the series and rename it
        entity_series = data_tables[table_name][variable_name].rename(column_name)
        data_list.append(entity_series)

    if len(data_list) == 0:
        return None

    return pd.concat(data_list, axis=1)


def pluck_entity_data(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    entity_attrs: dict[str, list[dict]] | list[dict],
    data_type: str,
    custom_transformations: Optional[dict[str, callable]] = None,
    transform: bool = True,
) -> pd.DataFrame | None:
    """
    Pluck Entity Attributes from an sbml_dfs based on a set of tables and variables to look for.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A mechanistic model.
    entity_attrs : dict[str, list[dict]] | list[dict]
        A list of dicts containing the species/reaction attributes to pull out. Of the form:
        [
            "to_be_created_graph_attr_name": {
                "table": "species/reactions data table",
                "variable": "variable in the data table",
                "trans": "optionally, a transformation to apply to the variable (where applicable)"
            }
        ]

        This can also be a dict of the form but this will result in a deprecation warning:
        {
            "species": << entity attributes list >>
            "reactions" : << entity attributes list >>
        }
    data_type : str
        "species" or "reactions" to pull out species_data or reactions_data.
    custom_transformations : dict[str, callable], optional
        A dictionary mapping transformation names to functions. If provided, these
        will be checked before built-in transformations. Example:
            custom_transformations = {"square": lambda x: x**2}
    transform : bool, default=True
        Whether to apply transformations to the extracted data. In a future version,
        this function will not support transformations by default.

    Returns
    -------
    pd.DataFrame or None
        A table where all extracted attributes are merged based on a common index or None
        if no attributes were extracted. If the requested data_type is not present in
        graph_attrs, or if the attribute dict is empty, returns None. This is intended
        to allow optional annotation blocks.

    Raises
    ------
    ValueError
        If data_type is not valid or if requested tables/variables are missing.
    """

    if data_type not in ENTITIES_W_DATA:
        raise ValueError(
            f'"data_type" was {data_type} and must be in {", ".join(ENTITIES_W_DATA)}'
        )

    if data_type in entity_attrs.keys():
        logger.warning(
            f"The provided entity_attrs is a dict of the form {entity_attrs}. This will be deprecated in a future release. Please provide a species- or reactions-level entity_attrs list of dicts."
        )
        entity_attrs = entity_attrs[data_type]

    # validating dict structure (skip transformation validation during data extraction)
    if transform:
        _validate_entity_attrs(
            entity_attrs, custom_transformations=custom_transformations
        )
    else:
        _validate_entity_attrs(entity_attrs, validate_transformations=False)

    if transform:
        logger.warning(
            "pluck_entity_data is applying transformations. In a future version, "
            "this function will not support transformations by default. "
            "Use transform=False to disable transformations."
        )

    if len(entity_attrs) == 0:
        logger.warning(
            f"No {data_type} attributes were provided in entity_attrs; returning None"
        )
        return None

    data_type_attr = ENTITIES_TO_ENTITY_DATA[data_type]
    entity_data_tbls = getattr(sbml_dfs, data_type_attr)

    data_list = list()
    for k, v in entity_attrs.items():
        # v["table"] is always present if entity_attrs is non-empty and validated
        if v[WEIGHTING_SPEC.TABLE] not in entity_data_tbls.keys():
            raise ValueError(
                f'{v[WEIGHTING_SPEC.TABLE]} was defined as a table in "graph_attrs" but '
                f'it is not present in the "{data_type_attr}" of the sbml_dfs'
            )

        if (
            v[WEIGHTING_SPEC.VARIABLE]
            not in entity_data_tbls[v[WEIGHTING_SPEC.TABLE]].columns.tolist()
        ):
            raise ValueError(
                f'{v[WEIGHTING_SPEC.VARIABLE]} was defined as a variable in "graph_attrs" but '
                f'it is not present in the {v[WEIGHTING_SPEC.TABLE]} of the "{data_type_attr}" of '
                "the sbml_dfs"
            )

        entity_series = entity_data_tbls[v[WEIGHTING_SPEC.TABLE]][
            v[WEIGHTING_SPEC.VARIABLE]
        ].rename(k)
        if transform:
            trans_name = v.get(WEIGHTING_SPEC.TRANSFORMATION, DEFAULT_WT_TRANS)
            # Look up transformation
            if custom_transformations:
                valid_transformations = {
                    **DEFINED_WEIGHT_TRANSFORMATION,
                    **custom_transformations,
                }
            else:
                valid_transformations = DEFINED_WEIGHT_TRANSFORMATION

            if trans_name in valid_transformations:
                trans_fxn = valid_transformations[trans_name]
            else:
                # This should never be hit if _validate_entity_attrs is called correctly.
                raise ValueError(
                    f"Transformation '{trans_name}' not found in custom_transformations or DEFINED_WEIGHT_TRANSFORMATION."
                )
            entity_series = entity_series.apply(trans_fxn)
        data_list.append(entity_series)

    if len(data_list) == 0:
        return None

    result_df = pd.concat(data_list, axis=1)

    # Preserve the index name from the original data
    # The index name should match the entity type (s_id for species, r_id for reactions)
    result_df.index.name = SBML_DFS_SCHEMA.SCHEMA[data_type][SCHEMA_DEFS.PK]

    return result_df


def prepare_entity_data_extraction(
    graph,
    entity_type: str,
    target_entity: str,
    mode: str = ADDING_ENTITY_DATA_DEFS.FRESH,
    overwrite: bool = False,
) -> tuple[dict, set] | None:
    """
    Prepare entity data extraction by validating inputs and determining which attributes to extract.

    This utility captures the logic from _add_entity_data up to the pluck_entity_data call,
    handling validation, conflict checking, and attribute filtering.

    Parameters
    ----------
    graph : NapistuGraph
        The graph object containing entity attributes metadata
    entity_type : str
        Either "reactions" or "species"
    target_entity : str
        Either "edges" or "vertices" - determines where attributes will be added
    mode : str, default="fresh"
        Either "fresh" (replace existing) or "extend" (add new attributes only)
    overwrite : bool, default=False
        Whether to allow overwriting existing attributes when conflicts arise

    Returns
    -------
    entity_attrs_to_extract | None
        If successful: entity_attrs_to_extract - a dictionary of entity attributes to extract drawn from the vertex/edge attributes metadata
        If failed: None

    Raises
    ------
    ValueError
        If target_entity is invalid, mode is invalid, or conflicts exist without overwrite
    """

    # Get entity_attrs from stored metadata
    entity_attrs = graph._get_entity_attrs(entity_type)
    if entity_attrs is None or not entity_attrs:
        logger.warning(
            f"No {entity_type}_attrs found. Use set_graph_attrs() to configure {entity_type} attributes before extracting {target_entity} data."
        )
        return None

    # Check for conflicts with existing attributes
    if target_entity == NAPISTU_GRAPH.EDGES:
        existing_attrs = set(graph.es.attributes())
    elif target_entity == NAPISTU_GRAPH.VERTICES:  # vertices
        existing_attrs = set(graph.vs.attributes())
    else:
        raise ValueError(
            f"Unknown target_entity: {target_entity}. Must be '{NAPISTU_GRAPH.EDGES}' or '{NAPISTU_GRAPH.VERTICES}'"
        )
    # Get a singular name for logging
    entity_name = SINGULAR_GRAPH_ENTITIES[target_entity]

    new_attrs = set(entity_attrs.keys())
    if mode == ADDING_ENTITY_DATA_DEFS.FRESH:
        overlapping_attrs = existing_attrs & new_attrs
        if overlapping_attrs and not overwrite:
            raise ValueError(
                f"{entity_name.capitalize()} attributes already exist: {overlapping_attrs}. "
                f"Use overwrite=True to replace or mode='{ADDING_ENTITY_DATA_DEFS.EXTEND}' to add only new attributes"
            )
        attrs_to_add = new_attrs

    elif mode == ADDING_ENTITY_DATA_DEFS.EXTEND:
        # In extend mode, only add attributes that don't exist (unless overwrite=True)
        attrs_to_add = new_attrs - existing_attrs

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Must be one of: {VALID_ADDING_ENTITY_DATA_DEFS}"
        )

    if not attrs_to_add:
        logger.info("No new attributes to add")
        return None

    # Only extract the attributes we're actually going to add
    entity_attrs_to_extract = {attr: entity_attrs[attr] for attr in attrs_to_add}

    return entity_attrs_to_extract


def separate_entity_attrs_by_source(
    entity_attrs: dict[str, dict],
    entity_type: str,
    sbml_dfs: Optional[Any] = None,
    side_loaded_attributes: Optional[dict[str, pd.DataFrame]] = None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Separate entity attributes by data source (SBML vs side-loaded).

    This function categorizes entity attributes based on their table content:
    - SBML attributes: table names that exist in the sbml_dfs object for the specified entity_type
    - Side-loaded attributes: table names that exist in the side_loaded_attributes dict

    Both types use the same structure: table, variable, transformation

    Parameters
    ----------
    entity_attrs : dict[str, dict]
        Dictionary of entity attributes to separate
    entity_type : str
        Either "reactions" or "species" - determines which SBML tables to check
    sbml_dfs : SBML_dfs, optional
        SBML_dfs object to check for valid table names
    side_loaded_attributes : dict[str, pd.DataFrame], optional
        Dictionary mapping table names to DataFrames for side-loaded data

    Returns
    -------
    tuple[dict[str, dict], dict[str, dict]]
        (sbml_attrs, side_loaded_attrs) - two dictionaries containing separated attributes

    Raises
    ------
    ValueError
        If an attribute has an invalid structure, if entity_type is invalid, if both
        sbml_dfs and side_loaded_attributes have overlapping table names, if neither data
        source is provided, or if required table names are missing from both sources
    """

    # Validate entity_type
    if entity_type not in [SBML_DFS.REACTIONS, SBML_DFS.SPECIES]:
        raise ValueError(
            f"Invalid entity_type: '{entity_type}'. Must be either '{SBML_DFS.REACTIONS}' or '{SBML_DFS.SPECIES}'"
        )

    # Validate that at least one data source is provided
    if sbml_dfs is None and side_loaded_attributes is None:
        raise ValueError(
            "At least one of 'sbml_dfs' or 'side_loaded_attributes' must be provided"
        )

    # Get valid table names from both sources
    valid_sbml_tables = set()
    valid_extra_tables = set()

    if sbml_dfs is not None:
        # Get the correct attribute name for the entity type
        entity_data_attr = ENTITIES_TO_ENTITY_DATA[entity_type]

        # Check only the relevant entity type tables
        if hasattr(sbml_dfs, entity_data_attr):
            entity_data = getattr(sbml_dfs, entity_data_attr)
            if entity_data:
                valid_sbml_tables.update(entity_data.keys())

    logger.debug(f"Valid SBML tables: {valid_sbml_tables}")

    if side_loaded_attributes is not None:
        _validate_side_loaded_attributes(side_loaded_attributes)
        valid_extra_tables.update(side_loaded_attributes.keys())

    logger.debug(f"Valid extra tables: {valid_extra_tables}")

    # Check for overlapping table names
    overlapping_tables = valid_sbml_tables & valid_extra_tables
    if overlapping_tables:
        raise ValueError(
            f"Overlapping table names found between sbml_dfs and extra_attributes: "
            f"{overlapping_tables}. Each table name must be unique across data sources."
        )

    # Get all required table names from entity_attrs
    required_tables = set()
    for attr_config in entity_attrs.values():
        if WEIGHTING_SPEC.TABLE in attr_config:
            required_tables.add(attr_config[WEIGHTING_SPEC.TABLE])

    # Check that all required tables are available
    available_tables = valid_sbml_tables | valid_extra_tables
    missing_tables = required_tables - available_tables

    if missing_tables:
        raise ValueError(
            f"Required table names not found in either data source: {missing_tables}. "
            f"Available SBML {entity_type} tables: {valid_sbml_tables}. "
            f"Available extra tables: {valid_extra_tables}."
        )

    sbml_attrs = {}
    side_loaded_attrs = {}

    for attr_name, attr_config in entity_attrs.items():
        # Validate structure - must have table and variable
        if (
            WEIGHTING_SPEC.TABLE not in attr_config
            or WEIGHTING_SPEC.VARIABLE not in attr_config
        ):
            raise ValueError(
                f"Invalid attribute structure for '{attr_name}': "
                f"must have both 'table' and 'variable' keys. "
                f"Found keys: {list(attr_config.keys())}"
            )

        table_name = attr_config[WEIGHTING_SPEC.TABLE]

        # Determine if it's SBML or side-loaded based on table name
        if table_name in valid_sbml_tables:
            sbml_attrs[attr_name] = attr_config
        elif table_name in valid_extra_tables:
            side_loaded_attrs[attr_name] = attr_config
        else:
            # This should not happen due to validation above, but just in case
            raise ValueError(
                f"Table '{table_name}' not found in any data source for attribute '{attr_name}'"
            )

    return sbml_attrs, side_loaded_attrs


def validate_assets(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    napistu_graph: Optional[Union["NapistuGraph", ig.Graph]] = None,
    precomputed_distances: Optional[pd.DataFrame] = None,
    identifiers_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Validate Assets

    Perform a few quick checks of inputs to catch inconsistencies.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A pathway representation. (Required)
        napistu_graph : "NapistuGraph", optional
    A network-based representation of `sbml_dfs`. NapistuGraph is a subclass of igraph.Graph.
    precomputed_distances : pandas.DataFrame, optional
        Precomputed distances between vertices in `napistu_graph`.
    identifiers_df : pandas.DataFrame, optional
        A table of systematic identifiers for compartmentalized species in `sbml_dfs`.

    Returns
    -------
    None

    Warns
    -----
    If only sbml_dfs is provided and no other assets are given, a warning is logged.

    Raises
    ------
    ValueError
        If precomputed_distances is provided but napistu_graph is not.
    """
    if (
        napistu_graph is None
        and precomputed_distances is None
        and identifiers_df is None
    ):
        logger.warning(
            "validate_assets: Only sbml_dfs was provided; nothing to validate."
        )
        return None

    # Validate napistu_graph if provided
    if napistu_graph is not None:
        _validate_assets_sbml_graph(sbml_dfs, napistu_graph)

    # Validate precomputed_distances if provided (requires napistu_graph)
    if precomputed_distances is not None:
        if napistu_graph is None:
            raise ValueError(
                "napistu_graph must be provided if precomputed_distances is provided."
            )
        _validate_assets_graph_dist(napistu_graph, precomputed_distances)

    # Validate identifiers_df if provided
    if identifiers_df is not None:
        _validate_assets_sbml_ids(sbml_dfs, identifiers_df)

    return None


def read_graph_attrs_spec(graph_attrs_spec_uri: str) -> dict:
    """Read a YAML file containing the specification for adding reaction- and/or species-attributes to a napistu_graph."""
    with open(graph_attrs_spec_uri) as f:
        graph_attrs_spec = yaml.safe_load(f)

    VALID_SPEC_SECTIONS = [SBML_DFS.SPECIES, SBML_DFS.REACTIONS]
    defined_spec_sections = set(graph_attrs_spec.keys()).intersection(
        VALID_SPEC_SECTIONS
    )

    if len(defined_spec_sections) == 0:
        raise ValueError(
            f"The provided graph attributes spec did not contain either of the expected sections: {', '.join(VALID_SPEC_SECTIONS)}"
        )

    if SBML_DFS.REACTIONS in defined_spec_sections:
        _validate_entity_attrs(graph_attrs_spec[SBML_DFS.REACTIONS])

    if SBML_DFS.SPECIES in defined_spec_sections:
        _validate_entity_attrs(graph_attrs_spec[SBML_DFS.SPECIES])

    return graph_attrs_spec


# Internal utility functions


def _wt_transformation_identity(x):
    """
    Identity transformation for weights.

    Parameters
    ----------
    x : any
        Input value.

    Returns
    -------
    any
        The input value unchanged.
    """
    return x


def _wt_transformation_string(x):
    """
    Map STRING scores to a similar scale as topology weights.

    Parameters
    ----------
    x : float
        STRING score.

    Returns
    -------
    float
        Transformed STRING score.
    """
    return 250000 / np.power(x, 1.7)


def _wt_transformation_string_inv(x):
    """
    Map STRING scores so they work with source weights.

    Parameters
    ----------
    x : float
        STRING score.

    Returns
    -------
    float
        Inverse transformed STRING score.
    """
    # string scores are bounded on [0, 1000]
    # and score/1000 is roughly a probability that
    # there is a real interaction (physical, genetic, ...)
    # reported string scores are currently on [150, 1000]
    # so this transformation will map these onto {6.67, 1}
    return 1 / (x / 1000)


# Define weight transformations mapping directly to functions
DEFINED_WEIGHT_TRANSFORMATION = {
    WEIGHT_TRANSFORMATIONS.IDENTITY: _wt_transformation_identity,
    WEIGHT_TRANSFORMATIONS.STRING: _wt_transformation_string,
    WEIGHT_TRANSFORMATIONS.STRING_INV: _wt_transformation_string_inv,
}


def _validate_assets_sbml_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs, napistu_graph: Union["NapistuGraph", ig.Graph]
) -> None:
    """
    Check an sbml_dfs model and NapistuGraph for inconsistencies.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The pathway representation.
    napistu_graph : "NapistuGraph"
        The network representation (subclass of igraph.Graph).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If species names do not match between sbml_dfs and napistu_graph.
    """
    vertices = pd.DataFrame(
        [{**{"index": v.index}, **v.attributes()} for v in napistu_graph.vs]
    )
    matched_cspecies = sbml_dfs.compartmentalized_species.reset_index()[
        [SBML_DFS.SC_ID, SBML_DFS.SC_NAME]
    ].merge(
        vertices.query(
            f"{NAPISTU_GRAPH_VERTICES.NODE_TYPE} == '{NAPISTU_GRAPH_NODE_TYPES.SPECIES}'"
        ),
        left_on=[SBML_DFS.SC_ID],
        right_on=[NAPISTU_GRAPH_VERTICES.NAME],
    )
    mismatched_names = [
        f"{x} != {y}"
        for x, y in zip(
            matched_cspecies[SBML_DFS.SC_NAME],
            matched_cspecies[NAPISTU_GRAPH_VERTICES.NODE_NAME],
        )
        if x != y
    ]
    if len(mismatched_names) > 0:
        example_names = mismatched_names[: min(10, len(mismatched_names))]
        raise ValueError(
            f"{len(mismatched_names)} species names do not match between sbml_dfs and napistu_graph: {example_names}"
        )
    return None


def _validate_assets_graph_dist(
    napistu_graph: "NapistuGraph", precomputed_distances: pd.DataFrame
) -> None:
    """
    Check a NapistuGraph and precomputed distances table for inconsistencies.

    Parameters
    ----------
    napistu_graph : "NapistuGraph"
        The network representation (subclass of igraph.Graph).
    precomputed_distances : pandas.DataFrame
        Precomputed distances between vertices in the network.

    Returns
    -------
    None

    Warns
    -----
    If edge weights are inconsistent between the graph and precomputed distances.
    """
    edges = pd.DataFrame(
        [{**{"index": e.index}, **e.attributes()} for e in napistu_graph.es]
    )
    direct_interactions = precomputed_distances.query(f"{DISTANCES.PATH_LENGTH} == 1")
    edges_with_distances = direct_interactions.merge(
        edges[
            [
                NAPISTU_GRAPH_EDGES.FROM,
                NAPISTU_GRAPH_EDGES.TO,
                NAPISTU_GRAPH_EDGES.WEIGHT,
                NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
            ]
        ],
        left_on=[DISTANCES.SC_ID_ORIGIN, DISTANCES.SC_ID_DEST],
        right_on=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO],
    )
    inconsistent_weights = edges_with_distances.query(
        # shortest weighted paths cannot be greater than the path weight for a direct connection
        f"{DISTANCES.PATH_WEIGHT} > {NAPISTU_GRAPH_EDGES.WEIGHT}"
    )
    if inconsistent_weights.shape[0] > 0:
        logger.warning(
            f"{inconsistent_weights.shape[0]} edges' weights are inconsistent between "
            f"edges in the napistu_graph and length 1 paths in precomputed_distances. "
            f"This is {inconsistent_weights.shape[0] / edges_with_distances.shape[0]:.2%} of all edges."
        )
    return None


def _validate_entity_attrs(
    entity_attrs: dict,
    validate_transformations: bool = True,
    custom_transformations: Optional[dict] = None,
) -> None:
    """
    Validate that graph attributes are a valid format.

    Parameters
    ----------
    entity_attrs : dict
        Dictionary of entity attributes to validate. The structure should be:
        {
            "attr_name": {
                "table": "table_name",
                "variable": "variable_name",
                "trans": "transformation_name"
            }
        }
        where "table" is the name of the table in the sbml_dfs to look for the variable,
        "variable" is the name of the variable in the table,
        "trans" (optional) is the name of the transformation to apply to the variable.
    validate_transformations : bool, optional
        Whether to validate transformation names, by default True.
    custom_transformations : dict, optional
        Dictionary of custom transformation functions, by default None. Keys are transformation names, values are transformation functions.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If entity_attrs is not a dictionary.
    ValueError
        If a transformation is not found in DEFINED_WEIGHT_TRANSFORMATION or custom_transformations.
    """
    assert isinstance(entity_attrs, dict), "entity_attrs must be a dictionary"

    for _, v in entity_attrs.items():
        # check structure against pydantic config
        validated_attrs = _EntityAttrValidator(**v).model_dump()

        if validate_transformations:
            trans_name = validated_attrs.get(
                WEIGHTING_SPEC.TRANSFORMATION, DEFAULT_WT_TRANS
            )
            if custom_transformations:
                valid_transformations = {
                    **DEFINED_WEIGHT_TRANSFORMATION,
                    **custom_transformations,
                }
            else:
                valid_transformations = DEFINED_WEIGHT_TRANSFORMATION

            if trans_name not in valid_transformations:
                raise ValueError(
                    f"transformation '{trans_name}' was not defined as an alias in "
                    "DEFINED_WEIGHT_TRANSFORMATION or custom_transformations. The defined transformations "
                    f"are {', '.join(sorted(valid_transformations.keys()))}"
                )

    return None


def _validate_side_loaded_attributes(
    side_loaded_attributes: dict[str, pd.DataFrame],
) -> None:
    """
    Validate that side_loaded_attributes is a dict of DataFrames with consistent index names.

    This function ensures that all DataFrames in the dictionary can be concatenated
    by checking that they have the same index structure (single or multi-index).

    Parameters
    ----------
    side_loaded_attributes : dict[str, pd.DataFrame]
        Dictionary mapping table names to DataFrames

    Raises
    ------
    TypeError
        If side_loaded_attributes is not a dict or contains non-DataFrame values
    ValueError
        If DataFrames have inconsistent index names or structures
    """
    if not isinstance(side_loaded_attributes, dict):
        raise TypeError("side_loaded_attributes must be a dictionary")

    if not side_loaded_attributes:
        return  # Empty dict is valid

    # Check that all values are DataFrames
    for table_name, df in side_loaded_attributes.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"side_loaded_attributes[{table_name}] must be a pandas DataFrame, got {type(df)}"
            )

    # Get the first DataFrame's index structure as reference
    first_df = next(iter(side_loaded_attributes.values()))
    reference_index_names = first_df.index.names
    reference_is_multiindex = isinstance(first_df.index, pd.MultiIndex)

    # Check that all other DataFrames have the same index structure
    for table_name, df in side_loaded_attributes.items():
        current_index_names = df.index.names
        current_is_multiindex = isinstance(df.index, pd.MultiIndex)

        # Check if index types match (both single or both multi)
        if current_is_multiindex != reference_is_multiindex:
            raise ValueError(
                f"side_loaded_attributes[{table_name}] has {'MultiIndex' if current_is_multiindex else 'single index'}, "
                f"but other tables have {'MultiIndex' if reference_is_multiindex else 'single index'}. "
                f"All DataFrames must have the same index structure for concatenation."
            )

        # Check if index names match
        if current_index_names != reference_index_names:
            raise ValueError(
                f"side_loaded_attributes[{table_name}] has index names {current_index_names}, "
                f"but other tables have index names {reference_index_names}. "
                f"All DataFrames must have the same index names for concatenation."
            )

        # check that index is unique
        if not df.index.is_unique:
            raise ValueError(
                f"side_loaded_attributes[{table_name}] has non-unique index. "
                f"All DataFrames must have a unique index for concatenation."
            )


class _EntityAttrValidator(BaseModel):
    table: str
    variable: str
    trans: Optional[str] = DEFAULT_WT_TRANS
