from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, List, Mapping, MutableMapping, Optional, Union

import igraph as ig
import pandas as pd

if TYPE_CHECKING:
    pass

from napistu import utils
from napistu.constants import (
    ENTITIES_TO_ENTITY_DATA,
    ENTITIES_W_DATA,
    MINI_SBO_TO_NAME,
    SBML_DFS,
)
from napistu.ingestion.constants import DEFAULT_PRIORITIZED_PATHWAYS
from napistu.network import data_handling, ig_utils, ng_utils
from napistu.network.constants import (
    ADDING_ENTITY_DATA_DEFS,
    DEFAULT_WT_TRANS,
    EDGE_DIRECTION_MAPPING,
    EDGE_REVERSAL_ATTRIBUTE_MAPPING,
    ENTITIES_TO_ATTRS,
    IGRAPH_DEFS,
    NAPISTU_GRAPH,
    NAPISTU_GRAPH_EDGE_ENDPOINT_ATTRIBUTES,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
    NAPISTU_METADATA_KEYS,
    NAPISTU_WEIGHTING_STRATEGIES,
    SINGULAR_GRAPH_ENTITIES,
    SOURCE_VARS_DICT,
    VALID_ADDING_ENTITY_DATA_DEFS,
    VALID_VERTEX_SBML_DFS_SUMMARIES,
    VALID_WEIGHTING_STRATEGIES,
    WEIGHT_TRANSFORMATIONS,
    WEIGHTING_SPEC,
)
from napistu.sbml_dfs_core import SBML_dfs

logger = logging.getLogger(__name__)


class NapistuGraph(ig.Graph):
    """
    NapistuGraph - Molecular Network Analysis Graph.

    A subclass of igraph.Graph with additional functionality for molecular network analysis.
    This class extends igraph.Graph with domain-specific methods and metadata tracking
    for biological pathway and molecular interaction networks. All standard igraph
    methods are available, plus additional functionality for edge reversal, weighting,
    and metadata management.

    Attributes
    ----------
    is_reversed : bool
        Whether the graph edges have been reversed from their original direction
    wiring_approach : str or None
        Type of graph (e.g., 'bipartite', 'regulatory', 'surrogate')
    weighting_strategy : str or None
        Strategy used for edge weighting (e.g., 'topology', 'mixed')
    weight_by : list[str] or None
        List of attributes used for edge weighting

    Public Methods (alphabetical)
    ----------------------------
    add_degree_attributes(inplace=True)
        Add degree-based attributes to vertices and edges.
    add_edge_data(sbml_dfs, side_loaded_attributes, mode='fresh', overwrite=False, inplace=True)
        Add edge data from SBML_dfs to the graph.
    add_topology_weights(base_score=2, protein_multiplier=1, metabolite_multiplier=3, unknown_multiplier=10, scale_multiplier_by_meandegree=True, inplace=True)
        Add topology-based weights to graph edges.
    add_sbml_dfs_summaries(sbml_dfs, summary_types=VALID_VERTEX_SBML_DFS_SUMMARIES, priority_pathways=DEFAULT_PRIORITIZED_PATHWAYS, stratify_by_bqb=True, characteristic_only=False, dogmatic=False, add_name_prefixes = False, binarize=True, inplace=True)
        Add vertex summaries from SBML_dfs to the graph vertices.
    add_vertex_data(sbml_dfs, side_loaded_attributes, mode='fresh', overwrite=False, inplace=True)
        Add vertex data from SBML_dfs to the graph.
    copy()
        Create a deep copy of the NapistuGraph.
    deduplicate_edges(verbose=False)
        Deduplicate edges with the same FROM -> TO pair, keeping only the first occurrence.
    from_igraph(graph, **metadata)
        Create a NapistuGraph from an existing igraph.Graph.
    from_pickle(path)
        Load a NapistuGraph from a pickle file.
    get_edge_dataframe()
        Return graph edges as a pandas DataFrame.
    get_edge_series(attribute_name)
        Return a single edge attribute as a pandas Series.
    get_metadata(key=None)
        Get metadata from the graph.
    get_vertex_dataframe()
        Return graph vertices as a pandas DataFrame.
    get_vertex_series(attribute_name)
        Return a single vertex attribute as a pandas Series.
    remove_attributes(attribute_type, attributes)
        Remove specified attributes from vertices or edges.
    remove_isolated_vertices(node_types='reactions')
        Remove isolated vertices from the graph.
    reverse_edges()
        Reverse all edges in the graph in-place.
    set_graph_attrs(graph_attrs, mode='fresh', overwrite=False)
        Set graph attributes from SBML_dfs or dictionary.
    set_metadata(**kwargs)
        Set metadata for the graph in-place.
    set_weights(weighting_strategy='unweighted', weight_by=None, reaction_edge_multiplier=0.5)
        Set edge weights using various strategies.
    to_pandas_dfs()
        Convert graph to pandas DataFrames for vertices and edges.
    to_pickle(path)
        Save the NapistuGraph to a pickle file.
    transform_edges(keep_raw_attributes=False, custom_transformations=None)
        Transform edge attributes using predefined or custom transformations.
    transform_vertices(keep_raw_attributes=False, custom_transformations=None)
        Transform vertex attributes using predefined or custom transformations.
    validate()
        Validate the graph structure and metadata.

    Private/Hidden Methods (alphabetical, appear after public methods)
    -----------------------------------------------------------------
    _add_graph_weights_mixed(weight_by=None)
        Add mixed weighting strategy to graph edges.
    _add_entity_data(sbml_dfs, entity_type, target_entity, mode, overwrite, inplace)
        Add entity data from SBML_dfs to the graph.
    _apply_reaction_edge_multiplier(multiplier=0.5)
        Apply multiplier to reaction edges.
    _compare_and_merge_attrs(new_attrs, attr_type, mode='fresh', overwrite=False)
        Compare and merge attributes with existing ones.
    _create_source_weights(edges_df, source_wt_var='source_wt', source_vars_dict=SOURCE_VARS_DICT, source_wt_default=1)
        Create source-based weights for edges.
    _get_entity_attrs(entity_type)
        Get entity-specific attributes from metadata.
    _get_weight_variables(weight_by=None)
        Get weight variables for edge weighting.

    Examples
    --------
    Create a NapistuGraph from scratch:

    >>> ng = NapistuGraph(directed=True)
    >>> ng.add_vertices(3)
    >>> ng.add_edges([(0, 1), (1, 2)])

    Convert from existing igraph:

    >>> import igraph as ig
    >>> g = ig.Graph.Erdos_Renyi(10, 0.3)
    >>> ng = NapistuGraph.from_igraph(g, wiring_approach='regulatory')

    Reverse edges and check state:

    >>> ng.reverse_edges()
    >>> print(ng.is_reversed)
    True

    Set and retrieve metadata:

    >>> ng.set_metadata(experiment_id='exp_001', date='2024-01-01')
    >>> print(ng.get_metadata('experiment_id'))
    'exp_001'

    Notes
    -----
    NapistuGraph inherits from igraph.Graph, so all standard igraph methods
    (degree, shortest_paths, betweenness, etc.) are available. The additional
    functionality is designed specifically for molecular network analysis.

    Edge reversal swaps 'from'/'to' attributes, negates stoichiometry values,
    and updates direction metadata according to predefined mapping rules.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a NapistuGraph.

        Accepts all the same arguments as igraph.Graph constructor.
        """
        super().__init__(*args, **kwargs)

        # Initialize metadata
        self._metadata = {
            NAPISTU_METADATA_KEYS.IS_REVERSED: False,
            NAPISTU_METADATA_KEYS.WIRING_APPROACH: None,
            NAPISTU_METADATA_KEYS.WEIGHTING_STRATEGY: None,
            NAPISTU_METADATA_KEYS.WEIGHT_BY: None,
            NAPISTU_METADATA_KEYS.CREATION_PARAMS: {},
            NAPISTU_METADATA_KEYS.SPECIES_ATTRS: {},
            NAPISTU_METADATA_KEYS.REACTION_ATTRS: {},
        }

    @property
    def is_reversed(self) -> bool:
        """Check if the graph has been reversed."""
        return self._metadata[NAPISTU_METADATA_KEYS.IS_REVERSED]

    @property
    def wiring_approach(self) -> Optional[str]:
        """Get the graph type (bipartite, regulatory, etc.)."""
        return self._metadata[NAPISTU_METADATA_KEYS.WIRING_APPROACH]

    @property
    def weighting_strategy(self) -> Optional[str]:
        """Get the weighting strategy used."""
        return self._metadata[NAPISTU_METADATA_KEYS.WEIGHTING_STRATEGY]

    @property
    def weight_by(self) -> Optional[list[str]]:
        """Get the weight_by attributes used."""
        return self._metadata[NAPISTU_METADATA_KEYS.WEIGHT_BY]

    def add_all_entity_data(
        self,
        sbml_dfs: SBML_dfs,
        entity_type: str,
        target_entity: Optional[str] = None,
        table_names: Optional[list[str]] = None,
        add_name_prefixes: bool = True,
        mode: str = ADDING_ENTITY_DATA_DEFS.FRESH,
        overwrite: bool = False,
        inplace: bool = True,
    ) -> Optional["NapistuGraph"]:
        """
        Add all data tables for a specific entity type to the graph.

        This is a convenience method that automatically adds all available
        data tables for species or reactions without requiring manual
        configuration of each table and column.

        Parameters
        ----------
        sbml_dfs : SBML_dfs
            The SBML_dfs object containing entity data
        entity_type : str
            Either "species" or "reactions"
        target_entity : Optional[str], default=None
            Where to add the data: "vertices" or "edges". If None, uses default mapping:
            - "species" -> "vertices"
            - "reactions" -> "edges"
            Explicit specification allows reactions data on vertices (bipartite graphs)
            or species data on edges (if such use case exists).
        table_names : Optional[list[str]], default=None
            Specific table names to include. If None, includes all available tables.
        add_name_prefixes : bool, default=True
            Whether to prefix attribute names with table name (e.g., "table_name_column_name")
        mode : str, default="fresh"
            Either "fresh" (replace existing) or "extend" (add new attributes only)
        overwrite : bool, default=False
            Whether to allow overwriting existing attributes when conflicts arise
        inplace : bool, default=True
            Whether to modify the graph in place

        Returns
        -------
        Optional[NapistuGraph]
            If inplace=True, returns None. Otherwise returns modified copy.

        Raises
        ------
        ValueError
            If entity_type is invalid or requested tables don't exist

        Examples
        --------
        Add all species data to vertices (default):
        >>> graph.add_all_entity_data(sbml_dfs, "species")

        Add reactions data to reaction vertices (bipartite graph):
        >>> graph.add_all_entity_data(sbml_dfs, "reactions", target_entity="vertices")

        Add reactions data to edges (interaction graph):
        >>> graph.add_all_entity_data(sbml_dfs, "reactions", target_entity="edges")

        Add specific tables:
        >>> graph.add_all_entity_data(sbml_dfs, "reactions",
        ...                          table_names=["experiments", "literature"])
        """

        if not inplace:
            graph = self.copy()
        else:
            graph = self

        # Validate entity_type
        if entity_type not in ENTITIES_W_DATA:
            raise ValueError(f"entity_type must be one of {ENTITIES_W_DATA}")

        # Determine target_entity if not specified
        if target_entity is None:
            if entity_type == SBML_DFS.SPECIES:
                target_entity = NAPISTU_GRAPH.VERTICES
            elif entity_type == SBML_DFS.REACTIONS:
                target_entity = NAPISTU_GRAPH.EDGES
        else:
            # Validate explicit target_entity
            if target_entity not in [NAPISTU_GRAPH.VERTICES, NAPISTU_GRAPH.EDGES]:
                raise ValueError(
                    f"target_entity must be '{NAPISTU_GRAPH.VERTICES}' or '{NAPISTU_GRAPH.EDGES}'"
                )

        # Get available data tables
        entity_data_attr = ENTITIES_TO_ENTITY_DATA[entity_type]
        entity_data_dict = getattr(sbml_dfs, entity_data_attr)

        if len(entity_data_dict) == 0:
            logger.warning(f"No data tables found in {entity_data_attr}")
            return None if inplace else graph

        # Validate and filter table names
        if table_names is None:
            table_names = list(entity_data_dict.keys())
        else:
            invalid_tables = set(table_names) - set(entity_data_dict.keys())
            if invalid_tables:
                available_tables = list(entity_data_dict.keys())
                raise ValueError(
                    f"Requested tables not found in {entity_data_attr}: {invalid_tables}. "
                    f"Available tables: {available_tables}"
                )

        # Build entity_attrs configuration using utility function
        entity_attrs = ng_utils.create_entity_attrs_from_data_tables(
            entity_data_dict=entity_data_dict,
            table_names=table_names,
            add_name_prefixes=add_name_prefixes,
        )

        if not entity_attrs:
            logger.warning("No attributes to add")
            return None if inplace else graph

        # Set graph attributes using existing infrastructure
        graph_attrs_config = {entity_type: entity_attrs}
        graph.set_graph_attrs(graph_attrs_config, mode=mode, overwrite=overwrite)

        # Add the actual data using existing methods
        if target_entity == NAPISTU_GRAPH.VERTICES:
            graph.add_vertex_data(sbml_dfs, mode=mode, overwrite=overwrite)
        else:  # edges
            graph.add_edge_data(sbml_dfs, mode=mode, overwrite=overwrite)

        logger.info(
            f"Added {len(entity_attrs)} {entity_type} attributes to {target_entity} from "
            f"{len(table_names)} tables: {table_names}"
        )

        return None if inplace else graph

    def add_degree_attributes(self, inplace: bool = True) -> Optional["NapistuGraph"]:
        """
        Calculate and add degree-based attributes (parents, children, degree) to the graph.

        This method calculates the number of parents, children, and total degree for each node
        and stores these as edge attributes to support topology weighting. The attributes are
        calculated from the current graph's edge data.

        Parameters
        ----------
        inplace : bool, default=True
            Whether to modify the graph in place. If False, returns a copy with degree attributes.

        Returns
        -------
        Optional[NapistuGraph]
            If inplace=True, returns None.
            If inplace=False, returns a new NapistuGraph with degree attributes added to edges.
        """
        # If not inplace, make a copy
        if not inplace:
            graph = self.copy()
        else:
            graph = self

        # Check if degree attributes already exist
        existing_attrs = set(graph.es.attributes())
        degree_attrs = {
            NAPISTU_GRAPH_EDGES.SC_DEGREE,
            NAPISTU_GRAPH_EDGES.SC_CHILDREN,
            NAPISTU_GRAPH_EDGES.SC_PARENTS,
        }

        existing_degree_attrs = degree_attrs.intersection(existing_attrs)

        if existing_degree_attrs and not degree_attrs.issubset(existing_attrs):
            # Some but not all degree attributes exist - this is pathological
            missing_attrs = degree_attrs - existing_attrs
            raise ValueError(
                f"Some degree attributes already exist ({existing_degree_attrs}) but others are missing ({missing_attrs}). "
                f"This indicates an inconsistent state. Please remove all degree attributes before recalculating."
            )
        elif degree_attrs.issubset(existing_attrs):
            logger.warning("Degree attributes already exist. Skipping calculation.")
            return None if inplace else graph

        # Get current edge data
        edges_df = graph.get_edge_dataframe()

        # Calculate undirected and directed degrees (i.e., # of parents and children)
        # based on the network's edgelist
        unique_edges = (
            edges_df.groupby([NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO])
            .first()
            .reset_index()
        )

        # Calculate children (out-degree)
        n_children = (
            unique_edges[NAPISTU_GRAPH_EDGES.FROM]
            .value_counts()
            .to_frame(name=NAPISTU_GRAPH_EDGES.SC_CHILDREN)
            .reset_index()
            .rename({NAPISTU_GRAPH_EDGES.FROM: "node_id"}, axis=1)
        )

        # Calculate parents (in-degree)
        n_parents = (
            unique_edges[NAPISTU_GRAPH_EDGES.TO]
            .value_counts()
            .to_frame(name=NAPISTU_GRAPH_EDGES.SC_PARENTS)
            .reset_index()
            .rename({NAPISTU_GRAPH_EDGES.TO: "node_id"}, axis=1)
        )

        # Merge children and parents data
        graph_degree_by_edgelist = n_children.merge(n_parents, how="outer").fillna(
            int(0)
        )

        # Calculate total degree
        graph_degree_by_edgelist[NAPISTU_GRAPH_EDGES.SC_DEGREE] = (
            graph_degree_by_edgelist[NAPISTU_GRAPH_EDGES.SC_CHILDREN]
            + graph_degree_by_edgelist[NAPISTU_GRAPH_EDGES.SC_PARENTS]
        )

        # Filter out reaction nodes (those with IDs matching "R[0-9]{8}")
        graph_degree_by_edgelist = (
            graph_degree_by_edgelist[
                ~graph_degree_by_edgelist["node_id"].str.contains("R[0-9]{8}")
            ]
            .set_index("node_id")
            .sort_index()
        )

        # Merge degree data back with edge data
        # For edges where FROM is a species (not reaction), use FROM node's degree
        # For edges where FROM is a reaction, use TO node's degree
        is_from_reaction = edges_df[NAPISTU_GRAPH_EDGES.FROM].str.contains("R[0-9]{8}")

        # Create degree data for edges
        edge_degree_data = pd.concat(
            [
                # Edges where FROM is a species - use FROM node's degree
                edges_df[~is_from_reaction].merge(
                    graph_degree_by_edgelist,
                    left_on=NAPISTU_GRAPH_EDGES.FROM,
                    right_index=True,
                    how="left",
                ),
                # Edges where FROM is a reaction - use TO node's degree
                edges_df[is_from_reaction].merge(
                    graph_degree_by_edgelist,
                    left_on=NAPISTU_GRAPH_EDGES.TO,
                    right_index=True,
                    how="left",
                ),
            ]
        ).fillna(int(0))

        # Add degree attributes to edges
        graph.es[NAPISTU_GRAPH_EDGES.SC_DEGREE] = edge_degree_data[
            NAPISTU_GRAPH_EDGES.SC_DEGREE
        ].values
        graph.es[NAPISTU_GRAPH_EDGES.SC_CHILDREN] = edge_degree_data[
            NAPISTU_GRAPH_EDGES.SC_CHILDREN
        ].values
        graph.es[NAPISTU_GRAPH_EDGES.SC_PARENTS] = edge_degree_data[
            NAPISTU_GRAPH_EDGES.SC_PARENTS
        ].values

        return None if inplace else graph

    def add_edge_data(
        self,
        sbml_dfs: Optional[SBML_dfs] = None,
        side_loaded_attributes: Optional[dict[str, pd.DataFrame]] = None,
        mode: str = ADDING_ENTITY_DATA_DEFS.FRESH,
        overwrite: bool = False,
        inplace: bool = True,
    ) -> Optional["NapistuGraph"]:
        """
        Extract and add reaction attributes to the graph edges.

        Parameters
        ----------
        sbml_dfs : SBML_dfs
            The SBML_dfs object containing reaction data. If None, only side-loaded attributes will be used.
        side_loaded_attributes : dict[str, pd.DataFrame], optional
            Dictionary mapping table names to DataFrames for side-loaded data
        mode : str
            Either "fresh" (replace existing) or "extend" (add new attributes only)
        overwrite : bool
            Whether to allow overwriting existing edge attributes when conflicts arise. Ignored if mode is "extend".
        inplace : bool, default=True
            Whether to modify the graph in place. If False, returns a copy with edge data.

        Returns
        -------
        Optional[NapistuGraph]
            If inplace=True, returns None.
            If inplace=False, returns a new NapistuGraph with edge data added.
        """
        return self._add_entity_data(
            entity_type=SBML_DFS.REACTIONS,
            target_entity=NAPISTU_GRAPH.EDGES,
            sbml_dfs=sbml_dfs,
            side_loaded_attributes=side_loaded_attributes,
            mode=mode,
            overwrite=overwrite,
            inplace=inplace,
        )

    def add_vertex_data(
        self,
        sbml_dfs: Optional[SBML_dfs] = None,
        side_loaded_attributes: Optional[dict[str, pd.DataFrame]] = None,
        mode: str = ADDING_ENTITY_DATA_DEFS.FRESH,
        overwrite: bool = False,
        inplace: bool = True,
    ) -> Optional["NapistuGraph"]:
        """
        Extract and add species attributes to the graph vertices.

        Parameters
        ----------
        sbml_dfs : SBML_dfs
            The SBML_dfs object containing species data. If None, only side-loaded attributes will be used.
        side_loaded_attributes : dict[str, pd.DataFrame], optional
            Dictionary mapping table names to DataFrames for side-loaded data
        mode : str
            Either "fresh" (replace existing) or "extend" (add new attributes only)
        overwrite : bool
            Whether to allow overwriting existing vertex attributes when conflicts arise. Ignored if mode is "extend".
        inplace : bool, default=True
            Whether to modify the graph in place. If False, returns a copy with vertex data.

        Returns
        -------
        Optional[NapistuGraph]
            If inplace=True, returns None.
            If inplace=False, returns a new NapistuGraph with vertex data added.
        """
        return self._add_entity_data(
            entity_type=SBML_DFS.SPECIES,
            target_entity=NAPISTU_GRAPH.VERTICES,
            sbml_dfs=sbml_dfs,
            side_loaded_attributes=side_loaded_attributes,
            mode=mode,
            overwrite=overwrite,
            inplace=inplace,
        )

    def add_sbml_dfs_summaries(
        self,
        sbml_dfs: SBML_dfs,
        summary_types: list[str] = VALID_VERTEX_SBML_DFS_SUMMARIES,
        priority_pathways: list[str] = DEFAULT_PRIORITIZED_PATHWAYS,
        stratify_by_bqb: bool = True,
        characteristic_only: bool = False,
        dogmatic: bool = False,
        add_name_prefixes: bool = False,
        binarize: bool = False,
        mode: str = ADDING_ENTITY_DATA_DEFS.FRESH,
        overwrite: bool = False,
        inplace: bool = True,
    ) -> Optional["NapistuGraph"]:
        """
        Add vertex summaries from SBML_dfs to the graph vertices.

        This method calls get_sbml_dfs_vertex_summaries and merges the results
        with the graph's vertices by name.

        Parameters
        ----------
        sbml_dfs : SBML_dfs
            The SBML_dfs object to extract vertex summaries from
        summary_types : list, optional
            Types of summaries to include. Defaults to all valid summary types.
        priority_pathways : list, optional
            Priority pathways for source occurrence calculations. Defaults to DEFAULT_PRIORITIZED_PATHWAYS.
        stratify_by_bqb : bool, optional
            Whether to stratify by BioQualifiers. Default is True.
        characteristic_only : bool, optional
            Whether to include only characteristic identifiers. Default is False.
        dogmatic : bool, optional
            Whether to use dogmatic mode. Default is False.
        add_name_prefixes : bool, default False
            If True, add prefixes to column names: 'source_' for source data
            and 'ontology_' for ontology data
        binarize: bool, optional
            Whether to convert the summary to binary values (0 vs 1+). Default is False.
        mode : str
            Either "fresh" (replace existing) or "extend" (add new attributes only)
        overwrite : bool
            Whether to allow overwriting existing vertex attributes when conflicts arise. Ignored if mode is "extend".
        inplace : bool, default=True
            Whether to modify the graph in place. If False, returns a copy with summary attributes.

        Returns
        -------
        Optional[NapistuGraph]
            If inplace=True, returns None.
            If inplace=False, returns a new NapistuGraph with summary attributes added to vertices.
        """
        # If not inplace, make a copy
        if not inplace:
            graph = self.copy()
        else:
            graph = self

        # Get vertex summaries from SBML_dfs
        logger.info("Loading SBML_dfs vertex summaries")
        summaries_df = ng_utils.get_sbml_dfs_vertex_summaries(
            sbml_dfs,
            summary_types=summary_types,
            priority_pathways=priority_pathways,
            stratify_by_bqb=stratify_by_bqb,
            characteristic_only=characteristic_only,
            dogmatic=dogmatic,
            add_name_prefixes=add_name_prefixes,
            binarize=binarize,
            has_reactions=graph.get_has_reactions(),
        )

        logger.info("Creating graph attributes")
        graph_attrs = data_handling._create_graph_attrs_config(
            column_mapping={v: v for v in summaries_df.columns},
            data_type=SBML_DFS.SPECIES,
            table_name="sbml_dfs_summaries",
            transformation="identity",
        )

        graph.set_graph_attrs(graph_attrs, mode=mode, overwrite=overwrite)

        logger.info("Adding vertex data")
        graph.add_vertex_data(
            side_loaded_attributes={"sbml_dfs_summaries": summaries_df},
            mode=mode,
            overwrite=overwrite,
        )

        return None if inplace else graph

    def add_topology_weights(
        self,
        base_score: float = 2,
        protein_multiplier: int = 1,
        metabolite_multiplier: int = 3,
        drug_multiplier: int = 1,
        complex_multiplier: int = 3,
        unknown_multiplier: int = 10,
        scale_multiplier_by_meandegree: bool = True,
        inplace: bool = True,
    ) -> Optional["NapistuGraph"]:
        """
        Create Topology Weights for a network based on its topology.

        Edges downstream of nodes with many connections receive a higher weight suggesting that any one
        of them is less likely to be regulatory. This is a simple and clearly flawed heuristic which can be
        combined with more principled weighting schemes.

        Parameters
        ----------
        base_score : float, optional
            Offset which will be added to all weights. Default is 2.
        protein_multiplier : int, optional
            Multiplier for non-metabolite species. Default is 1.
        metabolite_multiplier : int, optional
            Multiplier for metabolites. Default is 3.
        drug_multiplier : int, optional
            Multiplier for drugs. Default is 1.
        complex_multiplier : int, optional
            Multiplier for complexes. Default is 3.
        unknown_multiplier : int, optional
            Multiplier for species without any identifier. Default is 10.
        scale_multiplier_by_meandegree : bool, optional
            If True, multipliers will be rescaled by the average number of connections a node has. Default is True.
        inplace : bool, default=True
            Whether to modify the graph in place. If False, returns a copy with topology weights.

        Returns
        -------
        Optional[NapistuGraph]
            If inplace=True, returns None.
            If inplace=False, returns a new NapistuGraph with topology weights added.

        Raises
        ------
        ValueError
            If required attributes are missing or if parameters are invalid.
        """

        # If not inplace, make a copy
        if not inplace:
            graph = self.copy()
        else:
            graph = self

        # Check for required attributes and add degree attributes if missing
        degree_attrs = {
            NAPISTU_GRAPH_EDGES.SC_DEGREE,
            NAPISTU_GRAPH_EDGES.SC_CHILDREN,
            NAPISTU_GRAPH_EDGES.SC_PARENTS,
        }

        missing_degree_attrs = degree_attrs.difference(set(graph.es.attributes()))
        if missing_degree_attrs:
            logger.info(f"Adding missing degree attributes: {missing_degree_attrs}")
            graph.add_degree_attributes()

        # Check for species_type attribute
        if NAPISTU_GRAPH_EDGES.SPECIES_TYPE not in graph.es.attributes():
            raise ValueError(
                f"Missing required attribute: {NAPISTU_GRAPH_EDGES.SPECIES_TYPE}. "
                "Species type information is required for topology weighting."
            )

        if base_score < 0:
            raise ValueError(f"base_score was {base_score} and must be non-negative")
        if protein_multiplier > unknown_multiplier:
            raise ValueError(
                f"protein_multiplier was {protein_multiplier} and unknown_multiplier "
                f"was {unknown_multiplier}. unknown_multiplier must be greater than "
                "protein_multiplier"
            )
        if metabolite_multiplier > unknown_multiplier:
            raise ValueError(
                f"protein_multiplier was {metabolite_multiplier} and unknown_multiplier "
                f"was {unknown_multiplier}. unknown_multiplier must be greater than "
                "protein_multiplier"
            )

        # create a new weight variable
        weight_table = pd.DataFrame(
            {
                NAPISTU_GRAPH_EDGES.SC_DEGREE: graph.es[NAPISTU_GRAPH_EDGES.SC_DEGREE],
                NAPISTU_GRAPH_EDGES.SC_CHILDREN: graph.es[
                    NAPISTU_GRAPH_EDGES.SC_CHILDREN
                ],
                NAPISTU_GRAPH_EDGES.SC_PARENTS: graph.es[
                    NAPISTU_GRAPH_EDGES.SC_PARENTS
                ],
                NAPISTU_GRAPH_EDGES.SPECIES_TYPE: graph.es[
                    NAPISTU_GRAPH_EDGES.SPECIES_TYPE
                ],
            }
        )

        lookup_multiplier_dict = {
            "complex": complex_multiplier,
            "drug": drug_multiplier,
            "metabolite": metabolite_multiplier,
            "protein": protein_multiplier,
            "unknown": unknown_multiplier,
        }
        weight_table["multiplier"] = weight_table["species_type"].map(
            lookup_multiplier_dict
        )

        if any(weight_table["multiplier"].isna()):
            raise ValueError("Missing multiplier values")

        # calculate mean degree
        # since topology weights will differ based on the structure of the network
        # and it would be nice to have a consistent notion of edge weights and path weights
        # for interpretability and filtering, we can rescale topology weights by the
        # average degree of nodes
        if scale_multiplier_by_meandegree:
            mean_degree = len(self.es) / len(self.vs)
            if not self.is_directed():
                # for a directed network in- and out-degree are separately treated while
                # an undirected network's degree will be the sum of these two measures.
                mean_degree = mean_degree * 2

            weight_table["multiplier"] = weight_table["multiplier"] / mean_degree

        if self.is_directed():
            weight_table["connection_weight"] = weight_table[
                NAPISTU_GRAPH_EDGES.SC_CHILDREN
            ]
        else:
            weight_table["connection_weight"] = weight_table[
                NAPISTU_GRAPH_EDGES.SC_DEGREE
            ]

        # weight traveling through a species based on
        # - a constant
        # - how plausibly that species type mediates a change
        # - the number of connections that the node can bridge to
        weight_table["topo_weights"] = [
            base_score + (x * y)
            for x, y in zip(
                weight_table["multiplier"], weight_table["connection_weight"]
            )
        ]
        graph.es["topo_weights"] = weight_table["topo_weights"]

        # if directed and we want to use travel upstream define a corresponding weighting scheme
        if graph.is_directed():
            weight_table["upstream_topo_weights"] = [
                base_score + (x * y)
                for x, y in zip(weight_table["multiplier"], weight_table["sc_parents"])
            ]
            graph.es["upstream_topo_weights"] = weight_table["upstream_topo_weights"]

        return None if inplace else graph

    def copy(self) -> "NapistuGraph":
        """
        Create a deep copy of the NapistuGraph.

        Returns
        -------
        NapistuGraph
            A deep copy of this graph including metadata
        """
        # Use igraph's copy method to get the graph structure and attributes
        new_graph = super().copy()

        # Convert to NapistuGraph and copy metadata
        napistu_copy = NapistuGraph.from_igraph(new_graph)
        napistu_copy._metadata = copy.deepcopy(self._metadata)

        return napistu_copy

    @classmethod
    def from_igraph(cls, graph: ig.Graph, **metadata) -> "NapistuGraph":
        """
        Create a NapistuGraph from an existing igraph.Graph.

        Parameters
        ----------
        graph : ig.Graph
            The igraph to convert
        **metadata : dict
            Additional metadata to store with the graph

        Returns
        -------
        NapistuGraph
            A new NapistuGraph instance
        """
        # Create new instance with same structure
        new_graph = cls(
            n=graph.vcount(),
            edges=[(e.source, e.target) for e in graph.es],
            directed=graph.is_directed(),
        )

        # Copy all vertex attributes
        for attr in graph.vs.attributes():
            new_graph.vs[attr] = graph.vs[attr]

        # Copy all edge attributes
        for attr in graph.es.attributes():
            new_graph.es[attr] = graph.es[attr]

        # Copy graph attributes
        for attr in graph.attributes():
            new_graph[attr] = graph[attr]

        # Set metadata
        new_graph._metadata.update(metadata)

        return new_graph

    @classmethod
    def from_pickle(cls, path: str) -> "NapistuGraph":
        """
        Load a NapistuGraph from a pickle file.

        Parameters
        ----------
        path : str
            Path to the pickle file

        Returns
        -------
        NapistuGraph
            The loaded NapistuGraph object
        """
        napistu_graph = utils.load_pickle(path)
        if not isinstance(napistu_graph, cls):
            raise ValueError(
                f"Pickled input is not a NapistuGraph object but {type(napistu_graph)}: {path}"
            )
        return napistu_graph

    def get_edge_dataframe(self) -> pd.DataFrame:
        """
        Get edges as a Pandas DataFrame.
        Wrapper around igraph's get_edge_dataframe method.

        Returns
        -------
        pandas.DataFrame
            A table with one row per edge.
        """
        return super().get_edge_dataframe()

    def get_edge_endpoint_attributes(
        self, vertex_attribute_names: Union[str, List[str]]
    ) -> pd.DataFrame:
        """
        Get source and target vertex attributes for all edges.

        Creates a DataFrame with one row per edge containing the specified
        vertex attribute values for both the source and target nodes.

        Parameters
        ----------
        vertex_attribute_names : str or list of str
            Name(s) of vertex attribute(s) to extract for source and target nodes
            (e.g., 'species_type', ['species_type', 'node_type'])

        Returns
        -------
        pd.DataFrame
            DataFrame with row MultiIndex (from, to) and column MultiIndex
            (attribute_name, endpoint) where endpoint is 'source' or 'target'

        Raises
        ------
        KeyError
            If any vertex attribute does not exist

        Examples
        --------
        >>> # Get single attribute
        >>> species_df = graph.get_edge_endpoint_attributes('species_type')
        >>> species_df.head()
                                    species_type
                                        source    target
        from         to
        R00000000    SC00015559          reaction metabolite
        R00000214    SC00001716          reaction metabolite

        >>> # Get multiple attributes at once
        >>> edge_attrs = graph.get_edge_endpoint_attributes(['species_type', 'node_type'])
        >>> edge_attrs.head()
                                    species_type           node_type
                                        source    target    source    target
        from         to
        R00000000    SC00015559          reaction metabolite  reaction  species
        R00000214    SC00001716          reaction metabolite  reaction  species

        >>> # Access specific columns
        >>> edge_attrs[('species_type', 'source')]
        >>> edge_attrs['species_type']  # Both source and target for species_type
        >>> edge_attrs.xs('source', level='endpoint', axis=1)  # All source attributes
        """
        # Normalize input to list
        if isinstance(vertex_attribute_names, str):
            vertex_attribute_names = [vertex_attribute_names]

        # Validate all attributes exist
        available_attrs = self.vs.attributes()
        for attr_name in vertex_attribute_names:
            if attr_name not in available_attrs:
                raise KeyError(f"Vertex attribute '{attr_name}' does not exist")

        # Get all edges as (source, target) pairs
        edge_list = self.get_edgelist()

        # Create row MultiIndex from edge names (from, to)
        # edge_list contains (source_index, target_index) pairs
        from_names = [self.vs[src][NAPISTU_GRAPH_VERTICES.NAME] for src, _ in edge_list]
        to_names = [self.vs[tgt][NAPISTU_GRAPH_VERTICES.NAME] for _, tgt in edge_list]
        row_index = pd.MultiIndex.from_arrays(
            [from_names, to_names],
            names=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO],
        )

        # Build data dictionary for DataFrame construction
        data = {}
        for attr_name in vertex_attribute_names:
            # Get attribute values for all vertices
            attr_values = self.vs[attr_name]

            # Map to edge endpoints
            source_attrs = [attr_values[src] for src, _ in edge_list]
            target_attrs = [attr_values[tgt] for _, tgt in edge_list]

            data[(attr_name, IGRAPH_DEFS.SOURCE)] = source_attrs
            data[(attr_name, IGRAPH_DEFS.TARGET)] = target_attrs

        # Create column MultiIndex
        col_index = pd.MultiIndex.from_tuples(
            data.keys(),
            names=[
                NAPISTU_GRAPH_EDGE_ENDPOINT_ATTRIBUTES.ATTRIBUTE_NAME,
                NAPISTU_GRAPH_EDGE_ENDPOINT_ATTRIBUTES.ENDPOINT,
            ],
        )

        # Build DataFrame
        df = pd.DataFrame(data, index=row_index, columns=col_index)

        return df

    def get_edge_series(self, attribute_name: str) -> pd.Series:
        """
        Get a single edge attribute as a pandas Series.

        Parameters
        ----------
        attribute_name : str
            Name of the edge attribute to extract

        Returns
        -------
        pd.Series
            Series with MultiIndex (from, to) as index and attribute values as data

        Raises
        ------
        KeyError
            If the attribute does not exist on any edges
        """
        if attribute_name not in self.es.attributes():
            raise KeyError(f"Edge attribute '{attribute_name}' not found")

        edge_tuples = [
            (e[NAPISTU_GRAPH_EDGES.FROM], e[NAPISTU_GRAPH_EDGES.TO]) for e in self.es
        ]
        attribute_values = [e[attribute_name] for e in self.es]

        multi_index = pd.MultiIndex.from_tuples(
            edge_tuples, names=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
        )
        return pd.Series(attribute_values, index=multi_index, name=attribute_name)

    def get_has_reactions(self) -> bool:
        """
        Check if the graph contains reaction vertices.

        Returns
        -------
        bool
            True if the graph contains reaction vertices, False otherwise.
        """
        node_types = self.vs[NAPISTU_GRAPH_VERTICES.NODE_TYPE]
        return NAPISTU_GRAPH_NODE_TYPES.REACTION in node_types

    def get_metadata(self, key: Optional[str] = None) -> Any:
        """
        Get metadata from the graph.

        Parameters
        ----------
        key : str, optional
            Specific metadata key to retrieve. If None, returns all metadata.

        Returns
        -------
        Any
            The requested metadata value, or all metadata if key is None
        """
        if key is None:
            return self._metadata.copy()
        return self._metadata.get(key)

    def get_summary(self) -> Mapping[str, Any]:
        """
        Get summary statistics for the graph.

        Returns
        -------
        Mapping[str, Any]
            A dictionary of summary statistics.
            - n_vertices: Number of vertices
            - vertex_node_type_dict: Dictionary of node type counts
            - vertex_species_type_dict: Dictionary of species type counts
            - vertex_attributes: List of vertex attributes
            - n_edges: Number of edges
            - sbo_name_counts_dict: Dictionary of SBO name counts
            - edge_attributes: List of edge attributes
        """

        stats: MutableMapping[str, Any] = {}

        # vertex summaries
        vertex_df = self.get_vertex_dataframe()
        stats["n_vertices"] = vertex_df.shape[0]
        stats["vertex_node_type_dict"] = vertex_df.value_counts("node_type").to_dict()
        stats["vertex_species_type_dict"] = vertex_df.value_counts(
            "species_type"
        ).to_dict()
        stats["vertex_attributes"] = vertex_df.columns.tolist()

        # edge summaries
        stats["n_edges"] = len(self.es)

        # Concatenate and count both upstream and downstream SBO terms, then map to names
        upstream_sbo_terms = self.get_edge_series(NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM)
        downstream_sbo_terms = self.get_edge_series(
            NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM
        )

        # Concatenate both series, filter out None/NaN and empty strings, count, then map to names
        all_sbo_terms = pd.concat([upstream_sbo_terms, downstream_sbo_terms])
        all_sbo_terms = all_sbo_terms[all_sbo_terms.notna() & (all_sbo_terms != "")]
        sbo_term_counts = all_sbo_terms.value_counts()
        sbo_name_counts = sbo_term_counts.rename(index=MINI_SBO_TO_NAME)
        stats["sbo_name_counts_dict"] = sbo_name_counts.to_dict()
        stats["edge_attributes"] = self.es.attributes()

        return stats

    def get_vertex_dataframe(self) -> pd.DataFrame:
        """
        Get vertices as a Pandas DataFrame.
        Wrapper around igraph's get_vertex_dataframe method.

        Returns
        -------
        pandas.DataFrame
            A table with one row per vertex.
        """
        df = super().get_vertex_dataframe()
        df.index = df.index.rename(IGRAPH_DEFS.INDEX)
        return df

    def get_vertex_series(self, attribute_name: str) -> pd.Series:
        """
        Get a single vertex attribute as a pandas Series.

        Parameters
        ----------
        attribute_name : str
            Name of the vertex attribute to extract

        Returns
        -------
        pd.Series
            Series with vertex names as index and attribute values as data

        Raises
        ------
        KeyError
            If the attribute does not exist on any vertices
        """
        if attribute_name not in self.vs.attributes():
            raise KeyError(f"Vertex attribute '{attribute_name}' not found")

        vertex_names = [v[NAPISTU_GRAPH_VERTICES.NAME] for v in self.vs]
        attribute_values = [v[attribute_name] for v in self.vs]

        series = pd.Series(attribute_values, index=vertex_names, name=attribute_name)
        series.index.name = NAPISTU_GRAPH_VERTICES.NAME
        return series

    def set_graph_attrs(
        self,
        graph_attrs: Union[str, dict],
        mode: str = ADDING_ENTITY_DATA_DEFS.FRESH,
        overwrite: bool = False,
        validate_transformations: bool = True,
        custom_transformations: Optional[dict] = None,
    ) -> None:
        """
        Set graph attributes from YAML file or dictionary.

        Parameters
        ----------
        graph_attrs : str or dict
            Either path to YAML file or dictionary with 'species' and/or 'reactions' keys
        mode : str
            Either "fresh" (replace existing) or "extend" (add new keys)
        overwrite : bool
            Whether to allow overwriting existing data when conflicts arise
        validate_transformations : bool, default=True
            Whether to validate transformation names when setting up attributes
        custom_transformations : dict, optional
            Dictionary of custom transformation functions for validation. The values should be functions.
        """

        # Load from YAML if string path provided
        if isinstance(graph_attrs, str):
            graph_attrs = ng_utils.read_graph_attrs_spec(graph_attrs)

        # Process species attributes if present
        if SBML_DFS.SPECIES in graph_attrs:
            # Validate species attributes if requested
            if validate_transformations:
                ng_utils._validate_entity_attrs(
                    graph_attrs[SBML_DFS.SPECIES],
                    validate_transformations=True,
                    custom_transformations=custom_transformations,
                )

            merged_species = self._compare_and_merge_attrs(
                graph_attrs["species"],
                NAPISTU_METADATA_KEYS.SPECIES_ATTRS,
                mode,
                overwrite,
            )
            self.set_metadata(species_attrs=merged_species)

        # Process reaction attributes if present
        if SBML_DFS.REACTIONS in graph_attrs:
            # Validate reaction attributes if requested
            if validate_transformations:
                ng_utils._validate_entity_attrs(
                    graph_attrs[SBML_DFS.REACTIONS],
                    validate_transformations=True,
                    custom_transformations=custom_transformations,
                )

            merged_reactions = self._compare_and_merge_attrs(
                graph_attrs[SBML_DFS.REACTIONS],
                NAPISTU_METADATA_KEYS.REACTION_ATTRS,
                mode,
                overwrite,
            )
            self.set_metadata(reaction_attrs=merged_reactions)

    def show_summary(self) -> None:
        """
        Show summary statistics for the graph.
        """
        summary_stats = self.get_summary()
        summary_table = ng_utils.format_napistu_graph_summary(summary_stats)
        utils.show(summary_table)

    def remove_attributes(self, attribute_type: str, attributes: list[str]) -> None:
        """
        Remove specified attributes from vertices or edges.

        This method removes the specified attributes from either vertices or edges
        and warns if any of the attributes to be removed are not associated with
        any vertices/edges in the graph.

        Parameters
        ----------
        attribute_type : str
            Type of attributes to remove. Must be either NAPISTU_GRAPH.VERTICES or NAPISTU_GRAPH.EDGES.
        attributes : list[str]
            List of attribute names to remove from the specified attribute type.

        Returns
        -------
        None
            The graph is modified in-place.

        Raises
        ------
        ValueError
            If attribute_type is not NAPISTU_GRAPH.VERTICES or NAPISTU_GRAPH.EDGES.

        Examples
        --------
        Remove vertex attributes:
        >>> graph.remove_attributes(NAPISTU_GRAPH.VERTICES, ["species_type", "node_type"])

        Remove edge attributes:
        >>> graph.remove_attributes(NAPISTU_GRAPH.EDGES, ["weight", "stoichiometry"])

        Notes
        -----
        This method will warn about attributes that don't exist in the graph
        but will continue to remove the attributes that do exist.
        """

        # Get the appropriate attribute container
        if attribute_type == NAPISTU_GRAPH.VERTICES:
            attr_container = self.vs
        elif attribute_type == NAPISTU_GRAPH.EDGES:
            attr_container = self.es
        else:
            raise ValueError(
                f"attribute_type must be one of: {NAPISTU_GRAPH.VERTICES}, {NAPISTU_GRAPH.EDGES}, got '{attribute_type}'"
            )

        missing_attrs = set(attributes) - set(attr_container.attributes())
        if len(missing_attrs) > 0:
            logger.warning(
                f"The following {attribute_type} attributes are not associated with any {attribute_type} and will be skipped: {sorted(missing_attrs)}"
            )

        existing_attrs_to_remove = set(attributes) & set(attr_container.attributes())
        # Remove existing attributes
        if existing_attrs_to_remove:
            for attr_name in existing_attrs_to_remove:
                del attr_container[attr_name]

            # Clean up metadata for removed attributes
            self._cleanup_removed_attributes_metadata(
                attribute_type, existing_attrs_to_remove
            )

        return None

    def remove_isolated_vertices(self, node_types: str = SBML_DFS.REACTIONS):
        """
        Remove vertices that have no edges (degree 0) from the graph.

        By default, only removes reaction singletons since these are not included
        in wiring by-construction for interaction edges. Species singletons may
        reflect that their reactions were specifically removed (e.g., water if
        cofactors are removed).

        Parameters
        ----------
        node_types : str, default="reactions"
            Which type of isolated vertices to remove. Options:
            - "reactions": Remove only isolated reaction vertices (default)
            - "species": Remove only isolated species vertices
            - "all": Remove all isolated vertices regardless of type

        Returns
        -------
        None
            The graph is modified in-place.

        """

        # Find isolated vertices (degree 0)
        isolated_vertices = self.vs.select(_degree=0)

        if len(isolated_vertices) == 0:
            logger.info("No isolated vertices found to remove")
            return

        # Filter by node type if specified
        if node_types in [SBML_DFS.REACTIONS, SBML_DFS.SPECIES]:
            # Check if node_type attribute exists
            if NAPISTU_GRAPH_VERTICES.NODE_TYPE not in self.vs.attributes():
                raise ValueError(
                    f"Cannot filter by {node_types} - {NAPISTU_GRAPH_VERTICES.NODE_TYPE} "
                    "attribute not found. Please add the node_type attribute to the graph."
                )
            else:
                # Filter to only the specified type
                target_type = (
                    NAPISTU_GRAPH_NODE_TYPES.REACTION
                    if node_types == SBML_DFS.REACTIONS
                    else NAPISTU_GRAPH_NODE_TYPES.SPECIES
                )
                filtered_vertices = isolated_vertices.select(
                    **{NAPISTU_GRAPH_VERTICES.NODE_TYPE: target_type}
                )
        elif node_types == "all":
            filtered_vertices = isolated_vertices
        else:
            raise ValueError(
                f"Invalid node_types: {node_types}. "
                f"Must be one of: 'reactions', 'species', 'all'"
            )

        if len(filtered_vertices) == 0:
            logger.info(f"No isolated {node_types} vertices found to remove")
            return

        # Get vertex names/indices for logging (up to 5 examples)
        vertex_names = []
        for v in filtered_vertices[:5]:
            # Use vertex name if available, otherwise use index
            name = (
                v[NAPISTU_GRAPH_VERTICES.NAME]
                if NAPISTU_GRAPH_VERTICES.NAME in v.attributes()
                and v[NAPISTU_GRAPH_VERTICES.NAME] is not None
                else str(v.index)
            )
            vertex_names.append(name)

        # Create log message
        examples_str = ", ".join(f"'{name}'" for name in vertex_names)
        if len(filtered_vertices) > 5:
            examples_str += f" (and {len(filtered_vertices) - 5} more)"

        logger.info(
            f"Removed {len(filtered_vertices)} isolated {node_types} vertices: [{examples_str}]"
        )

        # Remove the filtered isolated vertices
        self.delete_vertices(filtered_vertices)

    def deduplicate_edges(self, verbose: bool = False) -> None:
        """
        Deduplicate edges with the same FROM -> TO pair, keeping only the first occurrence.

        This identifies and removes duplicate edges between the same pair of vertices,
        keeping only the first edge encountered. Modifies the graph in-place.

        Parameters
        ----------
        verbose : bool, optional
            Whether to show example duplicate edges if duplicates are found. Default is False.

        Returns
        -------
        None
        """
        # Get current edge dataframe
        edges_df = self.get_edge_dataframe()

        # Identify duplicate edges using vectorized operation (fast)
        # duplicated(keep='first') marks all duplicates except the first occurrence as True
        is_duplicate = edges_df.duplicated(
            subset=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO], keep="first"
        )

        # Get edge indices of duplicate edges to remove (vectorized boolean indexing)
        duplicate_edge_indices = edges_df[is_duplicate].index.tolist()

        n_dropped = len(duplicate_edge_indices)
        if n_dropped > 0:
            logger.warning(
                f"{n_dropped} edges were dropped "
                "due to duplicated origin -> target relationships, use verbose for "
                "more information"
            )

            if verbose:
                # report duplicated edges
                grouped_edges = edges_df.groupby(
                    [NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
                )
                duplicated_groups = [
                    grouped_edges.get_group(x)
                    for x in grouped_edges.groups
                    if grouped_edges.get_group(x).shape[0] > 1
                ]
                if duplicated_groups:
                    import random

                    example_duplicates = pd.concat(
                        random.sample(duplicated_groups, min(5, len(duplicated_groups)))
                    )
                    utils.show(example_duplicates, headers="keys")

            # Remove duplicate edges by their indices
            # Sort in reverse order to avoid index shifting issues when deleting
            duplicate_edge_indices.sort(reverse=True)
            self.delete_edges(duplicate_edge_indices)

            logger.info(
                f"Deduplicated graph edges. {n_dropped} edges were removed, "
                f"{self.ecount()} edges remain."
            )
        else:
            logger.info("No duplicate edges found. Graph unchanged.")

        return None

    def reverse_edges(self) -> None:
        """
        Reverse all edges in the graph.

        This swaps edge directions and updates all associated attributes
        according to the edge reversal mapping utilities. Modifies the graph in-place.

        Returns
        -------
        None
        """
        # Get current edge dataframe
        edges_df = self.get_edge_dataframe()

        # Apply systematic attribute swapping using utilities
        reversed_edges_df = _apply_edge_reversal_mapping(edges_df)

        # Handle special cases using utilities
        reversed_edges_df = _handle_special_reversal_cases(reversed_edges_df)

        # Update edge attributes
        for attr in reversed_edges_df.columns:
            if attr in self.es.attributes():
                self.es[attr] = reversed_edges_df[attr].values

        # Update metadata
        self._metadata["is_reversed"] = not self._metadata["is_reversed"]

        logger.info(
            f"Reversed graph edges. Current state: reversed={self._metadata['is_reversed']}"
        )

        return None

    def set_metadata(self, **kwargs) -> None:
        """
        Set metadata for the graph.

        Modifies the graph's metadata in-place.

        Parameters
        ----------
        **kwargs : dict
            Metadata key-value pairs to set
        """
        self._metadata.update(kwargs)

        return None

    def set_weights(
        self,
        weighting_strategy: str = NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED,
        weight_by: list[str] | None = None,
        reaction_edge_multiplier: float = 0.5,
    ) -> None:
        """
        Set Graph Weights for this NapistuGraph using a specified weighting strategy.

        Modifies the graph in-place. Now includes functionality to downweight edges
        connected to reaction vertices to account for increased path lengths through
        reaction intermediates (e.g., S → R → P vs direct S → P).

        Parameters:
            weight_by (list[str], optional): A list of edge attributes to weight by.
                How these are used depends on the weighting strategy.
            weighting_strategy (str, optional): A network weighting strategy. Options:
                'unweighted': all weights (and upstream_weight for directed graphs) are set to 1.
                'topology': weight edges by the degree of the source nodes favoring nodes
                    emerging from nodes with few connections.
                'mixed': transform edges with a quantitative score based on reaction_attrs;
                    and set edges without quantitative score as a source-specific weight.
            reaction_edge_multiplier (float, optional): Factor to multiply weights of edges
                connected to reaction vertices. Default 0.5 reduces reaction edge weights
                by 50% to normalize path lengths. Set to 1.0 to disable this feature.

        Raises:
            ValueError: If weighting_strategy is not valid.

        Notes:
            The reaction_edge_multiplier addresses the issue where SBML-derived networks
            have paths like S → R → P (length 2) compared to direct protein interactions
            S → P (length 1). A multiplier of 0.5 makes these path costs equivalent.
        """

        is_weights_provided = not ((weight_by is None) or (weight_by == []))

        # Apply base weighting strategy first
        if weighting_strategy not in VALID_WEIGHTING_STRATEGIES:
            raise ValueError(
                f"weighting_strategy was {weighting_strategy} and must be one of: "
                f"{', '.join(VALID_WEIGHTING_STRATEGIES)}"
            )

        if weighting_strategy == NAPISTU_WEIGHTING_STRATEGIES.TOPOLOGY:
            if is_weights_provided:
                logger.warning(
                    "weight_by is not used for topology weighting. "
                    "It will be ignored."
                )

            self.add_topology_weights()

            # count parents and children and create weights based on them
            self.es[NAPISTU_GRAPH_EDGES.WEIGHT] = self.es["topo_weights"]
            if self.is_directed():
                self.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] = self.es[
                    "upstream_topo_weights"
                ]

        elif weighting_strategy == NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED:

            if is_weights_provided:
                logger.warning(
                    "weight_by is not used for unweighted weighting. "
                    "It will be ignored."
                )

            # set weights as a constant
            self.es[NAPISTU_GRAPH_EDGES.WEIGHT] = 1
            if self.is_directed():
                self.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] = 1

        elif weighting_strategy == NAPISTU_WEIGHTING_STRATEGIES.MIXED:
            self._add_graph_weights_mixed(weight_by)

        else:
            raise NotImplementedError(
                f"No logic implemented for {weighting_strategy}. This error should not happen."
            )

        # Apply reaction edge multiplier if not 1.0
        if reaction_edge_multiplier != 1.0:
            self._apply_reaction_edge_multiplier(reaction_edge_multiplier)

        # Update metadata to track weighting configuration
        self.set_metadata(weighting_strategy=weighting_strategy, weight_by=weight_by)

        return None

    def to_pandas_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert this NapistuGraph to Pandas DataFrames for vertices and edges.

        Returns
        -------
        vertices : pandas.DataFrame
            A table with one row per vertex.
        edges : pandas.DataFrame
            A table with one row per edge.
        """
        return ig_utils.graph_to_pandas_dfs(self)

    def to_pickle(self, path: str) -> None:
        """
        Save the NapistuGraph to a pickle file.

        Parameters
        ----------
        path : str
            Path where to save the pickle file
        """
        utils.save_pickle(path, self)

    def transform_edges(
        self,
        keep_raw_attributes: bool = False,
        custom_transformations: Optional[dict] = None,
    ) -> None:
        """
        Apply transformations to edge attributes based on stored reaction_attrs.

        Parameters
        ----------
        keep_raw_attributes : bool
            If True, store untransformed attributes for future re-transformation
        custom_transformations : dict, optional
            Dictionary mapping transformation names to functions
        """
        self._transform_entity_attributes(
            entity_type=SBML_DFS.REACTIONS,
            target_entity=NAPISTU_GRAPH.EDGES,
            keep_raw_attributes=keep_raw_attributes,
            custom_transformations=custom_transformations,
        )

    def transform_vertices(
        self,
        keep_raw_attributes: bool = False,
        custom_transformations: Optional[dict] = None,
    ) -> None:
        """
        Apply transformations to vertex attributes based on stored species_attrs.

        Parameters
        ----------
        keep_raw_attributes : bool
            If True, store untransformed attributes for future re-transformation
        custom_transformations : dict, optional
            Dictionary mapping transformation names to functions
        """
        self._transform_entity_attributes(
            entity_type=SBML_DFS.SPECIES,
            target_entity=NAPISTU_GRAPH.VERTICES,
            keep_raw_attributes=keep_raw_attributes,
            custom_transformations=custom_transformations,
        )

    def validate(self) -> None:
        """
        Validate the NapistuGraph structure and attributes.

        This method performs various validation checks to ensure the graph
        is properly structured and has required attributes.

        Raises
        ------
        ValueError
            If validation fails with specific details about the issue
        """
        # Check if species_type is defined for all vertices
        if NAPISTU_GRAPH_VERTICES.SPECIES_TYPE in self.vs.attributes():
            species_types = self.vs[NAPISTU_GRAPH_VERTICES.SPECIES_TYPE]
            missing_species_types = [
                i for i, st in enumerate(species_types) if st is None or st == ""
            ]

            if missing_species_types:
                vertex_names = [
                    self.vs[i][NAPISTU_GRAPH_VERTICES.NAME]
                    for i in missing_species_types
                ]
                raise ValueError(
                    f"Found {len(missing_species_types)} vertices with missing species_type: {vertex_names[:10]}"
                    + (
                        f" and {len(missing_species_types) - 10} more..."
                        if len(missing_species_types) > 10
                        else ""
                    )
                )
        else:
            raise ValueError("species_type attribute is missing from all vertices")

    ### private methods

    def __str__(self) -> str:
        """String representation including metadata."""
        base_str = super().__str__()
        metadata_str = (
            f"Reversed: {self.is_reversed}, "
            f"Type: {self.wiring_approach}, "
            f"Weighting: {self.weighting_strategy}"
        )
        return f"{base_str}\nNapistuGraph metadata: {metadata_str}"

    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()

    def _add_graph_weights_mixed(self, weight_by: Optional[list[str]] = None) -> None:
        """
        Weight this NapistuGraph using a mixed approach combining source-specific weights and existing edge weights.

        Modifies the graph in-place.
        """
        # Get the variables to weight by
        reaction_attrs = self._get_weight_variables(weight_by)
        edges_df = self.get_edge_dataframe()

        # Use the already-transformed edge data (transformations should have been applied in transform_edges)
        edges_df = self._create_source_weights(edges_df, NAPISTU_GRAPH_EDGES.SOURCE_WT)

        score_vars = list(reaction_attrs.keys())
        score_vars.append(NAPISTU_GRAPH_EDGES.SOURCE_WT)

        logger.info(f"Creating mixed scores based on {', '.join(score_vars)}")

        edges_df[NAPISTU_GRAPH_EDGES.WEIGHT] = edges_df[score_vars].min(axis=1)

        self.es[NAPISTU_GRAPH_EDGES.WEIGHT] = edges_df[NAPISTU_GRAPH_EDGES.WEIGHT]
        if self.is_directed():
            self.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] = edges_df[
                NAPISTU_GRAPH_EDGES.WEIGHT
            ]

        # add other attributes and update transformed attributes
        self.es[NAPISTU_GRAPH_EDGES.SOURCE_WT] = edges_df[NAPISTU_GRAPH_EDGES.SOURCE_WT]
        for k in reaction_attrs.keys():
            self.es[k] = edges_df[k]

        return None

    def _apply_reaction_edge_multiplier(self, multiplier: float = 0.5) -> None:
        """
        Apply multiplier to edges connected to reaction vertices.

        This method modifies edge weights to account for path length differences
        between reaction-mediated connections (S → R → P) and direct connections (S → P).

        Parameters:
            multiplier (float): Factor to multiply edge weights by. Values < 1.0
                decrease weights, values > 1.0 increase weights.

        Notes:
            - Modifies both 'weight' and 'upstream_weight' attributes if they exist
            - Only affects edges that connect to/from reaction vertices
            - Preserves relative weight differences within modified edges
        """
        # Get reaction vertex indices and edges connected to them in one step
        reaction_vertices = {
            v.index
            for v in self.vs
            if v.attributes().get(NAPISTU_GRAPH_VERTICES.NODE_TYPE)
            == NAPISTU_GRAPH_NODE_TYPES.REACTION
        }
        edges_to_modify = [
            e.index
            for e in self.es
            if e.source in reaction_vertices or e.target in reaction_vertices
        ]

        if not edges_to_modify:
            # No reaction vertices found, nothing to modify
            return

        for edge_idx in edges_to_modify:
            edge = self.es[edge_idx]

            # Modify 'weight' attribute if it exists
            if NAPISTU_GRAPH_EDGES.WEIGHT in edge.attributes():
                current_weight = edge[NAPISTU_GRAPH_EDGES.WEIGHT]
                edge[NAPISTU_GRAPH_EDGES.WEIGHT] = current_weight * multiplier

            # Modify 'upstream_weight' attribute if it exists (for directed graphs)
            if NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM in edge.attributes():
                current_upstream = edge[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM]
                edge[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] = (
                    current_upstream * multiplier
                )

    def _cleanup_removed_attributes_metadata(
        self, attribute_type: str, removed_attributes: set[str]
    ) -> None:
        """
        Clean up metadata for removed attributes.

        This method removes entries from the metadata that are no longer relevant
        after attributes have been removed from the graph.

        Parameters
        ----------
        attribute_type : str
            Type of attributes that were removed (NAPISTU_GRAPH.VERTICES or NAPISTU_GRAPH.EDGES)
        removed_attributes : set[str]
            Set of attribute names that were removed
        """
        # Map attribute_type to entity_type for metadata
        if attribute_type == NAPISTU_GRAPH.VERTICES:
            entity_type = SBML_DFS.SPECIES
        else:  # edges
            entity_type = SBML_DFS.REACTIONS

        # Clean up raw attributes metadata
        if NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES in self._metadata:
            raw_attrs = self._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES]
            if entity_type in raw_attrs:
                for attr_name in removed_attributes:
                    raw_attrs[entity_type].pop(attr_name, None)
                # Remove empty entity type if no raw attributes left
                if not raw_attrs[entity_type]:
                    del raw_attrs[entity_type]

        # Clean up transformations applied metadata
        if NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED in self._metadata:
            transformations = self._metadata[
                NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED
            ]
            if entity_type in transformations:
                for attr_name in removed_attributes:
                    transformations[entity_type].pop(attr_name, None)
                # Remove empty entity type if no transformations left
                if not transformations[entity_type]:
                    del transformations[entity_type]

    def _compare_and_merge_attrs(
        self,
        new_attrs: dict,
        attr_type: str,
        mode: str = ADDING_ENTITY_DATA_DEFS.FRESH,
        overwrite: bool = False,
    ) -> dict:
        """
        Compare and merge new attributes with existing ones.

        Parameters
        ----------
        new_attrs : dict
            New attributes to add/merge
        attr_type : str
            Type of attributes ("species_attrs" or "reaction_attrs")
        mode : str
            Either "fresh" (replace) or "extend" (add new keys)
        overwrite : bool
            Whether to allow overwriting existing data (ignored if mode is "extend")

        Returns
        -------
        dict
            Merged attributes dictionary
        """
        existing_attrs = self.get_metadata(attr_type) or {}

        if mode == ADDING_ENTITY_DATA_DEFS.FRESH:
            if existing_attrs and not overwrite:
                raise ValueError(
                    f"Existing {attr_type} found. Use overwrite=True to replace or mode='extend' to add new keys. "
                    f"Existing keys: {list(existing_attrs.keys())}"
                )
            return new_attrs.copy()

        elif mode == ADDING_ENTITY_DATA_DEFS.EXTEND:
            # Merge dictionaries
            merged_attrs = existing_attrs.copy()
            merged_attrs.update(new_attrs)
            return merged_attrs

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be one of: {VALID_ADDING_ENTITY_DATA_DEFS}"
            )

    def _create_source_weights(
        self,
        edges_df: pd.DataFrame,
        source_wt_var: str = NAPISTU_GRAPH_EDGES.SOURCE_WT,
        source_vars_dict: dict = SOURCE_VARS_DICT,
        source_wt_default: float = 1,
    ) -> pd.DataFrame:
        """
        Create weights based on an edge's source.

        Parameters
        ----------
        edges_df : pd.DataFrame
            The edges dataframe to add the source weights to.
        source_wt_var : str, optional
            The name of the column to store the source weights. Default is "source_wt".
        source_vars_dict : dict, optional
            Dictionary with keys indicating edge attributes and values indicating the weight to assign to that attribute. Default is SOURCE_VARS_DICT.
        source_wt_default : float, optional
            The default weight to assign to an edge if no other weight attribute is found. Default is 0.5.

        Returns
        -------
        pd.DataFrame
            The edges dataframe with the source weights added.
        """
        # Check if any source variables are present in the dataframe
        included_weight_vars = set(source_vars_dict.keys()).intersection(
            set(edges_df.columns)
        )
        if len(included_weight_vars) == 0:
            logger.warning(
                f"No edge attributes were found which match those in source_vars_dict: {', '.join(source_vars_dict.keys())}"
            )
            edges_df[source_wt_var] = source_wt_default
            return edges_df

        # Create source weights based on available variables
        edges_df_source_wts = edges_df[list(included_weight_vars)].copy()
        for wt in list(included_weight_vars):
            edges_df_source_wts[wt] = [
                source_wt_default if x is True else source_vars_dict[wt]
                for x in edges_df[wt].isna()
            ]

        source_wt_edges_df = edges_df.join(
            edges_df_source_wts.max(axis=1).rename(source_wt_var)
        )

        return source_wt_edges_df

    def _get_entity_attrs(self, entity_type: str) -> Optional[dict]:
        """
        Get entity attributes (species or reactions) from graph metadata.

        Parameters
        ----------
        entity_type : str
            Either "species" or "reactions"

        Returns
        -------
        dict or None
            Valid entity_attrs dictionary, or None if none available
        """

        if entity_type not in ENTITIES_TO_ATTRS.keys():
            raise ValueError(
                f"Unknown entity_type: '{entity_type}'. Must be one of: {list(ENTITIES_TO_ATTRS.keys())}"
            )

        attr_key = ENTITIES_TO_ATTRS[entity_type]
        entity_attrs = self.get_metadata(attr_key)

        if entity_attrs is None:  # Key doesn't exist
            logger.warning(f"No {entity_type}_attrs found in graph metadata")
            return None
        elif not entity_attrs:  # Empty dict
            logger.warning(f"{entity_type}_attrs is empty")
            return None

        # Validate structure but skip transformation validation during setup
        # Transformations will be validated when actually used in transform methods
        ng_utils._validate_entity_attrs(entity_attrs, validate_transformations=False)
        return entity_attrs

    def _get_weight_variables(self, weight_by: Optional[list[str]] = None) -> dict:
        """
        Get the variables to weight by, either from weight_by or reaction_attrs.

        Parameters
        ----------
        weight_by : list[str], optional
            A list of edge attributes to weight by. If None, uses reaction_attrs from metadata.

        Returns
        -------
        dict
            Dictionary of reaction attributes to use for weighting.

        Raises
        ------
        ValueError
            If no weights are available or if specified weights do not exist as edge attributes.
        """
        if weight_by is None:
            # Use reaction attributes from stored metadata
            reaction_attrs = self._get_entity_attrs(SBML_DFS.REACTIONS)
            if reaction_attrs is None or not reaction_attrs:
                raise ValueError(
                    "No reaction_attrs found. Use set_graph_attrs() to configure reaction attributes "
                    "or add_reaction_data() to add reaction attributes."
                )
            return reaction_attrs
        else:
            # Use specified weight_by attributes
            logger.info(f"Using weight_by attributes: {weight_by}")

            # Ensure all attributes are present in the graph
            existing_edge_attrs = set(self.es.attributes())
            missing_attrs = set(weight_by) - existing_edge_attrs

            if missing_attrs:
                raise ValueError(
                    f"Edge attributes not found in graph: {missing_attrs}. "
                    "Please weight by an existing attribute with `weight_by` or use "
                    "`add_reaction_data()` to configure reaction attributes."
                )

            # Create a simple reaction_attrs dict from the weight_by attributes
            # This maintains compatibility with the existing weighting logic
            return {
                attr: {
                    WEIGHTING_SPEC.TABLE: "__edges__",
                    WEIGHTING_SPEC.VARIABLE: attr,
                    WEIGHTING_SPEC.TRANSFORMATION: WEIGHT_TRANSFORMATIONS.IDENTITY,
                }
                for attr in weight_by
            }

    def _add_entity_data(
        self,
        entity_type: str,
        target_entity: str,
        sbml_dfs: Optional[SBML_dfs] = None,
        side_loaded_attributes: Optional[dict[str, pd.DataFrame]] = None,
        mode: str = ADDING_ENTITY_DATA_DEFS.FRESH,
        overwrite: bool = False,
        inplace: bool = True,
    ) -> Optional["NapistuGraph"]:
        """
        Extract and add entity attributes to the graph edges or vertices.

        This is a shared utility method used by add_edge_data and add_vertex_data.

        Parameters
        ----------
        entity_type : str
            Either "reactions" or "species"
        target_entity : str
            Either "edges" or "vertices" - determines where to add the attributes
        sbml_dfs : SBML_dfs
            The SBML_dfs object containing entity data. If None, only side-loaded attributes will be used.
        side_loaded_attributes : dict[str, pd.DataFrame]
            A dictionary of side-loaded attributes. If None, only sbml_dfs will be used.
        mode : str
            Either "fresh" (replace existing) or "extend" (add new attributes only)
        overwrite : bool
            Whether to allow overwriting existing attributes when conflicts arise. Ignored if mode is "extend".
        inplace : bool, default=True
            Whether to modify the graph in place. If False, returns a copy with entity data.

        Returns
        -------
        Optional[NapistuGraph]
            If inplace=True, returns None.
            If inplace=False, returns a new NapistuGraph with entity data added.
        """

        # If not inplace, make a copy
        if not inplace:
            graph = self.copy()
        else:
            graph = self

        # Use utility function to prepare entity data extraction
        entity_attrs_to_extract = ng_utils.prepare_entity_data_extraction(
            graph, entity_type, target_entity, mode, overwrite
        )
        if entity_attrs_to_extract is None:
            return None if inplace else graph

        # separate the attributes into those which will be extracted from the sbml_dfs
        # and those which will be extracted from the side_loaded_attributes
        sbml_dfs_attrs, side_loaded_attrs = ng_utils.separate_entity_attrs_by_source(
            entity_attrs_to_extract, entity_type, sbml_dfs, side_loaded_attributes
        )

        logger.debug(f"{len(sbml_dfs_attrs)} sbml_dfs attributes to add")
        logger.debug(
            f"{len(side_loaded_attrs)} side_loaded_attributes attributes to add"
        )

        if len(sbml_dfs_attrs) > 0:

            logger.debug(f"sbml_dfs_attrs: {sbml_dfs_attrs}")
            # Get entity data from sbml_dfs
            sbml_dfs_entity_data = ng_utils.pluck_entity_data(
                sbml_dfs, sbml_dfs_attrs, entity_type, transform=False
            )

            if sbml_dfs_entity_data is not None:
                # add an index to help with merging
                # pk = SBML_DFS_SCHEMA.SCHEMA[entity_type][SCHEMA_DEFS.PK]
                # sbml_dfs_entity_data = sbml_dfs_entity_data.set_index(pk)

                self._add_attributes_df(sbml_dfs_entity_data, target_entity, overwrite)
            else:
                logger.warning(
                    f"No {entity_type} data could be extracted from sbml_dfs using the stored {entity_type}_attrs"
                )

        if len(side_loaded_attrs) > 0:
            # Get entity data from side_loaded_attributes
            side_loaded_entity_data = ng_utils.pluck_data(
                side_loaded_attributes, side_loaded_attrs
            )

            if side_loaded_entity_data is not None:
                self._add_attributes_df(
                    side_loaded_entity_data, target_entity, overwrite
                )
            else:
                logger.warning(
                    f"No {entity_type} data could be extracted from the side_loaded_attributes using the stored {entity_type}_attrs"
                )

        return None if inplace else graph

    def _add_attributes_df(
        self,
        entity_data: pd.DataFrame,
        target_entity: str,
        overwrite: bool = False,
    ) -> None:
        """
        Add attributes to a graph in-place by merging entity data with graph data.

        This private method performs the core operation of merging entity data
        with graph data and assigning the resulting attributes directly to the graph.
        It's extracted from the _add_entity_data method for reusability.

        Parameters
        ----------
        entity_data : pd.DataFrame
            DataFrame containing the entity data to add, indexed by entity IDs.
            The merge will occur on any combination of columns present in the
            vertices/edges that match the index names of entity_data.
            For single index: uses the index name (e.g., "s_id" for species).
            For multi-index: uses the index names (e.g., ["from", "to"] for edges).
        target_entity : str
            Either "edges" or "vertices" - determines where to add the attributes
        overwrite : bool, default=False
            Whether to allow overwriting existing attributes when conflicts arise

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If target_entity is not "edges" or "vertices"
        """
        if target_entity not in [NAPISTU_GRAPH.EDGES, NAPISTU_GRAPH.VERTICES]:
            raise ValueError(
                f"target_entity must be '{NAPISTU_GRAPH.EDGES}' or '{NAPISTU_GRAPH.VERTICES}'"
            )

        # Check that entity_data is a DataFrame
        if not isinstance(entity_data, pd.DataFrame):
            raise TypeError(
                f"Expected entity_data to be a pandas DataFrame, but got {type(entity_data)}. "
                f"entity_data value: {entity_data}"
            )

        # Get attributes to add from the entity_data columns
        attrs_to_add = list(entity_data.columns)

        # Get merge key(s) from entity_data index names
        if isinstance(entity_data.index, pd.MultiIndex):
            merge_keys = list(entity_data.index.names)
        else:
            merge_keys = [entity_data.index.name]

        # Check for missing index names (None values)
        missing_index_names = [i for i, key in enumerate(merge_keys) if key is None]
        if missing_index_names:
            if isinstance(entity_data.index, pd.MultiIndex):
                level_info = f" at level(s) {missing_index_names}"
            else:
                level_info = ""
            raise ValueError(
                f"Entity data DataFrame has unnamed index{level_info}. "
                f"Expected index name(s) for {target_entity}: {merge_keys}. "
                f"Please set the index name(s) on your DataFrame before adding entity data. "
                f"For example: df.index.name = 's_id' for species data or 'r_id' for reaction data."
            )

        # Get current graph data
        if target_entity == NAPISTU_GRAPH.EDGES:
            graph_df = self.get_edge_dataframe()
        else:  # vertices
            graph_df = self.get_vertex_dataframe()

        # Check that merge keys exist in graph_df
        missing_keys = [key for key in merge_keys if key not in graph_df.columns]
        if missing_keys:
            available_attrs = list(graph_df.columns)
            raise ValueError(
                f"Merge keys {missing_keys} from entity_data index are missing from graph {target_entity} DataFrame. "
                f"The merge is defined by entity_data's index names: {merge_keys}. "
                f"Available attributes in graph {target_entity} DataFrame: {available_attrs}"
            )

        # Remove overlapping attributes from graph_df if overwrite=True to avoid _x/_y suffixes
        if overwrite:
            overlapping_in_graph = [
                attr for attr in attrs_to_add if attr in graph_df.columns
            ]
            if overlapping_in_graph:
                graph_df = graph_df.drop(columns=overlapping_in_graph)

        # Merge entity data with graph data
        graph_with_attrs = graph_df.merge(
            entity_data, left_on=merge_keys, right_index=True, how="left"
        )

        # validate that there wasn't a 1-to-many merge
        if graph_with_attrs.shape[0] != graph_df.shape[0]:
            raise ValueError(
                f"1-to-many merge occurred when adding {entity_data.shape[0]} {entity_data.index.names} to graph {target_entity}"
            )

        # Add new attributes directly to the graph
        added_count = 0
        for attr_name in attrs_to_add:
            if target_entity == NAPISTU_GRAPH.EDGES:
                self.es[attr_name] = graph_with_attrs[attr_name].values
            else:  # vertices
                self.vs[attr_name] = graph_with_attrs[attr_name].values
            added_count += 1

        # Log the results
        entity_name = SINGULAR_GRAPH_ENTITIES[target_entity]
        logger.info(
            f"Added {added_count} {entity_name} attributes to graph: {attrs_to_add}"
        )

        return None

    def _transform_entity_attributes(
        self,
        entity_type: str,
        target_entity: str,
        keep_raw_attributes: bool = False,
        custom_transformations: Optional[dict] = None,
    ) -> None:
        """
        Apply transformations to entity attributes (edges or vertices).

        This is a shared utility method used by transform_edges and transform_vertices.

        Parameters
        ----------
        entity_type : str
            Either "reactions" or "species"
        target_entity : str
            Either "edges" or "vertices" - determines where to apply transformations
        keep_raw_attributes : bool
            If True, store untransformed attributes for future re-transformation
        custom_transformations : dict, optional
            Dictionary mapping transformation names to functions
        """
        # Get entity attributes from stored metadata
        entity_attrs = self._get_entity_attrs(entity_type)
        if entity_attrs is None or not entity_attrs:
            logger.warning(
                f"No {entity_type}_attrs found. Use set_graph_attrs() to configure {entity_type} attributes."
            )
            return

        if target_entity not in [NAPISTU_GRAPH.EDGES, NAPISTU_GRAPH.VERTICES]:
            raise ValueError(
                f"Unknown target_entity: {target_entity}. Must be '{NAPISTU_GRAPH.EDGES}' or '{NAPISTU_GRAPH.VERTICES}'"
            )

        # Validate transformations now that we have custom_transformations available
        ng_utils._validate_entity_attrs(
            entity_attrs,
            validate_transformations=True,
            custom_transformations=custom_transformations,
        )

        # Initialize metadata structures
        if NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED not in self._metadata:
            self._metadata[NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED] = {
                entity_type: {}
            }
        if NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES not in self._metadata:
            self._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES] = {entity_type: {}}

        # Determine what attributes need updating using set operations
        current_transformations = self._metadata[
            NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED
        ][entity_type]
        requested_attrs = set(entity_attrs.keys())

        # Attributes that have never been transformed
        never_transformed = requested_attrs - set(current_transformations.keys())

        # Attributes that need different transformations
        needs_retransform = set()
        for attr_name in requested_attrs & set(current_transformations.keys()):
            new_trans = entity_attrs[attr_name].get(
                WEIGHTING_SPEC.TRANSFORMATION, DEFAULT_WT_TRANS
            )
            current_trans = current_transformations[attr_name]
            if current_trans != new_trans:
                needs_retransform.add(attr_name)

        # Check if we can re-transform (need raw data)
        stored_raw_attrs = set(
            self._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES][entity_type].keys()
        )
        invalid_retransform = needs_retransform - stored_raw_attrs

        if invalid_retransform and not keep_raw_attributes:
            # Get transformation details for error message
            error_details = []
            for attr_name in invalid_retransform:
                current_trans = current_transformations[attr_name]
                new_trans = entity_attrs[attr_name].get(
                    WEIGHTING_SPEC.TRANSFORMATION, DEFAULT_WT_TRANS
                )
                error_details.append(f"'{attr_name}': {current_trans} -> {new_trans}")

            raise ValueError(
                f"Cannot re-transform attributes without raw data: {error_details}. "
                f"Raw attributes were not kept for these attributes."
            )

        attrs_to_transform = never_transformed | needs_retransform

        if not attrs_to_transform:
            logger.info(f"No {target_entity} attributes need transformation")
            return

        # Get current graph data
        if target_entity == NAPISTU_GRAPH.EDGES:
            graph_df = self.get_edge_dataframe()
        elif target_entity == NAPISTU_GRAPH.VERTICES:
            graph_df = self.get_vertex_dataframe()
        else:
            # not reachable; added for clarity
            raise ValueError("Unknown category for target_entity")

        # Check that all attributes to transform exist
        missing_attrs = attrs_to_transform - set(graph_df.columns)
        if missing_attrs:
            logger.warning(
                f"{target_entity.capitalize()} attributes not found in graph: {missing_attrs}. Skipping."
            )
            attrs_to_transform = attrs_to_transform - missing_attrs

        if not attrs_to_transform:
            return

        # Store raw attributes if requested (for never-transformed attributes)
        if keep_raw_attributes:
            for attr_name in never_transformed & attrs_to_transform:
                if (
                    attr_name
                    not in self._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES][
                        entity_type
                    ]
                ):
                    self._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES][entity_type][
                        attr_name
                    ] = graph_df[attr_name].copy()

        # Prepare data for transformation - always use raw data
        transform_data = graph_df.copy()
        for attr_name in attrs_to_transform:
            if (
                attr_name
                in self._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES][entity_type]
            ):
                # Use stored raw values
                transform_data[attr_name] = self._metadata[
                    NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES
                ][entity_type][attr_name]
            # If no raw data available, we must be in the never_transformed case with current data as raw

        # Apply transformations using existing function
        attrs_to_transform_config = {
            attr: entity_attrs[attr] for attr in attrs_to_transform
        }

        transformed_data = ng_utils.apply_weight_transformations(
            transform_data, attrs_to_transform_config, custom_transformations
        )

        # Update graph attributes
        for attr_name in attrs_to_transform:
            if target_entity == NAPISTU_GRAPH.EDGES:
                self.es[attr_name] = transformed_data[attr_name].values
            elif target_entity == NAPISTU_GRAPH.VERTICES:
                self.vs[attr_name] = transformed_data[attr_name].values
            else:
                # not reachable; added for clarity
                raise ValueError("Unknown category for target_entity")

        # Update transformations_applied metadata
        for attr_name in attrs_to_transform:
            self._metadata[NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED][entity_type][
                attr_name
            ] = entity_attrs[attr_name].get(
                WEIGHTING_SPEC.TRANSFORMATION, DEFAULT_WT_TRANS
            )

        logger.info(
            f"Transformed {len(attrs_to_transform)} {target_entity} attributes: {list(attrs_to_transform)}"
        )

        return None


def _apply_edge_reversal_mapping(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply systematic attribute mapping for edge reversal.

    This function swaps paired attributes according to EDGE_REVERSAL_ATTRIBUTE_MAPPING.
    For example, 'from' becomes 'to', 'weight' becomes 'upstream_weight', etc.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Current edge attributes

    Returns
    -------
    pd.DataFrame
        Edge dataframe with swapped attributes

    Warnings
    --------
    Logs warnings when expected attribute pairs are missing
    """
    # Find which attributes have pairs in the mapping
    available_attrs = set(edges_df.columns)

    # Find pairs where both attributes exist
    valid_mapping = {}
    missing_pairs = []

    for source_attr, target_attr in EDGE_REVERSAL_ATTRIBUTE_MAPPING.items():
        if source_attr in available_attrs:
            if target_attr in available_attrs:
                valid_mapping[source_attr] = target_attr
            else:
                missing_pairs.append(f"{source_attr} -> {target_attr}")

    # Warn about attributes that can't be swapped
    if missing_pairs:
        logger.warning(
            f"The following edge attributes cannot be swapped during reversal "
            f"because their paired attribute is missing: {', '.join(missing_pairs)}"
        )

    return edges_df.rename(columns=valid_mapping)


def _handle_special_reversal_cases(
    edges_df: pd.DataFrame, ignore_direction: bool = False
) -> pd.DataFrame:
    """
    Handle special cases that need more than simple attribute swapping.

    This includes:
    - Flipping stoichiometry signs (* -1) for upstream and downstream stoichiometries
    - Mapping direction enums (forward <-> reverse)

    Parameters
    ----------
    edges_df : pd.DataFrame
        Edge dataframe after basic attribute swapping (upstream/downstream already swapped)
    ignore_direction : bool, default=False
        If True, skip the direction attribute check and mapping. Useful when the direction
        attribute doesn't exist yet (e.g., during initial network creation).

    Returns
    -------
    pd.DataFrame
        Edge dataframe with special cases handled

    Warnings
    --------
    Logs warnings when expected attributes are missing (unless ignore_direction=True)
    """
    result_df = edges_df.copy()

    # Handle stoichiometry sign flip for upstream and downstream
    # Note: upstream/downstream attributes are already swapped by _apply_edge_reversal_mapping
    # so we just need to negate them here
    if NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM in result_df.columns:
        result_df[NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM] = result_df[
            NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM
        ].apply(lambda x: -1 * x if x is not None and x != 0 else (0 if x == 0 else x))
    else:
        logger.warning(
            f"Missing expected '{NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM}' attribute during edge reversal. "
            "Stoichiometry signs will not be flipped."
        )

    if NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM in result_df.columns:
        result_df[NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM] = result_df[
            NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM
        ].apply(lambda x: -1 * x if x is not None and x != 0 else (0 if x == 0 else x))
    else:
        logger.warning(
            f"Missing expected '{NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM}' attribute during edge reversal. "
            "Stoichiometry signs will not be flipped."
        )

    # Handle direction enum mapping
    if not ignore_direction:
        if NAPISTU_GRAPH_EDGES.DIRECTION in result_df.columns:
            result_df[NAPISTU_GRAPH_EDGES.DIRECTION] = result_df[
                NAPISTU_GRAPH_EDGES.DIRECTION
            ].map(EDGE_DIRECTION_MAPPING)
        else:
            logger.warning(
                f"Missing expected '{NAPISTU_GRAPH_EDGES.DIRECTION}' attribute during edge reversal. "
                "Direction metadata will not be updated."
            )

    return result_df
