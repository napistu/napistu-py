from __future__ import annotations

import copy
import logging
from typing import Any, Optional, Union

import igraph as ig
import pandas as pd

from napistu.sbml_dfs_core import SBML_dfs
from napistu.network import ng_utils
from napistu.constants import SBML_DFS
from napistu.network.constants import (
    EDGE_REVERSAL_ATTRIBUTE_MAPPING,
    EDGE_DIRECTION_MAPPING,
    ENTITIES_TO_ATTRS,
    NAPISTU_GRAPH_EDGES,
)

logger = logging.getLogger(__name__)


class NapistuGraph(ig.Graph):
    """
    A subclass of igraph.Graph with additional functionality for molecular network analysis.

    This class extends igraph.Graph with domain-specific methods and metadata tracking
    for biological pathway and molecular interaction networks. All standard igraph
    methods are available, plus additional functionality for edge reversal and
    metadata management.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to igraph.Graph constructor
    **kwargs : dict
        Keyword arguments passed to igraph.Graph constructor

    Attributes
    ----------
    is_reversed : bool
        Whether the graph edges have been reversed from their original direction
    wiring_approach : str or None
        Type of graph (e.g., 'bipartite', 'regulatory', 'surrogate')
    weighting_strategy : str or None
        Strategy used for edge weighting (e.g., 'topology', 'mixed', 'calibrated')

    Methods
    -------
    from_igraph(graph, **metadata)
        Create a NapistuGraph from an existing igraph.Graph
    reverse_edges()
        Reverse all edges in the graph in-place
    set_metadata(**kwargs)
        Set metadata for the graph in-place
    get_metadata(key=None)
        Get metadata from the graph
    copy()
        Create a deep copy of the NapistuGraph

    Examples
    --------
    Create a NapistuGraph from scratch:

    >>> ng = NapistuGraph(directed=True)
    >>> ng.add_vertices(3)
    >>> ng.add_edges([(0, 1), (1, 2)])

    Convert from existing igraph:

    >>> import igraph as ig
    >>> g = ig.Graph.Erdos_Renyi(10, 0.3)
    >>> ng = NapistuGraph.from_igraph(g, graph_type='random')

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
            "is_reversed": False,
            "wiring_approach": None,
            "weighting_strategy": None,
            "creation_params": {},
            "species_attrs": {},
            "reaction_attrs": {},
        }

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

    @property
    def is_reversed(self) -> bool:
        """Check if the graph has been reversed."""
        return self._metadata["is_reversed"]

    @property
    def wiring_approach(self) -> Optional[str]:
        """Get the graph type (bipartite, regulatory, etc.)."""
        return self._metadata["wiring_approach"]

    @property
    def weighting_strategy(self) -> Optional[str]:
        """Get the weighting strategy used."""
        return self._metadata["weighting_strategy"]

    def add_edge_data(
        self, sbml_dfs: SBML_dfs, mode: str = "fresh", overwrite: bool = False
    ) -> None:
        """
        Extract and add reaction attributes to the graph edges.

        Parameters
        ----------
        sbml_dfs : SBML_dfs
            The SBML_dfs object containing reaction data
        mode : str
            Either "fresh" (replace existing) or "extend" (add new attributes only)
        overwrite : bool
            Whether to allow overwriting existing edge attributes when conflicts arise
        """

        # Get reaction_attrs from stored metadata
        reaction_attrs = self._get_entity_attrs("reactions")
        if reaction_attrs is None or not reaction_attrs:
            logger.warning(
                "No reaction_attrs found. Use set_graph_attrs() to configure reaction attributes before extracting edge data."
            )
            return

        # Check for conflicts with existing edge attributes
        existing_edge_attrs = set(self.es.attributes())
        new_attrs = set(reaction_attrs.keys())

        if mode == "fresh":
            overlapping_attrs = existing_edge_attrs & new_attrs
            if overlapping_attrs and not overwrite:
                raise ValueError(
                    f"Edge attributes already exist: {overlapping_attrs}. "
                    f"Use overwrite=True to replace or mode='extend' to add only new attributes"
                )
            attrs_to_add = new_attrs

        elif mode == "extend":
            overlapping_attrs = existing_edge_attrs & new_attrs
            if overlapping_attrs and not overwrite:
                raise ValueError(
                    f"Overlapping edge attributes found: {overlapping_attrs}. "
                    f"Use overwrite=True to allow replacement"
                )
            # In extend mode, only add attributes that don't exist (unless overwrite=True)
            if overwrite:
                attrs_to_add = new_attrs
            else:
                attrs_to_add = new_attrs - existing_edge_attrs

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'fresh' or 'extend'")

        if not attrs_to_add:
            logger.info("No new attributes to add")
            return

        # Only extract the attributes we're actually going to add
        attrs_to_extract = {attr: reaction_attrs[attr] for attr in attrs_to_add}

        # Get reaction data using existing function - only for attributes we need
        reaction_data = ng_utils.pluck_entity_data(
            sbml_dfs, attrs_to_extract, SBML_DFS.REACTIONS
        )

        if reaction_data is None:
            logger.warning(
                "No reaction data could be extracted with the stored reaction_attrs"
            )
            return

        # Get current edges and merge with reaction data
        edges_df = self.get_edge_dataframe()

        # Remove overlapping attributes from edges_df if overwrite=True to avoid _x/_y suffixes
        if overwrite:
            overlapping_in_edges = [
                attr for attr in attrs_to_add if attr in edges_df.columns
            ]
            if overlapping_in_edges:
                edges_df = edges_df.drop(columns=overlapping_in_edges)

        edges_with_attrs = edges_df.merge(
            reaction_data, left_on=SBML_DFS.R_ID, right_index=True, how="left"
        )

        # Add new attributes directly to the graph
        added_count = 0
        for attr_name in attrs_to_add:
            if attr_name in reaction_data.columns:
                self.es[attr_name] = edges_with_attrs[attr_name].values
                added_count += 1

        logger.info(
            f"Added {added_count} edge attributes to graph: {list(attrs_to_add)}"
        )

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

    def set_graph_attrs(
        self,
        graph_attrs: Union[str, dict],
        mode: str = "fresh",
        overwrite: bool = False,
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
        """

        # Load from YAML if string path provided
        if isinstance(graph_attrs, str):
            graph_attrs = ng_utils.read_graph_attrs_spec(graph_attrs)

        # Process species attributes if present
        if "species" in graph_attrs:
            merged_species = self._compare_and_merge_attrs(
                graph_attrs["species"], "species_attrs", mode, overwrite
            )
            self.set_metadata(species_attrs=merged_species)

        # Process reaction attributes if present
        if "reactions" in graph_attrs:
            merged_reactions = self._compare_and_merge_attrs(
                graph_attrs["reactions"], "reaction_attrs", mode, overwrite
            )
            self.set_metadata(reaction_attrs=merged_reactions)

    def remove_isolated_vertices(self):
        """
        Remove vertices that have no edges (degree 0) from the graph.


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

        # Get vertex names/indices for logging (up to 5 examples)
        vertex_names = []
        for v in isolated_vertices[:5]:
            # Use vertex name if available, otherwise use index
            name = (
                v["name"]
                if "name" in v.attributes() and v["name"] is not None
                else str(v.index)
            )
            vertex_names.append(name)

        # Create log message
        examples_str = ", ".join(f"'{name}'" for name in vertex_names)
        if len(isolated_vertices) > 5:
            examples_str += f" (and {len(isolated_vertices) - 5} more)"

        logger.info(
            f"Removed {len(isolated_vertices)} isolated vertices: [{examples_str}]"
        )

        # Remove the isolated vertices
        self.delete_vertices(isolated_vertices)

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
        vertices = pd.DataFrame(
            [{**{"index": v.index}, **v.attributes()} for v in self.vs]
        )
        edges = pd.DataFrame(
            [
                {**{"source": e.source, "target": e.target}, **e.attributes()}
                for e in self.es
            ]
        )
        return vertices, edges

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

    def _compare_and_merge_attrs(
        self,
        new_attrs: dict,
        attr_type: str,
        mode: str = "fresh",
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
            Whether to allow overwriting existing data

        Returns
        -------
        dict
            Merged attributes dictionary
        """
        existing_attrs = self.get_metadata(attr_type) or {}

        if mode == "fresh":
            if existing_attrs and not overwrite:
                raise ValueError(
                    f"Existing {attr_type} found. Use overwrite=True to replace or mode='extend' to add new keys. "
                    f"Existing keys: {list(existing_attrs.keys())}"
                )
            return new_attrs.copy()

        elif mode == "extend":
            overlapping_keys = set(existing_attrs.keys()) & set(new_attrs.keys())
            if overlapping_keys and not overwrite:
                raise ValueError(
                    f"Overlapping keys found in {attr_type}: {overlapping_keys}. "
                    f"Use overwrite=True to allow key replacement"
                )

            # Merge dictionaries
            merged_attrs = existing_attrs.copy()
            merged_attrs.update(new_attrs)
            return merged_attrs

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'fresh' or 'extend'")

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

        # Validate and let any exceptions propagate
        ng_utils._validate_entity_attrs(entity_attrs)
        return entity_attrs


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


def _handle_special_reversal_cases(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle special cases that need more than simple attribute swapping.

    This includes:
    - Flipping stoichiometry signs (* -1)
    - Mapping direction enums (forward <-> reverse)

    Parameters
    ----------
    edges_df : pd.DataFrame
        Edge dataframe after basic attribute swapping

    Returns
    -------
    pd.DataFrame
        Edge dataframe with special cases handled

    Warnings
    --------
    Logs warnings when expected attributes are missing
    """
    result_df = edges_df.copy()

    # Handle stoichiometry sign flip
    if NAPISTU_GRAPH_EDGES.STOICHIOMETRY in result_df.columns:
        result_df[NAPISTU_GRAPH_EDGES.STOICHIOMETRY] *= -1
    else:
        logger.warning(
            f"Missing expected '{NAPISTU_GRAPH_EDGES.STOICHIOMETRY}' attribute during edge reversal. "
            "Stoichiometry signs will not be flipped."
        )

    # Handle direction enum mapping
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
