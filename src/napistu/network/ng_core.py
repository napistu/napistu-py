from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import igraph as ig
import pandas as pd

from napistu.network.constants import (
    NAPISTU_GRAPH_EDGES,
    EDGE_REVERSAL_ATTRIBUTE_MAPPING,
    EDGE_DIRECTION_MAPPING,
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
