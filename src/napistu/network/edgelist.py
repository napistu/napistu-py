"""
Edgelist class for representing and validating graph edges.

Classes
-------
Edgelist
    A class representing an edgelist with validation and merging capabilities.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import pandas as pd
from igraph import Graph

from napistu.network.constants import IGRAPH_DEFS, NAPISTU_GRAPH, NAPISTU_GRAPH_EDGES
from napistu.utils.pd_utils import validate_merge

logger = logging.getLogger(__name__)


class Edgelist:
    """
    A class representing an edgelist with validation and merging capabilities.

    Wraps a pandas DataFrame containing edge information with standardized
    column names (source/target or from/to) plus any additional attributes.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with edge information. Must have either:
        - 'source' and 'target' columns, or
        - 'from' and 'to' columns
    source_col : str, optional
        Name of source column. If None, auto-detects from 'source' or 'from'
    target_col : str, optional
        Name of target column. If None, auto-detects from 'target' or 'to'

    Attributes
    ----------
    df : pd.DataFrame
        The underlying DataFrame containing edge data
    source_col : str
        Name of the source column
    target_col : str
        Name of the target column

    Properties
    ----------------
    standard_merge_by : list[str]
        List of supported merge_by values based on column conventions.

    Public Methods
    --------------
    ensure(data: Union[Edgelist, pd.DataFrame]) -> Edgelist:
        Ensure the input is an Edgelist.
    merge_edgelists(other: Union[Edgelist, pd.DataFrame], how: str = "inner", suffixes: tuple[str, str] = ("_x", "_y"), relationship: Optional[str] = None) -> Edgelist:
        Merge this edgelist with another edgelist.
    to_dataframe() -> pd.DataFrame:
        Return the underlying DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from napistu.network.edgelist import Edgelist
    >>> df = pd.DataFrame({
    ...     'source': ['A', 'B'],
    ...     'target': ['B', 'C'],
    ...     'weight': [1.0, 2.0]
    ... })
    >>> el = Edgelist(df)
    >>> el.validate_subset(graph)  # Validate against a graph
    >>> merged = el.merge_edgelists(other_edgelist)  # Merge with another edgelist
    """

    def __init__(
        self,
        data: pd.DataFrame,
        source_col: Optional[str] = None,
        target_col: Optional[str] = None,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, got {type(data)}")

        # Auto-detect source/target columns if not provided
        if source_col is None or target_col is None:
            if (
                NAPISTU_GRAPH_EDGES.FROM in data.columns
                and NAPISTU_GRAPH_EDGES.TO in data.columns
            ):
                source_col = NAPISTU_GRAPH_EDGES.FROM
                target_col = NAPISTU_GRAPH_EDGES.TO

            elif (
                IGRAPH_DEFS.SOURCE in data.columns
                and IGRAPH_DEFS.TARGET in data.columns
            ):
                source_col = IGRAPH_DEFS.SOURCE
                target_col = IGRAPH_DEFS.TARGET

                for col, col_name in [(source_col, "Source"), (target_col, "Target")]:
                    if not pd.api.types.is_integer_dtype(data[col]):
                        raise ValueError(
                            f"{col_name} column '{col}' must contain integers (vertex indices) "
                            f"when using source/target convention, but got dtype: {data[col].dtype}"
                        )

            else:
                raise ValueError(
                    f"DataFrame must have either ('{IGRAPH_DEFS.SOURCE}', '{IGRAPH_DEFS.TARGET}') "
                    f"or ('{NAPISTU_GRAPH_EDGES.FROM}', '{NAPISTU_GRAPH_EDGES.TO}') columns. "
                    f"Found columns: {list(data.columns)}"
                )

            logger.info(
                f"Auto-detected source/target columns: {source_col}/{target_col}"
            )

        # Validate columns exist
        if source_col not in data.columns:
            raise ValueError(f"Source column '{source_col}' not found in DataFrame")
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        self.df = data.copy()
        self.source_col = source_col
        self.target_col = target_col

    @property
    def standard_merge_by(self) -> str:
        """
        Suggest a default merge_by value based on column conventions.

        Returns
        -------
        str
            name for 'source'/'target' style columns,
            index for 'from'/'to' style columns.
            custom merge attribute based on source_col and target_col
        """
        if self.source_col in {
            IGRAPH_DEFS.SOURCE,
            IGRAPH_DEFS.TARGET,
        } or self.target_col in {
            IGRAPH_DEFS.SOURCE,
            IGRAPH_DEFS.TARGET,
        }:
            return IGRAPH_DEFS.INDEX

        if self.source_col in {
            NAPISTU_GRAPH_EDGES.FROM,
            NAPISTU_GRAPH_EDGES.TO,
        } or self.target_col in {
            NAPISTU_GRAPH_EDGES.FROM,
            NAPISTU_GRAPH_EDGES.TO,
        }:
            return IGRAPH_DEFS.NAME

        # custom merge attribute based on source_col and target_col
        return f"{self.source_col}/{self.target_col}"

    @property
    def has_duplicated_edges(self) -> bool:
        """
        Check if the edgelist contains duplicate edges.

        Duplicate edges are multiple rows with the same (source, target) pair,
        regardless of other attributes.

        Returns
        -------
        bool
            True if duplicate edges exist, False otherwise.

        Examples
        --------
        >>> import pandas as pd
        >>> el = Edgelist(pd.DataFrame({
        ...     'source': ['A', 'B', 'A'],
        ...     'target': ['B', 'C', 'B'],
        ...     'weight': [1.0, 2.0, 1.5]
        ... }))
        >>> el.has_duplicated_edges
        True  # A->B appears twice
        """
        # Check for duplicates based only on source and target columns
        duplicates = self.df.duplicated(
            subset=[self.source_col, self.target_col], keep=False
        )
        return duplicates.any()

    @property
    def has_reciprocal_edges(self) -> bool:
        """
        Check if the edgelist contains reciprocal edges.

        A reciprocal edge is a pair (A, B) where both (A, B) and (B, A) exist
        in the edgelist.

        Returns
        -------
        bool
            True if reciprocal edges exist, False otherwise.

        Examples
        --------
        >>> import pandas as pd
        >>> el = Edgelist(pd.DataFrame({
        ...     'source': ['A', 'B', 'C'],
        ...     'target': ['B', 'A', 'D']
        ... }))
        >>> el.has_reciprocal_edges
        True  # A->B and B->A both exist
        """
        # Filter out self-loops upfront (A-A edges don't count as reciprocal)
        non_self_loops = self.df[self.df[self.source_col] != self.df[self.target_col]]

        if len(non_self_loops) == 0:
            return False

        # Create a set of (source, target) tuples
        edges = set(
            zip(non_self_loops[self.source_col], non_self_loops[self.target_col])
        )

        # Check if any (target, source) pair exists in the edges set
        for source, target in edges:
            if (target, source) in edges:
                return True

        return False

    @classmethod
    def ensure(self, data: Union[pd.DataFrame, Edgelist]) -> Edgelist:
        """
        Ensure the input is an Edgelist.

        Parameters
        ----------
        data : pd.DataFrame or Edgelist
            Data to ensure is an Edgelist.
        """
        if isinstance(data, pd.DataFrame):
            return Edgelist(data)
        elif isinstance(data, Edgelist):
            return data
        else:
            raise TypeError(
                f"data must be a pandas DataFrame or Edgelist, got {type(data)}"
            )

    def merge_edgelists(
        self,
        other: Union[Edgelist, pd.DataFrame],
        how: str = "inner",
        suffixes: tuple[str, str] = ("_x", "_y"),
        relationship: Optional[str] = None,
    ) -> Edgelist:
        """
        Merge this edgelist with another edgelist.

        This merges on the two edge key columns (source/target or from/to).
        If `relationship` is provided, the merge keys are validated via
        `napistu.utils.pd_utils.validate_merge` before merging.

        Parameters
        ----------
        other : Edgelist or pd.DataFrame
            Other edgelist to merge with
        how : str
            Type of merge: 'inner', 'outer', 'left', 'right' (default: 'inner')
        suffixes : tuple[str, str]
            Suffixes to apply to overlapping column names
        relationship : str, optional
            Expected relationship type to validate:
            - '1:1' (one-to-one): both keys are unique
            - '1:m' (one-to-many): left keys can be matched to multiple right keys, but each right key can only be matched to one left key
            - 'm:1' (many-to-one): right keys can be matched to multiple left keys, but each left key can only be matched to one right key
            - 'm:m' (many-to-many): both keys may have duplicates
            - '1:0' (one-to-zero-or-one): left keys can be matched to zero or more right keys, but each right key can only be matched to one left key
            - '0:1' (zero-or-one-to-one): right keys can be matched to zero or more left keys, but each left key can only be matched to one right key
            If None, no validation is performed.

        Returns
        -------
        Edgelist
            Merged edgelist

        Raises
        ------
        ValueError
            If edgelists have no overlapping supported merge_by conventions
            If relationship validation fails
        """
        if isinstance(other, pd.DataFrame):
            other_el = Edgelist(other)
        elif isinstance(other, Edgelist):
            other_el = other
        else:
            raise TypeError(f"other must be Edgelist or DataFrame, got {type(other)}")

        if self.standard_merge_by != other_el.standard_merge_by:
            raise ValueError(
                f"Cannot merge edgelists with different merge_by conventions: "
                f"self uses '{self.standard_merge_by}', other uses '{other_el.standard_merge_by}'. "
                f"Both edgelists must use the same column convention (source/target or from/to)."
            )

        if relationship is not None:
            validate_merge(
                self.df,
                other_el.df,
                left_on=[self.source_col, self.target_col],
                right_on=[other_el.source_col, other_el.target_col],
                relationship=relationship,
            )

        merged = self.df.merge(
            other_el.df,
            left_on=[self.source_col, self.target_col],
            right_on=[other_el.source_col, other_el.target_col],
            how=how,
            suffixes=suffixes,
        )

        return Edgelist(merged, source_col=self.source_col, target_col=self.target_col)

    def remove_duplicated_edges(
        self, keep: str = "first", inplace: bool = False
    ) -> Optional[Edgelist]:
        """
        Remove duplicate edges from the edgelist.

        For edges with the same (source, target) pair, keeps only one based
        on the `keep` parameter. Only considers source and target columns,
        ignoring all other attributes.

        Parameters
        ----------
        keep : str, default "first"
            Which duplicate edge to keep:
            - "first": Keep the first occurrence (by DataFrame order)
            - "last": Keep the last occurrence (by DataFrame order)
        inplace : bool, default False
            If True, modify the edgelist in place. If False, return a new Edgelist.

        Returns
        -------
        Edgelist or None
            If inplace=False, returns a new Edgelist with duplicate edges removed.
            If inplace=True, returns None.

        Examples
        --------
        >>> import pandas as pd
        >>> el = Edgelist(pd.DataFrame({
        ...     'source': ['A', 'B', 'A'],
        ...     'target': ['B', 'C', 'B'],
        ...     'weight': [1.0, 2.0, 1.5]
        ... }))
        >>> el_cleaned = el.remove_duplicated_edges(keep="first")
        >>> len(el_cleaned)
        2  # One of the A->B duplicates is removed
        """
        if keep not in ["first", "last"]:
            raise ValueError(f"keep must be 'first' or 'last', got '{keep}'")

        # Use pandas drop_duplicates on source and target columns only
        df_cleaned = self.df.drop_duplicates(
            subset=[self.source_col, self.target_col], keep=keep
        ).reset_index(drop=True)

        if inplace:
            self.df = df_cleaned
            return None
        else:
            return Edgelist(
                df_cleaned, source_col=self.source_col, target_col=self.target_col
            )

    def remove_reciprocal_edges(
        self, keep: str = "first", inplace: bool = False
    ) -> Optional[Edgelist]:
        """
        Remove reciprocal edges from the edgelist.

        For pairs of edges (A, B) and (B, A), keeps only one direction based
        on the `keep` parameter.

        Parameters
        ----------
        keep : str, default "first"
            Which edge to keep when a reciprocal pair is found:
            - "first": Keep the first edge encountered (by DataFrame order)
            - "lexicographic": Keep the edge where source < target (lexicographically)
        inplace : bool, default False
            If True, modify the edgelist in place. If False, return a new Edgelist.

        Returns
        -------
        Edgelist or None
            If inplace=False, returns a new Edgelist with reciprocal edges removed.
            If inplace=True, returns None.

        Examples
        --------
        >>> import pandas as pd
        >>> el = Edgelist(pd.DataFrame({
        ...     'source': ['A', 'B', 'C'],
        ...     'target': ['B', 'A', 'D']
        ... }))
        >>> el_cleaned = el.remove_reciprocal_edges(keep="first")
        >>> len(el_cleaned)
        2  # One of A->B or B->A is removed
        """
        if keep not in ["first", "lexicographic"]:
            raise ValueError(f"keep must be 'first' or 'lexicographic', got '{keep}'")

        # Check for duplicate edges upfront
        if self.has_duplicated_edges:
            n_duplicates = self.df.duplicated(
                subset=[self.source_col, self.target_col], keep=False
            ).sum()
            logger.warning(
                f"Edgelist contains {n_duplicates} duplicate edge(s). "
                f"remove_reciprocal_edges will only keep one edge per (source, target) pair, "
                f"so duplicates will be removed along with reciprocal pairs. "
                f"Consider calling remove_duplicated_edges() first if you want to preserve duplicates."
            )

        df = self.df.copy()

        # Create canonical columns for grouping reciprocal pairs
        # _source = min(source, target), _target = max(source, target)
        df["_source"] = df[[self.source_col, self.target_col]].min(axis=1)
        df["_target"] = df[[self.source_col, self.target_col]].max(axis=1)

        # Filter out self-loops (they can't be reciprocal)
        non_self_loops = df[self.source_col] != df[self.target_col]
        self_loops = df[~non_self_loops]
        df_non_self = df[non_self_loops].copy()

        if len(df_non_self) == 0:
            # No non-self-loop edges, return original
            df_cleaned = df.drop(columns=["_source", "_target"])
        else:
            if keep == "lexicographic":
                # For lexicographic, we want source < target
                # Filter to only edges where original source < target
                df_non_self = df_non_self[
                    df_non_self[self.source_col] < df_non_self[self.target_col]
                ]
            else:  # keep == "first"
                # For "first", group by canonical pair and keep first occurrence
                df_non_self = df_non_self.groupby(
                    ["_source", "_target"], as_index=False
                ).first()

            # Combine with self-loops (always kept)
            df_cleaned = pd.concat([df_non_self, self_loops], ignore_index=True)
            df_cleaned = df_cleaned.drop(columns=["_source", "_target"]).reset_index(
                drop=True
            )

        if inplace:
            self.df = df_cleaned
            return None
        else:
            return Edgelist(
                df_cleaned, source_col=self.source_col, target_col=self.target_col
            )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the underlying DataFrame.

        Returns
        -------
        pd.DataFrame
            The edgelist DataFrame
        """
        return self.df.copy()

    def validate_subset(
        self,
        graph: Graph,
        validate: str = "both",
        edgelist_name: str = "edgelist",
        graph_name: str = "graph",
    ) -> None:
        """
        Validate that this edgelist is a subset of the graph's edges.

        The merge_by convention is automatically determined from the edgelist's
        source/target column conventions:
        - If using 'source'/'target' columns, validates against vertex indices
        - If using 'from'/'to' columns, validates against vertex names
        - For custom columns, validates against vertex names

        Parameters
        ----------
        graph : Graph
            Graph to validate against.
        validate: str
            Entities to validate: 'vertices', 'edges', or 'both'. If 'both', validates both vertices and edges.
        edgelist_name : str
            Name to use for edgelist in error messages.
        graph_name : str
            Name to use for graph in error messages.

        Raises
        ------
        ValueError
            If edgelist contains vertices or edges not in graph
            If source/target columns are not edge attributes in the graph
        """

        # Get actual edge attributes if graph has edges
        if graph.ecount() == 0:
            raise ValueError(f"Graph {graph_name} has no edges")

        # 'source' and 'target' are built-in igraph properties (vertex indices),
        # not edge attributes. 'from' and 'to' are actual edge attributes.
        # Built-in igraph properties (always available)
        available_edge_attrs = {IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET} | set(
            graph.es.attributes()
        )

        missing_cols = []
        if self.source_col not in available_edge_attrs:
            missing_cols.append(self.source_col)
        if self.target_col not in available_edge_attrs:
            missing_cols.append(self.target_col)
        if missing_cols:
            raise ValueError(
                f"Column(s) {missing_cols} not found as edge attributes in {graph_name}. "
                f"Available: {available_edge_attrs}"
            )

        # Auto-determine merge_by from edgelist's column conventions
        merge_by = self.standard_merge_by

        # Determine vertex attribute to use for validation
        # If using source/target columns, check against vertex indices
        # Otherwise (from/to or custom), check against vertex names
        if merge_by == IGRAPH_DEFS.INDEX:
            vertex_id_attr = IGRAPH_DEFS.INDEX
        elif merge_by == IGRAPH_DEFS.NAME:
            vertex_id_attr = IGRAPH_DEFS.NAME
        else:
            # Custom merge_by - use name attribute for validation
            logger.info(
                f"Since merge_by is custom, '{self.source_col}' and '{self.target_col}' will be compared to vertex '{IGRAPH_DEFS.NAME}' for validation"
            )
            vertex_id_attr = IGRAPH_DEFS.NAME

        # Validate vertex attribute exists
        # Note: 'index' is a built-in igraph property (always available), not a vertex attribute
        if (
            vertex_id_attr == IGRAPH_DEFS.NAME
            and IGRAPH_DEFS.NAME not in graph.vs.attributes()
        ):
            raise ValueError(
                "Edge validation requires the 'name' vertex attribute unless merging by source/target indices"
            )

        VALID_VALIDATE = [NAPISTU_GRAPH.VERTICES, NAPISTU_GRAPH.EDGES, "both"]
        if validate not in VALID_VALIDATE:
            raise ValueError(
                f"validate must be one of {VALID_VALIDATE}, got {validate}"
            )

        validate_vertices = False
        if validate in [NAPISTU_GRAPH.VERTICES, "both"]:
            validate_vertices = True

        validate_edges = False
        if validate in [NAPISTU_GRAPH.EDGES, "both"]:
            validate_edges = True

        if validate_vertices:
            # check for vertices which are not in the graph
            if vertex_id_attr == IGRAPH_DEFS.INDEX:
                vertex_ids = graph.vs.indices
            else:
                vertex_ids = graph.vs[vertex_id_attr]

            invalid_vertices = set(
                self.df[self.source_col].tolist() + self.df[self.target_col].tolist()
            ) - set(vertex_ids)
            if invalid_vertices:
                example_invalid_vertices = list(invalid_vertices)[
                    : min(5, len(invalid_vertices))
                ]
                raise ValueError(
                    f"Found {len(invalid_vertices)} vertex(s) in {edgelist_name} not in {graph_name}: {example_invalid_vertices}"
                )

        if validate_edges:
            # Build set of universe edges
            universe_edges = set()
            for e in graph.es:
                if vertex_id_attr == IGRAPH_DEFS.INDEX:
                    src_name = e.source
                    tgt_name = e.target
                else:
                    src_name = graph.vs[e.source][vertex_id_attr]
                    tgt_name = graph.vs[e.target][vertex_id_attr]
                universe_edges.add((src_name, tgt_name))
                if not graph.is_directed():
                    # For undirected, also add reverse
                    universe_edges.add((tgt_name, src_name))

            # Check all observed edges are in universe
            invalid_edges = []
            for _, row in self.df.iterrows():
                edge = (row[self.source_col], row[self.target_col])
                if edge not in universe_edges:
                    invalid_edges.append(edge)
                    if len(invalid_edges) >= 10:  # Limit reporting to first 10
                        break

            if invalid_edges:
                raise ValueError(
                    f"Found {len(invalid_edges)} edge(s) in {edgelist_name} not in {graph_name}. "
                    f"First few: {invalid_edges[:5]}"
                )

        return None

    def __len__(self) -> int:
        """Return the number of edges."""
        return len(self.df)

    def __repr__(self) -> str:
        """String representation of the Edgelist."""
        return f"Edgelist(n_edges={len(self)}, source_col='{self.source_col}', target_col='{self.target_col}')"

    def __getitem__(self, key):
        """Allow indexing into the underlying DataFrame."""
        return self.df[key]

    def __setitem__(self, key, value):
        """Allow setting values in the underlying DataFrame."""
        self.df[key] = value
