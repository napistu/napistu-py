"""
Utilities for igraph operations.

Public Functions
----------------
find_weakly_connected_subgraphs(edgelist: pd.DataFrame) -> pd.DataFrame:
    Find all cliques of loosely connected components using igraph.

"""

from __future__ import annotations

import igraph as ig
import pandas as pd


def find_weakly_connected_subgraphs(edgelist: pd.DataFrame) -> pd.DataFrame:
    """Find all cliques of loosly connected components."""

    if edgelist.shape[1] != 2:
        raise ValueError("edgelist must have exactly 2 columns")
    if edgelist.columns.tolist() != ["ind", "id"]:
        raise ValueError("edgelist columns must be ['ind', 'id']")
    if not any(edgelist["ind"].str.startswith("ind")):
        raise ValueError("At least some entries in 'ind' must start with 'ind'")

    id_graph = ig.Graph.TupleList(edgelist.itertuples(index=False))

    id_graph_names = [v.attributes()["name"] for v in id_graph.vs]
    id_graphs_clusters = id_graph.connected_components().membership
    id_graph_df = pd.DataFrame({"name": id_graph_names, "cluster": id_graphs_clusters})
    # clusters based on index or identifiers will be the same when joined to id table
    ind_clusters = id_graph_df[id_graph_df.name.str.startswith("ind")].rename(
        columns={"name": "ind"}
    )

    return ind_clusters
