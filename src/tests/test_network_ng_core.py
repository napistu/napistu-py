import igraph as ig
import logging

from napistu.network.ng_core import NapistuGraph

logger = logging.getLogger(__name__)


def test_remove_isolated_vertices():
    """Test removing isolated vertices from a graph."""

    g = ig.Graph()
    g.add_vertices(5, attributes={"name": ["A", "B", "C", "D", "E"]})
    g.add_edges([(0, 1), (2, 3)])  # A-B, C-D connected; E isolated

    napstu_graph = NapistuGraph.from_igraph(g)
    napstu_graph.remove_isolated_vertices()
    assert napstu_graph.vcount() == 4  # Should have 4 vertices after removing E
    assert "E" not in [v["name"] for v in napstu_graph.vs]  # E should be gone
