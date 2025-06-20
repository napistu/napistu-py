from __future__ import annotations

import pytest

from napistu.network import ig_utils
from napistu.network import net_create


def test_validate_graph_attributes(sbml_dfs):

    napistu_graph = net_create.process_napistu_graph(
        sbml_dfs, directed=True, weighting_strategy="topology"
    )

    assert (
        ig_utils.validate_edge_attributes(
            napistu_graph, ["weights", "upstream_weights"]
        )
        is None
    )
    assert ig_utils.validate_vertex_attributes(napistu_graph, "node_type") is None
    with pytest.raises(ValueError):
        ig_utils.validate_vertex_attributes(napistu_graph, "baz")
