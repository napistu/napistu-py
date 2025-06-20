from __future__ import annotations

import pandas as pd
from napistu.ingestion import napistu_edgelist


def test_edgelist_remove_reciprocal_reactions():
    edgelist = pd.DataFrame({"from": ["A", "B", "C", "D"], "to": ["B", "A", "D", "C"]})

    nondegenerate_edgelist = napistu_edgelist.remove_reciprocal_interactions(edgelist)

    assert nondegenerate_edgelist.shape == (2, 2)


################################################
# __main__
################################################

if __name__ == "__main__":
    test_edgelist_remove_reciprocal_reactions()
