from __future__ import annotations

from napistu import sbml_dfs_utils


def test_id_formatter():
    input_vals = range(50, 100)

    # create standard IDs
    ids = sbml_dfs_utils.id_formatter(input_vals, "s_id", id_len=8)
    # invert standard IDs
    inv_ids = sbml_dfs_utils.id_formatter_inv(ids)

    assert list(input_vals) == inv_ids


################################################
# __main__
################################################

if __name__ == "__main__":
    test_id_formatter()
