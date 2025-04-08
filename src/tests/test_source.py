from __future__ import annotations

import os

import pandas as pd
from napistu import indices
from napistu import source

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")


def test_source():
    source_example_df = pd.DataFrame(
        [
            {"model": "fun", "id": "baz", "pathway_id": "fun"},
            {"model": "fun", "id": "bot", "pathway_id": "fun"},
            {"model": "time", "id": "boof", "pathway_id": "time"},
            {"model": "time", "id": "bor", "pathway_id": "time"},
        ]
    )

    source_obj = source.Source(source_example_df)
    source_init = source.Source(init=True)

    assert source.merge_sources([source_init, source_init]) == source_init

    pd._testing.assert_frame_equal(
        source.merge_sources([source_obj, source_init]).source, source_example_df
    )

    assert source.merge_sources([source_obj, source_obj]).source.shape[0] == 8

    alt_source_df = pd.DataFrame(
        [
            {"model": "fun", "identifier": "baz", "pathway_id": "fun"},
            {"model": "fun", "identifier": "baz", "pathway_id": "fun"},
        ]
    )
    alt_source_obj = source.Source(alt_source_df)

    assert source.merge_sources([source_obj, alt_source_obj]).source.shape == (6, 4)


def test_source_w_pwindex():
    # pathway_id not provided since this and other attributes will be found
    # in pw_index.tsv
    source_example_df = pd.DataFrame(
        [
            {"model": "R-HSA-1237044", "id": "baz"},
            {"model": "R-HSA-1237044", "id": "bot"},
        ]
    )

    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index.tsv"))

    source_obj = source.Source(source_example_df, pw_index=pw_index)
    assert source_obj.source.shape == (2, 8)


################################################
# __main__
################################################

if __name__ == "__main__":
    test_source()
    test_source_w_pwindex()
