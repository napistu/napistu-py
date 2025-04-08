from __future__ import annotations

import os

import pandas as pd
import pytest
from napistu import indices

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")


def test_pwindex_from_file():
    pw_index_path = os.path.join(test_data, "pw_index.tsv")
    pw_index = indices.PWIndex(pw_index_path)

    assert pw_index.index.shape == (5, 6)


def test_pwindex_from_df():
    stub_pw_df = pd.DataFrame(
        {
            "file": "DNE",
            "source": "The farm",
            "species": "Gallus gallus",
            "pathway_id": "chickens",
            "name": "Chickens",
            "date": "2020-01-01",
        },
        index=[0],
    )

    assert indices.PWIndex(pw_index=stub_pw_df, validate_paths=False).index.equals(
        stub_pw_df
    )

    with pytest.raises(FileNotFoundError) as _:
        indices.PWIndex(pw_index=stub_pw_df, pw_index_base_path="missing_directory")

    with pytest.raises(FileNotFoundError) as _:
        indices.PWIndex(pw_index=stub_pw_df, pw_index_base_path=test_data)


@pytest.fixture
def pw_testindex():
    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index.tsv"))
    return pw_index


def test_index(pw_testindex):
    pw_index = pw_testindex
    full_index_shape = (5, 6)
    assert pw_index.index.shape == full_index_shape

    ref_index = pw_index.index.copy()
    pw_index.filter(sources="Reactome")
    assert pw_index.index.shape == full_index_shape

    pw_index.filter(species="Homo sapiens")
    assert pw_index.index.shape == full_index_shape

    pw_index.filter(sources=("Reactome",))
    assert pw_index.index.shape == full_index_shape

    pw_index.filter(species=("Homo sapiens",))
    assert pw_index.index.shape == full_index_shape

    pw_index.filter(sources="NotValid")
    assert pw_index.index.shape == (0, full_index_shape[1])
    pw_index.index = ref_index.copy()

    pw_index.filter(species="NotValid")
    assert pw_index.index.shape == (0, full_index_shape[1])
    pw_index.index = ref_index.copy()

    pw_index.search("erythrocytes")
    assert pw_index.index.shape == (2, 6)
    pw_index.index = ref_index.copy()

    pw_index.search("erythrocytes|HYDROCARBON")
    assert pw_index.index.shape == (3, 6)


def test_missing_file(pw_testindex):
    pw_index = pw_testindex
    pw_index.index.loc[0, "file"] = "not_existing.sbml"
    with pytest.raises(FileNotFoundError) as _:
        pw_index._check_files()
