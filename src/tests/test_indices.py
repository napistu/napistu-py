from __future__ import annotations

import datetime
import os

import pandas as pd
import pytest

from napistu import indices
from napistu.constants import SOURCE_SPEC

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")


def test_create_pathway_index_df():
    """Test the creation of pathway index DataFrame."""
    # Test input data
    model_keys = {"human": "model1", "mouse": "model2"}
    model_urls = {
        "human": "http://example.com/model1.xml",
        "mouse": "http://example.com/model2.xml",
    }
    model_organismal_species = {"human": "Homo sapiens", "mouse": "Mus musculus"}
    base_path = "/test/path"
    data_source = "TestSource"

    # Create pathway index
    result = indices.create_pathway_index_df(
        model_keys=model_keys,
        model_urls=model_urls,
        model_organismal_species=model_organismal_species,
        base_path=base_path,
        data_source=data_source,
    )

    # Expected date in YYYYMMDD format
    expected_date = datetime.date.today().strftime("%Y%m%d")

    # Assertions
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    assert len(result) == 2, "Should have 2 rows for 2 models"

    # Check required columns exist
    required_columns = {
        SOURCE_SPEC.URL,
        SOURCE_SPEC.ORGANISMAL_SPECIES,
        SOURCE_SPEC.SBML_PATH,
        SOURCE_SPEC.FILE,
        SOURCE_SPEC.DATE,
        SOURCE_SPEC.PATHWAY_ID,
        SOURCE_SPEC.NAME,
        SOURCE_SPEC.DATA_SOURCE,
    }
    assert set(result.columns) == required_columns, "Missing required columns"

    # Check content for first model (human)
    human_row = result[result[SOURCE_SPEC.PATHWAY_ID] == "model1"].iloc[0]
    assert human_row[SOURCE_SPEC.URL] == "http://example.com/model1.xml"
    assert human_row[SOURCE_SPEC.ORGANISMAL_SPECIES] == "Homo sapiens"
    assert human_row[SOURCE_SPEC.FILE] == "model1.sbml"
    assert human_row[SOURCE_SPEC.DATE] == expected_date
    assert human_row[SOURCE_SPEC.DATA_SOURCE] == "TestSource"
    assert human_row[SOURCE_SPEC.SBML_PATH] == os.path.join(base_path, "model1.sbml")

    # Test with custom file extension
    result_custom_ext = indices.create_pathway_index_df(
        model_keys=model_keys,
        model_urls=model_urls,
        model_organismal_species=model_organismal_species,
        base_path=base_path,
        data_source=data_source,
        file_extension=".xml",
    )
    assert result_custom_ext.iloc[0][SOURCE_SPEC.FILE].endswith(
        ".xml"
    ), "Custom extension not applied"


def test_pwindex_from_file():
    pw_index_path = os.path.join(test_data, "pw_index.tsv")
    pw_index = indices.PWIndex(pw_index_path)

    assert pw_index.index.shape == (5, 6)


def test_pwindex_from_df():
    stub_pw_df = pd.DataFrame(
        {
            SOURCE_SPEC.FILE: "DNE",
            SOURCE_SPEC.DATA_SOURCE: "The farm",
            SOURCE_SPEC.ORGANISMAL_SPECIES: "Gallus gallus",
            SOURCE_SPEC.PATHWAY_ID: "chickens",
            SOURCE_SPEC.NAME: "Chickens",
            SOURCE_SPEC.DATE: "2020-01-01",
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
    pw_index.filter(data_sources="Reactome")
    assert pw_index.index.shape == full_index_shape

    pw_index.filter(organismal_species="Homo sapiens")
    assert pw_index.index.shape == full_index_shape

    pw_index.filter(data_sources=("Reactome",))
    assert pw_index.index.shape == full_index_shape

    pw_index.filter(organismal_species=("Homo sapiens",))
    assert pw_index.index.shape == full_index_shape

    pw_index.filter(data_sources="NotValid")
    assert pw_index.index.shape == (0, full_index_shape[1])
    pw_index.index = ref_index.copy()

    pw_index.filter(organismal_species="NotValid")
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
