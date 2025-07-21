from __future__ import annotations

import os

import numpy as np
import pandas as pd
from napistu import identifiers
from napistu.constants import IDENTIFIERS
import pytest

# logger = logging.getLogger()
# logger.setLevel("DEBUG")

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
identifier_examples = pd.read_csv(
    os.path.join(test_path, "test_data", "identifier_examples.tsv"),
    sep="\t",
    header=0,
)


def test_identifiers():
    assert (
        identifiers.Identifiers(
            [{"ontology": "KEGG", "identifier": "C00031", "bqb": "BQB_IS"}]
        ).ids[0]["ontology"]
        == "KEGG"
    )

    example_identifiers = identifiers.Identifiers(
        [
            {"ontology": "SGD", "identifier": "S000004535", "bqb": "BQB_IS"},
            {"ontology": "foo", "identifier": "bar", "bqb": "BQB_IS"},
        ]
    )

    assert type(example_identifiers) is identifiers.Identifiers

    assert example_identifiers.filter("SGD") is True
    assert example_identifiers.filter("baz") is False
    assert example_identifiers.filter("SGD", summarize=False) == [True, False]
    assert example_identifiers.filter(["SGD", "foo"], summarize=False) == [True, True]
    assert example_identifiers.filter(["foo", "SGD"], summarize=False) == [True, True]
    assert example_identifiers.filter(["baz", "bar"], summarize=False) == [False, False]

    assert example_identifiers.hoist("SGD") == "S000004535"
    assert example_identifiers.hoist("baz") is None


def test_identifiers_from_urls():
    for i in range(0, identifier_examples.shape[0]):
        # print(identifier_examples["url"][i])
        testIdentifiers = identifiers.Identifiers(
            [
                identifiers.format_uri(
                    identifier_examples["url"][i], biological_qualifier_type="BQB_IS"
                )
            ]
        )

        # print(f"ontology = {testIdentifiers.ids[0]['ontology']}; identifier = {testIdentifiers.ids[0]['identifier']}")
        assert (
            testIdentifiers.ids[0]["ontology"] == identifier_examples["ontology"][i]
        ), f"ontology {testIdentifiers.ids[0]['ontology']} does not equal {identifier_examples['ontology'][i]}"

        assert (
            testIdentifiers.ids[0]["identifier"] == identifier_examples["identifier"][i]
        ), f"identifier {testIdentifiers.ids[0]['identifier']} does not equal {identifier_examples['identifier'][i]}"


def test_url_from_identifiers():
    for row in identifier_examples.iterrows():
        # some urls (e.g., chebi) will be converted to a canonical url (e.g., chebi) since multiple URIs exist

        if row[1]["canonical_url"] is not np.nan:
            expected_url_out = row[1]["canonical_url"]
        else:
            expected_url_out = row[1]["url"]

        url_out = identifiers.create_uri_url(
            ontology=row[1]["ontology"], identifier=row[1]["identifier"]
        )

        # print(f"expected: {expected_url_out}; observed: {url_out}")
        assert url_out == expected_url_out

    # test non-strict treatment

    assert (
        identifiers.create_uri_url(ontology="chebi", identifier="abc", strict=False)
        is None
    )


def test_parsing_ensembl_ids():
    ensembl_examples = {
        # human foxp2
        "ENSG00000128573": ("ENSG00000128573", "ensembl_gene", "Homo sapiens"),
        "ENST00000441290": ("ENST00000441290", "ensembl_transcript", "Homo sapiens"),
        "ENSP00000265436": ("ENSP00000265436", "ensembl_protein", "Homo sapiens"),
        # mouse leptin
        "ENSMUSG00000059201": ("ENSMUSG00000059201", "ensembl_gene", "Mus musculus"),
        "ENSMUST00000069789": (
            "ENSMUST00000069789",
            "ensembl_transcript",
            "Mus musculus",
        ),
        # substrings are okay
        "gene=ENSMUSG00000017146": (
            "ENSMUSG00000017146",
            "ensembl_gene",
            "Mus musculus",
        ),
    }

    for k, v in ensembl_examples.items():
        assert identifiers.parse_ensembl_id(k) == v


def test_reciprocal_ensembl_dicts():
    assert len(identifiers.ENSEMBL_SPECIES_TO_CODE) == len(
        identifiers.ENSEMBL_SPECIES_FROM_CODE
    )
    for k in identifiers.ENSEMBL_SPECIES_TO_CODE.keys():
        assert (
            identifiers.ENSEMBL_SPECIES_FROM_CODE[
                identifiers.ENSEMBL_SPECIES_TO_CODE[k]
            ]
            == k
        )

    assert len(identifiers.ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY) == len(
        identifiers.ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY
    )
    for k in identifiers.ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY.keys():
        assert (
            identifiers.ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY[
                identifiers.ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY[k]
            ]
            == k
        )


def test_df_to_identifiers_basic():
    """Test basic conversion of DataFrame to Identifiers objects."""
    # Create a simple test DataFrame
    df = pd.DataFrame(
        {
            "s_id": ["s1", "s1", "s2"],
            IDENTIFIERS.ONTOLOGY: ["ncbi_entrez_gene", "uniprot", "ncbi_entrez_gene"],
            IDENTIFIERS.IDENTIFIER: ["123", "P12345", "456"],
            IDENTIFIERS.URL: [
                "http://ncbi/123",
                "http://uniprot/P12345",
                "http://ncbi/456",
            ],
            IDENTIFIERS.BQB: ["is", "is", "is"],
        }
    )

    # Convert to Identifiers objects
    result = identifiers.df_to_identifiers(df)

    # Check basic properties
    assert isinstance(result, pd.Series)
    assert len(result) == 2  # Two unique s_ids
    assert all(isinstance(x, identifiers.Identifiers) for x in result)

    # Check specific values
    s1_ids = result["s1"].ids
    assert len(s1_ids) == 2  # Two identifiers for s1
    assert any(x[IDENTIFIERS.IDENTIFIER] == "123" for x in s1_ids)
    assert any(x[IDENTIFIERS.IDENTIFIER] == "P12345" for x in s1_ids)

    s2_ids = result["s2"].ids
    assert len(s2_ids) == 1  # One identifier for s2
    assert s2_ids[0][IDENTIFIERS.IDENTIFIER] == "456"


def test_df_to_identifiers_duplicates():
    """Test that duplicates are handled correctly."""
    # Create DataFrame with duplicate entries
    df = pd.DataFrame(
        {
            "s_id": ["s1", "s1", "s1"],
            IDENTIFIERS.ONTOLOGY: [
                "ncbi_entrez_gene",
                "ncbi_entrez_gene",
                "ncbi_entrez_gene",
            ],
            IDENTIFIERS.IDENTIFIER: ["123", "123", "123"],  # Same identifier repeated
            IDENTIFIERS.URL: ["http://ncbi/123"] * 3,
            IDENTIFIERS.BQB: ["is"] * 3,
        }
    )

    result = identifiers.df_to_identifiers(df)

    # Should collapse duplicates
    assert len(result) == 1  # One unique s_id
    assert len(result["s1"].ids) == 1  # One unique identifier


def test_df_to_identifiers_missing_columns():
    """Test that missing required columns raise an error."""
    # Create DataFrame missing required columns
    df = pd.DataFrame(
        {
            "s_id": ["s1"],
            IDENTIFIERS.ONTOLOGY: ["ncbi_entrez_gene"],
            IDENTIFIERS.IDENTIFIER: ["123"],
            # Missing URL and BQB
        }
    )

    with pytest.raises(
        ValueError, match="The DataFrame does not contain the required columns"
    ):
        identifiers.df_to_identifiers(df)
