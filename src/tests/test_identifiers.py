from __future__ import annotations

import os

import numpy as np
import pandas as pd
from napistu import identifiers

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


################################################
# __main__
################################################

if __name__ == "__main__":
    test_identifiers()
    test_identifiers_from_urls()
    test_url_from_identifiers()
    test_parsing_ensembl_ids()
    test_reciprocal_ensembl_dicts()
