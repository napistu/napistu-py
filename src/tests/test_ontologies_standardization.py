import logging
import os

import numpy as np
import pandas as pd
import pytest

from napistu.constants import BQB, IDENTIFIERS
from napistu.identifiers import Identifiers
from napistu.ontologies.constants import ONTOLOGIES
from napistu.ontologies.standardization import (
    create_uri_url,
    format_uri,
    format_uri_url,
    parse_ensembl_id,
)


def get_identifier_examples(test_data_path):
    return pd.read_csv(
        os.path.join(test_data_path, "identifier_examples.tsv"),
        sep="\t",
        header=0,
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
        assert parse_ensembl_id(k) == v


def test_identifiers_from_urls(test_data_path):
    identifier_examples = get_identifier_examples(test_data_path)
    for i in range(0, identifier_examples.shape[0]):
        testIdentifiers = Identifiers(
            [format_uri(identifier_examples[IDENTIFIERS.URL][i], bqb=BQB.IS)]
        )

        assert (
            testIdentifiers.df.iloc[0][IDENTIFIERS.ONTOLOGY]
            == identifier_examples[IDENTIFIERS.ONTOLOGY][i]
        ), f"ontology {testIdentifiers.df.iloc[0][IDENTIFIERS.ONTOLOGY]} does not equal {identifier_examples[IDENTIFIERS.ONTOLOGY][i]}"

        assert (
            testIdentifiers.df.iloc[0][IDENTIFIERS.IDENTIFIER]
            == identifier_examples[IDENTIFIERS.IDENTIFIER][i]
        ), f"identifier {testIdentifiers.df.iloc[0][IDENTIFIERS.IDENTIFIER]} does not equal {identifier_examples[IDENTIFIERS.IDENTIFIER][i]}"


def test_url_from_identifiers(test_data_path):
    identifier_examples = get_identifier_examples(test_data_path)
    for row in identifier_examples.iterrows():
        # some urls (e.g., chebi) will be converted to a canonical url (e.g., chebi) since multiple URIs exist

        if row[1]["canonical_url"] is not np.nan:
            expected_url_out = row[1]["canonical_url"]
        else:
            expected_url_out = row[1][IDENTIFIERS.URL]

        url_out = create_uri_url(
            ontology=row[1][IDENTIFIERS.ONTOLOGY],
            identifier=row[1][IDENTIFIERS.IDENTIFIER],
        )

        # print(f"expected: {expected_url_out}; observed: {url_out}")
        assert url_out == expected_url_out

    # test non-strict treatment

    assert (
        create_uri_url(ontology=ONTOLOGIES.CHEBI, identifier="abc", strict=False)
        is None
    )


def test_proteinatlas_uri_error():
    """Test that proteinatlas.org URIs are not supported and raise NotImplementedError."""
    proteinatlas_uri = "https://www.proteinatlas.org"

    with pytest.raises(NotImplementedError) as exc_info:
        format_uri(proteinatlas_uri, bqb=BQB.IS)

    assert f"{proteinatlas_uri} is not a valid way of specifying a uri" in str(
        exc_info.value
    )


def test_format_uri_url_unrecognized_netloc_strict_modes(caplog):
    """Test that format_uri_url handles unrecognized netlocs in both strict modes."""
    unrecognized_uri = "https://unknown-domain.com/some/path"

    # Test strict=True (should raise NotImplementedError)
    with pytest.raises(NotImplementedError) as exc_info:
        format_uri_url(unrecognized_uri, strict=True)

    assert "has not been associated with a known ontology" in str(exc_info.value)

    # Test strict=False (should log warning and return None)
    with caplog.at_level(logging.WARNING):
        result = format_uri_url(unrecognized_uri, strict=False)

    assert result is None
    assert len(caplog.records) > 0
    assert any(
        "has not been associated with a known ontology" in record.message
        for record in caplog.records
    )


def test_format_uri_url_pathological_ensembl_id_strict_modes(caplog):
    """Test that format_uri_url handles pathological Ensembl IDs in both strict modes."""
    # Test with pathological Ensembl gene ID that will trigger AttributeError
    pathological_ensembl_uri = (
        "https://www.ensembl.org/Homo_sapiens/geneview?gene=INVALID_ID"
    )

    # Test strict=True (should exit with sys.exit(1) - we can't easily test this)
    # So we'll just test that it would trigger the exception path by testing strict=False

    # Test strict=False (should log warning and return None)
    with caplog.at_level(logging.WARNING):
        result = format_uri_url(pathological_ensembl_uri, strict=False)

    assert result is None
    assert len(caplog.records) > 0
    assert any(
        "Could not extract identifier from URI using regex" in record.message
        for record in caplog.records
    )
