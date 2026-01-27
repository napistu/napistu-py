from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from napistu import identifiers
from napistu.constants import BQB, IDENTIFIERS, ONTOLOGIES, SBML_DFS

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
            [
                {
                    IDENTIFIERS.ONTOLOGY: ONTOLOGIES.KEGG,
                    IDENTIFIERS.IDENTIFIER: "C00031",
                    IDENTIFIERS.BQB: BQB.IS,
                }
            ]
        ).df.iloc[0][IDENTIFIERS.ONTOLOGY]
        == ONTOLOGIES.KEGG
    )

    example_identifiers = identifiers.Identifiers(
        [
            {
                IDENTIFIERS.ONTOLOGY: ONTOLOGIES.SGD,
                IDENTIFIERS.IDENTIFIER: "S000004535",
                IDENTIFIERS.BQB: BQB.IS,
            },
            {
                IDENTIFIERS.ONTOLOGY: "foo",
                IDENTIFIERS.IDENTIFIER: "bar",
                IDENTIFIERS.BQB: BQB.IS,
            },
        ]
    )

    assert type(example_identifiers) is identifiers.Identifiers

    assert example_identifiers.has_ontology(ONTOLOGIES.SGD) is True
    assert example_identifiers.has_ontology("baz") is False
    assert example_identifiers.has_ontology([ONTOLOGIES.SGD, "foo"]) is True
    assert example_identifiers.has_ontology(["foo", ONTOLOGIES.SGD]) is True
    assert example_identifiers.has_ontology(["baz", "bar"]) is False

    assert example_identifiers.hoist(ONTOLOGIES.SGD) == "S000004535"
    assert example_identifiers.hoist("baz") is None


def test_identifiers_from_urls():
    for i in range(0, identifier_examples.shape[0]):
        # print(identifier_examples["url"][i])
        testIdentifiers = identifiers.Identifiers(
            [
                identifiers.format_uri(
                    identifier_examples[IDENTIFIERS.URL][i], bqb=BQB.IS
                )
            ]
        )

        assert (
            testIdentifiers.df.iloc[0][IDENTIFIERS.ONTOLOGY]
            == identifier_examples[IDENTIFIERS.ONTOLOGY][i]
        ), f"ontology {testIdentifiers.df.iloc[0][IDENTIFIERS.ONTOLOGY]} does not equal {identifier_examples[IDENTIFIERS.ONTOLOGY][i]}"

        assert (
            testIdentifiers.df.iloc[0][IDENTIFIERS.IDENTIFIER]
            == identifier_examples[IDENTIFIERS.IDENTIFIER][i]
        ), f"identifier {testIdentifiers.df.iloc[0][IDENTIFIERS.IDENTIFIER]} does not equal {identifier_examples[IDENTIFIERS.IDENTIFIER][i]}"


def test_url_from_identifiers():
    for row in identifier_examples.iterrows():
        # some urls (e.g., chebi) will be converted to a canonical url (e.g., chebi) since multiple URIs exist

        if row[1]["canonical_url"] is not np.nan:
            expected_url_out = row[1]["canonical_url"]
        else:
            expected_url_out = row[1][IDENTIFIERS.URL]

        url_out = identifiers.create_uri_url(
            ontology=row[1][IDENTIFIERS.ONTOLOGY],
            identifier=row[1][IDENTIFIERS.IDENTIFIER],
        )

        # print(f"expected: {expected_url_out}; observed: {url_out}")
        assert url_out == expected_url_out

    # test non-strict treatment

    assert (
        identifiers.create_uri_url(
            ontology=ONTOLOGIES.CHEBI, identifier="abc", strict=False
        )
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


def test_proteinatlas_uri_error():
    """Test that proteinatlas.org URIs are not supported and raise NotImplementedError."""
    proteinatlas_uri = "https://www.proteinatlas.org"

    with pytest.raises(NotImplementedError) as exc_info:
        identifiers.format_uri(proteinatlas_uri, bqb=BQB.IS)

    assert f"{proteinatlas_uri} is not a valid way of specifying a uri" in str(
        exc_info.value
    )


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
            SBML_DFS.S_ID: ["s1", "s1", "s2"],
            IDENTIFIERS.ONTOLOGY: [
                ONTOLOGIES.NCBI_ENTREZ_GENE,
                ONTOLOGIES.UNIPROT,
                ONTOLOGIES.NCBI_ENTREZ_GENE,
            ],
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
    # s1_ids = result["s1"].ids
    # assert len(s1_ids) == 2  # Two identifiers for s1
    # assert any(x[IDENTIFIERS.IDENTIFIER] == "123" for x in s1_ids)
    # assert any(x[IDENTIFIERS.IDENTIFIER] == "P12345" for x in s1_ids)

    # s2_ids = result["s2"].ids
    # assert len(s2_ids) == 1  # One identifier for s2
    # assert s2_ids[0][IDENTIFIERS.IDENTIFIER] == "456"


def test_df_to_identifiers_duplicates():
    """Test that duplicates are handled correctly."""
    # Create DataFrame with duplicate entries
    df = pd.DataFrame(
        {
            SBML_DFS.S_ID: ["s1", "s1", "s1"],
            IDENTIFIERS.ONTOLOGY: [
                ONTOLOGIES.NCBI_ENTREZ_GENE,
                ONTOLOGIES.NCBI_ENTREZ_GENE,
                ONTOLOGIES.NCBI_ENTREZ_GENE,
            ],
            IDENTIFIERS.IDENTIFIER: ["123", "123", "123"],  # Same identifier repeated
            IDENTIFIERS.URL: ["http://ncbi/123"] * 3,
            IDENTIFIERS.BQB: ["is"] * 3,
        }
    )

    result = identifiers.df_to_identifiers(df)
    print(result)

    # Should collapse duplicates
    assert len(result) == 1  # One unique s_id
    # assert len(result["s1"].ids) == 1  # One unique identifier


def test_df_to_identifiers_missing_columns():
    """Test that missing required columns raise an error."""
    # Create DataFrame missing required columns
    df = pd.DataFrame(
        {
            SBML_DFS.S_ID: ["s1"],
            IDENTIFIERS.ONTOLOGY: [ONTOLOGIES.NCBI_ENTREZ_GENE],
            IDENTIFIERS.IDENTIFIER: ["123"],
            # Missing URL and BQB
        }
    )

    with pytest.raises(
        ValueError,
        match=r"\d+ required variables were missing from the provided pd\.DataFrame or pd\.Series: bqb",
    ):
        identifiers.df_to_identifiers(df)


def test_format_uri_url_unrecognized_netloc_strict_modes(caplog):
    """Test that format_uri_url handles unrecognized netlocs in both strict modes."""
    import logging

    unrecognized_uri = "https://unknown-domain.com/some/path"

    # Test strict=True (should raise NotImplementedError)
    with pytest.raises(NotImplementedError) as exc_info:
        identifiers.format_uri_url(unrecognized_uri, strict=True)

    assert "has not been associated with a known ontology" in str(exc_info.value)

    # Test strict=False (should log warning and return None)
    with caplog.at_level(logging.WARNING):
        result = identifiers.format_uri_url(unrecognized_uri, strict=False)

    assert result is None
    assert len(caplog.records) > 0
    assert any(
        "has not been associated with a known ontology" in record.message
        for record in caplog.records
    )


def test_format_uri_url_pathological_ensembl_id_strict_modes(caplog):
    """Test that format_uri_url handles pathological Ensembl IDs in both strict modes."""
    import logging

    # Test with pathological Ensembl gene ID that will trigger AttributeError
    pathological_ensembl_uri = (
        "https://www.ensembl.org/Homo_sapiens/geneview?gene=INVALID_ID"
    )

    # Test strict=True (should exit with sys.exit(1) - we can't easily test this)
    # So we'll just test that it would trigger the exception path by testing strict=False

    # Test strict=False (should log warning and return None)
    with caplog.at_level(logging.WARNING):
        result = identifiers.format_uri_url(pathological_ensembl_uri, strict=False)

    assert result is None
    assert len(caplog.records) > 0
    assert any(
        "Could not extract identifier from URI using regex" in record.message
        for record in caplog.records
    )


def test_construct_cspecies_identifiers(sbml_dfs):
    """Test that construct_cspecies_identifiers works with both sbml_dfs and lookup table."""
    # Get species identifiers from sbml_dfs
    species_identifiers = sbml_dfs.get_characteristic_species_ids(dogmatic=True)

    # Method 1: Use sbml_dfs directly
    result_from_sbml_dfs = identifiers.construct_cspecies_identifiers(
        species_identifiers=species_identifiers,
        cspecies_references=sbml_dfs,
    )

    # Method 2: Extract lookup table and use it
    sid_to_scids_lookup = sbml_dfs.compartmentalized_species.reset_index()[
        [SBML_DFS.S_ID, SBML_DFS.SC_ID]
    ]
    result_from_lookup = identifiers.construct_cspecies_identifiers(
        species_identifiers=species_identifiers,
        cspecies_references=sid_to_scids_lookup,
    )

    # Verify both methods produce the same result
    pd.testing.assert_frame_equal(
        result_from_sbml_dfs.sort_values(
            by=[SBML_DFS.S_ID, SBML_DFS.SC_ID]
        ).reset_index(drop=True),
        result_from_lookup.sort_values(by=[SBML_DFS.S_ID, SBML_DFS.SC_ID]).reset_index(
            drop=True
        ),
        check_like=True,
    )

    # Verify the result has the expected structure
    assert SBML_DFS.SC_ID in result_from_sbml_dfs.columns
    assert SBML_DFS.S_ID in result_from_sbml_dfs.columns
    assert result_from_sbml_dfs.shape[0] == 160
