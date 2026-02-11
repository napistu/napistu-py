from __future__ import annotations

import pandas as pd
import pytest

from napistu import identifiers
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    SBML_DFS,
)
from napistu.ontologies.constants import (
    ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY,
    ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY,
    ENSEMBL_SPECIES_FROM_CODE,
    ENSEMBL_SPECIES_TO_CODE,
    ONTOLOGIES,
)


def test_identifiers_empty_list_has_df():
    """Identifiers([]) has a df attribute with expected schema and zero rows."""
    obj = identifiers.Identifiers([])
    assert hasattr(obj, "df")
    assert obj.df is not None
    assert list(obj.df.columns) == [
        IDENTIFIERS.ONTOLOGY,
        IDENTIFIERS.IDENTIFIER,
        IDENTIFIERS.URL,
        IDENTIFIERS.BQB,
    ]
    assert len(obj.df) == 0


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


def test_reciprocal_ensembl_dicts():
    assert len(ENSEMBL_SPECIES_TO_CODE) == len(ENSEMBL_SPECIES_FROM_CODE)
    for k in ENSEMBL_SPECIES_TO_CODE.keys():
        assert ENSEMBL_SPECIES_FROM_CODE[ENSEMBL_SPECIES_TO_CODE[k]] == k

    assert len(ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY) == len(
        ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY
    )
    for k in ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY.keys():
        assert (
            ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY[ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY[k]]
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
