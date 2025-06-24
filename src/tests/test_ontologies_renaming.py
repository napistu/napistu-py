"""Tests for the ontology aliases module."""

from unittest.mock import patch

import pytest
import pandas as pd
from napistu import identifiers
from napistu.constants import IDENTIFIERS, SBML_DFS, ONTOLOGIES
from napistu.ontologies import renaming


@pytest.fixture
def mock_sbml_dfs(sbml_dfs):
    """Create a mock SBML_dfs object for testing."""
    # Create a simple species DataFrame with identifiers
    s1_ids = identifiers.Identifiers(
        [
            {
                IDENTIFIERS.ONTOLOGY: "ncbigene",
                IDENTIFIERS.IDENTIFIER: "123",
                IDENTIFIERS.URL: "http://ncbi/123",
                IDENTIFIERS.BQB: "is",
            },
            {
                IDENTIFIERS.ONTOLOGY: "uniprot_id",
                IDENTIFIERS.IDENTIFIER: "P12345",
                IDENTIFIERS.URL: "http://uniprot/P12345",
                IDENTIFIERS.BQB: "is",
            },
        ]
    )

    s2_ids = identifiers.Identifiers(
        [
            {
                IDENTIFIERS.ONTOLOGY: "ncbigene",
                IDENTIFIERS.IDENTIFIER: "456",
                IDENTIFIERS.URL: "http://ncbi/456",
                IDENTIFIERS.BQB: "is",
            }
        ]
    )

    s3_ids = identifiers.Identifiers([])

    species_df = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["gene1", "gene2", "gene3"],
            SBML_DFS.S_IDENTIFIERS: [s1_ids, s2_ids, s3_ids],
        }
    )

    # Patch the species attribute only for the duration of the test
    with patch.object(sbml_dfs, "species", new=species_df):
        yield sbml_dfs  # All methods are real, only .species is patched


def test_rename_species_ontologies_basic(mock_sbml_dfs):
    """Test basic alias updating functionality."""
    # Define test aliases
    test_aliases = {
        ONTOLOGIES.NCBI_ENTREZ_GENE: {"ncbigene"},
        ONTOLOGIES.UNIPROT: {"uniprot_id"},
    }

    # Update aliases
    renaming.rename_species_ontologies(mock_sbml_dfs, test_aliases)

    # Get updated identifiers
    updated_ids = mock_sbml_dfs.get_identifiers(SBML_DFS.SPECIES)

    # Check that ontologies were updated correctly
    assert ONTOLOGIES.NCBI_ENTREZ_GENE in set(updated_ids[IDENTIFIERS.ONTOLOGY])
    assert ONTOLOGIES.UNIPROT in set(updated_ids[IDENTIFIERS.ONTOLOGY])
    assert "ncbigene" not in set(updated_ids[IDENTIFIERS.ONTOLOGY])
    assert "uniprot_id" not in set(updated_ids[IDENTIFIERS.ONTOLOGY])

    # verify that all the species have Identifiers object
    for row in mock_sbml_dfs.species.itertuples():
        val = getattr(row, SBML_DFS.S_IDENTIFIERS)
        assert val is not None and isinstance(
            val, identifiers.Identifiers
        ), f"Bad value: {val} in row {row}"


def test_rename_species_ontologies_no_overlap(mock_sbml_dfs):
    """Test that error is raised when no aliases overlap with data."""
    # Define aliases that don't match any existing ontologies
    test_aliases = {"ensembl_gene": {"ensembl"}}

    # Should raise ValueError due to no overlap
    with pytest.raises(ValueError, match="do not overlap"):
        renaming.rename_species_ontologies(mock_sbml_dfs, test_aliases)


def test_rename_species_ontologies_partial_update(mock_sbml_dfs):
    """Test that partial updates work correctly."""
    # Define aliases that only update some ontologies
    test_aliases = {
        ONTOLOGIES.NCBI_ENTREZ_GENE: {"ncbigene"}
        # Don't include uniprot_id mapping
    }

    # Update aliases
    renaming.rename_species_ontologies(mock_sbml_dfs, test_aliases)

    # Get updated identifiers
    updated_ids = mock_sbml_dfs.get_identifiers(SBML_DFS.SPECIES)

    # Check that only ncbigene was updated
    assert ONTOLOGIES.NCBI_ENTREZ_GENE in set(updated_ids[IDENTIFIERS.ONTOLOGY])
    assert "uniprot_id" in set(
        updated_ids[IDENTIFIERS.ONTOLOGY]
    )  # Should remain unchanged
