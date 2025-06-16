"""Tests for the ontology aliases module."""

import pytest
import pandas as pd
from napistu import identifiers
from napistu.constants import IDENTIFIERS, SBML_DFS
from napistu.ontologies import renaming


@pytest.fixture
def mock_sbml_dfs():
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

    species_df = pd.DataFrame(
        {"s_name": ["gene1", "gene2"], SBML_DFS.S_IDENTIFIERS: [s1_ids, s2_ids]}
    )

    # Create mock SBML_dfs object
    class MockSBMLDfs:
        def __init__(self):
            self.species = species_df
            self.schema = {"species": {"pk": "s_id", "id": SBML_DFS.S_IDENTIFIERS}}

        def get_identifiers(self, table_name):
            if table_name == SBML_DFS.SPECIES:
                all_ids = []
                for idx, row in self.species.iterrows():
                    for id_dict in row[SBML_DFS.S_IDENTIFIERS].ids:
                        all_ids.append({"s_id": idx, **id_dict})
                return pd.DataFrame(all_ids)
            return pd.DataFrame()

    return MockSBMLDfs()


def test_rename_species_ontologies_basic(mock_sbml_dfs):
    """Test basic alias updating functionality."""
    # Define test aliases
    test_aliases = {"ncbi_entrez_gene": {"ncbigene"}, "uniprot": {"uniprot_id"}}

    # Update aliases
    renaming.rename_species_ontologies(mock_sbml_dfs, test_aliases)

    # Get updated identifiers
    updated_ids = mock_sbml_dfs.get_identifiers(SBML_DFS.SPECIES)

    # Check that ontologies were updated correctly
    assert "ncbi_entrez_gene" in set(updated_ids[IDENTIFIERS.ONTOLOGY])
    assert "uniprot" in set(updated_ids[IDENTIFIERS.ONTOLOGY])
    assert "ncbigene" not in set(updated_ids[IDENTIFIERS.ONTOLOGY])
    assert "uniprot_id" not in set(updated_ids[IDENTIFIERS.ONTOLOGY])


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
        "ncbi_entrez_gene": {"ncbigene"}
        # Don't include uniprot_id mapping
    }

    # Update aliases
    renaming.rename_species_ontologies(mock_sbml_dfs, test_aliases)

    # Get updated identifiers
    updated_ids = mock_sbml_dfs.get_identifiers(SBML_DFS.SPECIES)

    # Check that only ncbigene was updated
    assert "ncbi_entrez_gene" in set(updated_ids[IDENTIFIERS.ONTOLOGY])
    assert "uniprot_id" in set(
        updated_ids[IDENTIFIERS.ONTOLOGY]
    )  # Should remain unchanged
