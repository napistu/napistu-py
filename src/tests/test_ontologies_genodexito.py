import pandas as pd
import pytest

from napistu.ontologies.genodexito import Genodexito
from napistu.ontologies.constants import (
    GENODEXITO_DEFS,
    INTERCONVERTIBLE_GENIC_ONTOLOGIES,
    PROTEIN_ONTOLOGIES,
)


@pytest.skip_on_timeout(5)
def test_genodexito_mapping_operations():
    """Test Genodexito mapping table creation and operations."""
    # Initialize with test mode and Python method to avoid R dependencies
    geno = Genodexito(
        species="Saccharomyces cerevisiae",
        preferred_method=GENODEXITO_DEFS.PYTHON,
        allow_fallback=False,
        test_mode=True,
    )

    # Test subset of mappings
    test_mappings = {"ensembl_gene", "symbol", "uniprot"}

    # Verify test mappings are valid
    assert test_mappings.issubset(
        INTERCONVERTIBLE_GENIC_ONTOLOGIES
    ), "Test mappings must be valid ontologies"

    # Create mapping tables
    geno.create_mapping_tables(mappings=test_mappings)

    # Verify mappings were created
    assert geno.mappings is not None, "Mappings should be created"
    assert geno.mapper_used == GENODEXITO_DEFS.PYTHON, "Should use Python mapper"
    assert test_mappings.issubset(
        set(geno.mappings.keys())
    ), "All requested mappings should be present"

    # Test merge_mappings
    geno.merge_mappings(test_mappings)
    assert isinstance(
        geno.merged_mappings, pd.DataFrame
    ), "merge_mappings should create a DataFrame"
    assert not geno.merged_mappings.empty, "Merged mappings should not be empty"
    assert (
        set(geno.merged_mappings.columns) & test_mappings == test_mappings
    ), "Merged mappings should contain all requested ontologies"

    # Test stack_mappings with protein ontologies
    protein_test_mappings = set(PROTEIN_ONTOLOGIES) & test_mappings
    if protein_test_mappings:  # Only test if we have protein ontologies
        geno.stack_mappings(protein_test_mappings)
        assert isinstance(
            geno.stacked_mappings, pd.DataFrame
        ), "stack_mappings should create a DataFrame"
        assert not geno.stacked_mappings.empty, "Stacked mappings should not be empty"
        assert (
            set(geno.stacked_mappings["ontology"].unique()) == protein_test_mappings
        ), "Stacked mappings should contain all requested protein ontologies"
