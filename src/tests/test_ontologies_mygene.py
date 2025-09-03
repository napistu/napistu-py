import pytest

from napistu.ontologies.constants import INTERCONVERTIBLE_GENIC_ONTOLOGIES
from napistu.ontologies.mygene import create_python_mapping_tables


@pytest.skip_on_timeout(5)
def test_create_python_mapping_tables_yeast():
    """Test create_python_mapping_tables with yeast species."""
    # Test with a subset of mappings to keep test runtime reasonable
    test_mappings = {"ensembl_gene", "symbol", "uniprot"}

    # Verify test mappings are valid
    assert test_mappings.issubset(
        INTERCONVERTIBLE_GENIC_ONTOLOGIES
    ), "Test mappings must be valid ontologies"

    # Call function with yeast species
    mapping_tables = create_python_mapping_tables(
        mappings=test_mappings,
        species="Saccharomyces cerevisiae",
        test_mode=True,  # Limit to 1000 genes for faster testing
    )

    # Basic validation of results
    assert isinstance(mapping_tables, dict), "Should return a dictionary"

    # Check that all requested mappings are present (ignoring extras like ncbi_entrez_gene)
    assert test_mappings.issubset(
        set(mapping_tables.keys())
    ), "All requested mappings should be present"

    # Check each mapping table
    for ontology in test_mappings:
        df = mapping_tables[ontology]
        assert not df.empty, f"Mapping table for {ontology} should not be empty"
        assert (
            df.index.name == "ncbi_entrez_gene"
        ), f"Index should be entrez gene IDs for {ontology}"
        assert (
            not df.index.duplicated().any()
        ), f"Should not have duplicate indices in {ontology}"
