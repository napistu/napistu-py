from unittest.mock import MagicMock

import pytest

from napistu.ontologies.constants import (
    INTERCONVERTIBLE_GENIC_ONTOLOGIES,
    MYGENE_QUERY_DEFS,
)
from napistu.ontologies.mygene import (
    _fetch_mygene_data_all_queries,
    create_python_mapping_tables,
)


def test_create_python_mapping_tables_rejects_invalid_query_strategies():
    """Invalid query_strategies should fail before any MyGene fetch runs."""
    with pytest.raises(ValueError, match="Invalid queries"):
        create_python_mapping_tables(
            mappings={"symbol"},
            species="Homo sapiens",
            test_mode=True,
            query_strategies=["type_of_gene:not-a-real-mygene-query"],
        )


def test_fetch_mygene_data_all_queries_runs_each_strategy_once():
    """Custom query_strategies should drive one MyGene query per strategy (mocked)."""
    strategies = [MYGENE_QUERY_DEFS.PROTEIN_CODING, MYGENE_QUERY_DEFS.NCRNA]
    mg = MagicMock()

    def query_impl(query, species, fields, fetch_all=True):
        yield {"entrezgene": 1, "symbol": f"hit-for-{query[:20]}"}

    mg.query.side_effect = query_impl

    df = _fetch_mygene_data_all_queries(
        mg=mg,
        taxa_id=9606,
        fields=["entrezgene", "symbol"],
        query_strategies=strategies,
        test_mode=True,
    )

    assert mg.query.call_count == 2
    assert list(df["query_type"]) == strategies


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
