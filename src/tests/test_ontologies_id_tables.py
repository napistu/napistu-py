from unittest.mock import patch

import pandas as pd
import pytest

from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    SBML_DFS,
    VALID_BQB_TERMS,
)
from napistu.ontologies import id_tables


@pytest.fixture
def sample_id_table():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            SBML_DFS.S_ID: ["s1", "s2", "s3", "s4"],
            IDENTIFIERS.ONTOLOGY: [
                ONTOLOGIES.GO,
                ONTOLOGIES.KEGG,
                ONTOLOGIES.REACTOME,
                ONTOLOGIES.WIKIPATHWAYS,
            ],
            IDENTIFIERS.IDENTIFIER: ["GO:0001", "hsa00010", "R-HSA-123", "WP123"],
            IDENTIFIERS.BQB: [BQB.IS, BQB.HAS_PART, BQB.IS_PART_OF, BQB.IS_VERSION_OF],
            IDENTIFIERS.URL: ["foo", "bar", "baz", "qux"],
            "other_col": ["a", "b", "c", "d"],
        }
    )


def test_sanitize_id_table_values_valid_cases(sample_id_table):
    """Test all valid use cases for _sanitize_id_table_values function."""

    # Test string input conversion
    result = id_tables._sanitize_id_table_values(
        ONTOLOGIES.GO, sample_id_table, IDENTIFIERS.ONTOLOGY
    )
    assert result == {ONTOLOGIES.GO}
    assert isinstance(result, set)

    # Test list input conversion
    result = id_tables._sanitize_id_table_values(
        [ONTOLOGIES.GO, ONTOLOGIES.KEGG], sample_id_table, IDENTIFIERS.ONTOLOGY
    )
    assert result == {ONTOLOGIES.GO, ONTOLOGIES.KEGG}
    assert isinstance(result, set)

    # Test set input unchanged
    input_set = {ONTOLOGIES.GO, ONTOLOGIES.KEGG}
    result = id_tables._sanitize_id_table_values(
        input_set, sample_id_table, IDENTIFIERS.ONTOLOGY
    )
    assert result == input_set
    assert isinstance(result, set)

    # Test successful validation against valid_values
    result = id_tables._sanitize_id_table_values(
        BQB.IS, sample_id_table, IDENTIFIERS.BQB, set(VALID_BQB_TERMS)
    )
    assert result == {BQB.IS}

    # Test duplicate values in input list are handled correctly
    result = id_tables._sanitize_id_table_values(
        [ONTOLOGIES.GO, ONTOLOGIES.GO, ONTOLOGIES.KEGG],
        sample_id_table,
        IDENTIFIERS.ONTOLOGY,
    )
    assert result == {
        ONTOLOGIES.GO,
        ONTOLOGIES.KEGG,
    }  # Duplicates removed by set conversion

    # Test all values present in table
    result = id_tables._sanitize_id_table_values(
        [ONTOLOGIES.GO, ONTOLOGIES.KEGG, ONTOLOGIES.REACTOME],
        sample_id_table,
        IDENTIFIERS.ONTOLOGY,
    )
    assert result == {ONTOLOGIES.GO, ONTOLOGIES.KEGG, ONTOLOGIES.REACTOME}

    # Test single value present in table
    result = id_tables._sanitize_id_table_values(
        ONTOLOGIES.WIKIPATHWAYS, sample_id_table, IDENTIFIERS.ONTOLOGY
    )
    assert result == {ONTOLOGIES.WIKIPATHWAYS}

    # Test with different column (BQB)
    result = id_tables._sanitize_id_table_values(
        BQB.HAS_PART, sample_id_table, IDENTIFIERS.BQB
    )
    assert result == {BQB.HAS_PART}


@patch("napistu.ontologies.id_tables.logger")
def test_sanitize_id_table_values_error_cases(mock_logger, sample_id_table):
    """Test error cases and edge cases for _sanitize_id_table_values function."""

    # Test invalid input types raise ValueError
    with pytest.raises(ValueError, match="ontology must be a string, a set, or list"):
        id_tables._sanitize_id_table_values(123, sample_id_table, IDENTIFIERS.ONTOLOGY)

    with pytest.raises(ValueError, match="ontology must be a string, a set, or list"):
        id_tables._sanitize_id_table_values(
            {"key": "value"}, sample_id_table, IDENTIFIERS.ONTOLOGY
        )

    # Test validation failure against valid_values
    with pytest.raises(
        ValueError, match="The following bqb are not valid: INVALID_BQB"
    ):
        id_tables._sanitize_id_table_values(
            "INVALID_BQB", sample_id_table, IDENTIFIERS.BQB, set(VALID_BQB_TERMS), "bqb"
        )

    # Test multiple invalid values against valid_values
    with pytest.raises(ValueError, match="The following bqb are not valid"):
        id_tables._sanitize_id_table_values(
            ["INVALID1", "INVALID2"],
            sample_id_table,
            IDENTIFIERS.BQB,
            set(VALID_BQB_TERMS),
            "bqb",
        )

    # Test all values missing from table raises error
    missing_values = {"MISSING1", "MISSING2"}
    with pytest.raises(ValueError, match="None of the requested ontology are present"):
        id_tables._sanitize_id_table_values(
            missing_values, sample_id_table, IDENTIFIERS.ONTOLOGY
        )

    # Test case-sensitive matching (lowercase 'go' should fail)
    with pytest.raises(ValueError, match="None of the requested ontology are present"):
        id_tables._sanitize_id_table_values(
            "INVALID_ONTOLOGY", sample_id_table, IDENTIFIERS.ONTOLOGY
        )

    # Test custom value_type_name in error messages
    with pytest.raises(ValueError, match="custom_type must be a string"):
        id_tables._sanitize_id_table_values(
            123, sample_id_table, IDENTIFIERS.ONTOLOGY, value_type_name="custom_type"
        )

    # Test default value_type_name uses column_name
    with pytest.raises(ValueError, match="test_column must be a string"):
        id_tables._sanitize_id_table_values(123, sample_id_table, "test_column")

    # Test empty dataframe column
    empty_df = pd.DataFrame({"ontology": []})
    with pytest.raises(ValueError, match="None of the requested ontology are present"):
        id_tables._sanitize_id_table_values("GO", empty_df, IDENTIFIERS.ONTOLOGY)

    # Test partial values missing logs warning but doesn't raise error
    mixed_values = {ONTOLOGIES.GO, "MISSING"}  # GO exists, MISSING doesn't
    result = id_tables._sanitize_id_table_values(
        mixed_values, sample_id_table, IDENTIFIERS.ONTOLOGY
    )

    assert result == mixed_values
    mock_logger.warning.assert_called_once()
    warning_call = mock_logger.warning.call_args[0][0]
    assert "MISSING" in warning_call
    assert "not present in the id_table" in warning_call

    # Test multiple partial missing values
    mock_logger.reset_mock()
    mixed_values = {ONTOLOGIES.GO, ONTOLOGIES.KEGG, "MISSING1", "MISSING2"}
    result = id_tables._sanitize_id_table_values(
        mixed_values, sample_id_table, IDENTIFIERS.ONTOLOGY
    )

    assert result == mixed_values
    mock_logger.warning.assert_called_once()
    warning_call = mock_logger.warning.call_args[0][0]
    assert "MISSING1" in warning_call and "MISSING2" in warning_call


def test_filter_id_table_basic(sample_id_table):
    """Basic test for filter_id_table filtering by identifier, ontology, and bqb."""

    # Use a known identifier, ontology, and bqb from the fixture
    filtered = id_tables.filter_id_table(
        id_table=sample_id_table,
        identifiers=["GO:0001"],
        ontologies=[ONTOLOGIES.GO],
        bqbs=[BQB.IS],
    )
    # Should return a DataFrame with only the matching row
    assert isinstance(filtered, pd.DataFrame)
    assert len(filtered) == 1
    row = filtered.iloc[0]
    assert row[IDENTIFIERS.ONTOLOGY] == ONTOLOGIES.GO
    assert row[IDENTIFIERS.IDENTIFIER] == "GO:0001"
    assert row[IDENTIFIERS.BQB] == BQB.IS
