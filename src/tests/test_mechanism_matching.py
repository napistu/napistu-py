from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from napistu import mechanism_matching
from napistu.network import net_create
from napistu.network import precompute
from napistu.mechanism_matching import _validate_wide_ontologies
from napistu.mechanism_matching import match_by_ontology_and_identifier
from napistu.mechanism_matching import resolve_matches

from napistu.constants import SBML_DFS
from napistu.constants import IDENTIFIERS
from napistu.constants import ONTOLOGIES
from napistu.constants import RESOLVE_MATCHES_AGGREGATORS
from napistu.constants import FEATURE_ID_VAR_DEFAULT


def test_features_to_pathway_species(sbml_dfs):

    species_identifiers = sbml_dfs.get_identifiers("species")
    feature_identifiers = pd.DataFrame({"chebis": ["17627", "15379", "29105", "-1"]})

    matching_df = (
        mechanism_matching.features_to_pathway_species(
            feature_identifiers, species_identifiers, {"chebi"}, "chebis"
        )
        .value_counts("identifier")
        .sort_index()
    )

    assert matching_df.index.tolist() == ["15379", "17627", "29105"]
    assert matching_df.tolist() == [2, 3, 2]


def test_features_to_pathway_species_basic_and_expansion():

    # Mock species_identifiers table
    species_identifiers = pd.DataFrame(
        {
            "ontology": ["chebi", "chebi", "uniprot", "uniprot"],
            "identifier": ["A", "B", "X", "Y"],
            "s_id": [1, 2, 3, 4],
            "s_name": ["foo", "bar", "baz", "qux"],
            "bqb": ["BQB_IS", "BQB_IS", "BQB_IS", "BQB_IS"],
        }
    )
    # Basic: no expansion, single identifier per row
    features = pd.DataFrame({"my_id": ["A", "B", "X"], "other_col": [10, 20, 30]})
    result = mechanism_matching.features_to_pathway_species(
        feature_identifiers=features,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"},
        feature_identifiers_var="my_id",
        expand_identifiers=False,
    )
    # Should map all three
    assert set(result["my_id"]) == {"A", "B", "X"}
    assert set(result["identifier"]) == {"A", "B", "X"}
    assert set(result["s_name"]) == {"foo", "bar", "baz"}
    # Expansion: one row with multiple IDs
    features2 = pd.DataFrame({"my_id": ["A / B / X", "Y"], "other_col": [100, 200]})
    result2 = mechanism_matching.features_to_pathway_species(
        feature_identifiers=features2,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"},
        feature_identifiers_var="my_id",
        expand_identifiers=True,
        identifier_delimiter="/",
    )
    # Should expand to 4 rows (A, B, X, Y)
    assert set(result2["identifier"]) == {"A", "B", "X", "Y"}
    assert set(result2["s_name"]) == {"foo", "bar", "baz", "qux"}
    # Whitespace trimming
    features3 = pd.DataFrame({"my_id": ["  A  /  B  /X  ", " Y"], "other_col": [1, 2]})
    result3 = mechanism_matching.features_to_pathway_species(
        feature_identifiers=features3,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"},
        feature_identifiers_var="my_id",
        expand_identifiers=True,
        identifier_delimiter="/",
    )
    # Should expand and trim whitespace
    assert set(result3["identifier"]) == {"A", "B", "X", "Y"}
    assert set(result3["s_name"]) == {"foo", "bar", "baz", "qux"}


def test_edgelist_to_pathway_species(sbml_dfs):

    edgelist = pd.DataFrame(
        [
            {"identifier_upstream": "17996", "identifier_downstream": "16526"},
            {"identifier_upstream": "15377", "identifier_downstream": "17544"},
            {"identifier_upstream": "15378", "identifier_downstream": "57945"},
            {"identifier_upstream": "57540", "identifier_downstream": "17996"},
        ]
    )
    species_identifiers = sbml_dfs.get_identifiers("species").query("bqb == 'BQB_IS'")

    edgelist_w_sids = mechanism_matching.edgelist_to_pathway_species(
        edgelist, species_identifiers, ontologies={"chebi", "uniprot"}
    )
    assert edgelist_w_sids.shape == (4, 4)

    egelist_w_scids = mechanism_matching.edgelist_to_scids(
        edgelist, sbml_dfs, species_identifiers, ontologies={"chebi"}
    )

    assert egelist_w_scids.shape == (12, 6)

    direct_interactions = mechanism_matching.filter_to_direct_mechanistic_interactions(
        edgelist, sbml_dfs, species_identifiers, ontologies={"chebi"}
    )

    assert direct_interactions.shape == (2, 10)


def test_direct_and_indirect_mechanism_matching(sbml_dfs_glucose_metabolism):

    cpr_graph = net_create.process_cpr_graph(sbml_dfs_glucose_metabolism)

    edgelist = pd.DataFrame(
        [
            {
                "identifier_upstream": "17925",
                "identifier_downstream": "32966",
            },  # glu, fbp
            {
                "identifier_upstream": "57634",
                "identifier_downstream": "32966",
            },  # f6p, fbp
            {
                "identifier_upstream": "32966",
                "identifier_downstream": "57642",
            },  # fbp, dhap
            {
                "identifier_upstream": "17925",
                "identifier_downstream": "15361",
            },  # glu, pyr
        ]
    )

    species_identifiers = sbml_dfs_glucose_metabolism.get_identifiers("species")

    direct_interactions = mechanism_matching.filter_to_direct_mechanistic_interactions(
        formatted_edgelist=edgelist,
        sbml_dfs=sbml_dfs_glucose_metabolism,
        species_identifiers=species_identifiers,
        ontologies={"chebi"},
    )

    assert direct_interactions.shape == (2, 10)

    indirect_interactions = (
        mechanism_matching.filter_to_indirect_mechanistic_interactions(
            formatted_edgelist=edgelist,
            sbml_dfs=sbml_dfs_glucose_metabolism,
            species_identifiers=species_identifiers,
            cpr_graph=cpr_graph,
            ontologies={"chebi"},
            precomputed_distances=None,
            max_path_length=10,
        )
    )

    assert indirect_interactions.shape == (6, 12)

    # confirm that we get the same thing even when using precomputed distances
    precomputed_distances = precompute.precompute_distances(
        cpr_graph, weights_vars=["weights"]
    )

    indirect_interactions_w_precompute = (
        mechanism_matching.filter_to_indirect_mechanistic_interactions(
            formatted_edgelist=edgelist,
            sbml_dfs=sbml_dfs_glucose_metabolism,
            species_identifiers=species_identifiers,
            cpr_graph=cpr_graph,
            ontologies={"chebi"},
            precomputed_distances=precomputed_distances,
            max_path_length=10,
        )
    )

    assert all(
        indirect_interactions["weight"] == indirect_interactions_w_precompute["weight"]
    )


def test_validate_wide_ontologies():
    """Test the _validate_wide_ontologies function with various input types and error cases."""
    # Setup test data
    example_data_wide = pd.DataFrame(
        {
            "results": [-1.0, 0.0, 1.0],
            "chebi": ["15377", "16810", "17925"],
            "uniprot": ["P12345", "Q67890", "O43826"],
        }
    )

    # Test auto-detection of ontology columns
    assert _validate_wide_ontologies(example_data_wide) == {"chebi", "uniprot"}

    # Test string input
    assert _validate_wide_ontologies(example_data_wide, ontologies="chebi") == {"chebi"}

    # Test set input
    assert _validate_wide_ontologies(example_data_wide, ontologies={"chebi"}) == {
        "chebi"
    }
    assert _validate_wide_ontologies(
        example_data_wide, ontologies={"chebi", "uniprot"}
    ) == {"chebi", "uniprot"}

    # Test dictionary mapping for renaming
    assert _validate_wide_ontologies(
        example_data_wide, ontologies={"chebi": "reactome", "uniprot": "ensembl_gene"}
    ) == {"reactome", "ensembl_gene"}

    # Test error cases

    # Missing column in set input (checks existence first)
    with pytest.raises(
        ValueError, match="Specified ontology columns not found in DataFrame:.*"
    ):
        _validate_wide_ontologies(example_data_wide, ontologies={"invalid_ontology"})

    # Valid column name but invalid ontology
    df_with_invalid = pd.DataFrame(
        {
            "results": [-1.0, 0.0, 1.0],
            "invalid_ontology": ["a", "b", "c"],
        }
    )
    with pytest.raises(ValueError, match="Invalid ontologies in set:.*"):
        _validate_wide_ontologies(df_with_invalid, ontologies={"invalid_ontology"})

    # Missing source column in mapping
    with pytest.raises(ValueError, match="Source columns not found in DataFrame:.*"):
        _validate_wide_ontologies(
            example_data_wide, ontologies={"missing_column": "reactome"}
        )

    # Invalid target ontology in mapping
    with pytest.raises(ValueError, match="Invalid ontologies in mapping:.*"):
        _validate_wide_ontologies(
            example_data_wide, ontologies={"chebi": "invalid_ontology"}
        )

    # DataFrame with no valid ontology columns
    invalid_df = pd.DataFrame(
        {"results": [-1.0, 0.0, 1.0], "col1": ["a", "b", "c"], "col2": ["d", "e", "f"]}
    )
    with pytest.raises(
        ValueError, match="No valid ontology columns found in DataFrame.*"
    ):
        _validate_wide_ontologies(invalid_df)


def test_ensure_feature_id_var():
    """Test the _ensure_feature_id_var function with various input cases."""
    from napistu.mechanism_matching import _ensure_feature_id_var
    from napistu.constants import FEATURE_ID_VAR_DEFAULT

    # Test case 1: DataFrame already has feature_id column
    df1 = pd.DataFrame({"feature_id": [100, 200, 300], "data": ["a", "b", "c"]})
    result1 = _ensure_feature_id_var(df1)
    # Should return unchanged DataFrame
    pd.testing.assert_frame_equal(df1, result1)

    # Test case 2: DataFrame missing feature_id column
    df2 = pd.DataFrame({"data": ["x", "y", "z"]})
    result2 = _ensure_feature_id_var(df2)
    # Should add feature_id column with sequential integers
    assert FEATURE_ID_VAR_DEFAULT in result2.columns
    assert list(result2[FEATURE_ID_VAR_DEFAULT]) == [0, 1, 2]
    assert list(result2["data"]) == ["x", "y", "z"]  # Original data preserved

    # Test case 3: Custom feature_id column name
    df3 = pd.DataFrame({"data": ["p", "q", "r"]})
    custom_id = "custom_feature_id"
    result3 = _ensure_feature_id_var(df3, feature_id_var=custom_id)
    # Should add custom named feature_id column
    assert custom_id in result3.columns
    assert list(result3[custom_id]) == [0, 1, 2]
    assert list(result3["data"]) == ["p", "q", "r"]  # Original data preserved

    # Test case 4: Empty DataFrame
    df4 = pd.DataFrame()
    result4 = _ensure_feature_id_var(df4)
    # Should handle empty DataFrame gracefully
    assert FEATURE_ID_VAR_DEFAULT in result4.columns
    assert len(result4) == 0


def test_match_by_ontology_and_identifier():
    """Test the match_by_ontology_and_identifier function with various input types."""
    # Setup test data
    feature_identifiers = pd.DataFrame(
        {
            "ontology": ["chebi", "chebi", "uniprot", "uniprot", "reactome"],
            "identifier": ["15377", "16810", "P12345", "Q67890", "R12345"],
            "results": [1.0, 2.0, -1.0, -2.0, 0.5],
        }
    )

    species_identifiers = pd.DataFrame(
        {
            "ontology": ["chebi", "chebi", "uniprot", "uniprot", "ensembl_gene"],
            "identifier": ["15377", "17925", "P12345", "O43826", "ENSG123"],
            "s_id": ["s1", "s2", "s3", "s4", "s5"],
            "s_name": ["compound1", "compound2", "protein1", "protein2", "gene1"],
            "bqb": ["BQB_IS"] * 5,  # Add required bqb column with BQB_IS values
        }
    )

    # Test with single ontology (string)
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies="chebi",
    )
    assert len(result) == 1  # Only one matching chebi identifier
    assert result.iloc[0]["identifier"] == "15377"
    assert result.iloc[0]["results"] == 1.0
    assert result.iloc[0]["ontology"] == "chebi"  # From species_identifiers
    assert result.iloc[0]["s_name"] == "compound1"  # Verify join worked correctly
    assert result.iloc[0]["bqb"] == "BQB_IS"  # Verify bqb column is preserved

    # Test with multiple ontologies (set)
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"},
    )
    assert len(result) == 2  # One chebi and one uniprot match
    assert set(result["ontology"]) == {"chebi", "uniprot"}  # From species_identifiers
    assert set(result["identifier"]) == {"15377", "P12345"}
    # Verify results are correctly matched
    chebi_row = result[result["ontology"] == "chebi"].iloc[0]
    uniprot_row = result[result["ontology"] == "uniprot"].iloc[0]
    assert chebi_row["results"] == 1.0
    assert uniprot_row["results"] == -1.0
    assert chebi_row["s_name"] == "compound1"
    assert uniprot_row["s_name"] == "protein1"
    assert chebi_row["bqb"] == "BQB_IS"
    assert uniprot_row["bqb"] == "BQB_IS"

    # Test with list of ontologies
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies=["chebi", "uniprot"],
    )
    assert len(result) == 2
    assert set(result["ontology"]) == {"chebi", "uniprot"}  # From species_identifiers

    # Test with no matches
    no_match_features = pd.DataFrame(
        {"ontology": ["chebi"], "identifier": ["99999"], "results": [1.0]}
    )
    result = match_by_ontology_and_identifier(
        feature_identifiers=no_match_features,
        species_identifiers=species_identifiers,
        ontologies="chebi",
    )
    assert len(result) == 0

    # Test with empty features
    empty_features = pd.DataFrame({"ontology": [], "identifier": [], "results": []})
    result = match_by_ontology_and_identifier(
        feature_identifiers=empty_features,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"},
    )
    assert len(result) == 0

    # Test with invalid ontology
    with pytest.raises(ValueError, match="Invalid ontologies specified:.*"):
        match_by_ontology_and_identifier(
            feature_identifiers=feature_identifiers,
            species_identifiers=species_identifiers,
            ontologies="invalid_ontology",
        )

    # Test with ontology not in feature_identifiers
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies={"ensembl_gene"},  # Only in species_identifiers
    )
    assert len(result) == 0

    # Test with custom feature_identifiers_var
    feature_identifiers_custom = feature_identifiers.rename(
        columns={"identifier": "custom_id"}
    )
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers_custom,
        species_identifiers=species_identifiers,
        ontologies={"chebi"},
        feature_identifiers_var="custom_id",
    )
    assert len(result) == 1
    assert result.iloc[0]["custom_id"] == "15377"
    assert result.iloc[0]["ontology"] == "chebi"  # From species_identifiers
    assert result.iloc[0]["s_name"] == "compound1"
    assert result.iloc[0]["bqb"] == "BQB_IS"


def test_match_features_to_wide_pathway_species(sbml_dfs_glucose_metabolism):

    def compare_frame_contents(df1, df2):
        """
        Compare if two DataFrames have the same content, ignoring index and column ordering.

        Parameters
        ----------
        df1 : pd.DataFrame
            First DataFrame to compare
        df2 : pd.DataFrame
            Second DataFrame to compare

        Returns
        -------
        None
        """
        df1_sorted = (
            df1.reindex(columns=sorted(df1.columns))
            .sort_values(sorted(df1.columns))
            .reset_index(drop=True)
        )

        df2_sorted = (
            df2.reindex(columns=sorted(df2.columns))
            .sort_values(sorted(df2.columns))
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(df1_sorted, df2_sorted, check_like=True)

        return None

    species_identifiers = (
        sbml_dfs_glucose_metabolism.get_identifiers("species")
        .query("bqb == 'BQB_IS'")
        .query("ontology != 'reactome'")
    )

    # create a table whose index is s_ids and columns are faux-measurements
    example_data = species_identifiers.groupby("ontology").head(10)[
        ["ontology", "identifier"]
    ]

    example_data["results_a"] = np.random.randn(len(example_data))
    example_data["results_b"] = np.random.randn(len(example_data))
    # add a feature_id column to the example_data which tracks the row of the original data
    example_data["feature_id"] = range(0, len(example_data))

    # pivot (identifier, ontology) to columns for each ontology
    example_data_wide = (
        example_data.pivot(
            columns="ontology",
            values="identifier",
            index=["feature_id", "results_a", "results_b"],
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # options, for matching
    # 1. match by identifier and a set of ontologies (provided by arg).
    matched_s_ids = mechanism_matching.features_to_pathway_species(
        feature_identifiers=example_data.drop(columns="ontology"),
        species_identifiers=species_identifiers,
        ontologies={"uniprot", "chebi"},
        feature_identifiers_var="identifier",
    )

    # 2. match by identifier and ontology.
    matched_s_ids_w_ontologies = mechanism_matching.match_by_ontology_and_identifier(
        feature_identifiers=example_data,
        species_identifiers=species_identifiers,
        ontologies={"uniprot", "chebi"},
        feature_identifiers_var="identifier",
    )

    # 3. format wide identifier sets into a table with a single identifier column and apply strategy #2.
    matched_s_ids_from_wide = mechanism_matching.match_features_to_wide_pathway_species(
        example_data_wide,
        species_identifiers,
        ontologies={"uniprot", "chebi"},
        feature_identifiers_var="identifier",
    )

    compare_frame_contents(
        matched_s_ids.drop(columns="s_Source"),
        matched_s_ids_w_ontologies.drop(columns="s_Source"),
    )
    compare_frame_contents(
        matched_s_ids.drop(columns="s_Source"),
        matched_s_ids_from_wide.drop(columns="s_Source"),
    )


def test_resolve_matches_with_example_data():
    """Test resolve_matches function with example data for all aggregation methods."""
    # Setup example data with overlapping 1-to-many and many-to-1 cases
    example_data = pd.DataFrame(
        {
            FEATURE_ID_VAR_DEFAULT: ["A", "B", "C", "D", "D", "E", "B", "B", "C"],
            SBML_DFS.S_ID: [
                "s_id_1",
                "s_id_1",
                "s_id_1",
                "s_id_4",
                "s_id_5",
                "s_id_6",
                "s_id_2",
                "s_id_3",
                "s_id_3",
            ],
            "results_a": [1, 2, 3, 0.4, 5, 6, 0.7, 0.8, 9],
            "results_b": [
                "foo",
                "foo",
                "bar",
                "bar",
                "baz",
                "baz",
                "not",
                "not",
                "not",
            ],
        }
    )

    # Test that missing feature_id raises KeyError
    data_no_id = pd.DataFrame(
        {
            SBML_DFS.S_ID: ["s_id_1", "s_id_1", "s_id_2"],
            "results_a": [1, 2, 3],
            "results_b": ["foo", "bar", "baz"],
        }
    )
    with pytest.raises(KeyError, match=FEATURE_ID_VAR_DEFAULT):
        resolve_matches(data_no_id)

    # Test with keep_id_col=True (default)
    result_with_id = resolve_matches(
        example_data, keep_id_col=True, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MEAN
    )

    # Verify feature_id column is present and correctly aggregated
    assert FEATURE_ID_VAR_DEFAULT in result_with_id.columns
    assert result_with_id.loc["s_id_1", FEATURE_ID_VAR_DEFAULT] == "A,B,C"
    assert result_with_id.loc["s_id_3", FEATURE_ID_VAR_DEFAULT] == "B,C"

    # Test with keep_id_col=False
    result_without_id = resolve_matches(
        example_data, keep_id_col=False, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MEAN
    )

    # Verify feature_id column is not in output
    assert FEATURE_ID_VAR_DEFAULT not in result_without_id.columns

    # Verify other columns are still present and correctly aggregated
    assert "results_a" in result_without_id.columns
    assert "results_b" in result_without_id.columns
    assert "feature_id_match_count" in result_without_id.columns

    # Verify numeric aggregation still works
    actual_mean = result_without_id.loc["s_id_1", "results_a"]
    expected_mean = 2.0  # (1 + 2 + 3) / 3
    assert (
        actual_mean == expected_mean
    ), f"Expected mean {expected_mean}, but got {actual_mean}"

    # Verify string aggregation still works
    assert result_without_id.loc["s_id_1", "results_b"] == "bar,foo"

    # Verify match counts are still present
    assert result_without_id.loc["s_id_1", "feature_id_match_count"] == 3
    assert result_without_id.loc["s_id_3", "feature_id_match_count"] == 2

    # Test maximum aggregation
    max_result = resolve_matches(
        example_data, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MAX
    )

    # Verify maximum values are correct
    assert max_result.loc["s_id_1", "results_a"] == 3.0  # max of [1, 2, 3]
    assert max_result.loc["s_id_3", "results_a"] == 9.0  # max of [0.8, 9]
    assert max_result.loc["s_id_4", "results_a"] == 0.4  # single value
    assert max_result.loc["s_id_5", "results_a"] == 5.0  # single value
    assert max_result.loc["s_id_6", "results_a"] == 6.0  # single value

    # Test weighted mean (feature_id is used for weights regardless of keep_id_col)
    weighted_result = resolve_matches(
        example_data,
        numeric_agg=RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
        keep_id_col=True,
    )

    # For s_id_1:
    # A appears once in total (weight = 1/1)
    # B appears three times in total (weight = 1/3)
    # C appears twice in total (weight = 1/2)
    # Sum of unnormalized weights = 1 + 1/3 + 1/2 = 1.833
    # Normalized weights:
    # A: (1/1)/1.833 = 0.545
    # B: (1/3)/1.833 = 0.182
    # C: (1/2)/1.833 = 0.273
    # Weighted mean = 1×0.545 + 2×0.182 + 3×0.273 = 1.73
    actual_weighted_mean_1 = weighted_result.loc["s_id_1", "results_a"]
    expected_weighted_mean_1 = 1.73
    assert (
        abs(actual_weighted_mean_1 - expected_weighted_mean_1) < 0.01
    ), f"s_id_1 weighted mean: expected {expected_weighted_mean_1:.3f}, but got {actual_weighted_mean_1:.3f}"

    # For s_id_3:
    # B appears three times in total (weight = 1/3)
    # C appears twice in total (weight = 1/2)
    # Sum of unnormalized weights = 1/3 + 1/2 = 0.833
    # Normalized weights:
    # B: (1/3)/0.833 = 0.4
    # C: (1/2)/0.833 = 0.6
    # Weighted mean = 0.8×0.4 + 9×0.6 = 5.72
    actual_weighted_mean_3 = weighted_result.loc["s_id_3", "results_a"]
    expected_weighted_mean_3 = 5.72
    assert (
        abs(actual_weighted_mean_3 - expected_weighted_mean_3) < 0.01
    ), f"s_id_3 weighted mean: expected {expected_weighted_mean_3:.3f}, but got {actual_weighted_mean_3:.3f}"

    # Test weighted mean with keep_id_col=False (weights still use feature_id)
    weighted_result_no_id = resolve_matches(
        example_data,
        numeric_agg=RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
        keep_id_col=False,
    )

    # Verify weighted means are the same regardless of keep_id_col
    assert (
        abs(weighted_result_no_id.loc["s_id_1", "results_a"] - expected_weighted_mean_1)
        < 0.01
    ), "Weighted mean should be the same regardless of keep_id_col"
    assert (
        abs(weighted_result_no_id.loc["s_id_3", "results_a"] - expected_weighted_mean_3)
        < 0.01
    ), "Weighted mean should be the same regardless of keep_id_col"

    # Test that both versions preserve the same index structure
    expected_index = pd.Index(
        ["s_id_1", "s_id_2", "s_id_3", "s_id_4", "s_id_5", "s_id_6"], name="s_id"
    )
    pd.testing.assert_index_equal(result_with_id.index, expected_index)
    pd.testing.assert_index_equal(result_without_id.index, expected_index)


def test_resolve_matches_invalid_dtypes():
    """Test that resolve_matches raises an error for unsupported dtypes."""
    # Setup data with boolean and datetime columns
    data = pd.DataFrame(
        {
            FEATURE_ID_VAR_DEFAULT: ["A", "B", "B", "C"],
            "bool_col": [True, False, True, False],
            "datetime_col": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
            ],
            "s_id": ["s1", "s1", "s2", "s2"],
        }
    )

    # Should raise TypeError for unsupported dtypes
    with pytest.raises(TypeError, match="Unsupported data types"):
        resolve_matches(data)


def test_resolve_matches_first_method():
    """Test resolve_matches with first method."""
    # Setup data with known order
    data = pd.DataFrame(
        {
            FEATURE_ID_VAR_DEFAULT: ["A", "C", "B", "B", "A"],
            SBML_DFS.S_ID: ["s1", "s1", "s1", "s2", "s2"],
            "value": [1, 2, 3, 4, 5],
        }
    )

    result = resolve_matches(data, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.FIRST)

    # Should take first value after sorting by feature_id
    assert result.loc["s1", "value"] == 1  # A comes first
    assert result.loc["s2", "value"] == 5  # A comes first


def test_resolve_matches_deduplicate_feature_id_within_sid():
    """Test that only the first value for each (s_id, feature_id) is used in mean aggregation."""
    data = pd.DataFrame(
        {
            FEATURE_ID_VAR_DEFAULT: ["A", "A", "B"],
            SBML_DFS.S_ID: ["s1", "s1", "s1"],
            "value": [
                1,
                1,
                2,
            ],  # average should be 1.5 because the two A's are redundant
        }
    )

    result = resolve_matches(data, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MEAN)
    assert result.loc["s1", "value"] == 1.5


def test_bind_wide_results(sbml_dfs_glucose_metabolism):
    """
    Test that bind_wide_results correctly matches identifiers and adds results to species data.
    """
    # Get species identifiers, excluding reactome
    species_identifiers = (
        sbml_dfs_glucose_metabolism.get_identifiers(SBML_DFS.SPECIES)
        .query("bqb == 'BQB_IS'")
        .query("ontology != 'reactome'")
    )

    # Create example data with identifiers and results
    example_data = species_identifiers.groupby("ontology").head(10)[
        [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]
    ]
    example_data["results_a"] = np.random.randn(len(example_data))
    example_data["results_b"] = np.random.randn(len(example_data))
    example_data[FEATURE_ID_VAR_DEFAULT] = range(0, len(example_data))

    # Create wide format data
    example_data_wide = (
        example_data.pivot(
            columns=IDENTIFIERS.ONTOLOGY,
            values=IDENTIFIERS.IDENTIFIER,
            index=[FEATURE_ID_VAR_DEFAULT, "results_a", "results_b"],
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # Call bind_wide_results
    results_name = "test_results"
    sbml_dfs_result = mechanism_matching.bind_wide_results(
        sbml_dfs=sbml_dfs_glucose_metabolism,
        results_df=example_data_wide,
        results_name=results_name,
        ontologies={ONTOLOGIES.UNIPROT, ONTOLOGIES.CHEBI},
        dogmatic=False,
        species_identifiers=None,
        feature_id_var=FEATURE_ID_VAR_DEFAULT,
        verbose=True,
    )

    # Verify the results were added correctly
    assert (
        results_name in sbml_dfs_result.species_data
    ), f"{results_name} not found in species_data"

    # Get the bound results
    bound_results = sbml_dfs_result.species_data[results_name]

    # columns are feature_id, results_a, results_b
    assert set(bound_results.columns) == {
        FEATURE_ID_VAR_DEFAULT,
        "results_a",
        "results_b",
    }

    assert bound_results.shape == (23, 3)
    assert bound_results.loc["S00000056", "feature_id"] == "18,19"
    assert bound_results.loc["S00000057", "feature_id"] == "18"
    assert bound_results.loc["S00000010", "feature_id"] == "9"
