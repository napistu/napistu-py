from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napistu import mechanism_matching
from napistu.network import net_create
from napistu.network import precompute
from napistu.mechanism_matching import _validate_wide_ontologies
from napistu.mechanism_matching import match_by_ontology_and_identifier

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
    species_identifiers = pd.DataFrame({
        "ontology": ["chebi", "chebi", "uniprot", "uniprot"],
        "identifier": ["A", "B", "X", "Y"],
        "s_id": [1, 2, 3, 4],
        "s_name": ["foo", "bar", "baz", "qux"],
        "bqb": ["BQB_IS", "BQB_IS", "BQB_IS", "BQB_IS"]
    })
    # Basic: no expansion, single identifier per row
    features = pd.DataFrame({
        "my_id": ["A", "B", "X"],
        "other_col": [10, 20, 30]
    })
    result = mechanism_matching.features_to_pathway_species(
        feature_identifiers=features,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"},
        feature_identifiers_var="my_id",
        expand_identifiers=False
    )
    # Should map all three
    assert set(result["my_id"]) == {"A", "B", "X"}
    assert set(result["identifier"]) == {"A", "B", "X"}
    assert set(result["s_name"]) == {"foo", "bar", "baz"}
    # Expansion: one row with multiple IDs
    features2 = pd.DataFrame({
        "my_id": ["A / B / X", "Y"],
        "other_col": [100, 200]
    })
    result2 = mechanism_matching.features_to_pathway_species(
        feature_identifiers=features2,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"},
        feature_identifiers_var="my_id",
        expand_identifiers=True,
        identifier_delimiter="/"
    )
    # Should expand to 4 rows (A, B, X, Y)
    assert set(result2["identifier"]) == {"A", "B", "X", "Y"}
    assert set(result2["s_name"]) == {"foo", "bar", "baz", "qux"}
    # Whitespace trimming
    features3 = pd.DataFrame({
        "my_id": ["  A  /  B  /X  ", " Y"],
        "other_col": [1, 2]
    })
    result3 = mechanism_matching.features_to_pathway_species(
        feature_identifiers=features3,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"},
        feature_identifiers_var="my_id",
        expand_identifiers=True,
        identifier_delimiter="/"
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
    example_data_wide = pd.DataFrame({
        'results': [-1.0, 0.0, 1.0],
        'chebi': ['15377', '16810', '17925'],
        'uniprot': ['P12345', 'Q67890', 'O43826']
    })

    # Test auto-detection of ontology columns
    assert _validate_wide_ontologies(example_data_wide) == {"chebi", "uniprot"}

    # Test string input
    assert _validate_wide_ontologies(example_data_wide, ontologies="chebi") == {"chebi"}

    # Test set input
    assert _validate_wide_ontologies(example_data_wide, ontologies={"chebi"}) == {"chebi"}
    assert _validate_wide_ontologies(example_data_wide, ontologies={"chebi", "uniprot"}) == {"chebi", "uniprot"}

    # Test dictionary mapping for renaming
    assert _validate_wide_ontologies(
        example_data_wide, 
        ontologies={"chebi": "reactome", "uniprot": "ensembl_gene"}
    ) == {"reactome", "ensembl_gene"}

    # Test error cases
    
    # Missing column in set input (checks existence first)
    with pytest.raises(ValueError, match="Specified ontology columns not found in DataFrame:.*"):
        _validate_wide_ontologies(example_data_wide, ontologies={"invalid_ontology"})

    # Valid column name but invalid ontology
    df_with_invalid = pd.DataFrame({
        'results': [-1.0, 0.0, 1.0],
        'invalid_ontology': ['a', 'b', 'c'],
    })
    with pytest.raises(ValueError, match="Invalid ontologies in set:.*"):
        _validate_wide_ontologies(df_with_invalid, ontologies={"invalid_ontology"})

    # Missing source column in mapping
    with pytest.raises(ValueError, match="Source columns not found in DataFrame:.*"):
        _validate_wide_ontologies(example_data_wide, ontologies={"missing_column": "reactome"})

    # Invalid target ontology in mapping
    with pytest.raises(ValueError, match="Invalid ontologies in mapping:.*"):
        _validate_wide_ontologies(example_data_wide, ontologies={"chebi": "invalid_ontology"})

    # DataFrame with no valid ontology columns
    invalid_df = pd.DataFrame({
        'results': [-1.0, 0.0, 1.0],
        'col1': ['a', 'b', 'c'],
        'col2': ['d', 'e', 'f']
    })
    with pytest.raises(ValueError, match="No valid ontology columns found in DataFrame.*"):
        _validate_wide_ontologies(invalid_df)

def test_match_by_ontology_and_identifier():
    """Test the match_by_ontology_and_identifier function with various input types."""
    # Setup test data
    feature_identifiers = pd.DataFrame({
        "ontology": ["chebi", "chebi", "uniprot", "uniprot", "reactome"],
        "identifier": ["15377", "16810", "P12345", "Q67890", "R12345"],
        "results": [1.0, 2.0, -1.0, -2.0, 0.5]
    })

    species_identifiers = pd.DataFrame({
        "ontology": ["chebi", "chebi", "uniprot", "uniprot", "ensembl_gene"],
        "identifier": ["15377", "17925", "P12345", "O43826", "ENSG123"],
        "s_id": ["s1", "s2", "s3", "s4", "s5"],
        "s_name": ["compound1", "compound2", "protein1", "protein2", "gene1"],
        "bqb": ["BQB_IS"] * 5  # Add required bqb column with BQB_IS values
    })

    # Test with single ontology (string)
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies="chebi"
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
        ontologies={"chebi", "uniprot"}
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
        ontologies=["chebi", "uniprot"]
    )
    assert len(result) == 2
    assert set(result["ontology"]) == {"chebi", "uniprot"}  # From species_identifiers

    # Test with no matches
    no_match_features = pd.DataFrame({
        "ontology": ["chebi"],
        "identifier": ["99999"],
        "results": [1.0]
    })
    result = match_by_ontology_and_identifier(
        feature_identifiers=no_match_features,
        species_identifiers=species_identifiers,
        ontologies="chebi"
    )
    assert len(result) == 0

    # Test with empty features
    empty_features = pd.DataFrame({
        "ontology": [],
        "identifier": [],
        "results": []
    })
    result = match_by_ontology_and_identifier(
        feature_identifiers=empty_features,
        species_identifiers=species_identifiers,
        ontologies={"chebi", "uniprot"}
    )
    assert len(result) == 0

    # Test with invalid ontology
    with pytest.raises(ValueError, match="Invalid ontologies specified:.*"):
        match_by_ontology_and_identifier(
            feature_identifiers=feature_identifiers,
            species_identifiers=species_identifiers,
            ontologies="invalid_ontology"
        )

    # Test with ontology not in feature_identifiers
    result = match_by_ontology_and_identifier(
        feature_identifiers=feature_identifiers,
        species_identifiers=species_identifiers,
        ontologies={"ensembl_gene"}  # Only in species_identifiers
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
        feature_identifiers_var="custom_id"
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
        df1_sorted = (df1
                        .reindex(columns=sorted(df1.columns))
                        .sort_values(sorted(df1.columns))
                        .reset_index(drop=True))
        
        df2_sorted = (df2
                        .reindex(columns=sorted(df2.columns))
                        .sort_values(sorted(df2.columns))
                        .reset_index(drop=True))
        
        pd.testing.assert_frame_equal(df1_sorted, df2_sorted, check_like=True)

        return None

    species_identifiers = sbml_dfs_glucose_metabolism.get_identifiers("species").query("bqb == 'BQB_IS'").query("ontology != 'reactome'")

    # create a table whose index is s_ids and columns are faux-measurements
    example_data = species_identifiers.groupby("ontology").head(10)[["ontology", "identifier"]]

    example_data["results_a"] = np.random.randn(len(example_data))
    example_data["results_b"] = np.random.randn(len(example_data))
    # add a feature_id column to the example_data which tracks the row of the original data
    example_data["feature_id"] = range(0, len(example_data))

    # pivot (identifier, ontology) to columns for each ontology
    example_data_wide = example_data.pivot(columns = "ontology", values = "identifier", index = ["feature_id", "results_a", "results_b"]).reset_index().rename_axis(None, axis = 1)

    # options, for matching
    # 1. match by identifier and a set of ontologies (provided by arg).
    matched_s_ids = mechanism_matching.features_to_pathway_species(
        feature_identifiers = example_data.drop(columns = "ontology"), 
        species_identifiers = species_identifiers,
        ontologies = {"uniprot", "chebi"},
        feature_identifiers_var = "identifier",
    )

    # 2. match by identifier and ontology.
    matched_s_ids_w_ontologies = mechanism_matching.match_by_ontology_and_identifier(
        feature_identifiers = example_data, 
        species_identifiers = species_identifiers,
        ontologies = {"uniprot", "chebi"},
        feature_identifiers_var = "identifier",
    )

    # 3. format wide identifier sets into a table with a single identifier column and apply strategy #2.
    matched_s_ids_from_wide = mechanism_matching.match_features_to_wide_pathway_species(
        example_data_wide,
        species_identifiers,
        ontologies = {"uniprot", "chebi"},
        feature_identifiers_var = "identifier",
    )

    compare_frame_contents(matched_s_ids.drop(columns = "s_Source"), matched_s_ids_w_ontologies.drop(columns = "s_Source"))
    compare_frame_contents(matched_s_ids.drop(columns = "s_Source"), matched_s_ids_from_wide.drop(columns = "s_Source"))
