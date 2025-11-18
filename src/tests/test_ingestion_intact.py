import numpy as np
import pandas as pd

from napistu.ingestion import intact
from napistu.ingestion.constants import (
    INTACT_SCORES,
    INTACT_TERM_SCORES,
    INTERACTION_EDGELIST_DEFS,
    PSI_MI_DEFS,
    PSI_MI_SCORED_TERMS,
)


def test_miscore_calculation():
    """
    Test MIscore calculation with synthetic input data and expected output.

    Note: This is a unit test to verify implementation correctness against the
    MIscore mathematical formulas. It does NOT validate against published IntAct
    scores due to lack of detailed worked examples showing:
    - Specific interaction evidence (X studies of type Y using method Z)
    - The resulting component scores
    - The final MIscore

    This test uses synthetic data to verify the algorithm implementation matches
    the formulas from Villaveces et al. (2015).
    """

    # Test case 1: Aggregated evidence counts
    counts_df = pd.DataFrame(
        [
            {
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: "protein_A",
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: "protein_B",
                INTACT_SCORES.ATTRIBUTE_TYPE: PSI_MI_DEFS.INTERACTION_TYPE,
                INTACT_SCORES.SCORED_TERM: PSI_MI_SCORED_TERMS.DIRECT_INTERACTION,
                INTACT_SCORES.RAW_SCORE: 1.0,
                "count": 2,
            },
            {
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: "protein_A",
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: "protein_B",
                INTACT_SCORES.ATTRIBUTE_TYPE: PSI_MI_DEFS.INTERACTION_METHOD,
                INTACT_SCORES.SCORED_TERM: PSI_MI_SCORED_TERMS.BIOCHEMICAL,
                INTACT_SCORES.RAW_SCORE: 1.0,
                "count": 2,
            },
        ]
    )

    # Publication counts (as Series with MultiIndex)
    n_publications = pd.Series(
        [2],
        index=pd.MultiIndex.from_tuples(
            [("protein_A", "protein_B")],
            names=[
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ],
        ),
        name=INTACT_SCORES.N_PUBLICATIONS,
    )

    # Calculate scores using the implementation
    result_1 = intact._calculate_all_scores_vectorized(counts_df, n_publications)

    # Hand-calculated expected values based on MIscore formulas:
    # Publication score: n=2 publications, max=7 -> log(3)/log(8) ≈ 0.528
    expected_pub_score = np.log(3) / np.log(8)

    # Method score: a=2*1.0=2, b=2+2=4 -> log(3)/log(5) ≈ 0.683
    expected_method_score = np.log(3) / np.log(5)

    # Type score: Same calculation as method score ≈ 0.683
    expected_type_score = np.log(3) / np.log(5)

    # Final MIscore: weighted average with equal weights (1.0 each)
    expected_miscore = (
        expected_pub_score + expected_method_score + expected_type_score
    ) / 3

    # Assertions with tolerance for floating point precision
    assert (
        abs(result_1[INTACT_SCORES.PUBLICATION_SCORE].iloc[0] - expected_pub_score)
        < 0.001
    ), f"Publication score mismatch: {result_1[INTACT_SCORES.PUBLICATION_SCORE].iloc[0]} vs {expected_pub_score}"

    assert (
        abs(
            result_1[INTACT_SCORES.INTERACTION_METHOD_SCORE].iloc[0]
            - expected_method_score
        )
        < 0.001
    ), f"Method score mismatch: {result_1[INTACT_SCORES.INTERACTION_METHOD_SCORE].iloc[0]} vs {expected_method_score}"

    assert (
        abs(
            result_1[INTACT_SCORES.INTERACTION_TYPE_SCORE].iloc[0] - expected_type_score
        )
        < 0.001
    ), f"Type score mismatch: {result_1[INTACT_SCORES.INTERACTION_TYPE_SCORE].iloc[0]} vs {expected_type_score}"

    assert (
        abs(result_1[INTACT_SCORES.MI_SCORE].iloc[0] - expected_miscore) < 0.001
    ), f"MIscore mismatch: {result_1[INTACT_SCORES.MI_SCORE].iloc[0]} vs {expected_miscore}"

    # Test case 2: Single publication, lower-scoring evidence (edge case)
    counts_df_2 = pd.DataFrame(
        [
            {
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: "protein_C",
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: "protein_D",
                INTACT_SCORES.ATTRIBUTE_TYPE: PSI_MI_DEFS.INTERACTION_TYPE,
                INTACT_SCORES.SCORED_TERM: PSI_MI_SCORED_TERMS.ASSOCIATION,
                INTACT_SCORES.RAW_SCORE: 0.33,
                "count": 1,
            },
            {
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: "protein_C",
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: "protein_D",
                INTACT_SCORES.ATTRIBUTE_TYPE: PSI_MI_DEFS.INTERACTION_METHOD,
                INTACT_SCORES.SCORED_TERM: PSI_MI_SCORED_TERMS.GENETIC_INTERFERENCE,
                INTACT_SCORES.RAW_SCORE: 0.10,
                "count": 1,
            },
        ]
    )

    n_publications_2 = pd.Series(
        [1],
        index=pd.MultiIndex.from_tuples(
            [("protein_C", "protein_D")],
            names=[
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ],
        ),
        name=INTACT_SCORES.N_PUBLICATIONS,
    )

    result_2 = intact._calculate_all_scores_vectorized(counts_df_2, n_publications_2)

    # All scores should be between 0 and 1
    assert 0 <= result_2[INTACT_SCORES.PUBLICATION_SCORE].iloc[0] <= 1
    assert 0 <= result_2[INTACT_SCORES.INTERACTION_METHOD_SCORE].iloc[0] <= 1
    assert 0 <= result_2[INTACT_SCORES.INTERACTION_TYPE_SCORE].iloc[0] <= 1
    assert 0 <= result_2[INTACT_SCORES.MI_SCORE].iloc[0] <= 1


def test_ontology_term_scoring():
    """
    Test ontology parsing and term scoring with spot checks.

    Tests the complete pipeline from ontology parsing to term score assignment,
    including explicit scores, inherited scores, and fallback to unknown.
    """

    # Build ontology graph and term lookup
    psi_mi_ontology_graph = intact._build_psi_mi_ontology_graph()
    intact_term_lookup = intact._get_intact_term_with_score(
        psi_mi_ontology_graph, INTACT_TERM_SCORES
    )

    # Test case 1: Term with explicit score
    explicit_term = PSI_MI_SCORED_TERMS.DIRECT_INTERACTION
    explicit_result = intact._get_intact_scored_term(explicit_term, intact_term_lookup)
    assert (
        explicit_result == PSI_MI_SCORED_TERMS.DIRECT_INTERACTION
    ), f"Explicit term should map to itself: {explicit_result}"

    # Test case 2: Term that inherits score by propagation
    # (This assumes there's a child term of a scored parent in the ontology)
    inherited_term = "enzymatic reaction"  # Should inherit from "direct interaction"
    inherited_result = intact._get_intact_scored_term(
        inherited_term, intact_term_lookup
    )
    assert (
        inherited_result == PSI_MI_SCORED_TERMS.DIRECT_INTERACTION
    ), f"Child term should inherit parent score: {inherited_result}"

    # Test case 3: Missing/unknown term
    missing_term = "foo"
    missing_result = intact._get_intact_scored_term(missing_term, intact_term_lookup)
    assert (
        missing_result == PSI_MI_SCORED_TERMS.UNKNOWN
    ), f"Missing term should default to unknown: {missing_result}"

    # Test case 4: Verify all scored terms are present in ontology
    # Get all scored terms except "unknown"
    expected_scored_terms = [
        term
        for term in PSI_MI_SCORED_TERMS.__dict__.values()
        if term != PSI_MI_SCORED_TERMS.UNKNOWN
    ]

    missing_terms = []
    for term in expected_scored_terms:
        if term not in intact_term_lookup:
            missing_terms.append(term)

    assert (
        len(missing_terms) == 0
    ), f"The following scored terms are missing from the ontology: {missing_terms}"

    # Test case 5: Verify all scored terms map to themselves (explicit scores)
    for term in expected_scored_terms:
        result = intact._get_intact_scored_term(term, intact_term_lookup)
        assert (
            result == term
        ), f"Scored term '{term}' should map to itself, but maps to '{result}'"

    # Additional validation: Check that lookup contains expected entries
    assert len(intact_term_lookup) > 0, "Term lookup should not be empty"

    # Test that all lookup values are either scored terms or "unknown"
    valid_values = set(INTACT_TERM_SCORES.keys()) | {PSI_MI_SCORED_TERMS.UNKNOWN}
    invalid_mappings = []
    for term, mapped_term in intact_term_lookup.items():
        if mapped_term not in valid_values:
            invalid_mappings.append((term, mapped_term))

    assert (
        len(invalid_mappings) == 0
    ), f"Found invalid term mappings: {invalid_mappings[:5]}..."  # Show first 5 if any
