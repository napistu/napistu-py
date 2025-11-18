from __future__ import annotations

import pandas as pd
import pytest

from napistu import sbml_dfs_utils
from napistu.constants import (
    BQB,
    BQB_DEFINING_ATTRS,
    BQB_DEFINING_ATTRS_LOOSE,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    MINI_SBO_TO_NAME,
    ONTOLOGIES,
    POLARITIES,
    POLARITY_TO_SYMBOL,
    SBML_DFS,
    SBOTERM_NAMES,
    VALID_SBO_TERM_NAMES,
    VALID_SBO_TERMS,
)
from napistu.identifiers import Identifiers
from napistu.ingestion.constants import (
    COMPARTMENTS,
    COMPARTMENTS_GO_TERMS,
    GENERIC_COMPARTMENT,
    INTERACTION_EDGELIST_DEFAULTS,
    INTERACTION_EDGELIST_DEFS,
    INTERACTION_EDGELIST_OPTIONAL_VARS,
)
from napistu.ontologies.constants import SPECIES_TYPES


def test_id_formatter():
    input_vals = range(50, 100)

    # create standard IDs
    ids = sbml_dfs_utils.id_formatter(input_vals, "s_id", id_len=8)
    # invert standard IDs
    inv_ids = sbml_dfs_utils.id_formatter_inv(ids)

    assert list(input_vals) == inv_ids


def test_filter_to_characteristic_species_ids():

    species_ids_dict = {
        SBML_DFS.S_ID: ["large_complex"] * 6
        + ["small_complex"] * 2
        + ["proteinA", "proteinB"]
        + ["proteinC"] * 3
        + [
            "promiscuous_complexA",
            "promiscuous_complexB",
            "promiscuous_complexC",
            "promiscuous_complexD",
            "promiscuous_complexE",
        ],
        IDENTIFIERS.ONTOLOGY: ["complexportal"]
        + ["HGNC"] * 7
        + ["GO"] * 2
        + ["ENSG", "ENSP", "pubmed"]
        + ["HGNC"] * 5,
        IDENTIFIERS.IDENTIFIER: [
            "CPX-BIG",
            "mem1",
            "mem2",
            "mem3",
            "mem4",
            "mem5",
            "part1",
            "part2",
            "GO:1",
            "GO:2",
            "dna_seq",
            "protein_seq",
            "my_cool_pub",
        ]
        + ["promiscuous_complex"] * 5,
        IDENTIFIERS.BQB: [BQB.IS]
        + [BQB.HAS_PART] * 7
        + [BQB.IS] * 2
        + [
            # these are retained if BQB_DEFINING_ATTRS_LOOSE is used
            BQB.ENCODES,
            BQB.IS_ENCODED_BY,
            # this should always be removed
            BQB.IS_DESCRIBED_BY,
        ]
        + [BQB.HAS_PART] * 5,
    }

    species_ids = pd.DataFrame(species_ids_dict)

    characteristic_ids_narrow = sbml_dfs_utils.filter_to_characteristic_species_ids(
        species_ids,
        defining_biological_qualifiers=BQB_DEFINING_ATTRS,
        max_complex_size=4,
        max_promiscuity=4,
    )

    EXPECTED_IDS = ["CPX-BIG", "GO:1", "GO:2", "part1", "part2"]
    assert characteristic_ids_narrow[IDENTIFIERS.IDENTIFIER].tolist() == EXPECTED_IDS

    characteristic_ids_loose = sbml_dfs_utils.filter_to_characteristic_species_ids(
        species_ids,
        # include encodes and is_encoded_by as equivalent to is
        defining_biological_qualifiers=BQB_DEFINING_ATTRS_LOOSE,
        max_complex_size=4,
        # expand promiscuity to default value
        max_promiscuity=20,
    )

    EXPECTED_IDS = [
        "CPX-BIG",
        "GO:1",
        "GO:2",
        "dna_seq",
        "protein_seq",
        "part1",
        "part2",
    ] + ["promiscuous_complex"] * 5
    assert characteristic_ids_loose[IDENTIFIERS.IDENTIFIER].tolist() == EXPECTED_IDS


def test_formula(sbml_dfs):
    # create a formula string

    an_r_id = sbml_dfs.reactions.index[0]

    reaction_species_df = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species[SBML_DFS.R_ID] == an_r_id
    ].merge(
        sbml_dfs.compartmentalized_species, left_on=SBML_DFS.SC_ID, right_index=True
    )

    formula_str = sbml_dfs_utils.construct_formula_string(
        reaction_species_df, sbml_dfs.reactions, name_var=SBML_DFS.SC_NAME
    )

    assert isinstance(formula_str, str)
    assert (
        formula_str
        == "CO2 [extracellular region] -> CO2 [cytosol] ---- modifiers: AQP1 tetramer [plasma membrane]]"
    )


def test_find_underspecified_reactions():

    reaction_w_regulators = pd.DataFrame(
        {
            SBML_DFS.SC_ID: ["A", "B", "C", "D", "E", "F", "G"],
            SBML_DFS.STOICHIOMETRY: [-1, -1, 1, 1, 0, 0, 0],
            SBML_DFS.SBO_TERM: [
                SBOTERM_NAMES.REACTANT,
                SBOTERM_NAMES.REACTANT,
                SBOTERM_NAMES.PRODUCT,
                SBOTERM_NAMES.PRODUCT,
                SBOTERM_NAMES.CATALYST,
                SBOTERM_NAMES.CATALYST,
                SBOTERM_NAMES.STIMULATOR,
            ],
        }
    ).assign(r_id="bar")
    reaction_w_regulators[SBML_DFS.RSC_ID] = [
        f"rsc_{i}" for i in range(len(reaction_w_regulators))
    ]
    reaction_w_regulators.set_index(SBML_DFS.RSC_ID, inplace=True)
    reaction_w_regulators = sbml_dfs_utils.add_sbo_role(reaction_w_regulators)

    reaction_w_interactors = pd.DataFrame(
        {
            SBML_DFS.SC_ID: ["A", "B"],
            SBML_DFS.STOICHIOMETRY: [-1, 1],
            SBML_DFS.SBO_TERM: [SBOTERM_NAMES.REACTANT, SBOTERM_NAMES.REACTANT],
        }
    ).assign(r_id="baz")
    reaction_w_interactors[SBML_DFS.RSC_ID] = [
        f"rsc_{i}" for i in range(len(reaction_w_interactors))
    ]
    reaction_w_interactors.set_index(SBML_DFS.RSC_ID, inplace=True)
    reaction_w_interactors = sbml_dfs_utils.add_sbo_role(reaction_w_interactors)

    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_0", "new"] = False
    working_reactions
    result = sbml_dfs_utils._find_underspecified_reactions(working_reactions)
    assert result == {"bar"}

    # missing one enzyme -> operable
    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_4", "new"] = False
    working_reactions
    result = sbml_dfs_utils._find_underspecified_reactions(working_reactions)
    assert result == set()

    # missing one product -> inoperable
    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_2", "new"] = False
    working_reactions
    result = sbml_dfs_utils._find_underspecified_reactions(working_reactions)
    assert result == {"bar"}

    # missing all enzymes -> inoperable
    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_4", "new"] = False
    working_reactions.loc["rsc_5", "new"] = False
    working_reactions
    result = sbml_dfs_utils._find_underspecified_reactions(working_reactions)
    assert result == {"bar"}

    # missing regulators -> operable
    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_6", "new"] = False
    working_reactions
    result = sbml_dfs_utils._find_underspecified_reactions(working_reactions)
    assert result == set()

    # remove an interactor
    working_reactions = reaction_w_interactors.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_0", "new"] = False
    working_reactions
    result = sbml_dfs_utils._find_underspecified_reactions(working_reactions)
    assert result == {"baz"}


def test_stubbed_compartment():
    compartment = sbml_dfs_utils.stub_compartments()

    compartment_ids = compartment[SBML_DFS.C_IDENTIFIERS].iloc[0].df
    expected_ids = pd.DataFrame(
        [
            {
                IDENTIFIERS.ONTOLOGY: ONTOLOGIES.GO,
                IDENTIFIERS.IDENTIFIER: COMPARTMENTS_GO_TERMS[GENERIC_COMPARTMENT],
                IDENTIFIERS.BQB: BQB.IS,
                IDENTIFIERS.URL: "https://www.ebi.ac.uk/QuickGO/term/GO:0005575",
            }
        ]
    )

    pd.testing.assert_frame_equal(compartment_ids, expected_ids, check_dtype=False)


def test_validate_sbo_values_success():
    # Should not raise
    sbml_dfs_utils._validate_sbo_values(pd.Series(VALID_SBO_TERMS), validate="terms")
    sbml_dfs_utils._validate_sbo_values(
        pd.Series(VALID_SBO_TERM_NAMES), validate="names"
    )


def test_validate_sbo_values_invalid_type():
    with pytest.raises(ValueError, match="Invalid validation type"):
        sbml_dfs_utils._validate_sbo_values(
            pd.Series(VALID_SBO_TERMS), validate="badtype"
        )


def test_validate_sbo_values_invalid_value():
    # Add an invalid term
    s = pd.Series(VALID_SBO_TERMS + ["SBO:9999999"])
    with pytest.raises(ValueError, match="unusable SBO terms"):
        sbml_dfs_utils._validate_sbo_values(s, validate="terms")
    # Add an invalid name
    s = pd.Series(VALID_SBO_TERM_NAMES + ["not_a_name"])
    with pytest.raises(ValueError, match="unusable SBO terms"):
        sbml_dfs_utils._validate_sbo_values(s, validate="names")


def test_sbo_constants_internal_consistency():
    # Every term should have a name and vice versa
    # MINI_SBO_FROM_NAME: name -> term, MINI_SBO_TO_NAME: term -> name
    terms_from_names = set(MINI_SBO_FROM_NAME.values())
    names_from_terms = set(MINI_SBO_TO_NAME.values())
    assert terms_from_names == set(VALID_SBO_TERMS)
    assert names_from_terms == set(VALID_SBO_TERM_NAMES)
    # Bijective mapping
    for name, term in MINI_SBO_FROM_NAME.items():
        assert MINI_SBO_TO_NAME[term] == name
    for term, name in MINI_SBO_TO_NAME.items():
        assert MINI_SBO_FROM_NAME[name] == term


def test_get_interaction_symbol():

    # Test SBO names (strings)
    assert (
        sbml_dfs_utils._get_interaction_symbol(SBOTERM_NAMES.CATALYST)
        == POLARITY_TO_SYMBOL[POLARITIES.ACTIVATION]
    )
    assert (
        sbml_dfs_utils._get_interaction_symbol(SBOTERM_NAMES.INHIBITOR)
        == POLARITY_TO_SYMBOL[POLARITIES.INHIBITION]
    )
    assert (
        sbml_dfs_utils._get_interaction_symbol(SBOTERM_NAMES.INTERACTOR)
        == POLARITY_TO_SYMBOL[POLARITIES.AMBIGUOUS]
    )
    assert (
        sbml_dfs_utils._get_interaction_symbol(SBOTERM_NAMES.PRODUCT)
        == POLARITY_TO_SYMBOL[POLARITIES.ACTIVATION]
    )
    assert (
        sbml_dfs_utils._get_interaction_symbol(SBOTERM_NAMES.REACTANT)
        == POLARITY_TO_SYMBOL[POLARITIES.ACTIVATION]
    )
    assert (
        sbml_dfs_utils._get_interaction_symbol(SBOTERM_NAMES.STIMULATOR)
        == POLARITY_TO_SYMBOL[POLARITIES.ACTIVATION]
    )

    # Test SBO terms (SBO:0000xxx format)
    assert (
        sbml_dfs_utils._get_interaction_symbol(
            MINI_SBO_FROM_NAME[SBOTERM_NAMES.CATALYST]
        )
        == POLARITY_TO_SYMBOL[POLARITIES.ACTIVATION]
    )
    assert (
        sbml_dfs_utils._get_interaction_symbol(
            MINI_SBO_FROM_NAME[SBOTERM_NAMES.INHIBITOR]
        )
        == POLARITY_TO_SYMBOL[POLARITIES.INHIBITION]
    )
    assert (
        sbml_dfs_utils._get_interaction_symbol(
            MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR]
        )
        == POLARITY_TO_SYMBOL[POLARITIES.AMBIGUOUS]
    )

    # Test invalid SBO term
    with pytest.raises(ValueError, match="Invalid SBO term"):
        sbml_dfs_utils._get_interaction_symbol("invalid_sbo_term")


def test_add_edgelist_defaults():

    # Test 1: No defaults needed - all optional columns present
    complete_edgelist = pd.DataFrame(
        {
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: ["A", "B"],
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: ["B", "C"],
            SBML_DFS.R_NAME: ["A->B", "B->C"],
            SBML_DFS.R_IDENTIFIERS: [[], []],
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: [
                COMPARTMENTS.CYTOPLASM,
                COMPARTMENTS.CYTOPLASM,
            ],
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: [
                COMPARTMENTS.CYTOPLASM,
                COMPARTMENTS.CYTOPLASM,
            ],
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: [
                SBOTERM_NAMES.CATALYST,
                SBOTERM_NAMES.CATALYST,
            ],
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: [
                SBOTERM_NAMES.PRODUCT,
                SBOTERM_NAMES.PRODUCT,
            ],
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM: [1, 1],
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM: [1, 1],
            SBML_DFS.R_ISREVERSIBLE: [False, False],
        }
    )

    result = sbml_dfs_utils._add_edgelist_defaults(complete_edgelist)
    assert result.equals(
        complete_edgelist
    ), "Should return unchanged DataFrame when all columns present"

    # Test 2: Missing optional columns - should add defaults
    incomplete_edgelist = pd.DataFrame(
        {
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: ["A", "B"],
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: ["B", "C"],
            SBML_DFS.R_NAME: ["A->B", "B->C"],
            SBML_DFS.R_IDENTIFIERS: [[], []],
        }
    )

    result = sbml_dfs_utils._add_edgelist_defaults(incomplete_edgelist)

    # Check that all optional columns were added
    for col in INTERACTION_EDGELIST_OPTIONAL_VARS:
        assert col in result.columns, f"Column {col} should be added"
        assert all(
            result[col] == INTERACTION_EDGELIST_DEFAULTS[col]
        ), f"Column {col} should have default values"

    # Test 3: Some columns present but with NaN values - should replace NaN with defaults
    edgelist_with_nan = pd.DataFrame(
        {
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: ["A", "B"],
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: ["B", "C"],
            SBML_DFS.R_NAME: ["A->B", "B->C"],
            SBML_DFS.R_IDENTIFIERS: [[], []],
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: [
                COMPARTMENTS.CYTOPLASM,
                None,
            ],
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: [
                None,
                COMPARTMENTS.CYTOPLASM,
            ],
            SBML_DFS.R_ISREVERSIBLE: [False, None],
        }
    )

    result = sbml_dfs_utils._add_edgelist_defaults(edgelist_with_nan)

    # Check that NaN values were replaced with defaults
    assert (
        result[INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM].iloc[1]
        == INTERACTION_EDGELIST_DEFAULTS[INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM]
    )
    assert (
        result[INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM].iloc[0]
        == INTERACTION_EDGELIST_DEFAULTS[
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM
        ]
    )
    assert (
        result[SBML_DFS.R_ISREVERSIBLE].iloc[1]
        == INTERACTION_EDGELIST_DEFAULTS[SBML_DFS.R_ISREVERSIBLE]
    )

    # Test 4: Custom defaults dictionary
    custom_defaults = {
        INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: COMPARTMENTS.NUCLEUS,
        INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: COMPARTMENTS.NUCLEUS,
        INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: SBOTERM_NAMES.CATALYST,
        INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: SBOTERM_NAMES.PRODUCT,
        INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM: 1,
        INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM: 1,
        SBML_DFS.R_ISREVERSIBLE: True,
    }

    minimal_edgelist = pd.DataFrame(
        {
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: ["A"],
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: ["B"],
            SBML_DFS.R_NAME: ["A->B"],
            SBML_DFS.R_IDENTIFIERS: [[]],
        }
    )

    result = sbml_dfs_utils._add_edgelist_defaults(minimal_edgelist, custom_defaults)

    # Check that custom defaults were applied
    assert (
        result[INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM].iloc[0]
        == COMPARTMENTS.NUCLEUS
    )
    assert (
        result[INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM].iloc[0]
        == COMPARTMENTS.NUCLEUS
    )
    assert result[SBML_DFS.R_ISREVERSIBLE].iloc[0]

    # Test 5: Missing defaults for required optional columns should raise ValueError
    incomplete_defaults = {
        INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: COMPARTMENTS.CYTOPLASM,
        # Missing other required defaults
    }

    # The function now uses the default INTERACTION_EDGELIST_DEFAULTS when incomplete defaults are provided
    # So this should work without raising an error
    result = sbml_dfs_utils._add_edgelist_defaults(
        minimal_edgelist, edgelist_defaults=incomplete_defaults
    )

    # Check that the missing defaults were filled from INTERACTION_EDGELIST_DEFAULTS
    assert (
        result[INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM].iloc[0]
        == INTERACTION_EDGELIST_DEFAULTS[
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM
        ]
    )
    assert (
        result[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM].iloc[0]
        == INTERACTION_EDGELIST_DEFAULTS[
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM
        ]
    )


def test_construct_formula_string():
    """Test construct_formula_string with various scenarios."""

    # Test pure interactor reaction (A ---- B format)
    reaction_species_df = pd.DataFrame(
        {
            SBML_DFS.R_ID: ["R001", "R001"],
            SBML_DFS.SC_ID: ["SC001", "SC002"],
            SBML_DFS.STOICHIOMETRY: [0, 0],
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR],
            ],
            "sc_name": ["Protein A", "Protein B"],
            SBML_DFS.RSC_ID: ["RSC001", "RSC002"],
        }
    )
    reactions_df = pd.DataFrame({SBML_DFS.R_ISREVERSIBLE: [False]}, index=["R001"])

    formula_str = sbml_dfs_utils.construct_formula_string(
        reaction_species_df, reactions_df, name_var="sc_name"
    )
    assert formula_str == "Protein A ---- Protein B"

    # Test mixed reaction with modifiers
    reaction_species_df = pd.DataFrame(
        {
            SBML_DFS.R_ID: ["R002", "R002", "R002", "R002"],
            SBML_DFS.SC_ID: ["SC003", "SC004", "SC005", "SC006"],
            SBML_DFS.STOICHIOMETRY: [-1, -1, 1, 0],
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.CATALYST],
            ],
            "sc_name": ["Substrate1", "Substrate2", "Product1", "Catalyst"],
            SBML_DFS.RSC_ID: ["RSC003", "RSC004", "RSC005", "RSC006"],
        }
    )
    reactions_df = pd.DataFrame({SBML_DFS.R_ISREVERSIBLE: [False]}, index=["R002"])

    formula_str = sbml_dfs_utils.construct_formula_string(
        reaction_species_df, reactions_df, name_var="sc_name"
    )
    assert (
        "Substrate1 + Substrate2 -> Product1 ---- modifiers: Catalyst]" in formula_str
    )

    # Test too many interactors (should return None)
    reaction_species_df = pd.DataFrame(
        {
            SBML_DFS.R_ID: ["R003", "R003", "R003"],
            SBML_DFS.SC_ID: ["SC007", "SC008", "SC009"],
            SBML_DFS.STOICHIOMETRY: [0, 0, 0],
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR],
            ],
            "sc_name": ["Protein A", "Protein B", "Protein C"],
            SBML_DFS.RSC_ID: ["RSC007", "RSC008", "RSC009"],
        }
    )
    reactions_df = pd.DataFrame({SBML_DFS.R_ISREVERSIBLE: [False]}, index=["R003"])

    formula_str = sbml_dfs_utils.construct_formula_string(
        reaction_species_df, reactions_df, name_var="sc_name"
    )
    assert formula_str is None


def test_reaction_formulas(sbml_dfs):
    """Test reaction_formulas method with various inputs."""
    # Test single reaction
    first_r_id = sbml_dfs.reactions.index[0]
    formulas = sbml_dfs.reaction_formulas(r_ids=first_r_id)
    assert isinstance(formulas, pd.Series)
    assert len(formulas) == 1
    assert first_r_id in formulas.index

    # Test multiple reactions
    r_ids = sbml_dfs.reactions.index[:2].tolist()
    formulas = sbml_dfs.reaction_formulas(r_ids=r_ids)
    assert len(formulas) == 2
    assert all(r_id in formulas.index for r_id in r_ids)

    # Test all reactions (default)
    formulas = sbml_dfs.reaction_formulas()
    assert len(formulas) == len(sbml_dfs.reactions)

    # Test invalid IDs
    with pytest.raises(ValueError, match="Reaction IDs.*not found"):
        sbml_dfs.reaction_formulas(r_ids=["invalid_id"])


def test_add_missing_ids_column(sbml_dfs):
    """
    Test add_missing_ids_column function with species IDs contingency table.

    This test generates a species IDs contingency table, drops some roles,
    and then uses add_missing_ids_column to add defaults for missing IDs.
    """

    # Generate a species IDs contingency table using get_ontology_occurrence
    # This creates a binary table where 1 indicates presence of an ontology for a species
    full_contingency_table = sbml_dfs.get_ontology_occurrence(SBML_DFS.SPECIES)

    # Verify we have a contingency table with binary values
    assert isinstance(full_contingency_table, pd.DataFrame)
    assert full_contingency_table.shape[0] > 0  # Should have some species
    assert full_contingency_table.shape[1] > 0  # Should have some ontologies

    # Create a subset contingency table by dropping some rows (simulating missing IDs)
    # Drop the last row to simulate missing data
    subset_contingency_table = full_contingency_table.iloc[:-1, :].copy()

    # Verify the subset is smaller than the full table
    assert subset_contingency_table.shape[0] <= full_contingency_table.shape[0]

    # Use add_missing_ids_column to add missing IDs back with defaults
    result_table = sbml_dfs_utils.add_missing_ids_column(
        subset_contingency_table, sbml_dfs.species
    )

    # Verify the result has the same number of rows as the reference table
    # and one additional column (the 'other' column)
    assert result_table.shape[0] == full_contingency_table.shape[0]
    assert result_table.shape[1] == full_contingency_table.shape[1] + 1
    assert result_table.index.equals(full_contingency_table.index)

    # Verify that missing IDs were added with 'other' column
    if "other" in result_table.columns:
        # Check that 'other' column exists and has values for previously missing rows
        other_column = result_table["other"]
        assert other_column.sum() > 0  # Should have some 1s in the 'other' column

        # Verify that rows that were missing now have 1 in the 'other' column
        missing_rows = full_contingency_table.index.difference(
            subset_contingency_table.index
        )
        if len(missing_rows) > 0:
            for missing_row in missing_rows:
                assert result_table.loc[missing_row, "other"] == 1

    # Verify that existing data is preserved
    for idx in subset_contingency_table.index:
        for col in subset_contingency_table.columns:
            if col in result_table.columns:
                assert (
                    result_table.loc[idx, col] == subset_contingency_table.loc[idx, col]
                )

    # Verify the result is a proper contingency table (integer values)
    assert result_table.dtypes.apply(
        lambda x: pd.api.types.is_integer_dtype(x)
    ).all(), "Result should contain only integer values"


def test_format_sbml_dfs_summary(sbml_dfs):
    """Test format_sbml_dfs_summary function with sbml_dfs fixture."""
    # Get network summary from the sbml_dfs
    summary_stats = sbml_dfs.get_summary()

    # Call format_sbml_dfs_summary
    result = sbml_dfs_utils.format_sbml_dfs_summary(summary_stats)

    # Test that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"

    # Test that it has the expected columns
    expected_columns = ["Metric", "Value"]
    assert (
        list(result.columns) == expected_columns
    ), f"Expected columns {expected_columns}, got {list(result.columns)}"

    # Test that it has some expected rows (basic structure)
    metrics = result["Metric"].tolist()
    assert "Species" in metrics, "Should contain 'Species' metric"
    assert "Reactions" in metrics, "Should contain 'Reactions' metric"
    assert "Compartments" in metrics, "Should contain 'Compartments' metric"
    assert (
        "Compartmentalized Species" in metrics
    ), "Should contain 'Compartmentalized Species' metric"

    # Test that all values are strings (formatted for display)
    assert (
        result["Value"].dtype == object
    ), "Values should be strings/objects for display"

    # Test that there are no empty values in the result
    assert not result["Value"].isna().any(), "No values should be NaN"
    assert not result["Metric"].isna().any(), "No metrics should be NaN"

    # Test that the DataFrame is not empty
    assert len(result) > 0, "Result should not be empty"


def test_find_unused_entities():

    # Create example SBML_dfs tables with unused entities
    CASCADING_CLEANUP_EXAMPLE_DATA = {
        # Compartments - c3 is unused (not referenced by any compartmentalized species)
        SBML_DFS.COMPARTMENTS: pd.DataFrame(
            {
                SBML_DFS.C_ID: ["c1", "c2", "c3"],
                SBML_DFS.C_NAME: [
                    "cytoplasm",
                    "unused_compartment",
                    "removed_due_to_cspecies_cleanup",
                ],
            }
        ).set_index(SBML_DFS.C_ID),
        # Species - s3 is unused (not referenced by any compartmentalized species)
        SBML_DFS.SPECIES: pd.DataFrame(
            {
                SBML_DFS.S_ID: ["s1", "s2", "s3", "s4"],
                SBML_DFS.S_NAME: [
                    "glucose",
                    "pyruvate",
                    "unused_metabolite",
                    "removed_due_to_cspecies_cleanup",
                ],
            }
        ).set_index(SBML_DFS.S_ID),
        # Compartmentalized species - sc3, sc4 are unused (not in reaction_species)
        SBML_DFS.COMPARTMENTALIZED_SPECIES: pd.DataFrame(
            {
                SBML_DFS.SC_ID: ["sc1", "sc2", "sc3", "sc4"],
                SBML_DFS.S_ID: ["s1", "s2", "s2", "s4"],  # Foreign key to species
                SBML_DFS.C_ID: ["c1", "c1", "c3", "c3"],  # Foreign key to compartments
            }
        ).set_index(SBML_DFS.SC_ID),
        # Reactions -
        SBML_DFS.REACTIONS: pd.DataFrame(
            {
                SBML_DFS.R_ID: ["r1", "r2"],
                SBML_DFS.R_NAME: ["glycolysis_step1", "unused_reaction"],
            }
        ).set_index(SBML_DFS.R_ID),
        # Reaction species - only sc1, sc2 are referenced (sc3, sc4 are unused)
        SBML_DFS.REACTION_SPECIES: pd.DataFrame(
            {
                SBML_DFS.RSC_ID: ["rsc1", "rsc2"],
                SBML_DFS.R_ID: ["r1", "r1"],  # Foreign key to reactions
                SBML_DFS.SC_ID: [
                    "sc1",
                    "sc2",
                ],  # Foreign key to compartmentalized_species
            }
        ).set_index(SBML_DFS.RSC_ID),
    }

    unused_entities = sbml_dfs_utils.find_unused_entities(
        CASCADING_CLEANUP_EXAMPLE_DATA
    )
    assert set(unused_entities[SBML_DFS.COMPARTMENTS]) == {"c2", "c3"}
    assert set(unused_entities[SBML_DFS.SPECIES]) == {"s3", "s4"}
    assert set(unused_entities[SBML_DFS.COMPARTMENTALIZED_SPECIES]) == {"sc3", "sc4"}
    assert set(unused_entities[SBML_DFS.REACTIONS]) == {"r2"}


def test_species_type_types():
    """Test the species_type_types function with various Identifiers objects."""

    # Test 1: Complex with HAS_PART (should return "complex" regardless of other ontologies)
    complex_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.CORUM,
            IDENTIFIERS.IDENTIFIER: "123",
            IDENTIFIERS.BQB: BQB.HAS_PART,
        },
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.UNIPROT,
            IDENTIFIERS.IDENTIFIER: "P12345",
            IDENTIFIERS.BQB: BQB.IS,
        },  # This should be ignored
    ]
    complex_identifiers = Identifiers(complex_ids)
    assert (
        sbml_dfs_utils.species_type_types(complex_identifiers) == SPECIES_TYPES.COMPLEX
    )

    # Test 2: Clear metabolite (only CHEBI)
    metabolite_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.CHEBI,
            IDENTIFIERS.IDENTIFIER: "CHEBI:15377",
            IDENTIFIERS.BQB: BQB.IS,
        }
    ]
    metabolite_identifiers = Identifiers(metabolite_ids)
    assert (
        sbml_dfs_utils.species_type_types(metabolite_identifiers)
        == SPECIES_TYPES.METABOLITE
    )

    # Test 3: Clear protein (only Ensembl gene)
    protein_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.ENSEMBL_GENE,
            IDENTIFIERS.IDENTIFIER: "ENSG00000139618",
            IDENTIFIERS.BQB: BQB.IS_ENCODED_BY,
        }
    ]
    protein_identifiers = Identifiers(protein_ids)
    assert (
        sbml_dfs_utils.species_type_types(protein_identifiers) == SPECIES_TYPES.PROTEIN
    )

    # Test 4: Clear regulatory RNA (only miRBase)
    rna_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.MIRBASE,
            IDENTIFIERS.IDENTIFIER: "MIMAT0000062",
            IDENTIFIERS.BQB: BQB.IS,
        }
    ]
    rna_identifiers = Identifiers(rna_ids)
    assert (
        sbml_dfs_utils.species_type_types(rna_identifiers)
        == SPECIES_TYPES.REGULATORY_RNA
    )

    # Test 5: Multiple ontologies from same species type (should work)
    multi_protein_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.ENSEMBL_GENE,
            IDENTIFIERS.IDENTIFIER: "ENSG00000139618",
            IDENTIFIERS.BQB: BQB.IS_ENCODED_BY,
        },
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.UNIPROT,
            IDENTIFIERS.IDENTIFIER: "P12345",
            IDENTIFIERS.BQB: BQB.IS,
        },
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.SYMBOL,
            IDENTIFIERS.IDENTIFIER: "BRCA2",
            IDENTIFIERS.BQB: BQB.IS_ENCODED_BY,
        },
    ]
    multi_protein_identifiers = Identifiers(multi_protein_ids)
    assert (
        sbml_dfs_utils.species_type_types(multi_protein_identifiers)
        == SPECIES_TYPES.PROTEIN
    )

    # Test 6: Mixed species types (should return "other")
    mixed_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.CHEBI,
            IDENTIFIERS.IDENTIFIER: "CHEBI:15377",
            IDENTIFIERS.BQB: BQB.IS,
        },
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.UNIPROT,
            IDENTIFIERS.IDENTIFIER: "P12345",
            IDENTIFIERS.BQB: BQB.IS,
        },
    ]
    mixed_identifiers = Identifiers(mixed_ids)
    assert sbml_dfs_utils.species_type_types(mixed_identifiers) == SPECIES_TYPES.UNKNOWN

    # Test 7: Empty identifiers (should return "unknown")
    empty_identifiers = Identifiers([])
    assert sbml_dfs_utils.species_type_types(empty_identifiers) == SPECIES_TYPES.UNKNOWN

    # Test 8: Ontologies not in mapping (should return "unknown")
    unmapped_ids = [
        {
            IDENTIFIERS.ONTOLOGY: "some_random_ontology",
            IDENTIFIERS.IDENTIFIER: "12345",
            IDENTIFIERS.BQB: BQB.IS,
        }
    ]
    unmapped_identifiers = Identifiers(unmapped_ids)
    assert (
        sbml_dfs_utils.species_type_types(unmapped_identifiers) == SPECIES_TYPES.OTHER
    )

    # Test 9: Non-Identifiers object (should return "unknown")
    assert (
        sbml_dfs_utils.species_type_types("not_an_identifiers_object")
        == SPECIES_TYPES.UNKNOWN
    )
    assert sbml_dfs_utils.species_type_types(None) == SPECIES_TYPES.UNKNOWN
    assert sbml_dfs_utils.species_type_types(123) == SPECIES_TYPES.UNKNOWN

    # Test 10: Complex with HAS_PART overrides everything else
    complex_override_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.CHEBI,
            IDENTIFIERS.IDENTIFIER: "CHEBI:15377",
            IDENTIFIERS.BQB: BQB.HAS_PART,
        },
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.UNIPROT,
            IDENTIFIERS.IDENTIFIER: "P12345",
            IDENTIFIERS.BQB: BQB.IS,
        },
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.MIRBASE,
            IDENTIFIERS.IDENTIFIER: "MIMAT0000062",
            IDENTIFIERS.BQB: BQB.IS,
        },
    ]
    complex_override_identifiers = Identifiers(complex_override_ids)
    assert (
        sbml_dfs_utils.species_type_types(complex_override_identifiers)
        == SPECIES_TYPES.COMPLEX
    )


def test_species_type_types_prioritized():
    """Test the species_type_types function with prioritized species types argument."""

    # Test 1: Drug should be prioritized over metabolite
    drug_metabolite_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.CHEBI,
            IDENTIFIERS.IDENTIFIER: "CHEBI:15377",  # metabolite
            IDENTIFIERS.BQB: BQB.IS,
        },
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.DRUGBANK,
            IDENTIFIERS.IDENTIFIER: "DB00001",  # drug
            IDENTIFIERS.BQB: BQB.IS,
        },
    ]
    drug_metabolite_identifiers = Identifiers(drug_metabolite_ids)

    # With default prioritized types (drug and complex), should return drug
    assert (
        sbml_dfs_utils.species_type_types(drug_metabolite_identifiers)
        == SPECIES_TYPES.DRUG
    )

    # Test 2: Custom prioritized types - prioritize protein over metabolite
    custom_prioritized = {SPECIES_TYPES.PROTEIN}
    protein_metabolite_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.CHEBI,
            IDENTIFIERS.IDENTIFIER: "CHEBI:15377",  # metabolite
            IDENTIFIERS.BQB: BQB.IS,
        },
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.UNIPROT,
            IDENTIFIERS.IDENTIFIER: "P12345",  # protein
            IDENTIFIERS.BQB: BQB.IS,
        },
    ]
    protein_metabolite_identifiers = Identifiers(protein_metabolite_ids)

    # With custom prioritized types, should return protein
    assert (
        sbml_dfs_utils.species_type_types(
            protein_metabolite_identifiers, prioritized_species_types=custom_prioritized
        )
        == SPECIES_TYPES.PROTEIN
    )

    # Test 3: Multiple prioritized types should return UNKNOWN
    multiple_prioritized_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.DRUGBANK,
            IDENTIFIERS.IDENTIFIER: "DB00001",  # drug
            IDENTIFIERS.BQB: BQB.IS,
        },
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.CORUM,
            IDENTIFIERS.IDENTIFIER: "123",  # complex
            IDENTIFIERS.BQB: BQB.IS,
        },
    ]
    multiple_prioritized_identifiers = Identifiers(multiple_prioritized_ids)

    # With default prioritized types (both drug and complex), should return UNKNOWN
    assert (
        sbml_dfs_utils.species_type_types(multiple_prioritized_identifiers)
        == SPECIES_TYPES.UNKNOWN
    )

    # Test 4: No prioritized types should return the single species type
    metabolite_only_ids = [
        {
            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.CHEBI,
            IDENTIFIERS.IDENTIFIER: "CHEBI:15377",  # metabolite
            IDENTIFIERS.BQB: BQB.IS,
        },
    ]
    metabolite_only_identifiers = Identifiers(metabolite_only_ids)

    # With empty prioritized types, should return metabolite
    assert (
        sbml_dfs_utils.species_type_types(
            metabolite_only_identifiers, prioritized_species_types=set()
        )
        == SPECIES_TYPES.METABOLITE
    )
