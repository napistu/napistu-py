from __future__ import annotations

import pandas as pd
import pytest

from napistu import sbml_dfs_utils
from napistu.constants import (
    BQB,
    BQB_DEFINING_ATTRS,
    BQB_DEFINING_ATTRS_LOOSE,
    SBML_DFS,
    IDENTIFIERS,
    POLARITIES,
    POLARITY_TO_SYMBOL,
    SBOTERM_NAMES,
    VALID_SBO_TERMS,
    VALID_SBO_TERM_NAMES,
    MINI_SBO_FROM_NAME,
    MINI_SBO_TO_NAME,
)
from napistu.ingestion.constants import (
    INTERACTION_EDGELIST_DEFS,
    INTERACTION_EDGELIST_OPTIONAL_VARS,
    INTERACTION_EDGELIST_DEFAULTS,
)


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
        sbml_dfs.reaction_species["r_id"] == an_r_id
    ].merge(sbml_dfs.compartmentalized_species, left_on="sc_id", right_index=True)

    formula_str = sbml_dfs_utils.construct_formula_string(
        reaction_species_df, sbml_dfs.reactions, name_var="sc_name"
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

    assert compartment["c_Identifiers"].iloc[0].ids[0] == {
        "ontology": "go",
        "identifier": "GO:0005575",
        "url": "https://www.ebi.ac.uk/QuickGO/term/GO:0005575",
        "bqb": "BQB_IS",
    }


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


def test_infer_entity_type():
    """Test entity type inference with valid keys"""
    # when index matches primary key.
    # Test compartments with index as primary key
    df = pd.DataFrame(
        {SBML_DFS.C_NAME: ["cytoplasm"], SBML_DFS.C_IDENTIFIERS: ["GO:0005737"]}
    )
    df.index.name = SBML_DFS.C_ID
    result = sbml_dfs_utils.infer_entity_type(df)
    assert result == SBML_DFS.COMPARTMENTS

    # Test species with index as primary key
    df = pd.DataFrame(
        {SBML_DFS.S_NAME: ["glucose"], SBML_DFS.S_IDENTIFIERS: ["CHEBI:17234"]}
    )
    df.index.name = SBML_DFS.S_ID
    result = sbml_dfs_utils.infer_entity_type(df)
    assert result == SBML_DFS.SPECIES

    # Test entity type inference by exact column matching.
    # Test compartmentalized_species (has foreign keys)
    df = pd.DataFrame(
        {
            SBML_DFS.SC_ID: ["glucose_c"],
            SBML_DFS.S_ID: ["glucose"],
            SBML_DFS.C_ID: ["cytoplasm"],
        }
    )
    result = sbml_dfs_utils.infer_entity_type(df)
    assert result == "compartmentalized_species"

    # Test reaction_species (has foreign keys)
    df = pd.DataFrame(
        {
            SBML_DFS.RSC_ID: ["rxn1_glc"],
            SBML_DFS.R_ID: ["rxn1"],
            SBML_DFS.SC_ID: ["glucose_c"],
        }
    )
    result = sbml_dfs_utils.infer_entity_type(df)
    assert result == SBML_DFS.REACTION_SPECIES

    # Test reactions (only primary key)
    df = pd.DataFrame({SBML_DFS.R_ID: ["rxn1"]})
    result = sbml_dfs_utils.infer_entity_type(df)
    assert result == SBML_DFS.REACTIONS


def test_infer_entity_type_errors():
    """Test error cases for entity type inference."""
    # Test no matching entity type
    df = pd.DataFrame({"random_column": ["value"], "another_col": ["data"]})
    with pytest.raises(ValueError, match="No entity type matches DataFrame"):
        sbml_dfs_utils.infer_entity_type(df)

    # Test partial match (missing required foreign key)
    df = pd.DataFrame(
        {SBML_DFS.SC_ID: ["glucose_c"], SBML_DFS.S_ID: ["glucose"]}
    )  # Missing c_id
    with pytest.raises(ValueError):
        sbml_dfs_utils.infer_entity_type(df)

    # Test extra primary keys that shouldn't be there
    df = pd.DataFrame(
        {SBML_DFS.R_ID: ["rxn1"], SBML_DFS.S_ID: ["glucose"]}
    )  # Two primary keys
    with pytest.raises(ValueError):
        sbml_dfs_utils.infer_entity_type(df)


def test_infer_entity_type_multindex_reactions():
    # DataFrame with MultiIndex (r_id, foo), should infer as reactions
    import pandas as pd
    from napistu.constants import SBML_DFS

    df = pd.DataFrame({"some_col": [1, 2]})
    df.index = pd.MultiIndex.from_tuples(
        [("rxn1", "a"), ("rxn2", "b")], names=[SBML_DFS.R_ID, "foo"]
    )
    result = sbml_dfs_utils.infer_entity_type(df)
    assert result == SBML_DFS.REACTIONS


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
            INTERACTION_EDGELIST_DEFS.UPSTREAM_NAME: ["A", "B"],
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_NAME: ["B", "C"],
            SBML_DFS.R_NAME: ["A->B", "B->C"],
            SBML_DFS.R_IDENTIFIERS: [[], []],
            INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT: ["cytoplasm", "cytoplasm"],
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT: [
                "cytoplasm",
                "cytoplasm",
            ],
            INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME: [
                SBOTERM_NAMES.CATALYST,
                SBOTERM_NAMES.CATALYST,
            ],
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_SBO_TERM_NAME: [
                SBOTERM_NAMES.PRODUCT,
                SBOTERM_NAMES.PRODUCT,
            ],
            INTERACTION_EDGELIST_DEFS.UPSTREAM_STOICHIOMETRY: [1, 1],
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_STOICHIOMETRY: [1, 1],
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
            INTERACTION_EDGELIST_DEFS.UPSTREAM_NAME: ["A", "B"],
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_NAME: ["B", "C"],
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
            INTERACTION_EDGELIST_DEFS.UPSTREAM_NAME: ["A", "B"],
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_NAME: ["B", "C"],
            SBML_DFS.R_NAME: ["A->B", "B->C"],
            SBML_DFS.R_IDENTIFIERS: [[], []],
            INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT: ["cytoplasm", None],
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT: [None, "cytoplasm"],
            SBML_DFS.R_ISREVERSIBLE: [False, None],
        }
    )

    result = sbml_dfs_utils._add_edgelist_defaults(edgelist_with_nan)

    # Check that NaN values were replaced with defaults
    assert (
        result[INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT].iloc[1]
        == INTERACTION_EDGELIST_DEFAULTS[INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT]
    )
    assert (
        result[INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT].iloc[0]
        == INTERACTION_EDGELIST_DEFAULTS[
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT
        ]
    )
    assert (
        result[SBML_DFS.R_ISREVERSIBLE].iloc[1]
        == INTERACTION_EDGELIST_DEFAULTS[SBML_DFS.R_ISREVERSIBLE]
    )

    # Test 4: Custom defaults dictionary
    custom_defaults = {
        INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT: "nucleus",
        INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT: "nucleus",
        INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME: SBOTERM_NAMES.CATALYST,
        INTERACTION_EDGELIST_DEFS.DOWNSTREAM_SBO_TERM_NAME: SBOTERM_NAMES.PRODUCT,
        INTERACTION_EDGELIST_DEFS.UPSTREAM_STOICHIOMETRY: 1,
        INTERACTION_EDGELIST_DEFS.DOWNSTREAM_STOICHIOMETRY: 1,
        SBML_DFS.R_ISREVERSIBLE: True,
    }

    minimal_edgelist = pd.DataFrame(
        {
            INTERACTION_EDGELIST_DEFS.UPSTREAM_NAME: ["A"],
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_NAME: ["B"],
            SBML_DFS.R_NAME: ["A->B"],
            SBML_DFS.R_IDENTIFIERS: [[]],
        }
    )

    result = sbml_dfs_utils._add_edgelist_defaults(minimal_edgelist, custom_defaults)

    # Check that custom defaults were applied
    assert result[INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT].iloc[0] == "nucleus"
    assert result[INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT].iloc[0] == "nucleus"
    assert result[SBML_DFS.R_ISREVERSIBLE].iloc[0]

    # Test 5: Missing defaults for required optional columns should raise ValueError
    incomplete_defaults = {
        INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT: "cytoplasm"
        # Missing other required defaults
    }

    # The function now uses the default INTERACTION_EDGELIST_DEFAULTS when incomplete defaults are provided
    # So this should work without raising an error
    result = sbml_dfs_utils._add_edgelist_defaults(
        minimal_edgelist, edgelist_defaults=incomplete_defaults
    )

    # Check that the missing defaults were filled from INTERACTION_EDGELIST_DEFAULTS
    assert (
        result[INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT].iloc[0]
        == INTERACTION_EDGELIST_DEFAULTS[
            INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT
        ]
    )
    assert (
        result[INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME].iloc[0]
        == INTERACTION_EDGELIST_DEFAULTS[
            INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME
        ]
    )
