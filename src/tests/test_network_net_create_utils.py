from __future__ import annotations

import pytest

import pandas as pd

from napistu.network import net_create_utils
from napistu.constants import (
    MINI_SBO_FROM_NAME,
    SBML_DFS,
    SBOTERM_NAMES,
    VALID_SBO_TERM_NAMES,
)
from napistu.network.constants import (
    DROP_REACTIONS_WHEN,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    VALID_GRAPH_WIRING_APPROACHES,
)


def test_format_interactors(reaction_species_examples):

    r_id = "foo"
    # interactions are formatted
    graph_hierarchy_df = net_create_utils.create_graph_hierarchy_df("regulatory")

    assert (
        net_create_utils.format_tiered_reaction_species(
            reaction_species_examples["valid_interactor"],
            r_id,
            graph_hierarchy_df,
            drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
        ).shape[0]
        == 1
    )

    # simple reaction with just substrates and products
    assert (
        net_create_utils.format_tiered_reaction_species(
            reaction_species_examples["sub_and_prod"],
            r_id,
            graph_hierarchy_df,
            drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
        ).shape[0]
        == 2
    )

    # add a stimulator (activator)
    rxn_edges = net_create_utils.format_tiered_reaction_species(
        reaction_species_examples["stimulator"],
        r_id,
        graph_hierarchy_df,
        drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
    )

    assert rxn_edges.shape[0] == 3
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim", "sub"]

    # add catalyst + stimulator
    rxn_edges = net_create_utils.format_tiered_reaction_species(
        reaction_species_examples["all_entities"],
        r_id,
        graph_hierarchy_df,
        drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
    )

    assert rxn_edges.shape[0] == 4
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim", "cat"]
    assert rxn_edges.iloc[1][["from", "to"]].tolist() == ["cat", "sub"]

    # no substrate
    rxn_edges = net_create_utils.format_tiered_reaction_species(
        reaction_species_examples["no_substrate"],
        r_id,
        graph_hierarchy_df,
        drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
    )

    assert rxn_edges.shape[0] == 5
    # stimulator -> reactant
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim1", "cat"]
    assert rxn_edges.iloc[1][["from", "to"]].tolist() == ["stim2", "cat"]
    assert rxn_edges.iloc[2][["from", "to"]].tolist() == ["inh", "cat"]

    # use the surrogate model tiered layout also

    graph_hierarchy_df = net_create_utils.create_graph_hierarchy_df("surrogate")

    rxn_edges = net_create_utils.format_tiered_reaction_species(
        reaction_species_examples["all_entities"],
        r_id,
        graph_hierarchy_df,
        drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
    )

    assert rxn_edges.shape[0] == 4
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim", "sub"]
    assert rxn_edges.iloc[1][["from", "to"]].tolist() == ["sub", "cat"]


def test_drop_reactions_when_parameters(reaction_species_examples):
    """Test different drop_reactions_when parameter values and edge cases."""

    r_id = "foo"
    graph_hierarchy = net_create_utils.create_graph_hierarchy_df("regulatory")

    # Test ALWAYS - should drop reaction regardless of tiers
    edges_always = net_create_utils.format_tiered_reaction_species(
        reaction_species_examples["all_entities"],
        r_id,
        graph_hierarchy,
        DROP_REACTIONS_WHEN.ALWAYS,
    )
    assert r_id not in edges_always[NAPISTU_GRAPH_EDGES.FROM].values
    assert r_id not in edges_always[NAPISTU_GRAPH_EDGES.TO].values

    # Test EDGELIST with 2 species (should drop reaction)
    edges_edgelist = net_create_utils.format_tiered_reaction_species(
        reaction_species_examples["sub_and_prod"],
        r_id,
        graph_hierarchy,
        DROP_REACTIONS_WHEN.EDGELIST,
    )
    assert r_id not in edges_edgelist[NAPISTU_GRAPH_EDGES.FROM].values

    # Test EDGELIST with >2 species (should keep reaction)
    edges_multi = net_create_utils.format_tiered_reaction_species(
        reaction_species_examples["all_entities"],
        r_id,
        graph_hierarchy,
        DROP_REACTIONS_WHEN.EDGELIST,
    )
    reaction_in_edges = (
        r_id in edges_multi[NAPISTU_GRAPH_EDGES.FROM].values
        or r_id in edges_multi[NAPISTU_GRAPH_EDGES.TO].values
    )
    assert reaction_in_edges

    # Test invalid parameter
    with pytest.raises(ValueError, match="Invalid drop_reactions"):
        net_create_utils.format_tiered_reaction_species(
            reaction_species_examples["sub_and_prod"],
            r_id,
            graph_hierarchy,
            "INVALID_OPTION",
        )


def test_edge_cases_and_validation(reaction_species_examples):
    """Test edge cases, empty inputs, and validation errors."""

    r_id = "foo"
    graph_hierarchy = net_create_utils.create_graph_hierarchy_df("regulatory")

    # Test single species
    edges_single = net_create_utils.format_tiered_reaction_species(
        reaction_species_examples["single_species"], r_id, graph_hierarchy
    )
    assert edges_single.empty

    # Test validation with incorrectly indexed DataFrame (should raise error)
    bad_df = reaction_species_examples[
        "sub_and_prod"
    ].reset_index()  # Remove proper index
    with pytest.raises(ValueError):
        net_create_utils.format_tiered_reaction_species(bad_df, r_id, graph_hierarchy)

    # Test activator and inhibitor only (should return empty DataFrame)
    edges_ai = net_create_utils.format_tiered_reaction_species(
        reaction_species_examples["activator_and_inhibitor_only"], r_id, graph_hierarchy
    )
    assert edges_ai.empty


def test_edgelist_should_have_one_edge():
    """EDGELIST with 2 species should create exactly 1 edge, not 2"""

    r_id = "foo"
    reaction_df = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
            ],
            SBML_DFS.SC_ID: ["sub", "prod"],
            SBML_DFS.STOICHIOMETRY: [-1.0, 1.0],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    graph_hierarchy = net_create_utils.create_graph_hierarchy_df("regulatory")
    edges = net_create_utils.format_tiered_reaction_species(
        reaction_df, r_id, graph_hierarchy, DROP_REACTIONS_WHEN.EDGELIST
    )

    # Should be 1 edge, actually gets 2
    assert len(edges) == 1, f"EDGELIST should create 1 edge, got {len(edges)}"


def test_edgelist_should_not_have_reaction_as_source():
    """EDGELIST should not have reaction ID in FROM column"""

    r_id = "foo"
    reaction_df = pd.DataFrame(
        {
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT],
                MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
            ],
            SBML_DFS.SC_ID: ["sub", "prod"],
            SBML_DFS.STOICHIOMETRY: [-1.0, 1.0],
        }
    ).set_index(SBML_DFS.SBO_TERM)

    graph_hierarchy = net_create_utils.create_graph_hierarchy_df("regulatory")
    edges = net_create_utils.format_tiered_reaction_species(
        reaction_df, r_id, graph_hierarchy, DROP_REACTIONS_WHEN.EDGELIST
    )

    # Should not have 'foo' in FROM column, but it does
    assert (
        r_id not in edges["from"].values
    ), f"Reaction {r_id} should not appear in FROM column"


def test_should_drop_reaction(reaction_species_examples):

    r_id = "foo"

    graph_hierarchy_df = net_create_utils.create_graph_hierarchy_df("regulatory")

    rxn_species = reaction_species_examples["sub_and_prod"]
    net_create_utils._validate_sbo_indexed_rsc_stoi(rxn_species)

    # map reaction species to the tiers of the graph hierarchy. higher levels point to lower levels
    # same-level entries point at each other only if there is only a single tier
    entities_ordered_by_tier = net_create_utils._reaction_species_to_tiers(
        rxn_species, graph_hierarchy_df, r_id
    )

    # this is an edgeliist (just 2 entries)
    assert net_create_utils._should_drop_reaction(
        entities_ordered_by_tier, drop_reactions_when=DROP_REACTIONS_WHEN.EDGELIST
    )

    # not the same tier
    assert not net_create_utils._should_drop_reaction(
        entities_ordered_by_tier, drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER
    )


def test_graph_hierarchy_layouts():
    REQUIRED_NAMES = VALID_SBO_TERM_NAMES + [NAPISTU_GRAPH_NODE_TYPES.REACTION]
    for value in VALID_GRAPH_WIRING_APPROACHES:
        layout_df = net_create_utils.create_graph_hierarchy_df(value)
        # all terms should be represented
        missing = set(REQUIRED_NAMES).difference(
            set(layout_df[NAPISTU_GRAPH_EDGES.SBO_NAME])
        )
        assert not missing, f"Missing SBO names in {value}: {missing}"
        # all terms should be unique
        duplicated = layout_df[layout_df[NAPISTU_GRAPH_EDGES.SBO_NAME].duplicated()]
        assert (
            duplicated.empty
        ), f"Duplicated SBO names in {value}: {duplicated[NAPISTU_GRAPH_EDGES.SBO_NAME].tolist()}"
        # check that reaction is present and its by itself
        reaction_tiers = layout_df[
            layout_df[NAPISTU_GRAPH_EDGES.SBO_NAME] == NAPISTU_GRAPH_NODE_TYPES.REACTION
        ]["tier"].unique()
        assert (
            len(reaction_tiers) == 1
        ), f"'reaction' appears in multiple tiers in {value}: {reaction_tiers}"
        reaction_tier = reaction_tiers[0]
        reaction_tier_df = layout_df[layout_df["tier"] == reaction_tier]
        assert (
            reaction_tier_df.shape[0] == 1
            and reaction_tier_df[NAPISTU_GRAPH_EDGES.SBO_NAME].iloc[0]
            == NAPISTU_GRAPH_NODE_TYPES.REACTION
        ), f"Tier {reaction_tier} in {value} should contain only 'reaction', but contains: {reaction_tier_df[NAPISTU_GRAPH_EDGES.SBO_NAME].tolist()}"


def test_identifying_and_formatting_interactor_duos(reaction_species_examples):

    # directly specify interactions as a speed up to same tier procedure
    interaction_template = reaction_species_examples["valid_interactor"]
    interactor_species = pd.concat(
        [
            interaction_template.reset_index().assign(
                r_id=str(i),
                sc_id=lambda df: df[SBML_DFS.SC_ID].apply(lambda x: f"r{i}_{x}"),
            )
            for i in range(0, 10)
        ]
    )

    invalid_interactor_template = reaction_species_examples["invalid_interactor"]
    invalid_interactor_species = pd.concat(
        [
            invalid_interactor_template.reset_index().assign(
                r_id=str(i),
                sc_id=lambda df: df[SBML_DFS.SC_ID].apply(lambda x: f"r{i}_{x}"),
            )
            for i in range(10, 12)
        ]
    )

    reaction_species = pd.concat([interactor_species, invalid_interactor_species])

    matching_r_ids = net_create_utils._find_sbo_duos(
        reaction_species, MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR]
    )
    assert set(matching_r_ids) == {str(i) for i in range(0, 10)}

    interactor_duos = reaction_species.loc[
        reaction_species[SBML_DFS.R_ID].isin(matching_r_ids)
    ]
    assert net_create_utils._interactor_duos_to_wide(interactor_duos).shape[0] == 10


def test_wire_reaction_species_mixed_interactions(reaction_species_examples):
    """
    Test wire_reaction_species function with a mix of interactor and non-interactor reactions.

    This test verifies that the function correctly processes:
    1. Interactor pairs (processed en-masse)
    2. Non-interactor reactions (processed with tiered algorithms)
    3. Different wiring approaches
    4. Different drop_reactions_when conditions
    """

    # Create a mixed dataset with both interactor and non-interactor reactions
    # Interactor reactions (should be processed en-masse)
    interactor_template = reaction_species_examples["valid_interactor"]
    interactor_species = pd.concat(
        [
            interactor_template.reset_index().assign(
                r_id=f"interactor_{i}",
                sc_id=lambda df: df[SBML_DFS.SC_ID].apply(
                    lambda x: f"interactor_{i}_{x}"
                ),
            )
            for i in range(3)  # 3 interactor reactions
        ]
    )

    # Non-interactor reactions (should be processed with tiered algorithms)
    non_interactor_species = pd.concat(
        [
            reaction_species_examples["sub_and_prod"]
            .reset_index()
            .assign(
                r_id=f"reaction_{i}",
                sc_id=lambda df: df[SBML_DFS.SC_ID].apply(
                    lambda x: f"reaction_{i}_{x}"
                ),
            )
            for i in range(2)  # 2 substrate-product reactions
        ]
    )

    # Add a complex reaction with multiple entity types
    complex_reaction = (
        reaction_species_examples["all_entities"]
        .reset_index()
        .assign(
            r_id="complex_reaction",
            sc_id=lambda df: df[SBML_DFS.SC_ID].apply(lambda x: f"complex_{x}"),
        )
    )

    # Combine all reaction species
    all_reaction_species = pd.concat(
        [interactor_species, non_interactor_species, complex_reaction]
    )

    # Test with regulatory wiring approach
    edges_regulatory = net_create_utils.wire_reaction_species(
        all_reaction_species,
        wiring_approach="regulatory",
        drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
    )

    # Verify the output structure
    assert not edges_regulatory.empty
    required_columns = [
        NAPISTU_GRAPH_EDGES.FROM,
        NAPISTU_GRAPH_EDGES.TO,
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY,
        NAPISTU_GRAPH_EDGES.SBO_TERM,
        SBML_DFS.R_ID,
    ]

    for col in required_columns:
        assert col in edges_regulatory.columns, f"Missing required column: {col}"

    # Check that interactor reactions were processed correctly
    interactor_edges = edges_regulatory[
        edges_regulatory[SBML_DFS.R_ID].str.startswith("interactor_")
    ]
    assert (
        len(interactor_edges) == 3
    ), f"Expected 3 interactor edges, got {len(interactor_edges)}"

    # Check that non-interactor reactions were processed correctly
    reaction_edges = edges_regulatory[
        edges_regulatory[SBML_DFS.R_ID].str.startswith("reaction_")
    ]
    assert (
        len(reaction_edges) == 4
    ), f"Expected 4 reaction edges, got {len(reaction_edges)}"

    # Check that complex reaction was processed correctly
    complex_edges = edges_regulatory[
        edges_regulatory[SBML_DFS.R_ID] == "complex_reaction"
    ]
    assert (
        len(complex_edges) == 4
    ), f"Expected 4 complex reaction edges, got {len(complex_edges)}"

    # Test with different drop_reactions_when condition
    edges_always_drop = net_create_utils.wire_reaction_species(
        all_reaction_species,
        wiring_approach="regulatory",
        drop_reactions_when=DROP_REACTIONS_WHEN.ALWAYS,
    )

    # With ALWAYS drop, reaction IDs should not appear in FROM or TO columns
    reaction_ids = [
        "interactor_0",
        "interactor_1",
        "interactor_2",
        "reaction_0",
        "reaction_1",
        "complex_reaction",
    ]
    for r_id in reaction_ids:
        assert (
            r_id not in edges_always_drop[NAPISTU_GRAPH_EDGES.FROM].values
        ), f"Reaction {r_id} should not be in FROM column"
        assert (
            r_id not in edges_always_drop[NAPISTU_GRAPH_EDGES.TO].values
        ), f"Reaction {r_id} should not be in TO column"

    # Test with EDGELIST drop condition
    edges_edgelist = net_create_utils.wire_reaction_species(
        all_reaction_species,
        wiring_approach="regulatory",
        drop_reactions_when=DROP_REACTIONS_WHEN.EDGELIST,
    )

    # Simple reactions (2 species) should not have reaction IDs, but complex reactions should
    simple_reaction_ids = ["reaction_0", "reaction_1"]
    complex_reaction_id = "complex_reaction"

    for r_id in simple_reaction_ids:
        assert (
            r_id not in edges_edgelist[NAPISTU_GRAPH_EDGES.FROM].values
        ), f"Simple reaction {r_id} should not be in FROM column"
        assert (
            r_id not in edges_edgelist[NAPISTU_GRAPH_EDGES.TO].values
        ), f"Simple reaction {r_id} should not be in TO column"

    # Complex reaction should still have reaction ID in edges
    complex_in_edges = (
        complex_reaction_id in edges_edgelist[NAPISTU_GRAPH_EDGES.FROM].values
        or complex_reaction_id in edges_edgelist[NAPISTU_GRAPH_EDGES.TO].values
    )
    assert (
        complex_in_edges
    ), f"Complex reaction {complex_reaction_id} should appear in edges"

    # Test edge case: only interactor reactions
    only_interactors = interactor_species
    edges_only_interactors = net_create_utils.wire_reaction_species(
        only_interactors,
        wiring_approach="regulatory",
        drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
    )

    assert not edges_only_interactors.empty
    assert (
        len(edges_only_interactors) == 3
    ), f"Expected 3 edges for only interactors, got {len(edges_only_interactors)}"

    # Test edge case: only non-interactor reactions
    only_reactions = pd.concat([non_interactor_species, complex_reaction])
    edges_only_reactions = net_create_utils.wire_reaction_species(
        only_reactions,
        wiring_approach="regulatory",
        drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
    )

    print(edges_only_reactions)

    assert not edges_only_reactions.empty
    assert (
        len(edges_only_reactions) == 8
    ), f"Expected 8 edges for only reactions, got {len(edges_only_reactions)}"


def test_wire_reaction_species_validation_errors():
    """Test wire_reaction_species function with invalid inputs."""

    # Test with invalid wiring approach
    reaction_species = pd.DataFrame(
        {
            SBML_DFS.R_ID: ["R1"],
            SBML_DFS.SC_ID: ["A"],
            SBML_DFS.STOICHIOMETRY: [0],
            SBML_DFS.SBO_TERM: ["SBO:0000336"],  # interactor
        }
    )

    with pytest.raises(ValueError, match="is not a valid wiring approach"):
        net_create_utils.wire_reaction_species(
            reaction_species,
            wiring_approach="invalid_approach",
            drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
        )

    # Test with invalid SBO terms
    invalid_sbo_species = pd.DataFrame(
        {
            SBML_DFS.R_ID: ["R1"],
            SBML_DFS.SC_ID: ["A"],
            SBML_DFS.STOICHIOMETRY: [0],
            SBML_DFS.SBO_TERM: ["INVALID_SBO_TERM"],
        }
    )

    with pytest.raises(
        ValueError, match="Some reaction species have unusable SBO terms"
    ):
        net_create_utils.wire_reaction_species(
            invalid_sbo_species,
            wiring_approach="regulatory",
            drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
        )
