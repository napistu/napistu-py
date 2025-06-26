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
