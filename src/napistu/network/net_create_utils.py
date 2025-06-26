import logging

import pandas as pd

from napistu import utils
from napistu.constants import MINI_SBO_FROM_NAME, SBML_DFS
from napistu.network.constants import (
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    DROP_REACTIONS_WHEN,
    VALID_DROP_REACTIONS_WHEN,
    GRAPH_WIRING_HIERARCHIES,
    VALID_GRAPH_WIRING_APPROACHES,
)

logger = logging.getLogger(__name__)


def format_tiered_reaction_species(
    r_id: str,
    sorted_reaction_species: pd.DataFrame,
    graph_hierarchy_df: pd.DataFrame,
    drop_reactions_when: str = DROP_REACTIONS_WHEN.SAME_TIER,
) -> pd.DataFrame:
    """
    Create a Napistu graph from a reaction and its species.

    Parameters
    ----------
    r_id : str
        The ID of the reaction.
    sorted_reaction_species : pd.DataFrame
        The reaction species.
    graph_hierarchy_df : pd.DataFrame
        The graph hierarchy.
    drop_reactions_when : str
        The condition under which to drop reactions as a network vertex.

    Returns
    -------
    edges : pd.DataFrame
        The edges of the Napistu graph for a single reaction.
    """

    rxn_species = sorted_reaction_species.loc[r_id]
    _validate_sbo_indexed_rsc_stoi(rxn_species)

    if sorted_reaction_species.shape[0] <= 1:
        logger.warning(
            f"Reaction {r_id} has {sorted_reaction_species.shape[0]} species. "
            "This reaction will be dropped."
        )
        return pd.DataFrame()

    # map reaction species to the tiers of the graph hierarchy. higher levels point to lower levels
    # same-level entries point at each other only if there is only a single tier
    entities_ordered_by_tier = _reaction_species_to_tiers(
        rxn_species, graph_hierarchy_df, r_id
    )
    n_tiers = len(entities_ordered_by_tier.index.get_level_values("tier").unique())

    # format edges for reactions where all participants are on the same tier of a wiring hierarcy
    if n_tiers == 2:
        edges = _format_same_tier_edges(rxn_species, r_id)
    else:
        edges = _format_cross_tier_edges(
            entities_ordered_by_tier, r_id, drop_reactions_when
        )

    return edges


def _should_drop_reaction(
    entities_ordered_by_tier: pd.DataFrame,
    drop_reactions_when: str = DROP_REACTIONS_WHEN.SAME_TIER,
):
    """
    Determine if a reaction should be dropped

    This is determined based on the regulatory relationships between the species and the desired stringency.

    Parameters
    ----------
    entities_ordered_by_tier : pd.DataFrame
        The entities ordered by tier.
    drop_reactions : str
        The desired stringency for dropping reactions.

    Returns
    -------
    bool
        True if the reaction should be dropped, False otherwise.

    Notes
    _____
    reactions are always dropped if they are on the same tier. This greatly decreases the number of vertices
    in a graph constructed from relatively dense interaction networks like STRING.

    """

    if drop_reactions_when == DROP_REACTIONS_WHEN.ALWAYS:
        return True

    elif drop_reactions_when == DROP_REACTIONS_WHEN.EDGELIST:
        if entities_ordered_by_tier.shape[0] == 3:  # 2 members + 1 for reaction
            return True
        else:
            return False

    elif drop_reactions_when == DROP_REACTIONS_WHEN.SAME_TIER:
        n_reactant_tiers = len(
            entities_ordered_by_tier.query("sbo_name != 'reaction'")
            .index.unique()
            .tolist()
        )
        if n_reactant_tiers == 1:
            return True
        else:
            return False

    else:
        raise ValueError(
            f"Invalid drop_reactions: {drop_reactions_when}; valid values are {VALID_DROP_REACTIONS_WHEN}"
        )


def _format_same_tier_edges(rxn_species: pd.DataFrame, r_id: str) -> pd.DataFrame:
    """Format edges for reactions where all participants are on the same tier of a wiring hierarcy."""

    # if they have the same SBO_term and stoichiometry, then the
    # reaction can be trivially treated as reversible

    valid_species = rxn_species.reset_index().assign(
        entry=range(0, rxn_species.shape[0])
    )
    distinct_metadata = valid_species[
        [SBML_DFS.SBO_TERM, SBML_DFS.STOICHIOMETRY]
    ].drop_duplicates()
    if distinct_metadata.shape[0] > 1:
        raise ValueError(
            f"Reaction {r_id} has multiple distinct metadata: {distinct_metadata}"
        )

    crossed_species = (
        valid_species.merge(valid_species, how="cross", suffixes=("_left", "_right"))
        .query("entry_left < entry_right")
        .rename(
            {
                "sc_id_left": NAPISTU_GRAPH_EDGES.FROM,
                "sc_id_right": NAPISTU_GRAPH_EDGES.TO,
                "stoichiometry_right": NAPISTU_GRAPH_EDGES.STOICHIOMETRY,
                "sbo_term_left": NAPISTU_GRAPH_EDGES.SBO_TERM,
            },
            axis=1,
        )
        .assign(r_id=r_id)
    )

    OUT_ATTRS = [
        NAPISTU_GRAPH_EDGES.FROM,
        NAPISTU_GRAPH_EDGES.TO,
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY,
        NAPISTU_GRAPH_EDGES.SBO_TERM,
        SBML_DFS.R_ID,
    ]

    return crossed_species[OUT_ATTRS]


def _format_cross_tier_edges(
    entities_ordered_by_tier: pd.DataFrame,
    r_id: str,
    drop_reactions_when: str = DROP_REACTIONS_WHEN.SAME_TIER,
):

    ordered_tiers = entities_ordered_by_tier.index.get_level_values("tier").unique()
    reaction_tier = entities_ordered_by_tier.query(
        "sbo_name == 'reaction'"
    ).index.tolist()[0]
    drop_reaction = _should_drop_reaction(entities_ordered_by_tier, drop_reactions_when)

    rxn_edges = list()
    past_reaction = False
    for i in range(0, len(ordered_tiers) - 1):

        if ordered_tiers[i] == reaction_tier:
            if drop_reaction:
                continue

        next_tier = ordered_tiers[i + 1]
        if ordered_tiers[i + 1] == reaction_tier:
            # hop over the reaction tier
            if drop_reaction:
                next_tier = ordered_tiers[i + 2]

        formatted_tier_combo = _format_tier_combo(
            entities_ordered_by_tier.loc[[ordered_tiers[i]]],
            entities_ordered_by_tier.loc[[next_tier]],
            past_reaction,
        )

        if ordered_tiers[i + 1] == reaction_tier:
            past_reaction = True

        rxn_edges.append(formatted_tier_combo)

    rxn_edges_df = (
        pd.concat(rxn_edges)[
            [
                NAPISTU_GRAPH_EDGES.FROM,
                NAPISTU_GRAPH_EDGES.TO,
                NAPISTU_GRAPH_EDGES.STOICHIOMETRY,
                NAPISTU_GRAPH_EDGES.SBO_TERM,
            ]
        ]
        .reset_index(drop=True)
        .assign(r_id=r_id)
    )

    return rxn_edges_df


def _validate_sbo_indexed_rsc_stoi(rxn_species) -> None:

    if not isinstance(rxn_species, pd.DataFrame):
        raise TypeError("rxn_species must be a pandas DataFrame")
    if list(rxn_species.index.names) != [SBML_DFS.SBO_TERM]:
        raise ValueError("rxn_species index names must be [SBML_DFS.SBO_TERM]")
    if rxn_species.columns.tolist() != [SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY]:
        raise ValueError(
            "rxn_species columns must be [SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY]"
        )

    return None


def _reaction_species_to_tiers(
    rxn_species: pd.DataFrame, graph_hierarchy_df: pd.DataFrame, r_id: str
) -> pd.DataFrame:

    entities_ordered_by_tier = (
        pd.concat(
            [
                (
                    rxn_species.reset_index()
                    .rename({SBML_DFS.SC_ID: "entity_id"}, axis=1)
                    .merge(graph_hierarchy_df)
                ),
                graph_hierarchy_df[
                    graph_hierarchy_df[NAPISTU_GRAPH_EDGES.SBO_NAME]
                    == NAPISTU_GRAPH_NODE_TYPES.REACTION
                ].assign(entity_id=r_id, r_id=r_id),
            ]
        )
        .sort_values(["tier"])
        .set_index("tier")
    )
    return entities_ordered_by_tier


def _format_tier_combo(
    upstream_tier: pd.DataFrame, downstream_tier: pd.DataFrame, past_reaction: bool
) -> pd.DataFrame:
    """
    Format Tier Combo

    Create a set of edges crossing two tiers of a tiered graph. This will involve an
    all x all combination of entries. Tiers form an ordering along the molecular entities
    in a reaction plus a tier for the reaction itself. Attributes such as stoichiometry
    and sbo_term will be passed from the tier which is furthest from the reaction tier
    to ensure that each tier of molecular data applies its attributes to a single set of
    edges while the "reaction" tier does not. Reaction entities have neither a
    stoichiometery or sbo_term annotation.

    Args:
        upstream_tier (pd.DataFrame): A table containing upstream entities in a reaction,
            e.g., regulators.
        downstream_tier (pd.DataFrame): A table containing downstream entities in a reaction,
            e.g., catalysts.
        past_reaction (bool): if True then attributes will be taken from downstream_tier and
            if False they will come from upstream_tier.

    Returns:
        formatted_tier_combo (pd.DataFrame): A table of edges containing (from, to, stoichiometry, sbo_term, r_id). The
        number of edges is the product of the number of entities in the upstream tier
        times the number in the downstream tier.

    """

    upstream_fields = ["entity_id", SBML_DFS.STOICHIOMETRY, SBML_DFS.SBO_TERM]
    downstream_fields = ["entity_id"]

    if past_reaction:
        # swap fields
        upstream_fields, downstream_fields = downstream_fields, upstream_fields

    formatted_tier_combo = (
        upstream_tier[upstream_fields]
        .rename({"entity_id": NAPISTU_GRAPH_EDGES.FROM}, axis=1)
        .assign(_joiner=1)
    ).merge(
        (
            downstream_tier[downstream_fields]
            .rename({"entity_id": NAPISTU_GRAPH_EDGES.TO}, axis=1)
            .assign(_joiner=1)
        ),
        left_on="_joiner",
        right_on="_joiner",
    )

    return formatted_tier_combo


def _create_graph_hierarchy_df(wiring_approach: str) -> pd.DataFrame:
    """
    Create Graph Hierarchy DataFrame

    Format a graph hierarchy list of lists and a pd.DataFrame

    Args:
        wiring_approach (str):
            The type of tiered graph to work with. Each type has its own specification in constants.py.

    Returns:
        A pandas DataFrame with sbo_name, tier, and sbo_term.

    """

    if wiring_approach not in VALID_GRAPH_WIRING_APPROACHES:
        raise ValueError(
            f"{wiring_approach} is not a valid wiring approach. Valid approaches are {', '.join(VALID_GRAPH_WIRING_APPROACHES)}"
        )

    sbo_names_hierarchy = GRAPH_WIRING_HIERARCHIES[wiring_approach]

    # format as a DF
    graph_hierarchy_df = pd.concat(
        [
            pd.DataFrame({"sbo_name": sbo_names_hierarchy[i]}).assign(tier=i)
            for i in range(0, len(sbo_names_hierarchy))
        ]
    ).reset_index(drop=True)
    graph_hierarchy_df[SBML_DFS.SBO_TERM] = graph_hierarchy_df["sbo_name"].apply(
        lambda x: (
            MINI_SBO_FROM_NAME[x] if x != NAPISTU_GRAPH_NODE_TYPES.REACTION else ""
        )
    )

    # ensure that the output is expected
    utils.match_pd_vars(
        graph_hierarchy_df,
        req_vars={NAPISTU_GRAPH_EDGES.SBO_NAME, "tier", SBML_DFS.SBO_TERM},
        allow_series=False,
    ).assert_present()

    return graph_hierarchy_df
