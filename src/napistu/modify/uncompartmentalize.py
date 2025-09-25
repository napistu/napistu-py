from __future__ import annotations

import logging

import pandas as pd

from napistu import (
    consensus,
    identifiers,
    indices,
    sbml_dfs_core,
    sbml_dfs_utils,
    source,
)
from napistu.constants import SBML_DFS, SOURCE_SPEC

logger = logging.getLogger(__name__)


def uncompartmentalize_sbml_dfs(
    sbml_dfs: sbml_dfs_core.SBML_dfs, inplace: bool = True
) -> None:
    """Uncompartmentalize SBML_dfs

    Take a compartmentalized mechanistic model and merge all of the compartments.

    To remove compartmentalization we can:
    1. update the compartments table to the stubbed default level: GO CELLULAR_COMPONENT
    2. ignore the species table (it will be the same in the compartmentalized and uncompartmenalzied model)
    3. create a 1-1 correspondence between species and new compartmentalized species. w/ GO CELLULAR_COMPONENT
    4. update reaction species to the new compartmentalized species
    5. drop reactions if:
       - they are redundant (e.g., the same reaction occurred in multiple compartments)
       - substrates and products are identical (e.g., a transportation reaction)

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object to uncompartmentalize
    inplace : bool
        Whether to modify the SBML_dfs object in-place or return a new object

    Returns
    -------
    None
        Modifies the SBML_dfs object in-place if inplace is True, otherwise returns a new SBML_dfs object
    """

    if not inplace:
        sbml_dfs = sbml_dfs.copy()

    if sbml_dfs.compartments.shape[0] == 1:
        logger.warning(
            "The sbml_dfs model only contains a single compartment, calling uncompartmentalize_sbml_dfs()"
            " may not be appropriate"
        )

    # 1. update the compartments table to the stubbed default level: GO CELLULAR_COMPONENT
    stubbed_compartment = sbml_dfs_utils.stub_compartments().assign(
        c_Source=_create_stubbed_source()
    )

    # 3. create a 1-1 correspondence between species and new compartmentalized species. w/ GO CELLULAR_COMPONENT
    compspec_consensus_instances, compspec_lookup_table = _uncompartmentalize_cspecies(
        sbml_dfs, stubbed_compartment
    )

    # 4. update reaction species to the new compartmentalized species
    # 5. drop reactions if:
    #   - they are redundant (e.g., the same reaction occurred in multiple compartments)
    #   - substrates and products are identical (e.g., a transportation reaction)
    reactions, reaction_species = _uncompartmentalize_reactions(
        sbml_dfs, compspec_lookup_table
    )

    sbml_dfs.compartments = stubbed_compartment
    sbml_dfs.compartmentalized_species = compspec_consensus_instances
    sbml_dfs.reactions = reactions
    sbml_dfs.reaction_species = reaction_species

    sbml_dfs.remove_unused()
    sbml_dfs.validate()

    return None if inplace else sbml_dfs


def _uncompartmentalize_cspecies(
    sbml_dfs: sbml_dfs_core.SBML_dfs, stubbed_compartment: identifiers.Identifiers
) -> tuple[pd.Dataframe, pd.DataFrame]:
    """Convert compartmetnalized species into uncompartmentalized ones."""

    updated_cspecies = (
        sbml_dfs.compartmentalized_species.drop(
            [SBML_DFS.SC_NAME, SBML_DFS.C_ID, SBML_DFS.SC_SOURCE], axis=1
        )
        .merge(
            sbml_dfs.species[[SBML_DFS.S_NAME, SBML_DFS.S_SOURCE]],
            left_on=SBML_DFS.S_ID,
            right_index=True,
        )
        .reset_index()
        .rename(
            {
                SBML_DFS.SC_ID: "sc_id_old",
                SBML_DFS.S_NAME: SBML_DFS.SC_NAME,
                SBML_DFS.S_SOURCE: SBML_DFS.SC_SOURCE,
            },
            axis=1,
        )
    )

    # define new sc_ids as a 1-1 match to s_ids
    new_sc_ids = updated_cspecies[SBML_DFS.S_ID].drop_duplicates().to_frame()
    new_sc_ids[SBML_DFS.SC_ID] = sbml_dfs_utils.id_formatter(
        range(new_sc_ids.shape[0]), SBML_DFS.SC_ID
    )

    # add new identifiers
    updated_cspecies = updated_cspecies.merge(new_sc_ids)
    # add new compartment
    updated_cspecies[SBML_DFS.C_ID] = stubbed_compartment.index.tolist()[0]

    # create a lookup table of old -> new sc_ids
    compspec_lookup_table = (
        updated_cspecies.assign(model="uncompartmentalization")
        .rename({"sc_id_old": SBML_DFS.SC_ID, SBML_DFS.SC_ID: "new_id"}, axis=1)
        .set_index([SOURCE_SPEC.MODEL, SBML_DFS.SC_ID])["new_id"]
    )

    compspec_consensus_instances = updated_cspecies.groupby(SBML_DFS.SC_ID).first()[
        [SBML_DFS.S_ID, SBML_DFS.C_ID, SBML_DFS.SC_NAME, SBML_DFS.SC_SOURCE]
    ]

    return compspec_consensus_instances, compspec_lookup_table


def _uncompartmentalize_reactions(
    sbml_dfs: sbml_dfs_core.SBML_dfs, compspec_lookup_table: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Update reactions and reaction species to include uncompartmentalized species"""

    stubbed_index = _create_stubbed_index()

    # format sbml_dfs as a dict to take advantage of the consensus functions
    sbml_dfs_dict = {"uncompartmentalization": sbml_dfs}

    # merge reactions with identical stoichiometry
    rxn_consensus_species, rxn_lookup_table = consensus.construct_meta_entities_members(
        sbml_dfs_dict=sbml_dfs_dict,  # a single dict entry
        pw_index=stubbed_index,
        table=SBML_DFS.REACTIONS,
        defined_by=SBML_DFS.REACTION_SPECIES,
        defined_lookup_tables={SBML_DFS.SC_ID: compspec_lookup_table},
        defining_attrs=[SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY, SBML_DFS.SBO_TERM],
    )

    (
        rxnspec_consensus_instances,
        _,
    ) = consensus.construct_meta_entities_fk(
        sbml_dfs_dict=sbml_dfs_dict,  # a single dict entry
        pw_index=stubbed_index,
        table=SBML_DFS.REACTION_SPECIES,
        fk_lookup_tables={
            SBML_DFS.R_ID: rxn_lookup_table,
            SBML_DFS.SC_ID: compspec_lookup_table,
        },
        # retain species with different roles
        extra_defining_attrs=[SBML_DFS.SBO_TERM],
    )

    # drop reactions and reaction species where due to removal of compartments
    # the substrates and products are the same
    # this will mostly remove transporation reactions
    reactions, reaction_species = _filter_trivial_reactions(
        rxn_consensus_species, rxnspec_consensus_instances
    )

    return reactions, reaction_species


def _filter_trivial_reactions(
    rxn_consensus_species: pd.DataFrame, rxnspec_consensus_instances: pd.DataFrame
) -> tuple[pd.Dataframe, pd.DataFrame]:
    """Filter Trivial Reactions

    Filter reaction species which cancel out as substrates and products in the same reaction.

    Args:
        rxn_consensus_species (pd.DataFrame): reactions
        rxnspec_consensus_instances (pd.DataFrame): reaction species

    Returns:
        reactions (pd.DataFrame): reactions with trivial reactions dropped
        reaction_species (pd.DataFrame): reaction species with trivial reaction species dropped
    """

    # look for reactions where substrates and products cancel out
    reactants = rxnspec_consensus_instances.query("stoichiometry != 0")
    reactants_stoi_sum = (
        reactants[[SBML_DFS.R_ID, SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY]]
        .groupby([SBML_DFS.R_ID, SBML_DFS.SC_ID])
        .sum()
    )

    # identify cspecies which cancel out
    invalid_cspecies_in_reaction = reactants_stoi_sum.query("stoichiometry == 0")

    if invalid_cspecies_in_reaction.shape[0] > 0:
        logger.info(
            f"{invalid_cspecies_in_reaction.shape[0]} reactions species will be removed because they are substrates"
            " and products in the same reaction"
        )

    # find all cspecies which cancel outs original rsc_ids
    invalid_reaction_species = reactants.merge(
        invalid_cspecies_in_reaction,
        left_on=[SBML_DFS.R_ID, SBML_DFS.SC_ID],
        right_index=True,
    ).index.tolist()

    # update the reaction species table to reflect reaction_species which were dropped because
    # they were both substrates and products
    updated_reaction_species = rxnspec_consensus_instances[
        ~rxnspec_consensus_instances.index.isin(invalid_reaction_species)
    ]

    # identify valid reactions based on their presence in updated_reaction_species
    valid_reactions = rxn_consensus_species.index.isin(
        updated_reaction_species[SBML_DFS.R_ID]
    )

    invalid_reaction_names = rxn_consensus_species[~valid_reactions][
        SBML_DFS.R_NAME
    ].tolist()
    if len(invalid_reaction_names) > 0:
        logger.info(
            f"{len(invalid_reaction_names)} reactions where substrates and products cancel out"
            f" were dropped including: {' & '.join(invalid_reaction_names[0:5])}"
        )

    updated_reactions = rxn_consensus_species[valid_reactions]

    return updated_reactions, updated_reaction_species


def _create_stubbed_index() -> indices.PWIndex:
    """Create a default pathway index for the uncompartmentalized model."""

    stubbed_index_df = pd.DataFrame(
        {
            SOURCE_SPEC.FILE: None,
            SOURCE_SPEC.DATA_SOURCE: None,
            SOURCE_SPEC.ORGANISMAL_SPECIES: None,
            SOURCE_SPEC.PATHWAY_ID: "uncompartmentalization",
            SOURCE_SPEC.NAME: "Merging all compartments",
            SOURCE_SPEC.DATE: None,
        },
        index=[0],
    )
    stubbed_index = indices.PWIndex(stubbed_index_df, validate_paths=False)

    return stubbed_index


def _create_stubbed_source() -> source.Source:
    """Create a default Source object for the uncompartmetnalized model."""

    src = source.Source(
        pd.DataFrame([{"model": "uncompartmentalization"}]),
        pw_index=_create_stubbed_index(),
    )
    return src
