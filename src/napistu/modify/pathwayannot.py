from __future__ import annotations

import copy
import logging
import os
import re
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs
import numpy as np
import pandas as pd

from napistu import identifiers, sbml_dfs_core, sbml_dfs_utils, source, utils
from napistu.constants import (
    BQB,
    ENSEMBL_PREFIX_TO_ONTOLOGY,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    ONTOLOGIES,
    SBML_DFS,
    SBOTERM_NAMES,
)
from napistu.modify.constants import (
    NEO4_MEMBERS_SET,
    REACTOME_CROSSREF_SET,
)

logger = logging.getLogger(__name__)


def add_complex_formation_species(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add Complex Formation - Species

    Define all species in complexes and format newly created species

    Parameters
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A relational mechanistic network

    Returns
    -------
    merged_membership: pd.DataFrame
        A table of complexes and their component members
    new_species_for_sbml_dfs: pd.DataFrame
        New entries to add to sbml_dfs.species
    complex_component_species_ids: pd.DataFrame
        All complex components
    """

    # define all species
    # Use keep_source=True since we need s_Source for merging sources later
    species_ids = sbml_dfs.get_identifiers(SBML_DFS.SPECIES, keep_source=True)
    species_defining_attributes = species_ids[species_ids[IDENTIFIERS.BQB] == BQB.IS]
    complex_membership = species_ids[species_ids[IDENTIFIERS.BQB] == BQB.HAS_PART]

    # find the species corresponding to complex components (if they exist)
    merged_membership = complex_membership.merge(
        species_defining_attributes[
            [
                SBML_DFS.S_ID,
                IDENTIFIERS.ONTOLOGY,
                IDENTIFIERS.IDENTIFIER,
                IDENTIFIERS.URL,
            ]
        ].rename({SBML_DFS.S_ID: "component_s_id"}, axis=1),
        how="left",
    )

    # define unique component species
    complex_component_species = merged_membership[
        [
            "component_s_id",
            IDENTIFIERS.ONTOLOGY,
            IDENTIFIERS.IDENTIFIER,
            IDENTIFIERS.URL,
        ]
    ].drop_duplicates()

    # turn unnlisted identifiers back into identifier format
    complex_component_species[SBML_DFS.S_IDENTIFIERS] = [
        identifiers.Identifiers(
            [
                {
                    IDENTIFIERS.ONTOLOGY: complex_component_species[
                        IDENTIFIERS.ONTOLOGY
                    ].iloc[i],
                    IDENTIFIERS.IDENTIFIER: complex_component_species[
                        IDENTIFIERS.IDENTIFIER
                    ].iloc[i],
                    IDENTIFIERS.URL: complex_component_species[IDENTIFIERS.URL].iloc[i],
                    IDENTIFIERS.BQB: BQB.IS,
                }
            ]
        )
        for i in range(0, complex_component_species.shape[0])
    ]

    # create an identifier -> source lookup by collapsing all sources with the same defining id
    indexed_members = merged_membership.set_index(
        [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER, IDENTIFIERS.URL]
    ).sort_index()
    collapsed_sources = [
        source.merge_sources(indexed_members.loc[ind][SBML_DFS.S_SOURCE].tolist())
        for ind in indexed_members.index.unique()
    ]
    collapsed_sources = pd.Series(
        collapsed_sources, index=indexed_members.index.unique(), name=SBML_DFS.S_SOURCE
    )

    # add sources to unique complex components
    complex_component_species = complex_component_species.merge(
        collapsed_sources,
        left_on=[IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER, IDENTIFIERS.URL],
        right_index=True,
    )

    # define the maximum current id so that we can make new ids without collisions
    max_existing_sid = max(
        sbml_dfs_utils.id_formatter_inv(sbml_dfs.species.index.tolist())
    )
    # if s_ids used an alternative convention then they'll be nans here; which is fine
    if max_existing_sid is np.nan:
        max_existing_sid = int(-1)

    new_species = complex_component_species[
        complex_component_species["component_s_id"].isna()
    ]
    new_species["component_s_id"] = sbml_dfs_utils.id_formatter(
        range(max_existing_sid + 1, max_existing_sid + new_species.shape[0] + 1),
        SBML_DFS.S_ID,
    )

    # format new species and add to sbml_dfs.species
    new_species_for_sbml_dfs = (
        new_species.rename(
            {"component_s_id": SBML_DFS.S_ID, "identifier": SBML_DFS.S_NAME}, axis=1
        )[[SBML_DFS.S_ID, SBML_DFS.S_NAME, SBML_DFS.S_IDENTIFIERS, SBML_DFS.S_SOURCE]]
        .set_index("s_id")
        .sort_index()
    )

    # prepend zzauto so the string comes late alphanumerically. this way a properly named species will
    # be preferred when merging species by identifiers
    new_species_for_sbml_dfs[SBML_DFS.S_NAME] = (
        "zzauto " + new_species_for_sbml_dfs[SBML_DFS.S_NAME]
    )

    # combine existing and newly defined complex components
    complex_component_species_ids = pd.concat(
        [
            complex_component_species[
                ~complex_component_species["component_s_id"].isna()
            ],
            new_species,
        ]
    )

    return merged_membership, new_species_for_sbml_dfs, complex_component_species_ids


def add_complex_formation(sbml_dfs: sbml_dfs_core.SBML_dfs):
    """
    Add Complex Formation

    Using Reactome-style complex annotations,
    where complex components are an attribute of complexes,
    add explicit complex formation reactions.

    Reactome represents complexers using BQB_HAS_PART
    annotations, which are extracted into identifiers.Identifiers
    objects. This is sufficient to define membership but does
    not include stoichiometry. Also, in this approach components
    are defined by their identifiers (URIs) rather than internal
    s_ids/sc_ids.
    """

    raise NotImplementedError(
        "TO DO - Need to look closer to see if the unformed complexes really need a formation reaction"
    )


"""     # define species present in complexes
    (
        merged_membership,
        new_species_for_sbml_dfs,
        complex_component_species_ids,
    ) = add_complex_formation_species(sbml_dfs)

    # define compartmentalized species present in complexes
    (
        new_compartmentalized_species_for_sbml_dfs,
        updated_compartmentalized_membership,
    ) = _add_complex_formation_compartmentalized_species(
        sbml_dfs,
        merged_membership,
        new_species_for_sbml_dfs,
        complex_component_species_ids.drop("s_Source", axis=1),
    )

    # remove complex formation for reactions which already have clear formation reactions
    # to flag these complexes look for cases where the membership of the substrates
    # and products (including complex membership) are the same

    reaction_species_expanded_complexes = sbml_dfs.reaction_species.merge(
        updated_compartmentalized_membership[["sc_id", "component_sc_id"]], how="left"
    )

    # if a species is not a complex then it is its own component
    reaction_species_expanded_complexes["component_sc_id"] = [
        x if z else y
        for x, y, z in zip(
            reaction_species_expanded_complexes["sc_id"],
            reaction_species_expanded_complexes["component_sc_id"],
            reaction_species_expanded_complexes["component_sc_id"].isna(),
        )
    ]

    # check for equal membership of substrates and products
    reaction_species_expanded_complexes = reaction_species_expanded_complexes.set_index(
        "r_id"
    )

    complex_formation_reactions = list()
    for rxn in reaction_species_expanded_complexes.index.unique():
        rxn_species = reaction_species_expanded_complexes.loc[rxn]
        substrates = set(
            rxn_species[rxn_species["stoichiometry"] < 0]["component_sc_id"].tolist()
        )
        products = set(
            rxn_species[rxn_species["stoichiometry"] > 0]["component_sc_id"].tolist()
        )

        if substrates == products:
            complex_formation_reactions.append(rxn)

    # find complexes which are products of complex formation reactions

    compartmentalized_complexes = updated_compartmentalized_membership["sc_id"].unique()

    # is a complex formation reaction
    formed_complexes = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species["r_id"].isin(complex_formation_reactions)
    ]
    # is a complex
    formed_complexes = formed_complexes[
        formed_complexes["sc_id"].isin(compartmentalized_complexes)
    ]
    # complex is product
    formed_complexes = formed_complexes[formed_complexes["stoichiometry"] > 0]

    formed_complexes = formed_complexes["sc_id"].unique()
    _ = set(compartmentalized_complexes).difference(set(formed_complexes))

    # add formation and dissolution reactions for all complexes without explicit formation reactions
     """


def add_entity_sets(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    neo4j_members: str,
) -> sbml_dfs_core.SBML_dfs:
    """
    Add Entity Sets

    Reactome represents some sets of interchangeable molecules as "entity sets".
    Common examples are ligands for a receptor. This function add members
    of each entity set as a "is a" style reaction.

    Parameters
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A relational mechanistic network
    neo4j_members: str
        Path to a table containing Reactome entity sets and corresponding members.
        This is currently extracted manually with Neo4j.

    Returns
    -------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        An updated database which includes entity set species and formation reactions

    """

    # read and reformat Reactome entity sets table
    reactome_members = _read_neo4j_members(neo4j_members)

    # create missing species and compartmentalized species
    logger.info("Adding entity set species")
    (
        merged_membership,
        new_species_for_sbml_dfs,
        set_component_species_ids,
    ) = _add_entity_sets_species(sbml_dfs, reactome_members)

    logger.info("Adding complex formation species")
    (
        new_compartmentalized_species_for_sbml_dfs,
        updated_compartmentalized_membership,
    ) = _add_complex_formation_compartmentalized_species(
        sbml_dfs,
        merged_membership,
        new_species_for_sbml_dfs,
        set_component_species_ids,
    )

    logger.info("Adding entity set reactions")
    (
        new_reactions_for_sbml_dfs,
        new_reaction_species_for_sbml_dfs,
    ) = _add_entity_sets_reactions(
        sbml_dfs,
        new_compartmentalized_species_for_sbml_dfs,
        updated_compartmentalized_membership,
    )

    # add all of the new entries to the sbml_dfs
    sbml_dfs_working = copy.copy(sbml_dfs)

    sbml_dfs_working.species = pd.concat(
        [sbml_dfs_working.species, new_species_for_sbml_dfs]
    )
    sbml_dfs_working.compartmentalized_species = pd.concat(
        [
            sbml_dfs_working.compartmentalized_species,
            new_compartmentalized_species_for_sbml_dfs,
        ]
    )
    sbml_dfs_working.reactions = pd.concat(
        [sbml_dfs_working.reactions, new_reactions_for_sbml_dfs]
    )
    sbml_dfs_working.reaction_species = pd.concat(
        [sbml_dfs_working.reaction_species, new_reaction_species_for_sbml_dfs]
    )

    return sbml_dfs_working


def add_reactome_identifiers(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    crossref_path: str,
) -> sbml_dfs_core.SBML_dfs:
    """
    Add Reactome Identifiers

    Add reactome-specific identifiers to existing species

    Params
    ------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A pathway model
    crossref_path:
        Path to the cross ref file extracted from Reactome's Neo4j database

    Returns
    -------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A pathway model with updated species' identifiers

    """

    logger.info("Reading Reactome crossref ids")
    select_reactome_ids = _read_reactome_crossref_ids(crossref_path)

    # read all current identifiers
    current_ids = sbml_dfs.get_identifiers(SBML_DFS.SPECIES, keep_source=True)
    # filter annotations of homologues and literature references
    current_molecular_ids = (
        current_ids[current_ids[IDENTIFIERS.BQB].isin([BQB.IS, BQB.HAS_PART])]
        .set_index([SBML_DFS.S_ID, IDENTIFIERS.BQB])
        .sort_index()
        .copy()
    )

    # combine existing s_ids with additional cross-ref annotations using uniprot ids
    logger.info("Merging Reactome crossref ids with existing identifiers")
    merged_crossrefs = _merge_reactome_crossref_ids(
        current_molecular_ids, select_reactome_ids
    )

    # create identifiers objects for each s_id
    combined_ids = (
        pd.concat(
            [
                current_ids[
                    [
                        SBML_DFS.S_ID,
                        IDENTIFIERS.ONTOLOGY,
                        IDENTIFIERS.IDENTIFIER,
                        IDENTIFIERS.URL,
                        IDENTIFIERS.BQB,
                    ]
                ],
                merged_crossrefs[
                    [
                        SBML_DFS.S_ID,
                        IDENTIFIERS.ONTOLOGY,
                        IDENTIFIERS.IDENTIFIER,
                        IDENTIFIERS.URL,
                        IDENTIFIERS.BQB,
                    ]
                ],
            ]
        )
        .reset_index(drop=True)
        .drop_duplicates()
    )

    updated_identifiers = {
        k: identifiers.Identifiers(
            list(
                v[
                    [
                        IDENTIFIERS.ONTOLOGY,
                        IDENTIFIERS.IDENTIFIER,
                        IDENTIFIERS.URL,
                        IDENTIFIERS.BQB,
                    ]
                ]
                .T.to_dict()
                .values()
            )
        )
        for k, v in combined_ids.groupby(SBML_DFS.S_ID)
    }
    updated_identifiers = pd.Series(
        updated_identifiers, index=updated_identifiers.keys()
    )
    updated_identifiers.index.name = SBML_DFS.S_ID
    updated_identifiers.name = "new_Identifiers"

    # add new identifiers to species tabl
    logger.info("Adding new identifiers to species table")
    updated_species = sbml_dfs.species.merge(
        updated_identifiers,
        left_index=True,
        right_index=True,
        how="outer",
        indicator=True,
    )

    if updated_species[updated_species["_merge"] == "right_only"].shape[0] > 0:
        raise ValueError("Reactome crossrefs added new sids; this shouldn't occur")

    updated_species = pd.concat(
        [
            updated_species[updated_species["_merge"] == "both"]
            .drop([SBML_DFS.S_IDENTIFIERS, "_merge"], axis=1)
            .rename({"new_Identifiers": SBML_DFS.S_IDENTIFIERS}, axis=1),
            # retain original Identifiers if there is not new_Identifiers object
            # (this would occur if there were not identifiers)
            updated_species[updated_species["_merge"] == "left_only"].drop(
                ["new_Identifiers", "_merge"], axis=1
            ),
        ]
    )

    n_species_diff = updated_species.shape[0] - sbml_dfs.species.shape[0]
    if n_species_diff != 0:
        raise ValueError(
            f"There are {n_species_diff} more species in the updated "
            "species table than the original one; this is unexpected behavior"
        )

    # create a copy to return a new object rather than update the provided one
    sbml_dfs_working = copy.copy(sbml_dfs)
    sbml_dfs_working.species = updated_species
    return sbml_dfs_working


def _add_entity_sets_species(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    reactome_members: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add Entity Sets - Species

    Define all species which are part of "entity sets" in the pathway

    Parameters
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A relational mechanistic network
    reactome_members: pd.DataFrame
        A table of all Reactome entity sets members - obtained using a Neo4j query

    Returns
    -------
    merged_membership: pd.DataFrame
        A table of complexes and their component members
    new_species_for_sbml_dfs: pd.DataFrame
        New entries to add to sbml_dfs.species
    set_component_species_ids: pd.DataFrame
        All set components
    """

    species_ids = sbml_dfs.get_identifiers(SBML_DFS.SPECIES, keep_source=True)
    reactome_ids = species_ids[
        species_ids[IDENTIFIERS.ONTOLOGY] == ONTOLOGIES.REACTOME
    ].copy()
    reactome_ids = reactome_ids[reactome_ids[IDENTIFIERS.BQB] == BQB.IS]

    # compare Reactome ids in sbml_dfs and reactome_members to make sure
    # they are for the same species
    identifiers.check_reactome_identifier_compatibility(
        reactome_members["member_id"], reactome_ids[IDENTIFIERS.IDENTIFIER]
    )

    # merge each species' entity sets to define entities which must exist in this pathway
    merged_membership = (
        reactome_ids[[SBML_DFS.S_ID, IDENTIFIERS.IDENTIFIER, SBML_DFS.S_SOURCE]]
        .rename({IDENTIFIERS.IDENTIFIER: "set_id"}, axis=1)
        .merge(reactome_members)
    )

    # define unique component species
    set_component_species = merged_membership[
        [
            "member_id",
            IDENTIFIERS.ONTOLOGY,
            IDENTIFIERS.IDENTIFIER,
            IDENTIFIERS.URL,
            "member_s_name",
        ]
    ].drop_duplicates()

    distinct_members = set_component_species.set_index(
        [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER, IDENTIFIERS.URL, "member_s_name"]
    ).sort_index()

    # since reactome IDs are compartmentalized, use external IDs only
    # to determine distinct species, but then add reactome IDs as well
    distinct_members = pd.Series(
        [
            identifiers.Identifiers(
                [
                    {
                        IDENTIFIERS.ONTOLOGY: ind[0],
                        IDENTIFIERS.IDENTIFIER: str(ind[1]),
                        IDENTIFIERS.URL: ind[2],
                        IDENTIFIERS.BQB: BQB.IS,
                    }
                ]
                + [
                    {
                        IDENTIFIERS.ONTOLOGY: ONTOLOGIES.REACTOME,
                        IDENTIFIERS.IDENTIFIER: x,
                        IDENTIFIERS.URL: "",
                        IDENTIFIERS.BQB: BQB.IS,
                    }
                    for x in utils.safe_series_tolist(
                        distinct_members.loc[ind, "member_id"]
                    )
                ]
            )
            for ind in distinct_members.index.unique()
        ],
        index=distinct_members.index.unique(),
        name=SBML_DFS.S_IDENTIFIERS,
    )

    utils.check_unique_index(distinct_members, "distinct_members")

    # combine identical species' sources
    indexed_members = merged_membership.set_index(
        [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER, IDENTIFIERS.URL]
    ).sort_index()

    collapsed_sources = [
        source._safe_source_merge(indexed_members.loc[ind][SBML_DFS.S_SOURCE])
        for ind in indexed_members.index.unique()
    ]
    collapsed_sources = pd.Series(
        collapsed_sources, index=indexed_members.index.unique(), name=SBML_DFS.S_SOURCE
    )

    # add sources to unique set components
    distinct_members = distinct_members.to_frame().join(collapsed_sources.to_frame())

    utils.check_unique_index(distinct_members, "distinct_members (with sources)")

    # define set members which already exist as species versus those that must be added
    set_component_species["is_already_included"] = set_component_species[
        "member_id"
    ].isin(reactome_ids[IDENTIFIERS.IDENTIFIER])

    # define the maximum current id so that we can make new ids without collisions
    max_existing_sid = max(
        sbml_dfs_utils.id_formatter_inv(sbml_dfs.species.index.tolist())
    )
    # if s_ids used an alternative convention then they'll be nans here; which is fine
    if max_existing_sid is np.nan:
        max_existing_sid = int(-1)

    new_species = set_component_species[
        ~set_component_species["is_already_included"]
    ].copy()
    new_species["component_s_id"] = sbml_dfs_utils.id_formatter(
        range(max_existing_sid + 1, max_existing_sid + new_species.shape[0] + 1),
        SBML_DFS.S_ID,
    )

    # define new unique species
    new_species_for_sbml_dfs = (
        new_species.merge(
            distinct_members,
            left_on=[
                IDENTIFIERS.ONTOLOGY,
                IDENTIFIERS.IDENTIFIER,
                IDENTIFIERS.URL,
                "member_s_name",
            ],
            right_index=True,
        )[
            [
                "component_s_id",
                "member_s_name",
                SBML_DFS.S_IDENTIFIERS,
                SBML_DFS.S_SOURCE,
            ]
        ]
        .rename(
            {"component_s_id": SBML_DFS.S_ID, "member_s_name": SBML_DFS.S_NAME}, axis=1
        )
        .set_index(SBML_DFS.S_ID)
        .sort_index()
    )

    utils.check_unique_index(new_species_for_sbml_dfs, "new_species_for_sbml_dfs")

    # combine existing and newly defined set components
    set_component_species_ids = pd.concat(
        [
            set_component_species[set_component_species["is_already_included"]].merge(
                reactome_ids[[SBML_DFS.S_ID, IDENTIFIERS.IDENTIFIER]].rename(
                    {
                        IDENTIFIERS.IDENTIFIER: "member_id",
                        SBML_DFS.S_ID: "component_s_id",
                    },
                    axis=1,
                )
            ),
            new_species,
        ]
    )

    return merged_membership, new_species_for_sbml_dfs, set_component_species_ids


def _add_entity_sets_reactions(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    new_compartmentalized_species_for_sbml_dfs: pd.DataFrame,
    updated_compartmentalized_membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add Entity Sets - Reactions

    Create reactions which indicate membership in an entity set

    Parameters
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A relational mechanistic network
    new_compartmentalized_species_for_sbml_dfs: pd.DataFrame
        New entries to add to sbml_dfs.compartmentalized_species
    updated_compartmentalized_membership: pd.DataFrame
        Compartmentalized complex components with updated IDs

    Returns
    -------
    new_reactions_for_sbml_dfs: pd.DataFrame
        New entries to add to sbml_dfs.reactions
    new_reaction_species_for_sbml_dfs: pd.DataFrame
        New entries to add to sbml_dfs.reaction_species
    """

    all_compartmentalized_species = pd.concat(
        [sbml_dfs.compartmentalized_species, new_compartmentalized_species_for_sbml_dfs]
    )

    # create a table with named "entity sets" and their members
    # each row will be turned into an "IS A" reaction
    named_set_components = updated_compartmentalized_membership[
        [SBML_DFS.SC_ID, SBML_DFS.SC_NAME, SBML_DFS.SC_SOURCE, "component_sc_id"]
    ].merge(
        all_compartmentalized_species[[SBML_DFS.SC_NAME]].rename(
            {SBML_DFS.SC_NAME: "component_sc_name"}, axis=1
        ),
        left_on="component_sc_id",
        right_index=True,
        how="left",
    )

    if any(named_set_components["component_sc_name"].isna()):
        raise ValueError("Some components could not be merged")

    # define newly added reactions
    max_existing_rid = max(
        sbml_dfs_utils.id_formatter_inv(sbml_dfs.reactions.index.tolist())
    )
    # if s_ids used an alternative convention then they'll be nans here; which is fine
    if max_existing_rid is np.nan:
        max_existing_rid = int(-1)

    # name the reaction following the "IS A" convention
    named_set_components[SBML_DFS.R_NAME] = [
        f"{comp_sc} IS A {sc}"
        for comp_sc, sc in zip(
            named_set_components["component_sc_name"], named_set_components["sc_name"]
        )
    ]

    named_set_components[SBML_DFS.R_ID] = sbml_dfs_utils.id_formatter(
        range(
            max_existing_rid + 1,
            max_existing_rid + named_set_components.shape[0] + 1,
        ),
        SBML_DFS.R_ID,
    )

    named_set_components[SBML_DFS.R_SOURCE] = named_set_components[SBML_DFS.SC_SOURCE]
    named_set_components[SBML_DFS.R_IDENTIFIERS] = [
        identifiers.Identifiers([]) for i in range(0, named_set_components.shape[0])
    ]

    new_reactions_for_sbml_dfs = (
        named_set_components[
            [SBML_DFS.R_ID, SBML_DFS.R_NAME, SBML_DFS.R_IDENTIFIERS, SBML_DFS.R_SOURCE]
        ]
        .set_index(SBML_DFS.R_ID)
        .sort_index()
        .assign(r_isreversible=False)
    )

    # define newly added reactions' species

    max_existing_rscid = max(
        sbml_dfs_utils.id_formatter_inv(sbml_dfs.reaction_species.index.tolist())
    )
    if max_existing_rscid is np.nan:
        max_existing_rscid = int(-1)

    new_reaction_species_for_sbml_dfs = pd.concat(
        [
            named_set_components[["component_sc_id", SBML_DFS.R_ID]]
            .rename({"component_sc_id": SBML_DFS.SC_ID}, axis=1)
            .assign(stoichiometry=-1)
            .assign(sbo_term=MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT]),
            named_set_components[[SBML_DFS.SC_ID, SBML_DFS.R_ID]]
            .assign(stoichiometry=1)
            .assign(sbo_term=MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT]),
        ]
    ).sort_values([SBML_DFS.R_ID, SBML_DFS.STOICHIOMETRY])

    new_reaction_species_for_sbml_dfs[SBML_DFS.RSC_ID] = sbml_dfs_utils.id_formatter(
        range(
            max_existing_rscid + 1,
            max_existing_rscid + new_reaction_species_for_sbml_dfs.shape[0] + 1,
        ),
        SBML_DFS.RSC_ID,
    )

    new_reaction_species_for_sbml_dfs = new_reaction_species_for_sbml_dfs.set_index(
        SBML_DFS.RSC_ID
    ).sort_index()

    return new_reactions_for_sbml_dfs, new_reaction_species_for_sbml_dfs


def _add_complex_formation_compartmentalized_species(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    merged_membership: pd.DataFrame,
    new_species_for_sbml_dfs: pd.DataFrame,
    complex_component_species_ids: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add Complex Formation - Compartmentalized Species

    Define all compartmentalized species in complexes and format newly created compartmentalized species

    Parameters
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A relational mechanistic network
    merged_membership: pd.DataFrame
        A table of complexes and their component members
    new_species_for_sbml_dfs: pd.DataFrame
        New entries to add to sbml_dfs.species
    complex_component_species_ids: pd.DataFrame
        All complex components

    Returns
    -------
    new_compartmentalized_species_for_sbml_dfs: pd.DataFrame
        New entries to add to sbml_dfs.compartmentalized_species
    updated_compartmentalized_membership: pd.DataFrame
        Compartmentalized complex components with updated IDs
    """

    # filter compartmentalized species to complexes
    complexes = merged_membership[SBML_DFS.S_ID].unique()
    compartmentalized_complexes = sbml_dfs.compartmentalized_species[
        sbml_dfs.compartmentalized_species[SBML_DFS.S_ID].isin(complexes)
    ]

    # create appropriate compartmentalized species
    # merge compartmentalized complexes with their membership
    merged_compartmentalized_membership = (
        compartmentalized_complexes.reset_index().merge(
            merged_membership[
                [
                    SBML_DFS.S_ID,
                    IDENTIFIERS.ONTOLOGY,
                    IDENTIFIERS.IDENTIFIER,
                    IDENTIFIERS.URL,
                ]
            ].merge(complex_component_species_ids)
        )
    )

    # define all of the compartmentalized species that should exist
    complex_component_compartmentalized_species = (
        merged_compartmentalized_membership[["component_s_id", SBML_DFS.C_ID]]
        .drop_duplicates()
        .merge(
            sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID, SBML_DFS.C_ID]]
            .reset_index()
            .rename({SBML_DFS.S_ID: "component_s_id"}, axis=1),
            how="left",
        )
    )

    new_compartmentalized_species = complex_component_compartmentalized_species[
        complex_component_compartmentalized_species[SBML_DFS.SC_ID].isna()
    ].copy()

    # add new identifiers
    max_existing_scid = max(
        sbml_dfs_utils.id_formatter_inv(
            sbml_dfs.compartmentalized_species.index.tolist()
        )
    )
    if max_existing_scid is np.nan:
        max_existing_scid = int(-1)

    new_compartmentalized_species[SBML_DFS.SC_ID] = sbml_dfs_utils.id_formatter(
        range(
            max_existing_scid + 1,
            max_existing_scid + new_compartmentalized_species.shape[0] + 1,
        ),
        SBML_DFS.SC_ID,
    )

    all_species = pd.concat([sbml_dfs.species, new_species_for_sbml_dfs])

    # name new sc_ids and inherit sources from their complexes
    new_compartmentalized_species_names = new_compartmentalized_species.merge(
        all_species[SBML_DFS.S_NAME],
        left_on="component_s_id",
        right_index=True,
        how="left",
    ).merge(
        sbml_dfs.compartments[SBML_DFS.C_NAME],
        left_on=SBML_DFS.C_ID,
        right_index=True,
        how="left",
    )

    if any(new_compartmentalized_species_names[SBML_DFS.S_NAME].isna()):
        raise ValueError("Some species were unnamed")
    if any(new_compartmentalized_species_names[SBML_DFS.C_NAME].isna()):
        raise ValueError("Some compartmnets were unnamed")

    # name compartmentalized species
    new_compartmentalized_species_names[SBML_DFS.SC_NAME] = [
        f"{s_name} [{c_name}]"
        for s_name, c_name in zip(
            new_compartmentalized_species_names[SBML_DFS.S_NAME],
            new_compartmentalized_species_names[SBML_DFS.C_NAME],
        )
    ]

    # add sources from the complexes that compartmentalized species belong to
    indexed_cmembers = (
        merged_compartmentalized_membership[
            ["component_s_id", SBML_DFS.C_ID, SBML_DFS.SC_SOURCE]
        ]
        .set_index(["component_s_id", SBML_DFS.C_ID])
        .sort_index()
    )

    collapsed_csources = [
        (
            source.merge_sources(indexed_cmembers.loc[ind][SBML_DFS.SC_SOURCE].tolist())
            if len(ind) == 1
            else indexed_cmembers.loc[ind][SBML_DFS.SC_SOURCE]
        )
        for ind in indexed_cmembers.index.unique()
    ]
    collapsed_csources = pd.Series(
        collapsed_csources,
        index=indexed_cmembers.index.unique(),
        name=SBML_DFS.SC_SOURCE,
    )

    new_compartmentalized_species_names = new_compartmentalized_species_names.merge(
        collapsed_csources, left_on=["component_s_id", SBML_DFS.C_ID], right_index=True
    )

    new_compartmentalized_species_for_sbml_dfs = (
        new_compartmentalized_species_names[
            [
                SBML_DFS.SC_ID,
                SBML_DFS.SC_NAME,
                "component_s_id",
                SBML_DFS.C_ID,
                SBML_DFS.SC_SOURCE,
            ]
        ]
        .rename({"component_s_id": SBML_DFS.S_ID}, axis=1)
        .set_index(SBML_DFS.SC_ID)
    )

    utils.check_unique_index(
        new_compartmentalized_species_for_sbml_dfs,
        "new_compartmentalized_species_for_sbml_dfs",
    )

    # combine old and new compartmentalized species using current sc_ids
    complex_compartmentalized_components_ids = pd.concat(
        [
            complex_component_compartmentalized_species[
                ~complex_component_compartmentalized_species[SBML_DFS.SC_ID].isna()
            ],
            new_compartmentalized_species,
        ]
    ).rename({SBML_DFS.SC_ID: "component_sc_id"}, axis=1)

    updated_compartmentalized_membership = merged_compartmentalized_membership[
        [
            SBML_DFS.SC_ID,
            SBML_DFS.SC_NAME,
            SBML_DFS.S_ID,
            SBML_DFS.C_ID,
            "component_s_id",
            SBML_DFS.SC_SOURCE,
        ]
    ].merge(complex_compartmentalized_components_ids)

    return (
        new_compartmentalized_species_for_sbml_dfs,
        updated_compartmentalized_membership,
    )


def _read_neo4j_members(neo4j_members: str) -> pd.DataFrame:
    """Read a table containing entity sets (members) derived from Reactome's Neo4J database."""

    # load a list containing Reactome entity sets -> members
    # entity sets are categories of molecular species that
    # share a common property such as serving as ligands for a receptor
    # these relationships are not represented in the Reactome .sbml
    # so they are pulled out of the Neo4j database.
    base, path = os.path.split(neo4j_members)
    with open_fs(base) as bfs:
        with bfs.open(path, "rb") as f:
            reactome_members = pd.read_csv(f).assign(url="")

    # check that the expected columns are present
    utils.match_pd_vars(reactome_members, NEO4_MEMBERS_SET).assert_present()

    reactome_members[IDENTIFIERS.ONTOLOGY] = reactome_members[
        IDENTIFIERS.ONTOLOGY
    ].str.lower()

    # add an uncompartmentalized name
    reactome_members["member_s_name"] = [
        re.sub(" \\[[A-Za-z ]+\\]$", "", x) for x in reactome_members["member_name"]
    ]
    reactome_members[IDENTIFIERS.IDENTIFIER] = reactome_members[
        IDENTIFIERS.IDENTIFIER
    ].astype(str)

    return reactome_members


def _merge_reactome_crossref_ids(
    current_molecular_ids: pd.DataFrame,
    select_reactome_ids: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Reactome CrossRef IDs

    Combine existing molecular IDs with Reactome crossref identifiers.

    Params
    ------
    current_molecular_ids: pd.DataFrame
        Molecular features in the current pathway model
    select_reactome_ids: pd.DataFrame
        Crossref identifiers produced by _format_reactome_crossref_ids()

    Returns
    -------
    merged_crossrefs: pd.DataFrame
        Molecular feature sids matched to crossref annotations

    """

    # reactome IDs to identifiers.Identifiers
    id_indices = current_molecular_ids.index.unique()
    # ind = id_indices[1]

    # loop through all s_id x bqb pairs
    uniprot_ids = list()
    uniprot_ids_w_reactome = list()
    for ind in id_indices:
        ind_ids = current_molecular_ids.loc[ind]
        ontologies_present = ind_ids[IDENTIFIERS.ONTOLOGY].unique()
        if ONTOLOGIES.UNIPROT in ontologies_present:
            # return all (s_id, bqb) -> uniprot entries
            # save the uniprot source since it will be propagated to new ids joined to the uniprot id
            entry_uniprot_ids = (
                ind_ids.loc[ind_ids[IDENTIFIERS.ONTOLOGY] == ONTOLOGIES.UNIPROT]
                .reset_index()[
                    [
                        SBML_DFS.S_ID,
                        IDENTIFIERS.BQB,
                        IDENTIFIERS.IDENTIFIER,
                        SBML_DFS.S_SOURCE,
                    ]
                ]
                .rename({IDENTIFIERS.IDENTIFIER: ONTOLOGIES.UNIPROT}, axis=1)
            )
            # remove trailing dashes in uniprot ids since they are not present in the crossref identifiers
            entry_uniprot_ids[ONTOLOGIES.UNIPROT] = entry_uniprot_ids[
                ONTOLOGIES.UNIPROT
            ].replace("\\-[0-9]+$", "", regex=True)

            uniprot_ids.append(entry_uniprot_ids)

            # add reactome ids to lookup if they exist (they won't for BQB_HAS_PART qualifiers)
            if ONTOLOGIES.REACTOME in ontologies_present:
                # create the all x all cross of bqb-matched reactome and uniprot ids
                entry_reactome = (
                    ind_ids.loc[ind_ids[IDENTIFIERS.ONTOLOGY] == ONTOLOGIES.REACTOME]
                    .reset_index()[
                        [SBML_DFS.S_ID, IDENTIFIERS.BQB, IDENTIFIERS.IDENTIFIER]
                    ]
                    .rename({IDENTIFIERS.IDENTIFIER: "reactome_id"}, axis=1)
                )
                uniprot_ids_w_reactome.append(entry_uniprot_ids.merge(entry_reactome))

    uniprot_ids = pd.concat(uniprot_ids)
    uniprot_ids_w_reactome = pd.concat(uniprot_ids_w_reactome)

    # uniprot_ids_w_reactome
    uni_rct_with_crossrefs = uniprot_ids_w_reactome.merge(select_reactome_ids)
    # check ontologies
    uni_rct_with_crossrefs_ensembl_genes = uni_rct_with_crossrefs.loc[
        uni_rct_with_crossrefs[IDENTIFIERS.ONTOLOGY] == ONTOLOGIES.ENSEMBL_GENE,
        SBML_DFS.S_ID,
    ].unique()

    failed_joins = uniprot_ids_w_reactome[
        ~uniprot_ids_w_reactome[SBML_DFS.S_ID].isin(
            uni_rct_with_crossrefs_ensembl_genes
        )
    ]
    # most of the failed joins are pathogens so they wouldn't match to human ensembl genes
    if failed_joins.shape[0] > 0:
        logged_join_fails = failed_joins.sample(min(failed_joins.shape[0], 5)).drop(
            SBML_DFS.S_SOURCE, axis=1
        )
        logger.warning(
            f"{failed_joins.shape[0]} network uniprot IDs were not matched to the Reactome Crossref IDs"
        )

        utils.show(logged_join_fails, headers="keys", hide_index=True)

    # entries without reactome IDs join just by uniprot
    # outer join back to uni_rct_with_crossrefs so we won't consider a uniprot-only match
    # when a uniprot + reactome match worked [its not entirely clear that this does anything]
    uni_no_rct_with_crossrefs = uniprot_ids.merge(select_reactome_ids).merge(
        uni_rct_with_crossrefs[[SBML_DFS.S_ID, IDENTIFIERS.BQB]].drop_duplicates(),
        how="outer",
        indicator=True,
    )
    uni_no_rct_with_crossrefs = uni_no_rct_with_crossrefs[
        uni_no_rct_with_crossrefs["_merge"] == "left_only"
    ].drop("_merge", axis=1)

    merged_crossrefs = pd.concat([uni_rct_with_crossrefs, uni_no_rct_with_crossrefs])
    if (
        not (uni_rct_with_crossrefs.shape[0] + uni_no_rct_with_crossrefs.shape[0])
        == merged_crossrefs.shape[0]
    ):
        raise ValueError(
            "The number of merged crossrefs does not match the sum of the number of uniprot + reactome crossrefs and uniprot-only crossrefs"
        )

    species_with_protein_and_no_gene = current_molecular_ids[
        current_molecular_ids[IDENTIFIERS.ONTOLOGY] == ONTOLOGIES.UNIPROT
    ].merge(
        merged_crossrefs.loc[
            merged_crossrefs[IDENTIFIERS.ONTOLOGY] == ONTOLOGIES.ENSEMBL_GENE,
            [SBML_DFS.S_ID, IDENTIFIERS.BQB],
        ].drop_duplicates(),
        how="outer",
        left_index=True,
        right_on=[SBML_DFS.S_ID, IDENTIFIERS.BQB],
        indicator=True,
    )
    species_with_protein_and_no_gene = species_with_protein_and_no_gene[
        species_with_protein_and_no_gene["_merge"] == "left_only"
    ][[SBML_DFS.S_ID, SBML_DFS.S_NAME, IDENTIFIERS.BQB]].drop_duplicates()

    if species_with_protein_and_no_gene.shape[0] > 0:
        logged_join_fails = species_with_protein_and_no_gene.sample(
            min(species_with_protein_and_no_gene.shape[0], 5)
        )

        logger.warning(
            f"A gene ID could not be found for {species_with_protein_and_no_gene.shape[0]} "
            "(species, bqb) pairs with a protein ID"
        )
        utils.show(logged_join_fails, headers="keys", hide_index=True)

    return merged_crossrefs


def _read_reactome_crossref_ids(
    crossref_path: str,
) -> pd.DataFrame:
    """
    Format Reactome CrossRef IDs

    Read and reformat Reactome's crossref identifiers

    Params
    ------
    crossref_path: str
        Path to the cross ref file extracted from Reactome's Neo4j database

    Returns
    -------
    select_reactome_ids: pd.DataFrame
        Crossref identifiers

    """

    base, path = os.path.split(crossref_path)
    with open_fs(base) as bfs:
        with bfs.open(path, "rb") as f:
            reactome_ids = pd.read_csv(f)

    # check that the expected columns are present
    utils.match_pd_vars(reactome_ids, REACTOME_CROSSREF_SET).assert_present()

    # only use ensembl and pharos for now

    # rename pharos ontology
    pharos_ids = reactome_ids[
        reactome_ids[IDENTIFIERS.ONTOLOGY] == "Pharos - Targets"
    ].copy()
    pharos_ids[IDENTIFIERS.ONTOLOGY] = ONTOLOGIES.PHAROS

    # format ensembl ids using conventions in identifiers.Identifiers
    ensembl_ids = reactome_ids[reactome_ids[IDENTIFIERS.ONTOLOGY] == "Ensembl"].copy()
    # distinguish ensembl genes/transcripts/proteins
    ensembl_ids["ontology_prefix"] = ensembl_ids[IDENTIFIERS.IDENTIFIER].str.slice(
        start=0, stop=4
    )
    ensembl_ids[IDENTIFIERS.ONTOLOGY] = [
        ENSEMBL_PREFIX_TO_ONTOLOGY[p] for p in ensembl_ids["ontology_prefix"]
    ]
    ensembl_ids = ensembl_ids.drop("ontology_prefix", axis=1)

    select_reactome_ids = pd.concat([pharos_ids, ensembl_ids])

    return select_reactome_ids
