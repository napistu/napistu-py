from __future__ import annotations

import os
import warnings
from typing import Any

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs
import numpy as np
import pandas as pd

from napistu import identifiers, sbml_dfs_core, sbml_dfs_utils, source
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    SBML_DFS,
    SBML_DFS_SCHEMA,
    SBOTERM_NAMES,
    SCHEMA_DEFS,
)
from napistu.modify.constants import CURATION_DEFS, VALID_ANNOTATION_TYPES

# Public functions (alphabetical order)


def curate_sbml_dfs(
    curation_dir: str, sbml_dfs: sbml_dfs_core.SBML_dfs, verbose: bool = True
) -> sbml_dfs_core.SBML_dfs:
    """
    Curate SBML_dfs

    Update a pathway model using manual annotations.

    The current workflow is to:
    - annotate pathways in https://docs.google.com/spreadsheets/d/1waVXSVMOthL5QAT0PITgLMDdXGHIS50LZ2P1_F_c-6s/edit#gid=101210748
    - parse annotations into flat files using parse_manual_annotation.Rmd
    - call this function to format flat files and update a current SBML_dfs pathway model

    Params
    ------
    curation_dir: str
        Directory containing annotations generated using parse_manual_annotation.Rmd
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A pathway model
    verbose: bool
        Extra reporting

    Returns
    -------
    sbml_df: sbml_dfs_core.SBML_dfs
        A curated pathway model

    """

    try:
        open_fs(curation_dir)
    except Exception as e:
        raise FileNotFoundError(f"{curation_dir} does not exist") from e

    if not isinstance(sbml_dfs, sbml_dfs_core.SBML_dfs):
        raise TypeError(
            f"sbml_dfs was a {type(sbml_dfs)} and must be an sbml_dfs_core.SBML_dfs"
        )
    if not isinstance(verbose, bool):
        raise TypeError(f"verbose was a {type(verbose)} and must be a bool")

    curation_dict = read_pathway_curations(curation_dir)

    # remove existing entities
    if CURATION_DEFS.REMOVE in curation_dict.keys():
        invalid_entities_dict = _find_invalid_entities(
            sbml_dfs, curation_dict[CURATION_DEFS.REMOVE]
        )
        if verbose:
            print(
                "removing "
                + ", ".join(
                    [
                        str(len(y)) + " " + x + "s"
                        for x, y in invalid_entities_dict.items()
                    ]
                )
            )
        sbml_dfs = _remove_entities(sbml_dfs, invalid_entities_dict)

    # add new entities
    new_entities = format_curations(curation_dict, sbml_dfs)
    if verbose:
        print(
            "adding "
            + ", ".join([str(y.shape[0]) + " " + x for x, y in new_entities.items()])
        )
    for entity_type in new_entities.keys():
        entity_df = getattr(sbml_dfs, entity_type)
        updated_entity_df = pd.concat([entity_df, new_entities[entity_type]])
        setattr(sbml_dfs, entity_type, updated_entity_df)
    sbml_dfs.validate()

    return sbml_dfs


def format_curated_entities(
    entity_type: str,
    new_curated_entities: dict[Any, pd.DataFrame],
    new_entities: dict[str, pd.DataFrame],
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    curation_id: str = "Calico curations",
) -> pd.DataFrame:
    """
    Format Curated Entities

    Convert entities from the curation format to the stucture of SBML_dfs tables

    Params
    ------
    entity_type: str
        The type of entity to update (e.g., reactions, species, ...)
    new_curated_entities: dict
        Curation pd.DataFrames generated using read_pathway_curations
    new_entities: dict
        Curations formatted as sbml_dfs_core.SBML_dfs tables
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A pathway model
    curation_id: str
        Name to use as a pathway id in source.Source objects

    Returns
    -------
    new_entity_df: pd.DataFrame
        Input for entity_type formatted as an SBML_dfs table

    """
    _validate_format_curated_entities_inputs(
        entity_type, new_curated_entities, new_entities, sbml_dfs, curation_id
    )

    type_schema = SBML_DFS_SCHEMA.SCHEMA[entity_type]

    new_curated_entities = _add_entity_labels(
        new_curated_entities, entity_type, type_schema
    )
    new_curated_entities = _add_entity_sources(
        new_curated_entities, type_schema, curation_id
    )
    new_curated_entities = _add_entity_primary_keys(
        new_curated_entities, entity_type, type_schema, sbml_dfs
    )
    new_curated_entities = _add_entity_foreign_keys(
        new_curated_entities, entity_type, type_schema, new_entities, sbml_dfs
    )
    new_curated_entities = _add_entity_identifiers(new_curated_entities, type_schema)

    return new_curated_entities.set_index(type_schema[SCHEMA_DEFS.PK])[
        type_schema[SCHEMA_DEFS.VARS]
    ]


def format_curations(
    curation_dict: dict[str, pd.DataFrame], sbml_dfs: sbml_dfs_core.SBML_dfs
) -> dict[str, pd.DataFrame]:
    """
    Format Curations

    Format manual curations into a set of table that can be appended to an sbml_dfs's tables

    Params
    ------
    curation_dict:
        Curations imported using read_pathway_curations
    sbml_dfs:
        A pathway model

    Returns
    -------
    new_entities: dict
        Curations formatted as sbml_dfs_core.SBML_dfs tables

    """

    new_entity_types = set(curation_dict.keys()).difference(
        {CURATION_DEFS.FOCI, CURATION_DEFS.REMOVE}
    )

    if SBML_DFS.COMPARTMENTS in new_entity_types:
        raise NotImplementedError("logic for adding compartments does not exist")

    new_entities = dict()  # type: dict[str, pd.DataFrame]

    # reorganize reaction species' annotations as a dict to allow for
    # annotations added expicitly in the curations sheet
    # and implicitly due to newly added reactions
    reaction_species_dict = dict()  # type: dict[str, pd.DataFrame | None]
    reaction_species_dict["explicit"] = _format_explicit_reaction_species(curation_dict)
    reaction_species_dict["implicit"] = None

    # create reaction species based on reaction stoichiometry
    if SBML_DFS.REACTIONS in new_entity_types:
        reaction_species_dict["implicit"] = _format_implicit_reaction_species(
            curation_dict
        )
        new_entity_types.add(SBML_DFS.REACTION_SPECIES)
    curation_dict[SBML_DFS.REACTION_SPECIES] = pd.concat(reaction_species_dict.values())  # type: ignore

    if SBML_DFS.REACTIONS in new_entity_types:
        # add "r_isreversible" to curation_dict["reactions"]
        curation_dict[SBML_DFS.REACTIONS][SBML_DFS.R_ISREVERSIBLE] = [
            (
                True
                if curation_dict[SBML_DFS.REACTIONS][SBML_DFS.STOICHIOMETRY]
                .iloc[0]
                .split("<->")
                == 2
                else False
            )
            for i in range(
                0, curation_dict[SBML_DFS.REACTIONS][SBML_DFS.STOICHIOMETRY].shape[0]
            )
        ]

    for entity_type in SBML_DFS_SCHEMA.SCHEMA.keys():
        if entity_type not in new_entity_types:
            continue

        # add in the order of compartments, species > reactions > compartmentalized_species > reaction_species
        new_entities[entity_type] = format_curated_entities(
            entity_type, curation_dict[entity_type], new_entities, sbml_dfs  # type: ignore
        )

    return new_entities


def read_pathway_curations(curation_dir: str) -> dict[str, pd.DataFrame]:
    """
    Read Pathway Curations

    Load curations that were prepared by parse_manual_annotations.Rmd

    Params
    ------
    curation_dir: str
        Directory containing annotations generated using parse_manual_annotation.Rmd

    Returns
    -------
    curations: dict
        Dictionary containing different types of annoations
    """
    with open_fs(curation_dir) as curation_fs:
        curation_files = curation_fs.listdir(".")

        annotations_types = set(curation_files).intersection(
            {x + ".tsv" for x in VALID_ANNOTATION_TYPES}
        )

        curation_dict = {}
        for annotation_file in annotations_types:
            with curation_fs.open(annotation_file) as f:
                key = os.path.splitext(annotation_file)[0]
                curation_dict[key] = pd.read_csv(f, sep="\t")

        return curation_dict


# Private functions (alphabetical order)


def _add_entity_foreign_keys(
    new_curated_entities: pd.DataFrame,
    entity_type: str,
    type_schema: dict[str, Any],
    new_entities: dict[str, pd.DataFrame],
    sbml_dfs: sbml_dfs_core.SBML_dfs,
) -> pd.DataFrame:
    """Add foreign key columns to curated entities."""
    if SCHEMA_DEFS.FK not in type_schema:
        return new_curated_entities

    # find primary keys corresponding to foreign keys, including both existing and newly added entities
    for fk in type_schema[SCHEMA_DEFS.FK]:
        # find the table that the fk belongs to
        fk_of = [
            x for x, y in SBML_DFS_SCHEMA.SCHEMA.items() if y[SCHEMA_DEFS.PK] == fk
        ][0]

        # pull up referenced entities table, including newly added entities
        if fk_of in new_entities.keys():
            ref_entities = pd.concat([new_entities[fk_of], getattr(sbml_dfs, fk_of)])
        else:
            ref_entities = getattr(sbml_dfs, fk_of)
        key_ref_schema = SBML_DFS_SCHEMA.SCHEMA[fk_of]
        # add primary key by joining on label
        new_curated_entities = new_curated_entities.merge(
            ref_entities[key_ref_schema[SCHEMA_DEFS.LABEL]].reset_index(),
            how="left",
        )

        # check that all fks were found
        failed_join_df = new_curated_entities[
            new_curated_entities[key_ref_schema[SCHEMA_DEFS.PK]].isna()
        ]
        if failed_join_df.shape[0] != 0:
            if SCHEMA_DEFS.LABEL in type_schema:
                fail_str = "\n".join(failed_join_df[type_schema[SCHEMA_DEFS.LABEL]])
            else:
                fail_str = "\n".join(failed_join_df["label"])
            raise ValueError(
                f"{failed_join_df.shape[0]} merges of {fk_of} "
                f"failed when updating the {entity_type} table:\n{fail_str}"
            )

    return new_curated_entities


def _add_entity_identifiers(
    new_curated_entities: pd.DataFrame,
    type_schema: dict[str, Any],
) -> pd.DataFrame:
    """Add identifier column to curated entities."""
    if SCHEMA_DEFS.ID not in type_schema:
        return new_curated_entities

    ids = list()
    for i in range(0, new_curated_entities.shape[0]):
        new_entity_series = new_curated_entities.iloc[i]

        is_identified = not new_entity_series.isna()[CURATION_DEFS.URI]
        if is_identified:
            id = [
                identifiers.format_uri(new_entity_series[CURATION_DEFS.URI], bqb=BQB.IS)
            ]
        else:
            id = [
                {
                    IDENTIFIERS.ONTOLOGY: "custom_species",
                    IDENTIFIERS.IDENTIFIER: new_entity_series[
                        type_schema[SCHEMA_DEFS.PK]
                    ],
                    IDENTIFIERS.BQB: BQB.IS,
                }
            ]
        # stub the id using the entity pk
        ids.append(identifiers.Identifiers(id))

    new_curated_entities[type_schema[SCHEMA_DEFS.ID]] = ids
    return new_curated_entities


def _add_entity_labels(
    new_curated_entities: pd.DataFrame,
    entity_type: str,
    type_schema: dict[str, Any],
) -> pd.DataFrame:
    """Add label column to curated entities."""
    if SCHEMA_DEFS.LABEL in type_schema:
        new_curated_entities[type_schema[SCHEMA_DEFS.LABEL]] = new_curated_entities[
            entity_type
        ]
    else:
        # add a temporary label to improve error messages
        new_curated_entities["label"] = [
            ", ".join(new_curated_entities.select_dtypes(include=["object"]).iloc[i])
            for i in range(0, new_curated_entities.shape[0])
        ]
    return new_curated_entities


def _add_entity_primary_keys(
    new_curated_entities: pd.DataFrame,
    entity_type: str,
    type_schema: dict[str, Any],
    sbml_dfs: sbml_dfs_core.SBML_dfs,
) -> pd.DataFrame:
    """Add primary key column to curated entities."""
    max_pk = max(
        sbml_dfs_utils.id_formatter_inv(getattr(sbml_dfs, entity_type).index.tolist())
    )
    if max_pk is np.nan:
        max_pk = int(-1)

    pk_attr = type_schema[SCHEMA_DEFS.PK]
    new_curated_entities[pk_attr] = sbml_dfs_utils.id_formatter(
        range(
            max_pk + 1,
            max_pk + new_curated_entities.shape[0] + 1,
        ),
        pk_attr,
    )
    return new_curated_entities


def _add_entity_sources(
    new_curated_entities: pd.DataFrame,
    type_schema: dict[str, Any],
    curation_id: str,
) -> pd.DataFrame:
    """Add source column to curated entities."""
    if SCHEMA_DEFS.SOURCE in type_schema:
        new_curated_entities[CURATION_DEFS.CURATOR] = new_curated_entities[
            CURATION_DEFS.CURATOR
        ].fillna(CURATION_DEFS.UNKNOWN)
        # convert curator entries to Sources
        new_curated_entities[type_schema[SCHEMA_DEFS.SOURCE]] = [
            source.Source(
                pd.DataFrame(
                    {"model": x, "name": "custom - " + x, "pathway_id": curation_id},
                    index=[0],
                )
            )
            for x in new_curated_entities[CURATION_DEFS.CURATOR]
        ]
    return new_curated_entities


def _expand_entities_by_fks(sbml_dfs: sbml_dfs_core.SBML_dfs, pk_dict: dict) -> dict:
    """
    Expand Entities By Foreign Keys

    Starting with a dictionary of foreign keys, add all primary keys that are defined by these foreign keys

    Params
    ------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A pathway model
    pk_dict: dict
        Dictionary where keys are types of primary keys in sbml_dfs

    Returns
    -------
    pk_dict: dict
        Input where additional primary keys may have been added

    """

    for tab in SBML_DFS_SCHEMA.SCHEMA.keys():
        tab_df = getattr(sbml_dfs, tab)
        tab_schema = SBML_DFS_SCHEMA.SCHEMA[tab]
        pk = tab_schema[SCHEMA_DEFS.PK]

        if SCHEMA_DEFS.FK in tab_schema:
            # check for foreign keys which are defined by primary keys
            # add these to the pk_dict
            for fk in tab_schema[SCHEMA_DEFS.FK]:
                if fk in pk_dict.keys():
                    fks = tab_df[tab_df[fk].isin(pk_dict[fk])]
                    if pk not in pk_dict.keys():
                        pk_dict[pk] = set()
                    for x in fks.index.tolist():
                        pk_dict[pk].add(x)

    return pk_dict


def _find_invalid_entities(
    sbml_dfs: sbml_dfs_core.SBML_dfs, invalid_entities: pd.DataFrame
) -> dict[str, set]:
    """
    Find Invalid Entities

    Based on a set of entity names or attributes, find each entities
    corresponding primary key

    Params
    ------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A pathway model
    invalid_entities: pd.DataFrame
        A table containing entities to be removed ("remove"),
        the table where the entity resides ("table") and variable used
        to find the entity ("variable")

    Returns
    -------
    invalid_entities_dict: dict
        A dictionary containing the primary keys of invalid entities

    """

    # find tables where removal will occur (or at least start)
    unique_tables = invalid_entities[CURATION_DEFS.TABLE].unique().tolist()
    invalid_tables = [
        x for x in unique_tables if x not in SBML_DFS_SCHEMA.SCHEMA.keys()
    ]

    if len(invalid_tables) > 0:
        raise ValueError(
            f"{', '.join(invalid_tables)} are not valid table names; "
            f"valid tables are {', '.join(SBML_DFS_SCHEMA.SCHEMA.keys())}"
        )

    invalid_entities_dict = dict()  # type: dict[str, set]
    for tab in unique_tables:
        tab_schema = SBML_DFS_SCHEMA.SCHEMA[tab]
        tab_vars = tab_schema[SCHEMA_DEFS.VARS] + [tab_schema[SCHEMA_DEFS.PK]]

        # pull out the annotations that start with the table being evaluated
        remove_df = invalid_entities[invalid_entities[CURATION_DEFS.TABLE] == tab]
        if not isinstance(remove_df, pd.DataFrame):
            raise TypeError(
                f"remove_df must be a pandas DataFrame, but got {type(remove_df).__name__}"
            )

        invalid_remove_vars = (
            remove_df[CURATION_DEFS.VARIABLE][
                ~remove_df[CURATION_DEFS.VARIABLE].isin(tab_vars)
            ]
            .unique()
            .tolist()
        )
        if len(invalid_remove_vars) > 0:
            raise ValueError(
                f"{', '.join(invalid_remove_vars)} are not valid variables"
                f" in the {tab} table; valid variables are {', '.join(tab_vars)}"
            )

        # find the pk corresponding to each removal annotation

        tab_df = getattr(sbml_dfs, tab)
        pk_attr = tab_schema[SCHEMA_DEFS.PK]

        invalid_entities_dict[pk_attr] = set()
        for i in range(0, remove_df.shape[0]):
            remove_series = remove_df.iloc[i]

            if remove_series[CURATION_DEFS.VARIABLE] == pk_attr:
                # check that pk exists and then add to invalid entities
                if remove_series[CURATION_DEFS.REMOVE] not in tab_df.index:
                    raise ValueError(
                        f"{remove_series[CURATION_DEFS.REMOVE]} was not found in the index of {tab}"
                    )
                invalid_entities_dict[pk_attr].add(remove_series[CURATION_DEFS.REMOVE])
            else:
                # lookup by
                matching_entity = tab_df[
                    tab_df[remove_series[CURATION_DEFS.VARIABLE]]
                    == remove_series[CURATION_DEFS.REMOVE]
                ]

                if matching_entity.shape[0] == 0:
                    raise ValueError(
                        f"{remove_series[CURATION_DEFS.REMOVE]} was not found in the {remove_series[CURATION_DEFS.VARIABLE]} column of {tab}"
                    )

                [invalid_entities_dict[pk_attr].add(x) for x in matching_entity.index.tolist()]  # type: ignore

    # iterate through primary key -> foreign key relationships
    # to define additional entities which should be removed based on
    # initial removal annotations
    new_invalid_entities_dict = invalid_entities_dict.copy()

    cont = True
    while cont:
        new_invalid_entities_dict = _expand_entities_by_fks(
            sbml_dfs, new_invalid_entities_dict
        )

        if new_invalid_entities_dict != invalid_entities_dict:
            invalid_entities_dict = new_invalid_entities_dict
            new_invalid_entities_dict = invalid_entities_dict.copy()
        else:
            cont = False

    return invalid_entities_dict


def _format_explicit_reaction_species(
    curation_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Format reaction species which are deirectly defined among curated species."""

    if SBML_DFS.REACTION_SPECIES not in curation_dict.keys():
        print("No explicitly curated reaction species")
        return None

    # convert from sbo_term_names to sbo_term
    mini_sbo_terms_df = pd.DataFrame(MINI_SBO_FROM_NAME, index=[SBML_DFS.SBO_TERM]).T

    augmented_reaction_species = (
        curation_dict[SBML_DFS.REACTION_SPECIES]
        .rename({SBML_DFS.REACTION_SPECIES: SBML_DFS.SC_NAME}, axis=1)
        .merge(
            mini_sbo_terms_df,
            left_on=CURATION_DEFS.SBO_TERM_NAME,
            right_index=True,
            how="left",
        )
    )

    # invalid terms
    invalid_terms_df = augmented_reaction_species[
        augmented_reaction_species[SBML_DFS.SBO_TERM].isna()
    ]
    if invalid_terms_df.shape[0] != 0:
        invalid_terms = invalid_terms_df[CURATION_DEFS.SBO_TERM_NAME].unique().tolist()
        raise ValueError(
            f'{", ".join(invalid_terms)} are invalid entries for "{CURATION_DEFS.SBO_TERM_NAME}", '
            f'valid entries are {", ".join(mini_sbo_terms_df.index.tolist())}'
        )

    # there currently isn't a good way to encode evidence and curator annotations
    # as source objects for reaction_species since they lack a source object
    # to date they have had the same source as their reaction
    augmented_reaction_species = augmented_reaction_species.drop(
        [CURATION_DEFS.SBO_TERM_NAME, CURATION_DEFS.EVIDENCE, CURATION_DEFS.CURATOR],
        axis=1,
    )

    return augmented_reaction_species


def _format_implicit_reaction_species(
    curation_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Construct reaction species which are defined in reactions' stoichiometry."""

    curated_reactions = curation_dict[SBML_DFS.REACTIONS][
        [SBML_DFS.REACTIONS, SBML_DFS.STOICHIOMETRY]
    ]

    reaction_species = list()
    for i in range(0, curated_reactions.shape[0]):
        reaction_stoi = curated_reactions[SBML_DFS.STOICHIOMETRY].iloc[i]
        if len(reaction_stoi.split("<->")) == 2:
            split_stoi = reaction_stoi.split("<->")
        elif len(reaction_stoi.split("->")) == 2:
            split_stoi = reaction_stoi.split("->")
        else:
            raise ValueError(
                f"{reaction_stoi} is not a valid reaction stoichiometry; "
                "there must be one and only one '->' to separate the substrates and products"
            )

        substrates = [x.strip() for x in split_stoi[0].strip().split("++")]
        products = [x.strip() for x in split_stoi[1].strip().split("++")]

        a_reactions_species = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            SBML_DFS.SC_NAME: x,
                            SBML_DFS.STOICHIOMETRY: -1,
                            SBML_DFS.SBO_TERM: MINI_SBO_FROM_NAME[
                                SBOTERM_NAMES.REACTANT
                            ],
                        }
                        for x in substrates
                    ]
                ),
                pd.DataFrame(
                    [
                        {
                            SBML_DFS.SC_NAME: x,
                            SBML_DFS.STOICHIOMETRY: 1,
                            SBML_DFS.SBO_TERM: MINI_SBO_FROM_NAME[
                                SBOTERM_NAMES.PRODUCT
                            ],
                        }
                        for x in products
                    ]
                ),
            ]
        ).assign(r_name=curated_reactions[SBML_DFS.REACTIONS].iloc[i])

        reaction_species.append(a_reactions_species)

    return pd.concat(reaction_species)


def _remove_entities(
    sbml_dfs: sbml_dfs_core.SBML_dfs, pk_dict: dict
) -> sbml_dfs_core.SBML_dfs:
    """
    Remove Entities

    Remove entities whose primary keys are in pk_dict

    Params
    ------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A pathway model
    pk_dict: dict
        Dictionary where keys are types of primary keys in sbml_dfs

    Returns
    -------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        Input with some entities removed

    """

    for tab in SBML_DFS_SCHEMA.SCHEMA.keys():
        tab_df = getattr(sbml_dfs, tab)
        tab_schema = SBML_DFS_SCHEMA.SCHEMA[tab]
        pk_attr = tab_schema[SCHEMA_DEFS.PK]

        if pk_attr in pk_dict.keys():
            updated_table = tab_df[~tab_df.index.isin(pk_dict[pk_attr])]
            setattr(sbml_dfs, tab, updated_table)

    return sbml_dfs


def _validate_format_curated_entities_inputs(
    entity_type: str,
    new_curated_entities: pd.DataFrame,
    new_entities: dict[str, pd.DataFrame],
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    curation_id: str,
) -> None:
    """Validate inputs for format_curated_entities."""
    if not isinstance(entity_type, str):
        raise TypeError(f"entity_type was a {type(entity_type)} and must be a str")
    if not isinstance(new_curated_entities, pd.DataFrame):
        raise TypeError(
            f"new_curated_entities was a {type(new_curated_entities)} and must be a pd.DataFrame"
        )
    if not isinstance(new_entities, dict):
        raise TypeError(f"new_entities was a {type(new_entities)} and must be a dict")
    if not isinstance(sbml_dfs, sbml_dfs_core.SBML_dfs):
        raise TypeError(
            f"sbml_dfs was a {type(sbml_dfs)} and must be an sbml_dfs_core.SBML_dfs"
        )
    if not isinstance(curation_id, str):
        raise TypeError(f"curation_id was a {type(curation_id)} and must be a str")
