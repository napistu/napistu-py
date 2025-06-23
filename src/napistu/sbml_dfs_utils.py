from __future__ import annotations

import copy

import logging
import re
from typing import Any
from typing import Iterable
from fs import open_fs

import numpy as np
import pandas as pd
from napistu import utils
from napistu import indices

from napistu import sbml_dfs_core
from napistu.constants import SBML_DFS
from napistu.constants import IDENTIFIERS
from napistu.constants import BQB_DEFINING_ATTRS
from napistu.constants import BQB_DEFINING_ATTRS_LOOSE
from napistu.constants import REQUIRED_REACTION_FROMEDGELIST_COLUMNS
from napistu.constants import INTERACTION_EDGELIST_EXPECTED_VARS
from napistu.constants import MINI_SBO_FROM_NAME

logger = logging.getLogger(__name__)


def unnest_identifiers(id_table: pd.DataFrame, id_var: str) -> pd.DataFrame:
    """
    Unnest Identifiers

    Take a pd.DataFrame containing an array of Identifiers and
    return one-row per identifier.

    Parameters:
    id_table: pd.DataFrame
        a table containing an array of Identifiers
    id_var: str
        variable containing Identifiers

    Returns:
    pd.Dataframe containing the index of id_table but expanded
    to include one row per identifier

    """

    # validate inputs
    utils.match_pd_vars(id_table, {id_var}).assert_present()

    N_invalid_ids = sum(id_table[id_var].isna())
    if N_invalid_ids != 0:
        raise ValueError(
            f'{N_invalid_ids} entries in "id_table" were missing',
            "entries with no identifiers should still include an Identifiers object",
        )

    # Get the identifier as a list of dicts
    df = id_table[id_var].apply(lambda x: x.ids if len(x.ids) > 0 else 0).to_frame()
    # Filter out zero length lists
    df = df.query(f"{id_var} != 0")
    # Unnest the list of dicts into one dict per row
    df = df.explode(id_var)
    # Unnest the dict into a dataframe
    df = pd.DataFrame(df[id_var].values.tolist(), index=df.index)
    # Add the entry number as an index
    df["entry"] = df.groupby(df.index).cumcount()
    df.set_index("entry", append=True, inplace=True)
    return df


def id_formatter(id_values: Iterable[Any], id_type: str, id_len: int = 8) -> list[str]:
    id_prefix = utils.extract_regex_match("^([a-zA-Z]+)_id$", id_type).upper()
    return [id_prefix + format(x, f"0{id_len}d") for x in id_values]


def id_formatter_inv(ids: list[str]) -> list[int]:
    """
    ID Formatter Inverter

    Convert from internal IDs back to integer IDs
    """

    id_val = list()
    for an_id in ids:
        if re.match("^[A-Z]+[0-9]+$", an_id):
            id_val.append(int(re.sub("^[A-Z]+", "", an_id)))
        else:
            id_val.append(np.nan)  # type: ignore

    return id_val


def get_current_max_id(sbml_dfs_table: pd.DataFrame) -> int:
    """
    Get Current Max ID

    Look at a table from an SBML_dfs object and find the largest primary key following
    the default naming convention for a the table.

    Params:
    sbml_dfs_table (pd.DataFrame):
        A table derived from an SBML_dfs object.

    Returns:
    current_max_id (int):
        The largest id which is already defined in the table using its expected naming
        convention. If no IDs following this convention are present then the default
        will be -1. In this way new IDs will be added starting with 0.

    """

    existing_ids_numeric = id_formatter_inv(sbml_dfs_table.index.tolist())

    # filter np.nan which will be introduced if the key is not the default format
    existing_ids_numeric_valid = [x for x in existing_ids_numeric if x is not np.nan]
    if len(existing_ids_numeric_valid) == 0:
        current_max_id = -1
    else:
        current_max_id = max(existing_ids_numeric_valid)

    return current_max_id


def construct_formula_string(
    reaction_species_df: pd.DataFrame,
    reactions_df: pd.DataFrame,
    name_var: str,
) -> str:
    """
    Construct Formula String

    Convert a table of reaction species into a formula string

    Parameters:
    ----------
    reaction_species_df: pd.DataFrame
        Table containing a reactions' species
    reactions_df: pd.DataFrame
        smbl.reactions
    name_var: str
        Name used to label species

    Returns:
    ----------
    formula_str: str
        String representation of a reactions substrates, products and
        modifiers

    """

    reaction_species_df["label"] = [
        _add_stoi_to_species_name(x, y)
        for x, y in zip(
            reaction_species_df[SBML_DFS.STOICHIOMETRY], reaction_species_df[name_var]
        )
    ]

    rxn_reversible = bool(
        reactions_df.loc[
            reaction_species_df[SBML_DFS.R_ID].iloc[0], SBML_DFS.R_ISREVERSIBLE
        ]
    )  # convert from a np.bool_ to bool if needed
    if not isinstance(rxn_reversible, bool):
        raise TypeError(
            f"rxn_reversible must be a bool, but got {type(rxn_reversible).__name__}"
        )

    if rxn_reversible:
        arrow_type = " <-> "
    else:
        arrow_type = " -> "

    substrates = " + ".join(
        reaction_species_df["label"][
            reaction_species_df[SBML_DFS.STOICHIOMETRY] < 0
        ].tolist()
    )
    products = " + ".join(
        reaction_species_df["label"][
            reaction_species_df[SBML_DFS.STOICHIOMETRY] > 0
        ].tolist()
    )
    modifiers = " + ".join(
        reaction_species_df["label"][
            reaction_species_df[SBML_DFS.STOICHIOMETRY] == 0
        ].tolist()
    )
    if modifiers != "":
        modifiers = f" ---- modifiers: {modifiers}]"

    return f"{substrates}{arrow_type}{products}{modifiers}"


def adapt_pw_index(
    source: str | indices.PWIndex,
    species: str | Iterable[str] | None,
    outdir: str | None = None,
) -> indices.PWIndex:
    """Adapts a pw_index

    Helpful to filter for species before reconstructing.

    Args:
        source (str | PWIndex): uri for pw_index.csv file or PWIndex object
        species (str):
        outdir (str | None, optional): Optional directory to write pw_index to.
            Defaults to None.

    Returns:
        indices.PWIndex: Filtered pw index
    """
    if isinstance(source, str):
        pw_index = indices.PWIndex(source)
    elif isinstance(source, indices.PWIndex):
        pw_index = copy.deepcopy(source)
    else:
        raise ValueError("'source' needs to be str or PWIndex.")
    pw_index.filter(species=species)

    if outdir is not None:
        with open_fs(outdir, create=True) as fs:
            with fs.open("pw_index.tsv", "w") as f:
                pw_index.index.to_csv(f, sep="\t")
    return pw_index


def _dogmatic_to_defining_bqbs(dogmatic: bool = False) -> str:
    if dogmatic:
        logger.info(
            "Running in dogmatic mode - differences genes, transcripts, and proteins will "
            "try to be maintained as separate species."
        )
        # preserve differences between genes, transcripts, and proteins
        defining_biological_qualifiers = BQB_DEFINING_ATTRS
    else:
        logger.info(
            "Running in non-dogmatic mode - genes, transcripts, and proteins will "
            "be merged if possible."
        )
        # merge genes, transcripts, and proteins (if they are defined with
        # bqb terms which specify their relationships).
        defining_biological_qualifiers = BQB_DEFINING_ATTRS_LOOSE

    return defining_biological_qualifiers


def match_entitydata_index_to_entity(
    entity_data_dict: dict,
    an_entity_data_type: str,
    consensus_entity_df: pd.DataFrame,
    entity_schema: dict,
    table: str,
) -> pd.DataFrame:
    """
    Match the index of entity_data_dict[an_entity_data_type] with the index of corresponding entity.
    Update entity_data_dict[an_entity_data_type]'s index to the same as consensus_entity_df's index
    Report cases where entity_data has indices not in corresponding entity's index.
    Args
        entity_data_dict (dict): dictionary containing all model's "an_entity_data_type" dictionaries
        an_entity_data_type (str): data_type from species/reactions_data in entity_data_dict
        consensus_entity_df (pd.DataFrame): the dataframe of the corresponding entity
        entity_schema (dict): schema for "table"
        table (str): table whose data is being consolidates (currently species or reactions)
    Returns:
        entity_data_df (pd.DataFrame) table for entity_data_dict[an_entity_data_type]
    """

    data_table = table + "_data"
    entity_data_df = entity_data_dict[an_entity_data_type]

    # ensure entity_data_df[an_entity_data_type]'s index doesn't have
    # reaction ids that are not in consensus_entity's index
    if len(entity_data_df.index.difference(consensus_entity_df.index)) == 0:
        logger.info(f"{data_table} ids are included in {table} ids")
    else:
        logger.warnning(
            f"{data_table} have ids are not matched to {table} ids,"
            f"please check mismatched ids first"
        )

    # when entity_data_df is only a subset of the index of consensus_entity_df
    # add ids only in consensus_entity_df to entity_data_df, and fill values with Nan
    if len(entity_data_df) != len(consensus_entity_df):
        logger.info(
            f"The {data_table} has {len(entity_data_df)} ids,"
            f"different from {len(consensus_entity_df)} ids in the {table} table,"
            f"updating {data_table} ids."
        )

        entity_data_df = pd.concat(
            [
                entity_data_df,
                consensus_entity_df[
                    ~consensus_entity_df.index.isin(entity_data_df.index)
                ],
            ],
            ignore_index=False,
        )

        entity_data_df.drop(entity_schema["vars"], axis=1, inplace=True)

    return entity_data_df


def check_entity_data_index_matching(sbml_dfs, table):
    """
    Update the input smbl_dfs's entity_data (dict) index
    with match_entitydata_index_to_entity,
    so that index for dataframe(s) in entity_data (dict) matches the sbml_dfs'
    corresponding entity, and then passes sbml_dfs.validate()
    Args
        sbml_dfs (cpr.SBML_dfs): a cpr.SBML_dfs
        table (str): table whose data is being consolidates (currently species or reactions)
    Returns
        sbml_dfs (cpr.SBML_dfs):
        sbml_dfs whose entity_data is checked to have the same index
        as the corresponding entity.
    """

    table_data = table + "_data"

    entity_data_dict = getattr(sbml_dfs, table_data)
    entity_schema = sbml_dfs.schema[table]
    sbml_dfs_entity = getattr(sbml_dfs, table)

    if entity_data_dict != {}:
        entity_data_types = set.union(set(entity_data_dict.keys()))

        entity_data_dict_checked = {
            x: match_entitydata_index_to_entity(
                entity_data_dict, x, sbml_dfs_entity, entity_schema, table
            )
            for x in entity_data_types
        }

        if table == SBML_DFS.REACTIONS:
            sbml_dfs.reactions_data = entity_data_dict_checked
        elif table == SBML_DFS.SPECIES:
            sbml_dfs.species_data = entity_data_dict_checked

    return sbml_dfs


def get_characteristic_species_ids(
    sbml_dfs: sbml_dfs_core.SBML_dfs, dogmatic: bool = True
) -> pd.DataFrame:
    """
    Get Characteristic Species IDs

    List the systematic identifiers which are characteristic of molecular species, e.g., excluding subcomponents, and optionally, treating proteins, transcripts, and genes equiavlently.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object.
    dogmatic : bool, default=True
        Whether to use the dogmatic flag to determine which BQB attributes are valid.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the systematic identifiers which are characteristic of molecular species.
    """

    # select valid BQB attributes based on dogmatic flag
    defining_biological_qualifiers = _dogmatic_to_defining_bqbs(dogmatic)

    # pre-summarize ontologies
    species_identifiers = sbml_dfs.get_identifiers(SBML_DFS.SPECIES)

    # drop some BQB_HAS_PART annotations
    species_identifiers = sbml_dfs_core.filter_to_characteristic_species_ids(
        species_identifiers,
        defining_biological_qualifiers=defining_biological_qualifiers,
    )

    return species_identifiers


def _dogmatic_to_defining_bqbs(dogmatic: bool = False) -> str:
    assert isinstance(dogmatic, bool)
    if dogmatic:
        logger.info(
            "Running in dogmatic mode - differences genes, transcripts, and proteins will "
            "try to be maintained as separate species."
        )
        # preserve differences between genes, transcripts, and proteins
        defining_biological_qualifiers = BQB_DEFINING_ATTRS
    else:
        logger.info(
            "Running in non-dogmatic mode - genes, transcripts, and proteins will "
            "be merged if possible."
        )
        # merge genes, transcripts, and proteins (if they are defined with
        # bqb terms which specify their relationships).
        defining_biological_qualifiers = BQB_DEFINING_ATTRS_LOOSE

    return defining_biological_qualifiers


def _stub_ids(ids):
    """Stub with a blank ID if an ids list is blank; otherwise create an Identifiers object from the provided ids"""
    if len(ids) == 0:
        return pd.DataFrame(
            {
                IDENTIFIERS.ONTOLOGY: [None],
                IDENTIFIERS.IDENTIFIER: [None],
                IDENTIFIERS.URL: [None],
                IDENTIFIERS.BQB: [None],
            }
        )
    else:
        return pd.DataFrame(ids)


def _edgelist_validate_inputs(
    interaction_edgelist: pd.DataFrame,
    species_df: pd.DataFrame,
    compartments_df: pd.DataFrame,
) -> None:
    """
    Validate input DataFrames have required columns.

    Parameters
    ----------
    interaction_edgelist : pd.DataFrame
        Interaction data to validate
    species_df : pd.DataFrame
        Species data to validate
    compartments_df : pd.DataFrame
        Compartments data to validate
    """

    # check compartments
    compartments_df_expected_vars = {SBML_DFS.C_NAME, SBML_DFS.C_IDENTIFIERS}
    compartments_df_columns = set(compartments_df.columns.tolist())
    missing_required_fields = compartments_df_expected_vars.difference(
        compartments_df_columns
    )
    if len(missing_required_fields) > 0:
        raise ValueError(
            f"{', '.join(missing_required_fields)} are required variables"
            ' in "compartments_df" but were not present in the input file.'
        )

    # check species
    species_df_expected_vars = {SBML_DFS.S_NAME, SBML_DFS.S_IDENTIFIERS}
    species_df_columns = set(species_df.columns.tolist())
    missing_required_fields = species_df_expected_vars.difference(species_df_columns)
    if len(missing_required_fields) > 0:
        raise ValueError(
            f"{', '.join(missing_required_fields)} are required"
            ' variables in "species_df" but were not present '
            "in the input file."
        )

    # check interactions
    interaction_edgelist_columns = set(interaction_edgelist.columns.tolist())
    missing_required_fields = INTERACTION_EDGELIST_EXPECTED_VARS.difference(
        interaction_edgelist_columns
    )
    if len(missing_required_fields) > 0:
        raise ValueError(
            f"{', '.join(missing_required_fields)} are required "
            'variables in "interaction_edgelist" but were not '
            "present in the input file."
        )

    return None


def _edgelist_identify_extra_columns(
    interaction_edgelist, species_df, keep_reactions_data, keep_species_data
):
    """
    Identify extra columns in input data that should be preserved.

    Parameters
    ----------
    interaction_edgelist : pd.DataFrame
        Interaction data containing potential extra columns
    species_df : pd.DataFrame
        Species data containing potential extra columns
    keep_reactions_data : bool or str
        Whether to keep extra reaction columns
    keep_species_data : bool or str
        Whether to keep extra species columns

    Returns
    -------
    dict
        Dictionary with 'reactions' and 'species' keys containing lists of extra column names
    """
    extra_reactions_columns = []
    extra_species_columns = []

    if keep_reactions_data is not False:
        extra_reactions_columns = [
            c
            for c in interaction_edgelist.columns
            if c not in INTERACTION_EDGELIST_EXPECTED_VARS
        ]

    if keep_species_data is not False:
        extra_species_columns = [
            c
            for c in species_df.columns
            if c not in {SBML_DFS.S_NAME, SBML_DFS.S_IDENTIFIERS}
        ]

    return {"reactions": extra_reactions_columns, "species": extra_species_columns}


def _edgelist_process_compartments(compartments_df, interaction_source):
    """
    Format compartments DataFrame with source and ID columns.

    Parameters
    ----------
    compartments_df : pd.DataFrame
        Raw compartments data
    interaction_source : source.Source
        Source object to assign to compartments

    Returns
    -------
    pd.DataFrame
        Processed compartments with IDs, indexed by compartment ID
    """
    compartments = compartments_df.copy()
    compartments[SBML_DFS.C_SOURCE] = interaction_source
    compartments[SBML_DFS.C_ID] = id_formatter(
        range(compartments.shape[0]), SBML_DFS.C_ID
    )
    return compartments.set_index(SBML_DFS.C_ID)[
        [SBML_DFS.C_NAME, SBML_DFS.C_IDENTIFIERS, SBML_DFS.C_SOURCE]
    ]


def _edgelist_process_species(species_df, interaction_source, extra_species_columns):
    """
    Format species DataFrame and extract extra data.

    Parameters
    ----------
    species_df : pd.DataFrame
        Raw species data
    interaction_source : source.Source
        Source object to assign to species
    extra_species_columns : list
        Names of extra columns to preserve separately

    Returns
    -------
    tuple of pd.DataFrame
        Processed species DataFrame and species extra data DataFrame
    """
    species = species_df.copy()
    species[SBML_DFS.S_SOURCE] = interaction_source
    species[SBML_DFS.S_ID] = id_formatter(range(species.shape[0]), SBML_DFS.S_ID)

    required_cols = [SBML_DFS.S_NAME, SBML_DFS.S_IDENTIFIERS, SBML_DFS.S_SOURCE]
    species_indexed = species.set_index(SBML_DFS.S_ID)[
        required_cols + extra_species_columns
    ]

    # Separate extra data from main species table
    species_data = species_indexed[extra_species_columns]
    processed_species = species_indexed[required_cols]

    return processed_species, species_data


def _edgelist_create_compartmentalized_species(
    interaction_edgelist, species_df, compartments_df, interaction_source
):
    """
    Create compartmentalized species from interactions.

    Parameters
    ----------
    interaction_edgelist : pd.DataFrame
        Interaction data containing species-compartment combinations
    species_df : pd.DataFrame
        Processed species data with IDs
    compartments_df : pd.DataFrame
        Processed compartments data with IDs
    interaction_source : source.Source
        Source object to assign to compartmentalized species

    Returns
    -------
    pd.DataFrame
        Compartmentalized species with formatted names and IDs
    """
    # Get all distinct upstream and downstream compartmentalized species
    comp_species = pd.concat(
        [
            interaction_edgelist[["upstream_name", "upstream_compartment"]].rename(
                {
                    "upstream_name": SBML_DFS.S_NAME,
                    "upstream_compartment": SBML_DFS.C_NAME,
                },
                axis=1,
            ),
            interaction_edgelist[["downstream_name", "downstream_compartment"]].rename(
                {
                    "downstream_name": SBML_DFS.S_NAME,
                    "downstream_compartment": SBML_DFS.C_NAME,
                },
                axis=1,
            ),
        ]
    ).drop_duplicates()

    # Add species and compartment IDs
    comp_species_w_ids = comp_species.merge(
        species_df[SBML_DFS.S_NAME].reset_index(), how="left", on=SBML_DFS.S_NAME
    ).merge(
        compartments_df[SBML_DFS.C_NAME].reset_index(), how="left", on=SBML_DFS.C_NAME
    )

    # Validate merge was successful
    _sbml_dfs_from_edgelist_check_cspecies_merge(comp_species_w_ids, comp_species)

    # Format compartmentalized species with names, source, and IDs
    comp_species_w_ids[SBML_DFS.SC_NAME] = [
        f"{s} [{c}]"
        for s, c in zip(
            comp_species_w_ids[SBML_DFS.S_NAME], comp_species_w_ids[SBML_DFS.C_NAME]
        )
    ]
    comp_species_w_ids[SBML_DFS.SC_SOURCE] = interaction_source
    comp_species_w_ids[SBML_DFS.SC_ID] = id_formatter(
        range(comp_species_w_ids.shape[0]), SBML_DFS.SC_ID
    )

    return comp_species_w_ids.set_index(SBML_DFS.SC_ID)[
        [SBML_DFS.SC_NAME, SBML_DFS.S_ID, SBML_DFS.C_ID, SBML_DFS.SC_SOURCE]
    ]


def _edgelist_create_reactions_and_species(
    interaction_edgelist,
    comp_species,
    species_df,
    compartments_df,
    interaction_source,
    upstream_stoichiometry,
    downstream_stoichiometry,
    downstream_sbo_name,
    extra_reactions_columns,
):
    """
    Create reactions and reaction species from interactions.

    Parameters
    ----------
    interaction_edgelist : pd.DataFrame
        Original interaction data
    comp_species : pd.DataFrame
        Compartmentalized species with IDs
    species_df : pd.DataFrame
        Processed species data with IDs
    compartments_df : pd.DataFrame
        Processed compartments data with IDs
    interaction_source : source.Source
        Source object for reactions
    upstream_stoichiometry : int
        Stoichiometry for upstream species
    downstream_stoichiometry : int
        Stoichiometry for downstream species
    downstream_sbo_name : str
        SBO term name for downstream species
    extra_reactions_columns : list
        Names of extra columns to preserve

    Returns
    -------
    tuple
        (reactions_df, reaction_species_df, reactions_data)
    """
    # Add compartmentalized species IDs to interactions
    comp_species_w_names = (
        comp_species.reset_index()
        .merge(species_df[SBML_DFS.S_NAME].reset_index())
        .merge(compartments_df[SBML_DFS.C_NAME].reset_index())
    )

    interaction_w_cspecies = interaction_edgelist.merge(
        comp_species_w_names[[SBML_DFS.SC_ID, SBML_DFS.S_NAME, SBML_DFS.C_NAME]].rename(
            {
                SBML_DFS.SC_ID: "sc_id_up",
                SBML_DFS.S_NAME: "upstream_name",
                SBML_DFS.C_NAME: "upstream_compartment",
            },
            axis=1,
        ),
        how="left",
    ).merge(
        comp_species_w_names[[SBML_DFS.SC_ID, SBML_DFS.S_NAME, SBML_DFS.C_NAME]].rename(
            {
                SBML_DFS.SC_ID: "sc_id_down",
                SBML_DFS.S_NAME: "downstream_name",
                SBML_DFS.C_NAME: "downstream_compartment",
            },
            axis=1,
        ),
        how="left",
    )[
        REQUIRED_REACTION_FROMEDGELIST_COLUMNS + extra_reactions_columns
    ]

    # Validate merge didn't create duplicates
    if interaction_edgelist.shape[0] != interaction_w_cspecies.shape[0]:
        raise ValueError(
            f"Merging compartmentalized species resulted in row count change "
            f"from {interaction_edgelist.shape[0]} to {interaction_w_cspecies.shape[0]}"
        )

    # Create reaction IDs FIRST - before using them
    interaction_w_cspecies[SBML_DFS.R_ID] = id_formatter(
        range(interaction_w_cspecies.shape[0]), SBML_DFS.R_ID
    )

    # Create reactions DataFrame
    interactions_copy = interaction_w_cspecies.copy()
    interactions_copy[SBML_DFS.R_SOURCE] = interaction_source

    reactions_columns = [
        SBML_DFS.R_NAME,
        SBML_DFS.R_IDENTIFIERS,
        SBML_DFS.R_SOURCE,
        SBML_DFS.R_ISREVERSIBLE,
    ]

    reactions_df = interactions_copy.set_index(SBML_DFS.R_ID)[
        reactions_columns + extra_reactions_columns
    ]

    # Separate extra data
    reactions_data = reactions_df[extra_reactions_columns]
    reactions_df = reactions_df[reactions_columns]

    # Create reaction species relationships - NOW r_id exists
    reaction_species_df = pd.concat(
        [
            # Upstream species (modifiers/stimulators/inhibitors)
            interaction_w_cspecies[["sc_id_up", "sbo_term", SBML_DFS.R_ID]]
            .assign(stoichiometry=upstream_stoichiometry)
            .rename({"sc_id_up": "sc_id"}, axis=1),
            # Downstream species (products)
            interaction_w_cspecies[["sc_id_down", SBML_DFS.R_ID]]
            .assign(
                stoichiometry=downstream_stoichiometry,
                sbo_term=MINI_SBO_FROM_NAME[downstream_sbo_name],
            )
            .rename({"sc_id_down": "sc_id"}, axis=1),
        ]
    )

    reaction_species_df["rsc_id"] = id_formatter(
        range(reaction_species_df.shape[0]), "rsc_id"
    )

    reaction_species_df = reaction_species_df.set_index("rsc_id")

    return reactions_df, reaction_species_df, reactions_data


def _sbml_dfs_from_edgelist_check_cspecies_merge(
    merged_species: pd.DataFrame, original_species: pd.DataFrame
) -> None:
    """Check for a mismatch between the provided species data and species implied by the edgelist."""

    # check for 1-many merge
    if merged_species.shape[0] != original_species.shape[0]:
        raise ValueError(
            "Merging compartmentalized species to species_df"
            " and compartments_df by names resulted in an "
            f"increase in the tables from {original_species.shape[0]}"
            f" to {merged_species.shape[0]} indicating that names were"
            " not unique"
        )

    # check for missing species and compartments
    missing_compartments = merged_species[merged_species[SBML_DFS.C_ID].isna()][
        SBML_DFS.C_NAME
    ].unique()
    if len(missing_compartments) >= 1:
        raise ValueError(
            f"{len(missing_compartments)} compartments were present in"
            ' "interaction_edgelist" but not "compartments_df":'
            f" {', '.join(missing_compartments)}"
        )

    missing_species = merged_species[merged_species[SBML_DFS.S_ID].isna()][
        SBML_DFS.S_NAME
    ].unique()
    if len(missing_species) >= 1:
        raise ValueError(
            f"{len(missing_species)} species were present in "
            '"interaction_edgelist" but not "species_df":'
            f" {', '.join(missing_species)}"
        )

    return None


def _add_stoi_to_species_name(stoi: float | int, name: str) -> str:
    """
    Add Stoi To Species Name

    Add # of molecules to a species name

    Parameters:
    ----------
    stoi: float or int
        Number of molecules
    name: str
        Name of species

    Returns:
    ----------
    name: str
        Name containing number of species

    """

    if stoi in [-1, 0, 1]:
        return name
    else:
        return str(abs(stoi)) + " " + name
