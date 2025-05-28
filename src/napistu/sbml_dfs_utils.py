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
