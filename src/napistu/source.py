from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from typing import Optional

from napistu import indices
from napistu import sbml_dfs_core
from napistu import sbml_dfs_utils
from napistu.statistics import hypothesis_testing
from napistu.constants import (
    SBML_DFS_SCHEMA,
    SCHEMA_DEFS,
    SOURCE_SPEC,
)
from napistu.statistics.constants import CONTINGENCY_TABLE

logger = logging.getLogger(__name__)


class Source:
    """
    An Entity's Source

    Attributes
    ----------
    source : pd.DataFrame
        A dataframe containing the model source and other optional variables

    Methods
    -------

    """

    def __init__(
        self,
        source_df: pd.DataFrame | None = None,
        init: bool = False,
        pw_index: indices.PWIndex | None = None,
    ) -> None:
        """
        Tracks the model(s) an entity (i.e., a compartment, species, reaction) came from.

        By convention sources exist only for the models that an entity came from rather
        than the current model they are part of. For example, when combining Reactome models
        into a consensus, a molecule which existed in multiple models would have a source entry
        for each, but it would not have a source entry for the consensus model itself.

        Parameters
        ----------
        source_df : pd.DataFrame
            A dataframe containing the model source and other optional variables
        init : bool
            Creates an empty source object. This is typically used when creating an SBML_dfs
            object from a single source.
        pw_index : indices.PWIndex
            a pathway index object containing the pathway_id and other metadata

        Returns
        -------
        None.

        Raises
        ------
        ValueError:
            if pw_index is not a indices.PWIndex
        ValueError:
            if SOURCE_SPEC.MODEL is not present in source_df
        """

        if init is True:
            # initialize with an empty Source
            self.source = None
        else:
            if isinstance(source_df, pd.DataFrame):
                # if pw_index is provided then it will be joined to source_df to add additional metadata
                if pw_index is not None:
                    if not isinstance(pw_index, indices.PWIndex):
                        raise ValueError(
                            f"pw_index must be a indices.PWIndex or None and was {type(pw_index).__name__}"
                        )
                    else:
                        # check that all models are present in the pathway index
                        missing_pathways = set(
                            source_df[SOURCE_SPEC.MODEL].tolist()
                        ).difference(
                            set(pw_index.index[SOURCE_SPEC.PATHWAY_ID].tolist())
                        )
                        if len(missing_pathways) > 0:
                            raise ValueError(
                                f"{len(missing_pathways)} pathway models are present"
                                f" in source_df but not the pw_index: {', '.join(missing_pathways)}"
                            )

                        source_df = source_df.merge(
                            pw_index.index,
                            left_on=SOURCE_SPEC.MODEL,
                            right_on=SOURCE_SPEC.PATHWAY_ID,
                        )

                self.source = source_df
            else:
                raise TypeError(
                    'source_df must be a pd.DataFrame if "init" is False, but was type '
                    f"{type(source_df).__name__}"
                )

            if SOURCE_SPEC.MODEL not in source_df.columns.values.tolist():
                raise ValueError(
                    f"{SOURCE_SPEC.MODEL} variable was not found, but is required in a Source object"
                )
            if SOURCE_SPEC.PATHWAY_ID not in source_df.columns.values.tolist():
                raise ValueError(
                    f"{SOURCE_SPEC.PATHWAY_ID} variable was not found, but is required in a Source object"
                )


def create_source_table(
    lookup_table: pd.Series, table_schema: dict, pw_index: indices.PWIndex | None
) -> pd.DataFrame:
    """
    Create Source Table

    Create a table with one row per "new_id" and a Source object created from the unionof "old_id" Source objects

    Parameters
    ----------
    lookup_table: pd.Series
        a pd.Series containing the index of the table to create a source table for
    table_schema: dict
        a dictionary containing the schema of the table to create a source table for
    pw_index: indices.PWIndex
        a pathway index object containing the pathway_id and other metadata

    Returns
    -------
    source_table: pd.DataFrame
        a pd.DataFrame containing the index of the table to create a source table for
        with one row per "new_id" and a Source object created from the union of "old_id" Source objects

    Raises
    ------
    ValueError:
        if SOURCE_SPEC.SOURCE is not present in table_schema
    """

    if SOURCE_SPEC.SOURCE not in table_schema.keys():
        raise ValueError(
            f"{SOURCE_SPEC.SOURCE} not present in schema, can't create source_table"
        )

    # take lookup_table and create an index on "new_id". Multiple rows may have the
    # same value for new_id so these are grouped together.
    lookup_table_rearranged = lookup_table.reset_index().set_index(["new_id"])

    # run a list comprehension over each value of new_id to create a Source
    # object based on the dataframe specific to new_id
    # pw_index is provided to fill out additional meta-information beyond the
    # pathway_id which defines a single source
    def create_source(group):
        return Source(
            group.reset_index(drop=True),
            pw_index=pw_index,
        )

    id_table = (
        lookup_table_rearranged.groupby("new_id")
        .apply(create_source)
        .rename(table_schema[SOURCE_SPEC.SOURCE])
        .to_frame()
    )

    id_table.index = id_table.index.rename(table_schema["pk"])

    return id_table


def merge_sources(source_list: list | pd.Series) -> Source:
    """
    Merge Sources

    Merge a list of Source objects into a single Source object

    Parameters
    ----------
    source_list: list | pd.Series
        a list of Source objects or a pd.Series of Source objects

    Returns
    -------
    source: Source
        a Source object created from the union of the Source objects in source_list

    Raises
    ------
    TypeError:
        if source_list is not a list or pd.Series
    """

    if not isinstance(source_list, (list, pd.Series)):
        raise TypeError(
            f"source_list must be a list or pd.Series, but was a {type(source_list).__name__}"
        )

    # filter to non-empty sources
    # empty sources have only been initialized; a merge hasn't occured
    existing_sources = [s.source is not None for s in source_list]
    if not any(existing_sources):
        if isinstance(source_list, list):
            return source_list[0]
        else:
            return source_list.iloc[0]

    existing_source_list = [
        x.source for x, y in zip(source_list, existing_sources) if y
    ]

    return Source(pd.concat(existing_source_list))


def unnest_sources(source_table: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Unnest Sources

    Take a pd.DataFrame containing an array of Sources and
    return one-row per source.

    Parameters
    ----------
    source_table: pd.DataFrame
        a table containing an array of Sources
    verbose: bool
        print progress

    Returns
    -------
    pd.Dataframe containing the index of source_table but expanded
    to include one row per source

    """

    sources = list()

    table_type = sbml_dfs_utils.infer_entity_type(source_table)
    source_table_schema = SBML_DFS_SCHEMA.SCHEMA[table_type]
    if SCHEMA_DEFS.SOURCE not in source_table_schema.keys():
        raise ValueError(f"{table_type} does not have a source attribute")

    source_var = source_table_schema[SCHEMA_DEFS.SOURCE]
    source_table_index = source_table.index.to_frame().reset_index(drop=True)

    for i in range(source_table.shape[0]):
        if verbose:
            logger.info(f"Processing {source_table_index.index.values[i]}")

        # check that the entries of sourcevar are Source objects
        source_value = source_table[source_var].iloc[i]

        if not isinstance(source_value, Source):
            raise TypeError(
                f"source_value must be a Source, but got {type(source_value).__name__}"
            )

        if source_value.source is None:
            logger.warning("Some sources were only missing - returning None")
            return None

        source_tbl = pd.DataFrame(source_value.source)
        source_tbl.index.name = SOURCE_SPEC.INDEX_NAME
        source_tbl = source_tbl.reset_index()

        # add original index as variables and then set index
        for j in range(source_table_index.shape[1]):
            source_tbl[source_table_index.columns[j]] = source_table_index.iloc[i, j]
        source_tbl = source_tbl.set_index(
            list(source_table_index.columns) + [SOURCE_SPEC.INDEX_NAME]
        )

        sources.append(source_tbl)

    return pd.concat(sources)


def source_set_coverage(
    select_sources_df: pd.DataFrame,
    source_total_counts: Optional[pd.Series | pd.DataFrame] = None,
    sbml_dfs: Optional[sbml_dfs_core.SBML_dfs] = None,
    min_pw_size: int = 3,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Greedy Set Coverage of Sources

    Find the set of pathways covering `select_sources_df`. If `all_sources_df`
    is provided pathways will be selected iteratively based on statistical
    enrichment. If `all_sources_df` is not provided, the largest pathways
    will be chosen iteratively.

    Parameters
    ----------
    select_sources_df: pd.DataFrame
        pd.Dataframe containing the index of source_table but expanded to
        include one row per source. As produced by source.unnest_sources()
    source_total_counts: pd.Series | pd.DataFrame
        pd.Series containing the total counts of each source. As produced by
        source.get_source_total_counts() or a pd.DataFrame with two columns:
        pathway_id and total_counts.
    sbml_dfs: sbml_dfs_core.SBML_dfs
        if `source_total_counts` is provided then `sbml_dfs` must be provided
        to calculate the total number of entities in the table.
    min_pw_size: int
        the minimum size of a pathway to be considered
    verbose: bool
        Whether to print verbose output

    Returns
    -------
    minimial_sources: [str]
        A list of pathway_ids of the minimal source set

    """

    table_type = sbml_dfs_utils.infer_entity_type(select_sources_df)
    pk = SBML_DFS_SCHEMA.SCHEMA[table_type][SCHEMA_DEFS.PK]

    if source_total_counts is not None:
        source_total_counts = _ensure_source_total_counts(
            source_total_counts, verbose=verbose
        )
        if source_total_counts is None:
            raise ValueError("`source_total_counts` is empty or invalid.")

        if sbml_dfs is None:
            raise ValueError(
                "If `source_total_counts` is provided, `sbml_dfs` must be provided to calculate the total number of entities in the table."
            )
        n_total_entities = sbml_dfs.get_table(table_type).shape[0]

        # Filter out pathways that aren't in source_total_counts before processing
        pathways_without_totals = set(select_sources_df[SOURCE_SPEC.PATHWAY_ID]) - set(
            source_total_counts.index
        )
        if len(pathways_without_totals) > 0:
            raise ValueError(
                f"The following pathways are present in `select_sources_df` but not in `source_total_counts`: {', '.join(sorted(pathways_without_totals))}"
            )

        if verbose:
            logger.info(
                f"Finding a minimal sources set based on enrichment of {SOURCE_SPEC.PATHWAY_ID}."
            )
    else:
        if verbose:
            logger.info(
                f"Finding a minimal sources set based on size of {SOURCE_SPEC.PATHWAY_ID}."
            )

    # rollup pathways with identical membership
    deduplicated_sources = _deduplicate_source_df(select_sources_df)

    unaccounted_for_members = deduplicated_sources
    retained_pathway_ids = []
    while unaccounted_for_members.shape[0] != 0:
        # find the pathway with the most members

        if source_total_counts is None:
            top_pathway = _select_top_pathway_by_size(
                unaccounted_for_members, min_pw_size=min_pw_size
            )
        else:
            top_pathway = _select_top_pathway_by_enrichment(
                unaccounted_for_members,
                source_total_counts,
                n_total_entities,
                pk,
                min_pw_size=min_pw_size,
            )

        if top_pathway is None:
            break

        retained_pathway_ids.append(top_pathway)

        # remove all members associated with the top pathway
        unaccounted_for_members = _update_unaccounted_for_members(
            top_pathway, unaccounted_for_members
        )

    minimial_sources = deduplicated_sources[
        deduplicated_sources[SOURCE_SPEC.PATHWAY_ID].isin(retained_pathway_ids)
    ].sort_index()

    return minimial_sources


def _deduplicate_source_df(source_df: pd.DataFrame) -> pd.DataFrame:
    """Combine entries in a source table when multiple models have the same members."""

    table_type = sbml_dfs_utils.infer_entity_type(source_df)
    source_table_schema = SBML_DFS_SCHEMA.SCHEMA[table_type]

    # drop entries which are missing required attributes and throw an error if none are left
    REQUIRED_NON_NA_ATTRIBUTES = [SOURCE_SPEC.PATHWAY_ID]
    indexed_sources = (
        source_df.reset_index()
        .merge(source_df[REQUIRED_NON_NA_ATTRIBUTES].dropna())
        .set_index(SOURCE_SPEC.PATHWAY_ID)
    )

    if indexed_sources.shape[0] == 0:
        raise ValueError(
            f"source_df was provided but zero entries had a defined {' OR '.join(REQUIRED_NON_NA_ATTRIBUTES)}"
        )

    pathways = indexed_sources.index.unique()

    # identify pathways with identical coverage

    pathway_member_string = (
        pd.DataFrame(
            [
                {
                    SOURCE_SPEC.PATHWAY_ID: p,
                    "membership_string": "_".join(
                        set(
                            indexed_sources.loc[[p]][
                                source_table_schema[SCHEMA_DEFS.PK]
                            ].tolist()
                        )
                    ),
                }
                for p in pathways
            ]
        )
        .drop_duplicates()
        .set_index("membership_string")
    )

    membership_categories = pathway_member_string.merge(
        source_df.groupby(SOURCE_SPEC.PATHWAY_ID).first(),
        left_on=SOURCE_SPEC.PATHWAY_ID,
        right_index=True,
    )

    category_index = membership_categories.index.unique()
    if not isinstance(category_index, pd.core.indexes.base.Index):
        raise TypeError(
            f"category_index must be a pandas Index, but got {type(category_index).__name__}"
        )

    merged_sources = pd.concat(
        [
            _collapse_by_membership_string(s, membership_categories, source_table_schema)  # type: ignore
            for s in category_index.tolist()
        ]
    )
    merged_sources[SOURCE_SPEC.INDEX_NAME] = merged_sources.groupby(
        source_table_schema[SCHEMA_DEFS.PK]
    ).cumcount()

    return merged_sources.set_index(
        [source_table_schema[SCHEMA_DEFS.PK], SOURCE_SPEC.INDEX_NAME]
    ).sort_index()


def _collapse_by_membership_string(
    membership_string: str, membership_categories: pd.DataFrame, table_schema: dict
) -> pd.DataFrame:
    """Assign each member of a membership-string to a set of pathways."""

    collapsed_source_membership = _collapse_source_df(
        membership_categories.loc[membership_string]
    )

    return pd.DataFrame(
        [
            pd.concat(
                [
                    pd.Series({table_schema[SCHEMA_DEFS.PK]: ms}),
                    collapsed_source_membership,
                ]
            )
            for ms in membership_string.split("_")
        ]
    )


def _collapse_source_df(source_df: pd.DataFrame) -> pd.Series:
    """Collapse a source_df table into a single entry."""

    if isinstance(source_df, pd.DataFrame):
        collapsed_source_series = pd.Series(
            {
                SOURCE_SPEC.PATHWAY_ID: " OR ".join(source_df[SOURCE_SPEC.PATHWAY_ID]),
                SOURCE_SPEC.MODEL: " OR ".join(source_df[SOURCE_SPEC.MODEL]),
                SOURCE_SPEC.SOURCE: " OR ".join(
                    set(source_df[SOURCE_SPEC.SOURCE].tolist())
                ),
                SOURCE_SPEC.SPECIES: " OR ".join(
                    set(source_df[SOURCE_SPEC.SPECIES].tolist())
                ),
                SOURCE_SPEC.NAME: " OR ".join(source_df[SOURCE_SPEC.NAME]),
                SOURCE_SPEC.N_COLLAPSED_PATHWAYS: source_df.shape[0],
            }
        )
    elif isinstance(source_df, pd.Series):
        collapsed_source_series = pd.Series(
            {
                SOURCE_SPEC.PATHWAY_ID: source_df[SOURCE_SPEC.PATHWAY_ID],
                SOURCE_SPEC.MODEL: source_df[SOURCE_SPEC.MODEL],
                SOURCE_SPEC.SOURCE: source_df[SOURCE_SPEC.SOURCE],
                SOURCE_SPEC.SPECIES: source_df[SOURCE_SPEC.SPECIES],
                SOURCE_SPEC.NAME: source_df[SOURCE_SPEC.NAME],
                SOURCE_SPEC.N_COLLAPSED_PATHWAYS: 1,
            }
        )
    else:
        raise TypeError(
            f"source_df must be a pd.DataFrame or pd.Series, but was a {type(source_df).__name__}"
        )

    return collapsed_source_series


def _safe_source_merge(member_Sources: Source | list) -> Source:
    """Combine either a Source or pd.Series of Sources into a single Source object."""

    if isinstance(member_Sources, Source):
        return member_Sources
    elif isinstance(member_Sources, pd.Series):
        return merge_sources(member_Sources.tolist())
    else:
        raise TypeError("Expecting source.Source or pd.Series")


def _select_top_pathway_by_size(
    unaccounted_for_members: pd.DataFrame, min_pw_size: int = 3
) -> str:

    pathway_members = unaccounted_for_members.value_counts(SOURCE_SPEC.PATHWAY_ID)
    pathway_members = pathway_members.loc[pathway_members >= min_pw_size]
    if pathway_members.shape[0] == 0:
        return None

    top_pathway = pathway_members[pathway_members == max(pathway_members)].index[0]

    return top_pathway


def _select_top_pathway_by_enrichment(
    unaccounted_for_members: pd.DataFrame,
    source_total_counts: pd.Series,
    n_total_entities: int,
    table_pk: str,
    min_pw_size: int = 3,
) -> str:

    n_observed_entities = len(
        unaccounted_for_members.index.get_level_values(table_pk).unique()
    )
    pathway_members = unaccounted_for_members.value_counts(
        SOURCE_SPEC.PATHWAY_ID
    ).rename(CONTINGENCY_TABLE.OBSERVED_MEMBERS)

    pathway_members = pathway_members.loc[pathway_members >= min_pw_size]
    if pathway_members.shape[0] == 0:
        return None

    wide_contingency_table = (
        pathway_members.to_frame()
        .join(source_total_counts)
        .assign(
            missing_members=lambda x: x[CONTINGENCY_TABLE.TOTAL_COUNTS]
            - x[CONTINGENCY_TABLE.OBSERVED_MEMBERS],
            observed_nonmembers=lambda x: n_observed_entities
            - x[CONTINGENCY_TABLE.OBSERVED_MEMBERS],
            nonobserved_nonmembers=lambda x: n_total_entities
            - x[CONTINGENCY_TABLE.OBSERVED_NONMEMBERS]
            - x[CONTINGENCY_TABLE.MISSING_MEMBERS]
            - x[CONTINGENCY_TABLE.OBSERVED_MEMBERS],
        )
        .drop(columns=[CONTINGENCY_TABLE.TOTAL_COUNTS])
    )

    # calculate enrichments using a fast vectorized normal approximation
    odds_ratios, _ = hypothesis_testing.fisher_exact_vectorized(
        wide_contingency_table["observed_members"],
        wide_contingency_table["missing_members"],
        wide_contingency_table["observed_nonmembers"],
        wide_contingency_table["nonobserved_nonmembers"],
    )

    return pathway_members.index[np.argmax(odds_ratios)]


def _update_unaccounted_for_members(
    top_pathway, unaccounted_for_members
) -> pd.DataFrame:
    """
    Update the unaccounted for members dataframe by removing the members
    associated with the top pathway.

    Parameters
    ----------
    top_pathway: str
        the pathway to remove from the unaccounted for members
    unaccounted_for_members: pd.DataFrame
        the dataframe of unaccounted for members

    Returns
    -------
    unaccounted_for_members: pd.DataFrame
        the dataframe of unaccounted for members with the top pathway removed
    """

    table_type = sbml_dfs_utils.infer_entity_type(unaccounted_for_members)
    pk = SBML_DFS_SCHEMA.SCHEMA[table_type][SCHEMA_DEFS.PK]

    members_captured = (
        unaccounted_for_members[
            unaccounted_for_members[SOURCE_SPEC.PATHWAY_ID] == top_pathway
        ]
        .index.get_level_values(pk)
        .tolist()
    )

    return unaccounted_for_members[
        ~unaccounted_for_members.index.get_level_values(pk).isin(members_captured)
    ]


def _ensure_source_total_counts(
    source_total_counts: Optional[pd.Series | pd.DataFrame], verbose: bool = False
) -> Optional[pd.Series]:
    if source_total_counts is None:
        return None

    if isinstance(source_total_counts, pd.DataFrame):
        if SOURCE_SPEC.PATHWAY_ID not in source_total_counts.columns:
            raise ValueError(
                f"`source_total_counts` must have a `{SOURCE_SPEC.PATHWAY_ID}` column. Observed columns are: {source_total_counts.columns.tolist()}"
            )
        if CONTINGENCY_TABLE.TOTAL_COUNTS not in source_total_counts.columns:
            raise ValueError(
                f"`source_total_counts` must have a `{CONTINGENCY_TABLE.TOTAL_COUNTS}` column. Observed columns are: {source_total_counts.columns.tolist()}"
            )
        if source_total_counts.shape[1] > 2:
            raise ValueError(
                f"`source_total_counts` must have only two columns: `{SOURCE_SPEC.PATHWAY_ID}` and `{CONTINGENCY_TABLE.TOTAL_COUNTS}`."
            )
        # convert to a pd.Series
        source_total_counts = source_total_counts.set_index(SOURCE_SPEC.PATHWAY_ID)[
            CONTINGENCY_TABLE.TOTAL_COUNTS
        ]

    if source_total_counts.shape[0] == 0:
        if verbose:
            logger.warning("`source_total_counts` is empty; returning None.")
        return None

    return source_total_counts
