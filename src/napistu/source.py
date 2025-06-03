from __future__ import annotations

import pandas as pd
from napistu import indices
from napistu.constants import SOURCE_SPEC


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

        Returns
        -------
        None.

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

    Create a table with one row per "new_id" and a Source object created from the union
      of "old_id" Source objects
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

    """

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


def unnest_sources(
    source_table: pd.DataFrame, source_var: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Unnest Sources

    Take a pd.DataFrame containing an array of Sources and
    return one-row per source.

    Parameters:
    source_table: pd.DataFrame
        a table containing an array of Sources
    source_var: str
        variable containing Sources

    Returns:
    pd.Dataframe containing the index of source_table but expanded
    to include one row per source

    """

    sources = list()
    source_table_index = source_table.index.to_frame().reset_index(drop=True)

    for i in range(source_table.shape[0]):
        if verbose:
            print(f"Processing {source_table_index.index.values[i]}")

        # check that the entries of sourcevar are Source objects
        source_value = source_table[source_var].iloc[i]

        if not isinstance(source_value, Source):
            raise TypeError(
                f"source_value must be a Source, but got {type(source_value).__name__}"
            )

        if source_value.source is None:
            print("Some sources were only missing - returning None")
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


def greedy_set_coverge_of_sources(
    source_df: pd.DataFrame, table_schema: dict
) -> pd.DataFrame:
    """
    Greedy Set Coverage of Sources

    Apply the greedy set coverge algorithm to find the minimal set of
      sources which cover all entries

    Parameters:
    source_df: pd.DataFrame
        pd.Dataframe containing the index of source_table but expanded to
        include one row per source. As produced by source.unnest_sources()

    Returns:
    minimial_sources: [str]
        A list of pathway_ids of the minimal source set

    """

    # rollup pathways with identical membership
    deduplicated_sources = _deduplicate_source_df(source_df, table_schema)

    unaccounted_for_members = deduplicated_sources
    retained_pathway_ids = []

    while unaccounted_for_members.shape[0] != 0:
        # find the pathway with the most members
        pathway_members = unaccounted_for_members.groupby(SOURCE_SPEC.PATHWAY_ID).size()
        top_pathway = pathway_members[pathway_members == max(pathway_members)].index[0]
        retained_pathway_ids.append(top_pathway)

        # remove all members associated with the top pathway
        members_captured = (
            unaccounted_for_members[
                unaccounted_for_members[SOURCE_SPEC.PATHWAY_ID] == top_pathway
            ]
            .index.get_level_values(table_schema["pk"])
            .tolist()
        )

        unaccounted_for_members = unaccounted_for_members[
            ~unaccounted_for_members.index.get_level_values(table_schema["pk"]).isin(
                members_captured
            )
        ]

    minimial_sources = deduplicated_sources[
        deduplicated_sources[SOURCE_SPEC.PATHWAY_ID].isin(retained_pathway_ids)
    ].sort_index()

    return minimial_sources


def _deduplicate_source_df(source_df: pd.DataFrame, table_schema: dict) -> pd.DataFrame:
    """Combine entries in a source table when multiple models have the same members."""

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
                        set(indexed_sources.loc[[p]][table_schema["pk"]].tolist())
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
            _collapse_by_membership_string(s, membership_categories, table_schema)  # type: ignore
            for s in category_index.tolist()
        ]
    )
    merged_sources[SOURCE_SPEC.INDEX_NAME] = merged_sources.groupby(
        table_schema["pk"]
    ).cumcount()

    return merged_sources.set_index(
        [table_schema["pk"], SOURCE_SPEC.INDEX_NAME]
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
                [pd.Series({table_schema["pk"]: ms}), collapsed_source_membership]
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
