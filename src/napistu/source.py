from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from napistu import indices, sbml_dfs_core, utils
from napistu.constants import (
    SBML_DFS_SCHEMA,
    SCHEMA_DEFS,
    SOURCE_SPEC,
    SOURCE_STANDARD_COLUMNS,
)
from napistu.statistics import hypothesis_testing
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
    empty() : classmethod
        Create an empty Source object
    single_entry(model, pathway_id, **kwargs) : classmethod
        Create a Source object with a single entry
    validate_single_source() : bool
        Check whether the Source object contains exactly 1 entry
    _validate_source_df(source_df) : None
        Validate that source_df is a pandas DataFrame with required columns
    _validate_pathway_index(pw_index, source_df) : None
        Validate pathway index and check for missing pathways
    _process_source_df(source_df, pw_index) : pd.DataFrame
        Process source DataFrame by merging with pathway index if provided
    _validate_final_df(df, required_columns) : None
        Validate that the final DataFrame has all required columns

    """

    def __init__(
        self,
        source_df: pd.DataFrame,
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
        pw_index : indices.PWIndex, optional
            A pathway index object containing the pathway_id and other metadata

        Returns
        -------
        None.

        Raises
        ------
        ValueError:
            If pw_index is not a indices.PWIndex
        ValueError:
            If required columns are not present in source_df
        TypeError:
            If source_df is not a pd.DataFrame
        """
        # Validate inputs
        self._validate_source_df(source_df)
        if pw_index is not None:
            self._validate_pathway_index(pw_index, source_df)

        # Process DataFrame (merge with pathway index if provided)
        processed_df = self._process_source_df(source_df, pw_index)

        # Final validation and assignment
        required_columns = [SOURCE_SPEC.MODEL, SOURCE_SPEC.PATHWAY_ID]
        self._validate_final_df(processed_df, required_columns)
        self.source = processed_df

    def validate_single_source(self) -> bool:
        """
        Check whether the Source object contains exactly 1 entry.

        Returns
        -------
        bool
            True if the Source contains exactly one row, False otherwise

        Raises
        ------
        ValueError:
            If the Source object is empty or contains more than one row
        """

        if self.source is None:
            raise ValueError("Source object is empty and must contain exactly one row")
        if len(self.source) != 1:
            raise ValueError(
                f"Source object must contain exactly one row, but has {len(self.source)} rows"
            )

        return True

    @classmethod
    def empty(cls) -> "Source":
        """
        Create an empty Source object.

        This is typically used when creating an SBML_dfs object from a single source.

        Returns
        -------
        Source
            An empty Source instance with source attribute set to None
        """
        instance = cls.__new__(cls)  # Create instance without calling __init__
        instance.source = None
        return instance

    @classmethod
    def single_entry(
        cls,
        model: str,
        pathway_id: str | None = None,
        file: str | None = None,
        data_source: str | None = None,
        organismal_species: str | None = None,
        name: str | None = None,
        date: str | None = None,
    ) -> "Source":
        """
        Create a Source object with a single entry.

        Convenience method for creating a Source with one row containing the core
        attributes from the pathway index schema.

        Parameters
        ----------
        model : str
            The model identifier (required)
        pathway_id : str, optional
            The pathway identifier. Defaults to same as model if not provided
        file : str, optional
            Source file path or identifier
        data_source : str, optional
            Source database or origin (e.g., 'Reactome', 'KEGG')
        organismal_species : str, optional
            Species the pathway is from
        name : str, optional
            Human-readable pathway/model name
        date : str, optional
            Date of pathway/model creation or last update

        Returns
        -------
        Source
            A Source instance with a single-row DataFrame

        Examples
        --------
        >>> source = Source.single_entry(
        ...     model="R-HSA-123",
        ...     name="Glycolysis",
        ...     source="Reactome",
        ...     organismal_species="Homo sapiens"
        ... )
        """
        # Set pathway_id to model if not provided
        if pathway_id is None:
            pathway_id = model

        # Build the data dictionary with core fields
        data = {
            SOURCE_SPEC.MODEL: model,
            SOURCE_SPEC.PATHWAY_ID: pathway_id,
        }

        # Add optional fields if provided (using actual SOURCE_SPEC column names)
        optional_fields = {
            SOURCE_SPEC.FILE: file,
            SOURCE_SPEC.DATA_SOURCE: data_source,
            SOURCE_SPEC.ORGANISMAL_SPECIES: organismal_species,
            SOURCE_SPEC.NAME: name,
            SOURCE_SPEC.DATE: date,
        }

        for field_name, value in optional_fields.items():
            data[field_name] = value

        # Create DataFrame with single row
        source_df = pd.DataFrame([data])

        # Create and return Source instance
        return cls(source_df)

    def _validate_source_df(self, source_df: pd.DataFrame) -> None:
        """Validate that source_df is a pandas DataFrame with required columns."""
        if not isinstance(source_df, pd.DataFrame):
            raise TypeError(
                f"source_df must be a pd.DataFrame, but was type {type(source_df).__name__}"
            )

        if SOURCE_SPEC.MODEL not in source_df.columns:
            raise ValueError(
                f"{SOURCE_SPEC.MODEL} variable was not found, but is required in a Source object"
            )

    def _validate_pathway_index(
        self, pw_index: indices.PWIndex, source_df: pd.DataFrame
    ) -> None:
        """Validate pathway index and check for missing pathways."""
        if not isinstance(pw_index, indices.PWIndex):
            raise ValueError(
                f"pw_index must be a indices.PWIndex or None and was {type(pw_index).__name__}"
            )

        # Check that all models are present in the pathway index
        source_models = set(source_df[SOURCE_SPEC.MODEL])
        index_pathways = set(pw_index.index[SOURCE_SPEC.PATHWAY_ID])
        missing_pathways = source_models - index_pathways

        if missing_pathways:
            raise ValueError(
                f"{len(missing_pathways)} pathway models are present in source_df "
                f"but not the pw_index: {', '.join(missing_pathways)}"
            )

    def _process_source_df(
        self, source_df: pd.DataFrame, pw_index: indices.PWIndex | None
    ) -> pd.DataFrame:
        """Process source DataFrame by merging with pathway index if provided."""
        if pw_index is None:
            return source_df

        return source_df.merge(
            pw_index.index,
            left_on=SOURCE_SPEC.MODEL,
            right_on=SOURCE_SPEC.PATHWAY_ID,
        )

    def _validate_final_df(self, df: pd.DataFrame, required_columns: list[str]) -> None:
        """Validate that the final DataFrame has all required columns."""
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Required columns {missing_columns} not found in final DataFrame"
            )


def create_source_table(
    lookup_table: pd.Series, table_schema: dict, pw_index: indices.PWIndex | None
) -> pd.DataFrame:
    """
    Create Source Table

    Create a table with one row per "new_id" and a Source object created from the union of "old_id" Source objects

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
        if SOURCE_SPEC.DATA_SOURCE is not present in table_schema
    """

    if SCHEMA_DEFS.SOURCE not in table_schema.keys():
        raise ValueError(
            f"{SCHEMA_DEFS.SOURCE} not present in schema, can't create source_table"
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
        .rename(table_schema[SCHEMA_DEFS.SOURCE])
        .to_frame()
    )

    id_table.index = id_table.index.rename(table_schema[SCHEMA_DEFS.PK])

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


def unnest_sources(source_table: pd.DataFrame) -> pd.DataFrame:
    """
    Unnest Sources - Optimized Version

    Take a pd.DataFrame containing an array of Sources and
    return one-row per source.

    Parameters
    ----------
    source_table: pd.DataFrame
        a table containing an array of Sources

    Returns
    -------
    pd.Dataframe containing the index of source_table but expanded
    to include one row per source
    """

    table_type = utils.infer_entity_type(source_table)
    source_table_schema = SBML_DFS_SCHEMA.SCHEMA[table_type]

    if SCHEMA_DEFS.SOURCE not in source_table_schema.keys():
        raise ValueError(f"{table_type} does not have a source attribute")

    source_var = source_table_schema[SCHEMA_DEFS.SOURCE]

    # Build dict mapping each source's primary key to its source DataFrame
    source_dict = {}
    for idx, source_value in source_table[source_var].items():
        if not isinstance(source_value, Source):
            raise TypeError(
                f"source_value must be a Source, but got {type(source_value).__name__}"
            )

        # Skip None sources - they just won't appear in output
        if source_value.source is not None:
            source_dict[idx] = source_value.source

    # If no valid sources, return None (maintains original behavior)
    if not source_dict:
        logger.warning("Some sources were only missing - returning None")
        return None

    # Use pd.concat with keys parameter to create MultiIndex directly
    result = pd.concat(source_dict, names=[source_table.index.name, SOURCE_SPEC.ENTRY])

    # Only keep columns that actually exist in the result
    available_columns = [
        col for col in list(SOURCE_STANDARD_COLUMNS) if col in result.columns
    ]
    result = result[available_columns]

    return result


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

    table_type = utils.infer_entity_type(select_sources_df)
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

    table_type = utils.infer_entity_type(source_df)
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
    merged_sources[SOURCE_SPEC.ENTRY] = merged_sources.groupby(
        source_table_schema[SCHEMA_DEFS.PK]
    ).cumcount()

    return merged_sources.set_index(
        [source_table_schema[SCHEMA_DEFS.PK], SOURCE_SPEC.ENTRY]
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


def _collapse_source_df(source_df: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    """Collapse a source_df table into a single entry.

    Combines multiple source entries into a single entry by joining values
    with " OR " separators. Handles None values by filtering them out before joining.

    Parameters
    ----------
    source_df : pd.DataFrame or pd.Series
        Source data to collapse. Must contain required columns MODEL and PATHWAY_ID.

    Returns
    -------
    pd.Series
        Collapsed source entry with joined values and count of collapsed pathways.

    Raises
    ------
    TypeError
        If source_df is not a DataFrame or Series.
    ValueError
        If required columns MODEL or PATHWAY_ID are missing.

    Notes
    -----
    - None values are filtered out before joining
    - For DataFrame input, unique values are used for DATA_SOURCE and ORGANISMAL_SPECIES
    - The N_COLLAPSED_PATHWAYS field tracks how many entries were collapsed
    """

    if isinstance(source_df, pd.DataFrame):
        # Validate required columns
        required_cols = [SOURCE_SPEC.MODEL, SOURCE_SPEC.PATHWAY_ID]
        missing_cols = [col for col in required_cols if col not in source_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        collapsed_source_series = pd.Series(
            {
                SOURCE_SPEC.PATHWAY_ID: utils.safe_join_set(
                    source_df[SOURCE_SPEC.PATHWAY_ID]
                ),
                SOURCE_SPEC.MODEL: utils.safe_join_set(source_df[SOURCE_SPEC.MODEL]),
                SOURCE_SPEC.DATA_SOURCE: (
                    utils.safe_join_set(source_df[SOURCE_SPEC.DATA_SOURCE])
                    if SOURCE_SPEC.DATA_SOURCE in source_df.columns
                    else None
                ),
                SOURCE_SPEC.ORGANISMAL_SPECIES: (
                    utils.safe_join_set(source_df[SOURCE_SPEC.ORGANISMAL_SPECIES])
                    if SOURCE_SPEC.ORGANISMAL_SPECIES in source_df.columns
                    else None
                ),
                SOURCE_SPEC.NAME: (
                    utils.safe_join_set(source_df[SOURCE_SPEC.NAME])
                    if SOURCE_SPEC.NAME in source_df.columns
                    else None
                ),
                SOURCE_SPEC.N_COLLAPSED_PATHWAYS: source_df.shape[0],
            }
        )
    elif isinstance(source_df, pd.Series):
        # Validate required columns
        required_cols = [SOURCE_SPEC.MODEL, SOURCE_SPEC.PATHWAY_ID]
        missing_cols = [col for col in required_cols if col not in source_df.index]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        collapsed_source_series = pd.Series(
            {
                SOURCE_SPEC.PATHWAY_ID: source_df[SOURCE_SPEC.PATHWAY_ID],
                SOURCE_SPEC.MODEL: source_df[SOURCE_SPEC.MODEL],
                SOURCE_SPEC.DATA_SOURCE: source_df.get(SOURCE_SPEC.DATA_SOURCE),
                SOURCE_SPEC.ORGANISMAL_SPECIES: source_df.get(
                    SOURCE_SPEC.ORGANISMAL_SPECIES
                ),
                SOURCE_SPEC.NAME: source_df.get(SOURCE_SPEC.NAME),
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

    source_total_counts = _ensure_source_total_counts(source_total_counts)
    if source_total_counts is None:
        raise ValueError("`source_total_counts` is empty or invalid.")

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

    table_type = utils.infer_entity_type(unaccounted_for_members)
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

    # Ensure the Series has the correct name and index name
    if source_total_counts.name != CONTINGENCY_TABLE.TOTAL_COUNTS:
        if verbose:
            logger.warning(
                f"source_total_counts has name '{source_total_counts.name}' but expected '{CONTINGENCY_TABLE.TOTAL_COUNTS}'. Renaming to '{CONTINGENCY_TABLE.TOTAL_COUNTS}'."
            )
        source_total_counts = source_total_counts.rename(CONTINGENCY_TABLE.TOTAL_COUNTS)

    if source_total_counts.index.name != SOURCE_SPEC.PATHWAY_ID:
        if verbose:
            logger.warning(
                f"source_total_counts has index name '{source_total_counts.index.name}' but expected '{SOURCE_SPEC.PATHWAY_ID}'. Renaming to '{SOURCE_SPEC.PATHWAY_ID}'."
            )
        source_total_counts.index.name = SOURCE_SPEC.PATHWAY_ID

    # index should be character and values should be integerish
    if not source_total_counts.index.dtype == "object":
        raise ValueError(
            f"source_total_counts index must be a string, but got {source_total_counts.index.dtype}"
        )
    if not np.issubdtype(source_total_counts.values.dtype, np.number):
        raise ValueError(
            f"source_total_counts values must be numeric, but got {source_total_counts.values.dtype}"
        )

    return source_total_counts
