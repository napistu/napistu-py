"""
Systematic identifiers for species, reactions, compartments, etc.

Classes
-------
Identifiers
    Identifiers for a single entity or relationship.

Public Functions
----------------
construct_cspecies_identifiers
    Construct compartmentalized species identifiers by adding sc_id to species_identifiers.
df_to_identifiers
    Convert a DataFrame of identifier information to a Series of Identifiers objects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd
from pydantic import BaseModel

from napistu.constants import (
    BQB_PRIORITIES,
    IDENTIFIERS,
    IDENTIFIERS_REQUIRED_VARS,
    SBML_DFS,
    SBML_DFS_SCHEMA,
    SCHEMA_DEFS,
    SPECIES_IDENTIFIERS_REQUIRED_VARS,
)
from napistu.utils.display_utils import show
from napistu.utils.pd_utils import infer_entity_type, match_pd_vars

if TYPE_CHECKING:
    from napistu.sbml_dfs_core import SBML_dfs


logger = logging.getLogger(__name__)


class Identifiers:
    """
    Identifiers for a single entity or relationship.

    Attributes
    ----------
    df : pd.DataFrame
        a DataFrame of identifiers with columns ontology, identifier, url, bqb

    Properties
    ----------
    ids : list
        (deprecated) a list of identifiers which are each a dict containing an ontology and identifier

    Public Methods
    -------
    get_all_bqbs()
        Returns a set of all BQB entries
    get_all_ontologies()
        Returns a set of all ontology entries
    has_ontology(ontologies)
        Returns a bool of whether 1+ of the ontologies was represented
    hoist(ontology)
        Returns value(s) from an ontology
    print
        Print a table of identifiers
    """

    def __init__(self, id_list: list, verbose: bool = False) -> None:
        """
        Tracks a set of identifiers and the ontologies they belong to.

        Parameters
        ----------
        id_list : list
            a list of identifier dictionaries containing ontology, identifier, and optionally url
        verbose : bool
            extra reporting, defaults to False

        Returns
        -------
        None.

        """

        # read list and validate format
        validated_id_list = _IdentifiersValidator(id_list=id_list).model_dump()[
            "id_list"
        ]

        if validated_id_list:
            df = _deduplicate_identifiers_by_priority(
                pd.DataFrame(validated_id_list),
                [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER],
            )
        else:
            # Empty DataFrame with expected schema
            df = pd.DataFrame(
                columns=[
                    IDENTIFIERS.ONTOLOGY,
                    IDENTIFIERS.IDENTIFIER,
                    IDENTIFIERS.URL,
                    IDENTIFIERS.BQB,
                ]
            )

        self.df = df.astype(
            {
                IDENTIFIERS.ONTOLOGY: "string",
                IDENTIFIERS.IDENTIFIER: "string",
                IDENTIFIERS.URL: "string",
                IDENTIFIERS.BQB: "string",
            }
        )

    def get_all_bqbs(self) -> set[str]:
        """Returns a set of all BQB entries

        Returns:
            set[str]: A set containing all unique BQB values from the identifiers
        """
        return set(self.df[IDENTIFIERS.BQB].dropna().unique())

    def get_all_ontologies(self, bqb_terms: list[str] = None) -> set[str]:
        """Returns a set of all ontology entries

        Returns:
            set[str]: A set containing all unique ontology names from the identifiers
        """

        if bqb_terms is not None:
            return set(
                self.df[self.df[IDENTIFIERS.BQB].isin(bqb_terms)][IDENTIFIERS.ONTOLOGY]
            )
        else:
            return set(self.df[IDENTIFIERS.ONTOLOGY])

    def has_ontology(self, ontologies: str | list[str]) -> bool:
        """
        Check if specified ontologies are present in the identifiers.

        Parameters
        ----------
        ontologies : str or list of str
            Ontology name(s) to search for

        Returns
        -------
        bool
            True if any specified ontologies are present
        """

        if isinstance(ontologies, str):
            ontologies = [ontologies]

        if self.df.empty:
            return False

        # Check if any rows have matching ontologies
        return bool(self.df[IDENTIFIERS.ONTOLOGY].isin(ontologies).any())

    def hoist(self, ontology: str, squeeze: bool = True) -> str | list[str] | None:
        """Returns value(s) from an ontology

        Args:
            ontology (str): the ontology of interest
            squeeze (bool): if True, return a single value if possible

        Returns:
            str or list: the value(s) of an ontology of interest

        """

        if not isinstance(ontology, str):
            raise TypeError(f"{ontology} must be a str")

        # return the value(s) of an ontology of interest
        matches = self.df[self.df[IDENTIFIERS.ONTOLOGY] == ontology]
        ontology_ids = matches[IDENTIFIERS.IDENTIFIER].tolist()

        if squeeze:
            if len(ontology_ids) == 0:
                return None
            elif len(ontology_ids) == 1:
                return ontology_ids[0]
        return ontology_ids

    @property
    def ids(self) -> list[dict]:

        logger.warning("Identifiers.ids is deprecated. Use Identifiers.df instead.")
        return self.df.to_dict("records") if self.df is not None else []

    @classmethod
    def merge(cls, identifier_series: pd.Series) -> "Identifiers":
        """
        Merge multiple Identifiers objects into a single Identifiers object.

        Parameters
        ----------
        identifier_series : pd.Series
            Series of Identifiers objects to merge

        Returns
        -------
        Identifiers
            New Identifiers object containing all unique identifiers
        """

        if len(identifier_series) == 1:
            return identifier_series.iloc[0]

        # Concatenate all DataFrames and let __init__ handle deduplication
        all_dfs = [
            identifiers.df
            for identifiers in identifier_series
            if not identifiers.df.empty
        ]

        if not all_dfs:
            return cls([])  # Return empty Identifiers

        merged_df = pd.concat(all_dfs, ignore_index=True)

        # Convert back to list format for __init__ to handle deduplication and validation
        merged_ids = merged_df.to_dict("records")

        return cls(merged_ids)

    def print(self):
        """Print a table of identifiers"""

        show(self.df, hide_index=True)

    def __repr__(self) -> str:
        """Return a string representation of the Identifiers object"""
        return f"Identifiers({self.df.shape[0]} identifiers)"


def construct_cspecies_identifiers(
    species_identifiers: pd.DataFrame,
    cspecies_references: Union["SBML_dfs", pd.DataFrame],
) -> pd.DataFrame:
    """
    Construct compartmentalized species identifiers by adding sc_id to species_identifiers.

    This function merges compartmentalized species IDs (sc_id) into a species_identifiers
    table, allowing you to work with compartmentalized species without loading the full
    sbml_dfs object.

    Parameters
    ----------
    species_identifiers : pd.DataFrame
        A species identifiers table with columns including s_id, ontology, identifier.
        Must satisfy SPECIES_IDENTIFIERS_REQUIRED_VARS.
    cspecies_references : Union[sbml_dfs_core.SBML_dfs, pd.DataFrame]
        Either an sbml_dfs object from which compartmentalized_species will be extracted,
        or a 2-column DataFrame with s_id and sc_id columns.

    Returns
    -------
    pd.DataFrame
        The species_identifiers table with an additional sc_id column. Each row
        in the original table will be expanded to include all corresponding sc_ids
        for that s_id.
    """

    from napistu.sbml_dfs_core import SBML_dfs

    # Validate input species_identifiers table
    _check_species_identifiers_table(species_identifiers)

    # Extract sid_to_scids table based on type of cspecies_references
    if isinstance(cspecies_references, SBML_dfs):
        sid_to_scids = cspecies_references.compartmentalized_species.reset_index()[
            [SBML_DFS.S_ID, SBML_DFS.SC_ID]
        ]
    elif isinstance(cspecies_references, pd.DataFrame):
        sid_to_scids = cspecies_references
        match_pd_vars(
            sid_to_scids,
            req_vars={SBML_DFS.S_ID, SBML_DFS.SC_ID},
            allow_series=False,
        ).assert_present()
    else:
        raise TypeError(
            f"cspecies_references must be either an SBML_dfs object or a pandas DataFrame, "
            f"got {type(cspecies_references)}"
        )

    species_identifiers_w_scids = species_identifiers.merge(
        sid_to_scids,
        on=SBML_DFS.S_ID,
        how="left",
    )

    if any(species_identifiers_w_scids[SBML_DFS.SC_ID].isna()):
        raise ValueError(
            "Some species identifiers were not found in the cspecies_references table"
        )

    return species_identifiers_w_scids


def df_to_identifiers(df: pd.DataFrame) -> pd.Series:
    """
    Convert a DataFrame of identifier information to a Series of Identifiers objects.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing identifier information with required columns:
        ontology, identifier, url, bqb

    Returns
    -------
    pd.Series
        Series indexed by index_col containing Identifiers objects
    """

    entity_type = infer_entity_type(df)
    table_schema = SBML_DFS_SCHEMA.SCHEMA[entity_type]
    if SCHEMA_DEFS.ID not in table_schema:
        raise ValueError(f"The entity type {entity_type} does not have an id column")

    table_pk_var = table_schema[SCHEMA_DEFS.PK]
    required_vars = {table_pk_var} | IDENTIFIERS_REQUIRED_VARS
    match_pd_vars(df, required_vars).assert_present()

    identifiers_dict = {}
    for pk_value in df[table_pk_var].unique():
        pk_rows = df[df[table_pk_var] == pk_value]
        # Convert to list of dicts format for Identifiers constructor
        id_list = pk_rows.drop(columns=[table_pk_var]).to_dict("records")
        identifiers_dict[pk_value] = Identifiers(
            id_list
        )  # Handles deduplication internally

    output = pd.Series(identifiers_dict, name=table_schema[SCHEMA_DEFS.ID])
    output.index.name = table_pk_var

    return output


# private utility functions


def _check_species_identifiers_table(
    species_identifiers: pd.DataFrame,
    required_vars: set = SPECIES_IDENTIFIERS_REQUIRED_VARS,
):
    missing_required_vars = required_vars.difference(
        set(species_identifiers.columns.tolist())
    )
    if len(missing_required_vars) > 0:
        raise ValueError(
            f"{len(missing_required_vars)} required variables "
            "were missing from the species_identifiers table: "
            f"{', '.join(missing_required_vars)}"
        )

    return None


def _deduplicate_identifiers_by_priority(
    df: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """
    Deduplicate identifiers by prioritizing BQB terms and URL presence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing identifier information with BQB and URL columns
    group_cols : list[str]
        Columns to group by for deduplication (e.g., [ontology, identifier] or [pk, ontology, identifier])

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with highest priority entries retained
    """
    return (
        df.merge(BQB_PRIORITIES, how="left")
        .assign(_has_url=lambda x: x[IDENTIFIERS.URL].notna().astype(int))
        .sort_values(["bqb_rank", "_has_url"], ascending=[True, False])
        .drop_duplicates(subset=group_cols)
        .drop(columns=["bqb_rank", "_has_url"])
    )


def _prepare_species_identifiers(
    sbml_dfs: "SBML_dfs",
    dogmatic: bool = False,
    species_identifiers: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Accepts and validates species_identifiers, or extracts a fresh table if None."""

    if species_identifiers is None:
        species_identifiers = sbml_dfs.get_characteristic_species_ids(dogmatic=dogmatic)
    else:
        # check for compatibility
        try:
            # check species_identifiers format

            _check_species_identifiers_table(species_identifiers)
            # quick check for compatibility between sbml_dfs and species_identifiers
            _validate_assets_sbml_ids(sbml_dfs, species_identifiers)
        except ValueError as e:
            logger.warning(
                f"The provided identifiers are not compatible with your `sbml_dfs` object. Extracting a fresh species identifier table. {e}"
            )
            species_identifiers = sbml_dfs.get_characteristic_species_ids(
                dogmatic=dogmatic
            )

    return species_identifiers


def _validate_assets_sbml_ids(
    sbml_dfs: "SBML_dfs", identifiers_df: pd.DataFrame
) -> None:
    """
    Check an sbml_dfs file and identifiers table for inconsistencies.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The sbml_dfs object to check
    identifiers_df : pd.DataFrame
        The identifiers table to check

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If there are inconsistencies between the sbml_dfs and identifiers_df
    """

    joined_species_w_ids = sbml_dfs.species.merge(
        identifiers_df[["s_id", "s_name"]].drop_duplicates(),
        left_index=True,
        right_on="s_id",
    )

    inconsistent_names_df = joined_species_w_ids.query("s_name_x != s_name_y").dropna()
    inconsistent_names_list = [
        f"{x} != {y}"
        for x, y in zip(
            inconsistent_names_df["s_name_x"], inconsistent_names_df["s_name_y"]
        )
    ]

    if len(inconsistent_names_list):
        example_inconsistent_names = inconsistent_names_list[
            0 : min(10, len(inconsistent_names_list))
        ]

        raise ValueError(
            f"{len(inconsistent_names_list)} species names do not match between "
            f"sbml_dfs and identifiers_df including: {', '.join(example_inconsistent_names)}"
        )

    return None


# validators


class _IdentifierValidator(BaseModel):
    ontology: str
    identifier: str
    bqb: str
    url: Optional[str] = None


class _IdentifiersValidator(BaseModel):
    id_list: list[_IdentifierValidator]
