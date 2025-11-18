from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

if TYPE_CHECKING:
    from napistu.sbml_dfs_core import SBML_dfs

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    # Fallback for non-Jupyter environments
    def display(obj):
        print(obj)


from napistu import identifiers, utils
from napistu.constants import (
    BQB,
    BQB_DEFINING_ATTRS,
    BQB_DEFINING_ATTRS_LOOSE,
    IDENTIFIERS,
    IDENTIFIERS_REQUIRED_VARS,
    MINI_SBO_FROM_NAME,
    MINI_SBO_NAME_TO_POLARITY,
    MINI_SBO_TO_NAME,
    ONTOLOGIES,
    POLARITY_TO_SYMBOL,
    SBML_DFS,
    SBML_DFS_SCHEMA,
    SBO_NAME_TO_ROLE,
    SBO_ROLES_DEFS,
    SBOTERM_NAMES,
    SCHEMA_DEFS,
    SOURCE_SPEC,
    VALID_SBO_TERM_NAMES,
    VALID_SBO_TERMS,
)
from napistu.ingestion.constants import (
    COMPARTMENTS_GO_TERMS,
    DEFAULT_PRIORITIZED_PATHWAYS,
    GENERIC_COMPARTMENT,
    INTERACTION_EDGELIST_DEFAULTS,
    INTERACTION_EDGELIST_DEFS,
    INTERACTION_EDGELIST_EXPECTED_VARS,
    INTERACTION_EDGELIST_OPTIONAL_VARS,
    VALID_COMPARTMENTS,
)
from napistu.ontologies.constants import (
    ONTOLOGY_TO_SPECIES_TYPE,
    PRIORITIZED_SPECIES_TYPES,
    SPECIES_TYPE_PLURAL,
    SPECIES_TYPES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC FUNCTIONS (ALPHABETICAL ORDER)
# =============================================================================


def add_missing_ids_column(
    contingency_table: pd.DataFrame,
    reference_table: pd.DataFrame,
    other_column_name: str = "other",
) -> pd.DataFrame:
    """
    Add an 'other' column to a contingency table for IDs that exist in a reference table
    but are missing from the contingency table.

    Parameters:
    -----------
    contingency_table : pd.DataFrame
        The contingency table with binary values (subset of IDs)
    reference_table : pd.DataFrame
        The reference table containing all possible IDs
    other_column_name : str, optional
        Name for the 'other' column, by default "other"

    Returns:
    --------
    pd.DataFrame
        Updated contingency table with 'other' column(s) added if there are missing IDs.
        If no IDs are missing, returns a copy of the original contingency table without
        adding an 'other' column.

    Raises:
    -------
    ValueError
        If the index names don't match between the two tables
    """

    logger.debug("Validating contingency and reference tables")
    # Check that index names match
    if contingency_table.index.name != reference_table.index.name:
        raise ValueError(
            f"Index names must match. Contingency table index name: '{contingency_table.index.name}', "
            f"Reference table index name: '{reference_table.index.name}'"
        )

    # Get the indices as sets (we'll use these for duplicate checking too)
    contingency_ids = set(contingency_table.index)
    reference_ids = set(reference_table.index)

    # Check that all index values are unique in both tables (using set length comparison)
    if len(contingency_ids) != len(contingency_table.index):
        raise ValueError(
            f"Contingency table has duplicate index values. Expected {len(contingency_ids)} unique values, got {len(contingency_table.index)} total values."
        )

    if len(reference_ids) != len(reference_table.index):
        raise ValueError(
            f"Reference table has duplicate index values. Expected {len(reference_ids)} unique values, got {len(reference_table.index)} total values."
        )

    # Check that contingency_ids is a subset of reference_ids
    if not contingency_ids.issubset(reference_ids):
        extra_ids = contingency_ids - reference_ids
        raise ValueError(
            f"Contingency table contains {len(extra_ids)} IDs not found in reference table. First few: {list(extra_ids)[:5]}"
        )

    # Find missing IDs
    missing_ids = reference_ids - contingency_ids

    # Create a copy of the contingency table to avoid modifying the original
    result_table = contingency_table.copy()

    # If no missing IDs, return the original table without adding 'other' column
    if len(missing_ids) == 0:
        return result_table

    # Create 'other' column name based on column structure
    if isinstance(result_table.columns, pd.MultiIndex):
        other_column = tuple([other_column_name] * result_table.columns.nlevels)
    else:
        other_column = other_column_name

    # Add the 'other' column
    result_table[other_column] = int(0)

    # Add missing IDs as new rows (optimized to avoid iterative concat)
    if missing_ids:
        logger.debug(f"Adding {len(missing_ids)} missing IDs to the result table")
        # Create all missing rows at once
        missing_data = pd.DataFrame(
            0, index=list(missing_ids), columns=result_table.columns
        )
        missing_data[other_column] = 1

        # Single concat operation instead of iterative ones
        result_table = pd.concat([result_table, missing_data])

    # Sort the index to maintain order
    result_table = result_table.sort_index()

    # Verify that the result has the expected number of rows (fast length check)
    if len(result_table.index) != len(reference_table.index):
        raise ValueError(
            f"Result table has {len(result_table.index)} rows, expected {len(reference_table.index)}. This is an internal error."
        )

    return result_table


def add_sbo_role(reaction_species: pd.DataFrame) -> pd.DataFrame:
    """
    Add an sbo_role column to the reaction_species table.

    The sbo_role column is a string column that contains the SBO role of the reaction species.
    The values in the sbo_role column are taken from the sbo_term column.

    The sbo_role column is added to the reaction_species table by mapping the sbo_term column to the SBO_NAME_TO_ROLE dictionary.
    """

    validate_sbml_dfs_table(reaction_species, SBML_DFS.REACTION_SPECIES)

    reaction_species = (
        reaction_species.assign(sbo_role=reaction_species[SBML_DFS.SBO_TERM])
        .replace({SBO_ROLES_DEFS.SBO_ROLE: MINI_SBO_TO_NAME})
        .replace({SBO_ROLES_DEFS.SBO_ROLE: SBO_NAME_TO_ROLE})
    )

    undefined_roles = set(reaction_species[SBO_ROLES_DEFS.SBO_ROLE].unique()) - set(
        SBO_NAME_TO_ROLE.values()
    )
    if len(undefined_roles) > 0:
        logger.warning(
            f"The following SBO roles are not defined: {undefined_roles}. They will be treated as {SBO_ROLES_DEFS.OPTIONAL} when determining reaction operability."
        )
        mask = reaction_species[SBO_ROLES_DEFS.SBO_ROLE].isin(undefined_roles)
        reaction_species.loc[mask, SBO_ROLES_DEFS.SBO_ROLE] = SBO_ROLES_DEFS.OPTIONAL

    return reaction_species


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

    reaction_species_df[SCHEMA_DEFS.LABEL] = [
        _add_stoi_to_species_name(x, y)
        for x, y in zip(
            reaction_species_df[SBML_DFS.STOICHIOMETRY], reaction_species_df[name_var]
        )
    ]

    if all(
        reaction_species_df[SBML_DFS.SBO_TERM]
        == MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR]
    ):
        labels = reaction_species_df[SCHEMA_DEFS.LABEL].tolist()
        if len(labels) != 2:
            logger.warning(
                f"Reaction {reaction_species_df[SBML_DFS.R_ID].iloc[0]} has {len(labels)} species, expected 2"
            )
            return None
        return f"{labels[0]} ---- {labels[1]}"

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
    product_sbo_terms = [
        MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
        MINI_SBO_FROM_NAME[SBOTERM_NAMES.MODIFIER],
    ]
    products = " + ".join(
        reaction_species_df["label"][
            reaction_species_df[SBML_DFS.SBO_TERM].isin(product_sbo_terms)
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


def create_reaction_formula_series(
    reaction_data,
    reactions_df,
    species_name_col,
    sort_cols,
    group_cols=None,
    add_compartment_prefix=False,
    r_id_col=SBML_DFS.R_ID,
    c_name_col=SBML_DFS.C_NAME,
):
    """
    Helper function to create reaction formula series.

    Parameters:
    -----------
    reaction_data : pd.DataFrame
        The reaction species data to process
    reactions_df : pd.DataFrame
        The reactions dataframe needed by construct_formula_string
    species_name_col : str
        Column name to use for species names in formulas
    sort_cols : list
        Columns to sort by before grouping
    group_cols : list, optional
        Columns to group by. If None, uses [r_id_col]
    add_compartment_prefix : bool
        Whether to add compartment name as prefix to formula
    r_id_col : str
        Column name for reaction ID
    c_name_col : str
        Column name for compartment name (used when add_compartment_prefix=True)

    Returns:
    --------
    pd.Series or None : Formula strings indexed by reaction ID, or None if no data
    """
    if reaction_data.shape[0] == 0:
        return None

    if group_cols is None:
        group_cols = [r_id_col]

    # Include all columns that might be needed by construct_formula_string
    # We include all original columns to avoid accidentally filtering out needed data
    all_cols = list(reaction_data.columns)

    formulas = (
        reaction_data.sort_values(sort_cols)
        .groupby(group_cols)[all_cols]  # Use all columns to be safe
        .apply(lambda x: construct_formula_string(x, reactions_df, species_name_col))
    )

    if add_compartment_prefix:
        # Add compartment prefix and reindex by reaction ID only
        formulas = pd.Series(
            [
                f"{compartment}: {formula}"
                for formula, compartment in zip(
                    formulas, formulas.index.get_level_values(c_name_col)
                )
            ],
            index=formulas.index.get_level_values(r_id_col),
        )

    return formulas.rename("r_formula_str")


def display_post_consensus_checks(checks_results: dict) -> None:
    """
    Display the results of post_consensus_checks in a formatted way.

    This function takes the results from the post_consensus_checks method and displays
    them using the same formatting as shown in the sandbox notebook.

    Parameters
    ----------
    checks_results : dict
        Dictionary returned by the post_consensus_checks method, containing nested
        dictionaries with entity types and check types as keys, and DataFrames as values.

    Returns
    -------
    None
        This function displays results but doesn't return anything.
    """
    for entity_type, entity_results in checks_results.items():
        for check_type, cooccurrences in entity_results.items():
            utils.show(f"Entity type: {entity_type}, Check type: {check_type}")
            utils.show(cooccurrences)


def find_underspecified_reactions(
    reaction_species_w_roles: pd.DataFrame,
) -> pd.DataFrame:

    # check that both sbo_role and "new" are present
    if SBO_ROLES_DEFS.SBO_ROLE not in reaction_species_w_roles.columns:
        raise ValueError(
            "The sbo_role column is not present in the reaction_species_w_roles table. Please call sbml_dfs_utils.add_sbo_role() first."
        )
    if "new" not in reaction_species_w_roles.columns:
        raise ValueError(
            "The new column is not present in the reaction_species_w_roles table. This should indicate what cspecies would be preserved in the reaction should it be preserved."
        )
    # check that new is a boolean column
    if reaction_species_w_roles["new"].dtype != bool:
        raise ValueError(
            "The new column is not a boolean column. Please ensure that the new column is a boolean column. This should indicate what cspecies would be preserved in the reaction should it be preserved."
        )

    reactions_with_lost_defining_members = set(
        reaction_species_w_roles.query("~new")
        .query("sbo_role == 'DEFINING'")[SBML_DFS.R_ID]
        .tolist()
    )

    N_reactions_with_lost_defining_members = len(reactions_with_lost_defining_members)
    if N_reactions_with_lost_defining_members > 0:
        logger.info(
            f"Removing {N_reactions_with_lost_defining_members} reactions which have lost at least one defining species"
        )

    # find the cases where all "new" values for a given (r_id, sbo_term) are False
    reactions_with_lost_requirements = set(
        reaction_species_w_roles
        # drop already filtered reactions
        .query("r_id not in @reactions_with_lost_defining_members")
        .query("sbo_role == 'REQUIRED'")
        # which entries which have some required attribute have all False values for that attribute
        .groupby([SBML_DFS.R_ID, SBML_DFS.SBO_TERM])
        .agg({"new": "any"})
        .query("new == False")
        .index.get_level_values(SBML_DFS.R_ID)
    )

    N_reactions_with_lost_requirements = len(reactions_with_lost_requirements)
    if N_reactions_with_lost_requirements > 0:
        logger.info(
            f"Removing {N_reactions_with_lost_requirements} reactions which have lost all required members"
        )

    underspecified_reactions = reactions_with_lost_defining_members.union(
        reactions_with_lost_requirements
    )

    return underspecified_reactions


def find_unused_entities(
    sbml_dfs_or_dict: Union[SBML_dfs, dict[str, pd.DataFrame]],
) -> dict[str, set[str]]:

    from napistu.sbml_dfs_core import SBML_dfs

    if isinstance(sbml_dfs_or_dict, SBML_dfs):
        d = sbml_dfs_or_dict.to_dict()
    else:
        d = sbml_dfs_or_dict

    EXPECTED_KEYS = {
        SBML_DFS.REACTION_SPECIES,
        SBML_DFS.REACTIONS,
        SBML_DFS.COMPARTMENTALIZED_SPECIES,
        SBML_DFS.SPECIES,
        SBML_DFS.COMPARTMENTS,
    }

    if set(d.keys()) != EXPECTED_KEYS:
        raise ValueError(f"sbml_dfs must contain the following keys: {EXPECTED_KEYS}")

    cleaned_entities = {}

    # cleanup reactions and compartmentalized species based on reaction_species
    defined_rxn_species_reactions = d[SBML_DFS.REACTION_SPECIES][SBML_DFS.R_ID].unique()
    defined_rxn_species_cspecies = d[SBML_DFS.REACTION_SPECIES][SBML_DFS.SC_ID].unique()
    cleaned_entities[SBML_DFS.REACTIONS] = (
        d[SBML_DFS.REACTIONS]
        .index[~d[SBML_DFS.REACTIONS].index.isin(defined_rxn_species_reactions)]
        .tolist()
    )
    cleaned_entities[SBML_DFS.COMPARTMENTALIZED_SPECIES] = (
        d[SBML_DFS.COMPARTMENTALIZED_SPECIES]
        .index[
            ~d[SBML_DFS.COMPARTMENTALIZED_SPECIES].index.isin(
                defined_rxn_species_cspecies
            )
        ]
        .tolist()
    )

    # cleanup species and compartments based on compartmentalized_species
    post_cleanup_cspecies = d[SBML_DFS.COMPARTMENTALIZED_SPECIES].loc[
        ~d[SBML_DFS.COMPARTMENTALIZED_SPECIES].index.isin(
            cleaned_entities[SBML_DFS.COMPARTMENTALIZED_SPECIES]
        )
    ]
    defined_cspecies_species = post_cleanup_cspecies[SBML_DFS.S_ID].unique()
    defined_cspecies_compartments = post_cleanup_cspecies[SBML_DFS.C_ID].unique()
    cleaned_entities[SBML_DFS.SPECIES] = (
        d[SBML_DFS.SPECIES]
        .index[~d[SBML_DFS.SPECIES].index.isin(defined_cspecies_species)]
        .tolist()
    )
    cleaned_entities[SBML_DFS.COMPARTMENTS] = (
        d[SBML_DFS.COMPARTMENTS]
        .index[~d[SBML_DFS.COMPARTMENTS].index.isin(defined_cspecies_compartments)]
        .tolist()
    )

    # summarize the cleanup
    non_empty_cleaned_entities = {
        k: v for k, v in cleaned_entities.items() if len(v) > 0
    }
    for k, v in non_empty_cleaned_entities.items():
        logger.info(f"Found {len(v)} unused {k} entities")

    return cleaned_entities


def filter_to_characteristic_species_ids(
    species_ids: pd.DataFrame,
    max_complex_size: int = 4,
    max_promiscuity: int = 20,
    defining_biological_qualifiers: list[str] = BQB_DEFINING_ATTRS,
) -> pd.DataFrame:
    """
    Filter to Characteristic Species IDs

    Remove identifiers corresponding to one component within a large protein
    complexes and non-characteristic annotations such as pubmed references and
    homologues.

        Parameters
        ----------
    species_ids: pd.DataFrame
        A table of identifiers produced by sdbml_dfs.get_identifiers("species")
    max_complex_size: int
        The largest size of a complex, where BQB_HAS_PART terms will be retained.
        In most cases, complexes are handled with specific formation and
        dissolutation reactions,but these identifiers will be pulled in when
        searching by identifiers or searching the identifiers associated with a
        species against an external resource such as Open Targets.
    max_promiscuity: int
        Maximum number of species where a single molecule can act as a
        BQB_HAS_PART component associated with a single identifier (and common ontology).
    defining_biological_qualifiers (list[str]):
        BQB codes which define distinct entities. Narrowly this would be BQB_IS, while more
        permissive settings would include homologs, different forms of the same gene.

    Returns:
    --------
    species_id: pd.DataFrame
        Input species filtered to characteristic identifiers

    """

    if not isinstance(species_ids, pd.DataFrame):
        raise TypeError(
            f"species_ids was a {type(species_ids)} but must be a pd.DataFrame"
        )

    if not isinstance(max_complex_size, int):
        raise TypeError(
            f"max_complex_size was a {type(max_complex_size)} but must be an int"
        )

    if not isinstance(max_promiscuity, int):
        raise TypeError(
            f"max_promiscuity was a {type(max_promiscuity)} but must be an int"
        )

    if not isinstance(defining_biological_qualifiers, list):
        raise TypeError(
            f"defining_biological_qualifiers was a {type(defining_biological_qualifiers)} but must be a list"
        )

    # primary annotations of a species
    bqb_is_species = species_ids.query("bqb in @defining_biological_qualifiers")

    # add components within modestly sized protein complexes
    # look at HAS_PART IDs
    bqb_has_parts_species = species_ids[species_ids[IDENTIFIERS.BQB] == BQB.HAS_PART]

    # number of species in a complex
    n_species_components = bqb_has_parts_species.value_counts(
        [IDENTIFIERS.ONTOLOGY, SBML_DFS.S_ID]
    )
    big_complex_sids = set(
        n_species_components[
            n_species_components > max_complex_size
        ].index.get_level_values(SBML_DFS.S_ID)
    )

    filtered_bqb_has_parts = _filter_promiscuous_components(
        bqb_has_parts_species, max_promiscuity
    )

    # drop species parts if there are many components
    filtered_bqb_has_parts = filtered_bqb_has_parts[
        ~filtered_bqb_has_parts[SBML_DFS.S_ID].isin(big_complex_sids)
    ]

    # combine primary identifiers and rare components
    characteristic_species_ids = pd.concat(
        [
            bqb_is_species,
            filtered_bqb_has_parts,
        ]
    )

    return characteristic_species_ids


def force_edgelist_consistency(
    interaction_edgelist: pd.DataFrame,
    species_df: pd.DataFrame,
    compartments_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Force the edgelist to be consistent with the species and compartments dataframes.

    Parameters
    ----------
    interaction_edgelist : pd.DataFrame
        The interaction edgelist to force consistency with
    species_df : pd.DataFrame
        The species dataframe to force consistency with
    compartments_df : pd.DataFrame
        The compartments dataframe to force consistency with

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the filtered interaction edgelist, species dataframe, and compartments dataframe
    """

    # Check what's missing (warnings only)
    _validate_edgelist_consistency(
        interaction_edgelist, species_df, compartments_df, raise_on_missing=False
    )

    # Filter to valid species
    available_species = set(species_df[SBML_DFS.S_NAME])
    valid_mask = interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM].isin(
        available_species
    ) & interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM].isin(
        available_species
    )

    filtered_interactions = interaction_edgelist[valid_mask]
    if filtered_interactions.shape[0] != interaction_edgelist.shape[0]:
        logger.warning(
            f"Filtered {interaction_edgelist.shape[0] - filtered_interactions.shape[0]} interactions out of {interaction_edgelist.shape[0]} interactions due to missing species"
        )

    # Filter species to used ones
    used_species = set(
        filtered_interactions[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM]
    ) | set(filtered_interactions[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM])
    filtered_species = species_df[species_df[SBML_DFS.S_NAME].isin(used_species)]

    if filtered_species.shape[0] != species_df.shape[0]:
        logger.warning(
            f"Filtered {species_df.shape[0] - filtered_species.shape[0]} species out of {species_df.shape[0]} species due to not being used in the interaction edgelist"
        )

    return filtered_interactions, filtered_species, compartments_df


def format_sbml_dfs_summary(data):
    """Format model data into a clean summary table for Jupyter display"""

    # Calculate species percentages
    total_species = data["n_entity_types"][SBML_DFS.SPECIES]
    total_compartments = data["n_entity_types"][SBML_DFS.COMPARTMENTS]
    total_cspecies = data["n_entity_types"][SBML_DFS.COMPARTMENTALIZED_SPECIES]
    total_reactions = data["n_entity_types"][SBML_DFS.REACTIONS]
    total_reaction_species = data["n_entity_types"][SBML_DFS.REACTION_SPECIES]
    species_data = data["n_species_per_type"]

    # Build the summary data
    summary_data = [["Species", f"{total_species:,}"]]

    # Add species breakdown sorted by count (descending)
    for species_type, count in sorted(
        species_data.items(), key=lambda x: x[1], reverse=True
    ):
        pct = count / total_species * 100
        summary_data.append(
            [
                f"- {utils.safe_capitalize(SPECIES_TYPE_PLURAL[species_type])}",
                f"{count:,} ({pct:.1f}%)",
            ]
        )

    # Add spacing and compartments section
    summary_data.extend(
        [
            ["", ""],  # Empty row for spacing
            ["Compartments", f"{total_compartments:,}"],
        ]
    )

    # Sort compartments by species count (descending) and add compartment breakdown
    sorted_compartments = sorted(
        data["dict_n_species_per_compartment"],
        key=lambda x: x["n_species"],
        reverse=True,
    )

    # Show top 10 compartments, bundle rest into "other"
    top_compartments = sorted_compartments[:10]
    other_compartments = sorted_compartments[10:]

    for comp in top_compartments:
        comp_pct = comp["n_species"] / total_cspecies * 100
        summary_data.append(
            [f"- {comp['c_name']}", f"{comp['n_species']:,} ({comp_pct:.1f}%)"]
        )

    # Add "other" category if there are more than 10 compartments
    if other_compartments:
        other_cspecies_count = sum(comp["n_species"] for comp in other_compartments)
        other_pct = other_cspecies_count / total_cspecies * 100
        n_other_compartments = len(other_compartments)
        summary_data.append(
            [
                f"- Other ({n_other_compartments} compartments)",
                f"{other_cspecies_count:,} ({other_pct:.1f}%)",
            ]
        )

    summary_data.extend(
        [
            ["", ""],  # Empty row for spacing
            ["Compartmentalized Species", f"{total_cspecies:,}"],
            ["Reactions", f"{total_reactions:,}"],
            ["Reaction Species", f"{total_reaction_species:,}"],
        ]
    )

    # Create DataFrame and display
    df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

    return df


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
        logger.warning(
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


def species_type_types(
    x,
    ontology_to_species_type: dict = ONTOLOGY_TO_SPECIES_TYPE,
    prioritized_species_types: set[str] = PRIORITIZED_SPECIES_TYPES,
) -> str:
    """
    Assign a high-level molecule type to a molecular species

    Parameters
    ----------
    x : identifiers.Identifiers
        The identifiers object to assign a species type to
    ontology_to_species_type : dict
        The mapping of ontologies to species types
    prioritized_species_types : set[str]
        The set of prioritized species types

    Returns
    -------
    str
        The high-level molecule type of the species

    Examples
    --------
    >>> identifiers = identifiers.Identifiers([{'ontology': 'CHEBI', 'identifier': '123456', 'bqb': 'BQB.IS'}])
    >>> species_type_types(identifiers)
    'metabolite'
    """

    if isinstance(x, identifiers.Identifiers):

        # Check for HAS_PART first (indicates complex)
        bqbs = x.get_all_bqbs()
        if BQB.HAS_PART in bqbs:
            return SPECIES_TYPES.COMPLEX

        ontologies = x.get_all_ontologies([BQB.IS, BQB.IS_ENCODED_BY, BQB.ENCODES])
        if len(ontologies) == 0:
            return SPECIES_TYPES.UNKNOWN

        # check for prioritized ontologies
        ontologies_w_species_types = ontologies & ontology_to_species_type.keys()

        # Then map to species types
        species_types = {
            ontology_to_species_type[ont] for ont in ontologies_w_species_types
        }

        prioritized_types = species_types & prioritized_species_types
        if len(prioritized_types) == 1:
            return prioritized_types.pop()
        elif len(prioritized_types) > 1:
            return SPECIES_TYPES.UNKNOWN

        if len(species_types) == 0:
            # none of the defined ontologies are associated with a species type
            return SPECIES_TYPES.OTHER
        elif len(species_types) == 1:
            return species_types.pop()
        elif len(species_types) > 1:
            return SPECIES_TYPES.UNKNOWN

    else:
        logger.warning(
            f"Invalid input type: {type(x)}; returning {SPECIES_TYPES.UNKNOWN}"
        )
        return SPECIES_TYPES.UNKNOWN


def stub_compartments(
    stubbed_compartment: str = GENERIC_COMPARTMENT,
    with_source: bool = False,
) -> pd.DataFrame:
    """Stub Compartments

    Create a compartments table with only a single compartment

    Parameters
    ----------
    stubbed_compartment : str
        the name of a compartment which should match the keys in
        ingestion.constants.VALID_COMPARTMENTS and ingestion.constants.COMPARTMENTS_GO_TERMS
    with_source : bool
        whether to include a source column in the compartments dataframe. Defaults to False which is the standard approach for edgelist creation. True will create a valid compartments table with a c_Source column.

    Returns
    -------
    compartments_df : pd.DataFrame
        compartments dataframe
    """

    # import Source here to avoid circular import
    from napistu.source import Source

    if stubbed_compartment not in VALID_COMPARTMENTS:
        raise ValueError(
            f"{stubbed_compartment} is not defined in ingestion.constants.VALID_COMPARTMENTS"
        )

    if stubbed_compartment not in COMPARTMENTS_GO_TERMS.keys():
        raise ValueError(
            f"{stubbed_compartment} is not defined in ingestion.constants.COMPARTMENTS_GO_TERMS"
        )

    stubbed_compartment_id = COMPARTMENTS_GO_TERMS[stubbed_compartment]

    formatted_uri = identifiers.format_uri(
        uri=identifiers.create_uri_url(
            ontology=ONTOLOGIES.GO,
            identifier=stubbed_compartment_id,
        ),
        bqb=BQB.IS,
    )

    compartments_df = pd.DataFrame(
        {
            SBML_DFS.C_NAME: [stubbed_compartment],
            SBML_DFS.C_IDENTIFIERS: [identifiers.Identifiers([formatted_uri])],
        }
    )
    compartments_df.index = id_formatter([0], SBML_DFS.C_ID)  # type: ignore
    compartments_df.index.name = SBML_DFS.C_ID

    if with_source:
        compartments_df[SBML_DFS.C_SOURCE] = [Source.empty()]

    return compartments_df


def unnest_identifiers(id_table: pd.DataFrame, id_var: str) -> pd.DataFrame:
    """
    Unnest Identifiers

    Take a pd.DataFrame containing an array of Identifiers and
    return one-row per identifier.

    Parameters
    ----------
    id_table : pd.DataFrame
        Table containing Identifiers objects
    id_var : str
        Column name containing Identifiers objects

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per identifier, MultiIndex with original index + entry
    """
    # Validate inputs
    utils.match_pd_vars(id_table, {id_var}).assert_present()

    N_invalid_ids = sum(id_table[id_var].isna())
    if N_invalid_ids != 0:
        utils.show("Rows with missing identifiers:")
        utils.show(id_table.loc[id_table[id_var].isna(), id_var])
        raise ValueError(
            f'{N_invalid_ids} entries in "id_table" were missing identifiers'
        )

    # Build dict mapping each index to its identifiers DataFrame
    identifier_dict = {}
    for idx, identifiers_obj in id_table[id_var].items():
        if not identifiers_obj.df.empty:
            identifier_dict[idx] = identifiers_obj.df

    # If no valid identifiers, return empty DataFrame
    if not identifier_dict:
        return pd.DataFrame()

    # Use pd.concat with keys to create MultiIndex directly
    result = pd.concat(
        identifier_dict, names=id_table.index.names + [SOURCE_SPEC.ENTRY]
    )

    return result


def validate_sbml_dfs_table(table_data: pd.DataFrame, table_name: str) -> None:
    """
    Validate a standalone table against the SBML_dfs schema.

    This function validates a table against the schema defined in SBML_DFS_SCHEMA,
    without requiring an SBML_dfs object. Useful for validating tables before
    creating an SBML_dfs object.

    Parameters
    ----------
    table_data : pd.DataFrame
        The table to validate
    table_name : str
        Name of the table in the SBML_dfs schema

    Raises
    ------
    ValueError
    If table_name is not in schema or validation fails
    """
    if table_name not in SBML_DFS_SCHEMA.SCHEMA:
        raise ValueError(
            f"{table_name} is not a valid table name in SBML_DFS_SCHEMA. "
            f"Valid tables are: {', '.join(SBML_DFS_SCHEMA.SCHEMA.keys())}"
        )

    table_schema = SBML_DFS_SCHEMA.SCHEMA[table_name]
    _perform_sbml_dfs_table_validation(table_data, table_schema, table_name)


# =============================================================================
# PRIVATE FUNCTIONS (ALPHABETICAL ORDER)
# =============================================================================


def _add_edgelist_defaults(
    interaction_edgelist: pd.DataFrame,
    edgelist_defaults: Optional[dict[str, Any]] = INTERACTION_EDGELIST_DEFAULTS,
) -> pd.DataFrame:
    """
    Add default values to the interaction edgelist

    Parameters
    ----------
    interaction_edgelist : pd.DataFrame
        The interaction edgelist to add defaults to
    edgelist_defaults : dict[str, Any]
        The defaults to add to the interaction edgelist

    Returns
    """

    interaction_edgelist_w_defaults = interaction_edgelist.copy()

    missing_vars_with_defaults = INTERACTION_EDGELIST_OPTIONAL_VARS.difference(
        interaction_edgelist_w_defaults.columns
    )

    if len(missing_vars_with_defaults) > 0:

        logger.debug(
            f"Adding default values to interaction edgelist for {len(missing_vars_with_defaults)} variables: {missing_vars_with_defaults}"
        )

        vars_missing_defaults = INTERACTION_EDGELIST_OPTIONAL_VARS.difference(
            edgelist_defaults.keys()
        )
        if len(vars_missing_defaults) > 0:
            # replace with global defaults
            for var in vars_missing_defaults:
                logger.debug(
                    f"Adding default value for {var} from INTERACTION_EDGELIST_DEFAULTS: {INTERACTION_EDGELIST_DEFAULTS[var]}"
                )
                edgelist_defaults[var] = INTERACTION_EDGELIST_DEFAULTS[var]

        for var_with_default in missing_vars_with_defaults:
            # pull out the default value for the variable
            default_value = edgelist_defaults[var_with_default]
            # add the default value to the interaction edgelist
            interaction_edgelist_w_defaults[var_with_default] = default_value

    # replace missing values with defaults
    for var_with_default in edgelist_defaults:
        na_values = interaction_edgelist_w_defaults[var_with_default].isna()

        if len(na_values) > 0:
            default_value = edgelist_defaults[var_with_default]
            logger.info(
                f"Replacing {len(na_values)} missing values with default value for {var_with_default}: {default_value}"
            )
            interaction_edgelist_w_defaults.loc[na_values, var_with_default] = (
                default_value
            )

    # are there columns which respect defaults and also have NaNs?
    columns_with_nans = interaction_edgelist_w_defaults.columns[
        interaction_edgelist_w_defaults.isna().any()
    ]
    columns_with_nans_and_respecting_defaults = set(columns_with_nans) & set(
        INTERACTION_EDGELIST_OPTIONAL_VARS
    )
    if len(columns_with_nans_and_respecting_defaults) > 0:
        raise ValueError(
            f"The following columns have NaNs and respect defaults: {columns_with_nans_and_respecting_defaults}. Either address the missing values or add a default value to the edgelist_defaults."
        )

    return interaction_edgelist_w_defaults


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
            interaction_edgelist[
                [
                    INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                    INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM,
                ]
            ].rename(
                {
                    INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: SBML_DFS.S_NAME,
                    INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: SBML_DFS.C_NAME,
                },
                axis=1,
            ),
            interaction_edgelist[
                [
                    INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                    INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM,
                ]
            ].rename(
                {
                    INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: SBML_DFS.S_NAME,
                    INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: SBML_DFS.C_NAME,
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
    processed_species,
    processed_compartments,
    interaction_source,
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
    processed_species : pd.DataFrame
        Processed species data with IDs
    processed_compartments : pd.DataFrame
        Processed compartments data with IDs
    interaction_source : source.Source
        Source object for reactions
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
        .merge(processed_species[SBML_DFS.S_NAME].reset_index())
        .merge(processed_compartments[SBML_DFS.C_NAME].reset_index())
    )

    interaction_w_cspecies = interaction_edgelist.merge(
        comp_species_w_names[[SBML_DFS.SC_ID, SBML_DFS.S_NAME, SBML_DFS.C_NAME]].rename(
            {
                SBML_DFS.SC_ID: "sc_id_up",
                SBML_DFS.S_NAME: INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                SBML_DFS.C_NAME: INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM,
            },
            axis=1,
        ),
        how="left",
    ).merge(
        comp_species_w_names[[SBML_DFS.SC_ID, SBML_DFS.S_NAME, SBML_DFS.C_NAME]].rename(
            {
                SBML_DFS.SC_ID: "sc_id_down",
                SBML_DFS.S_NAME: INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                SBML_DFS.C_NAME: INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM,
            },
            axis=1,
        ),
        how="left",
    )

    expected_interaction_w_cspecies_columns = (
        set(INTERACTION_EDGELIST_EXPECTED_VARS)
        - {
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
        }
        | {"sc_id_up", "sc_id_down"}
        | set(extra_reactions_columns)
    )
    missing_interaction_w_cspecies_columns = (
        expected_interaction_w_cspecies_columns - set(interaction_w_cspecies.columns)
    )
    if len(missing_interaction_w_cspecies_columns) > 0:
        raise ValueError(
            f"The following columns are missing from interaction_w_cspecies: {missing_interaction_w_cspecies_columns}"
        )

    interaction_w_cspecies = interaction_w_cspecies[
        list(expected_interaction_w_cspecies_columns)
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

    reactions_columns = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.REACTIONS][SCHEMA_DEFS.VARS]
    reactions_df = interactions_copy.set_index(SBML_DFS.R_ID)[
        reactions_columns + extra_reactions_columns
    ]

    # Separate extra data
    reactions_data = reactions_df[extra_reactions_columns]
    reactions_df = reactions_df[reactions_columns]

    # Create reaction species relationships - NOW r_id exists
    reaction_species_df = pd.concat(
        [
            # upstream
            interaction_w_cspecies[
                [
                    SBML_DFS.R_ID,
                    "sc_id_up",
                    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM,
                    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM,
                ]
            ].rename(
                {
                    "sc_id_up": SBML_DFS.SC_ID,
                    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: SBML_DFS.SBO_TERM,
                    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM: SBML_DFS.STOICHIOMETRY,
                },
                axis=1,
            ),
            # downstream
            interaction_w_cspecies[
                [
                    SBML_DFS.R_ID,
                    "sc_id_down",
                    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM,
                    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM,
                ]
            ].rename(
                {
                    "sc_id_down": SBML_DFS.SC_ID,
                    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: SBML_DFS.SBO_TERM,
                    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM: SBML_DFS.STOICHIOMETRY,
                },
                axis=1,
            ),
        ]
    )

    # switch from sbo_term_name to sbo_term
    reaction_species_df[SBML_DFS.SBO_TERM] = reaction_species_df[SBML_DFS.SBO_TERM].map(
        MINI_SBO_FROM_NAME
    )
    # add a unique id to each reaction species
    reaction_species_df[SBML_DFS.RSC_ID] = id_formatter(
        range(reaction_species_df.shape[0]), SBML_DFS.RSC_ID
    )

    reaction_species_df = reaction_species_df.set_index(SBML_DFS.RSC_ID)

    return reactions_df, reaction_species_df, reactions_data


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

        logger.info(
            f"Saving {len(extra_reactions_columns)} extra reaction columns as reaction_data: {extra_reactions_columns}"
        )

    if keep_species_data is not False:
        extra_species_columns = [
            c
            for c in species_df.columns
            if c not in {SBML_DFS.S_NAME, SBML_DFS.S_IDENTIFIERS}
        ]

        logger.info(
            f"Saving {len(extra_species_columns)} extra species columns as species_data: {extra_species_columns}"
        )

    return {
        SBML_DFS.REACTIONS: extra_reactions_columns,
        SBML_DFS.SPECIES: extra_species_columns,
    }


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

    # sbo term names should be in controlled vocabulary (excluding NaN values which will be filled by defaults)
    upstream_sbo_terms = set(
        interaction_edgelist[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM].dropna()
    )
    downstream_sbo_terms = set(
        interaction_edgelist[
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM
        ].dropna()
    )
    sbo_term_names = upstream_sbo_terms | downstream_sbo_terms
    invalid_sbo_term_names = sbo_term_names - set(MINI_SBO_FROM_NAME.keys())
    if len(invalid_sbo_term_names) > 0:
        raise ValueError(
            f"The following SBO term names are not in the controlled vocabulary: {invalid_sbo_term_names}"
        )

    # Check for non-null values in all required columns for each table
    _validate_non_null_values(
        compartments_df, compartments_df_expected_vars, "compartments_df"
    )
    _validate_non_null_values(species_df, species_df_expected_vars, "species_df")
    _validate_non_null_values(
        interaction_edgelist, INTERACTION_EDGELIST_EXPECTED_VARS, "interaction_edgelist"
    )

    # check for extra or missing species and compartments
    defined_interactors = set(
        interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM]
    ) | set(interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM])
    defined_interaction_compartments = set(
        interaction_edgelist[INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM]
    ) | set(interaction_edgelist[INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM])
    species_df_names = set(species_df[SBML_DFS.S_NAME])
    compartments_df_names = set(compartments_df[SBML_DFS.C_NAME])

    invalid_references = []

    extra_species = species_df_names - defined_interactors
    if len(extra_species) > 0:
        invalid_references.append(
            f"{len(extra_species)} species are defined in the interaction edgelist but not in the species_df: {extra_species}"
        )

    extra_compartments = compartments_df_names - defined_interaction_compartments
    if len(extra_compartments) > 0:
        invalid_references.append(
            f"{len(extra_compartments)} compartments are defined in the interaction edgelist but not in the compartments_df: {extra_compartments}"
        )

    missing_species = defined_interactors - species_df_names
    if len(missing_species) > 0:
        invalid_references.append(
            f"{len(missing_species)} species are defined in the interaction edgelist but not in the species_df: {missing_species}"
        )

    missing_compartments = defined_interaction_compartments - compartments_df_names
    if len(missing_compartments) > 0:
        invalid_references.append(
            f"{len(missing_compartments)} compartments are defined in the interaction edgelist but not in the compartments_df: {missing_compartments}"
        )

    if len(invalid_references) > 0:
        raise ValueError(f"Invalid references: {invalid_references}")

    return None


def _filter_promiscuous_components(
    bqb_has_parts_species: pd.DataFrame, max_promiscuity: int
) -> pd.DataFrame:

    # number of complexes a species is part of
    n_complexes_involvedin = bqb_has_parts_species.value_counts(
        [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]
    )
    promiscuous_component_identifiers_index = n_complexes_involvedin[
        n_complexes_involvedin > max_promiscuity
    ].index
    promiscuous_component_identifiers = pd.Series(
        data=[True] * len(promiscuous_component_identifiers_index),
        index=promiscuous_component_identifiers_index,
        name="is_shared_component",
        dtype=bool,
    )

    if len(promiscuous_component_identifiers) == 0:
        return bqb_has_parts_species

    filtered_bqb_has_parts = bqb_has_parts_species.merge(
        promiscuous_component_identifiers,
        left_on=[IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER],
        right_index=True,
        how="left",
    )

    filtered_bqb_has_parts["is_shared_component"] = (
        filtered_bqb_has_parts["is_shared_component"].astype("boolean").fillna(False)
    )
    # drop identifiers shared as components across many species
    filtered_bqb_has_parts = filtered_bqb_has_parts[
        ~filtered_bqb_has_parts["is_shared_component"]
    ].drop(["is_shared_component"], axis=1)

    return filtered_bqb_has_parts


def _filter_to_pathways(df: pd.DataFrame, pathways: list[str]) -> pd.DataFrame:
    """
    Filter a table to only include pathways in the list.
    """

    pathway_mask = df[SOURCE_SPEC.PATHWAY_ID].isin(pathways)
    return df.loc[pathway_mask]


def _find_underspecified_reactions(
    reaction_species_w_roles: pd.DataFrame,
) -> pd.DataFrame:

    # check that both sbo_role and "new" are present
    if SBO_ROLES_DEFS.SBO_ROLE not in reaction_species_w_roles.columns:
        raise ValueError(
            "The sbo_role column is not present in the reaction_species_w_roles table. Please call add_sbo_role() first."
        )
    if "new" not in reaction_species_w_roles.columns:
        raise ValueError(
            "The new column is not present in the reaction_species_w_roles table. This should indicate what cspecies would be preserved in the reaction should it be preserved."
        )
    # check that new is a boolean column
    if reaction_species_w_roles["new"].dtype != bool:
        raise ValueError(
            "The new column is not a boolean column. Please ensure that the new column is a boolean column. This should indicate what cspecies would be preserved in the reaction should it be preserved."
        )

    reactions_with_lost_defining_members = set(
        reaction_species_w_roles.query("~new")
        .query("sbo_role == 'DEFINING'")[SBML_DFS.R_ID]
        .tolist()
    )

    N_reactions_with_lost_defining_members = len(reactions_with_lost_defining_members)
    if N_reactions_with_lost_defining_members > 0:
        logger.info(
            f"Removing {N_reactions_with_lost_defining_members} reactions which have lost at least one defining species"
        )

    # find the cases where all "new" values for a given (r_id, sbo_term) are False
    reactions_with_lost_requirements = set(
        reaction_species_w_roles
        # drop already filtered reactions
        .query("r_id not in @reactions_with_lost_defining_members")
        .query("sbo_role == 'REQUIRED'")
        # which entries which have some required attribute have all False values for that attribute
        .groupby([SBML_DFS.R_ID, SBML_DFS.SBO_TERM])
        .agg({"new": "any"})
        .query("new == False")
        .index.get_level_values(SBML_DFS.R_ID)
    )

    N_reactions_with_lost_requirements = len(reactions_with_lost_requirements)
    if N_reactions_with_lost_requirements > 0:
        logger.info(
            f"Removing {N_reactions_with_lost_requirements} reactions which have lost all required members"
        )

    underspecified_reactions = reactions_with_lost_defining_members.union(
        reactions_with_lost_requirements
    )

    return underspecified_reactions


def _get_interaction_symbol(sbo_term_or_name: str) -> str:

    if sbo_term_or_name in MINI_SBO_TO_NAME.keys():
        sbo_term_name = MINI_SBO_TO_NAME[sbo_term_or_name]
    else:
        sbo_term_name = sbo_term_or_name

    if sbo_term_name not in MINI_SBO_NAME_TO_POLARITY.keys():
        raise ValueError(f"Invalid SBO term: {sbo_term_or_name}")

    return POLARITY_TO_SYMBOL[MINI_SBO_NAME_TO_POLARITY[sbo_term_name]]


def _id_dict_to_df(ids):
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


def _name_interaction(
    upstream_name: str,
    downstream_name: str,
    sbo_term_upstream: Optional[str] = SBOTERM_NAMES.INTERACTOR,
):
    """
    Name an interaction

    Parameters
    ----------
    upstream_name : str
        The name of the upstream species
    downstream_name : str
        The name of the downstream species
    sbo_term_upstream : str, optional
        The SBO term of the upstream species. If not provided, the interaction will be named "interactor"

    Returns
    -------
    str
        The name of the interaction
    """

    interaction_symbol = _get_interaction_symbol(sbo_term_upstream)
    return f"{upstream_name} {interaction_symbol} {downstream_name}"


def _perform_sbml_dfs_table_validation(
    table_data: pd.DataFrame,
    table_schema: dict,
    table_name: str,
) -> None:
    """
    Core validation logic for SBML_dfs tables.

    This function performs the actual validation checks for any table against its schema,
    regardless of whether it's part of an SBML_dfs object or standalone.

        Parameters
        ----------
    table_data : pd.DataFrame
        The table data to validate
    table_schema : dict
        Schema definition for the table
    table_name : str
        Name of the table (for error messages)

        Raises
        ------
        ValueError
        If the table does not conform to its schema:
        - Not a DataFrame
        - Wrong index name
        - Duplicate primary keys
        - Missing required variables
        - Empty table
    """
    if not isinstance(table_data, pd.DataFrame):
        raise ValueError(
            f"{table_name} must be a pd.DataFrame, but was a {type(table_data)}"
        )

    # check index
    expected_index_name = table_schema[SCHEMA_DEFS.PK]
    if table_data.index.name != expected_index_name:
        raise ValueError(
            f"the index name for {table_name} was not the pk: {expected_index_name}"
        )

    # check that all entries in the index are unique
    if len(set(table_data.index.tolist())) != table_data.shape[0]:
        duplicated_pks = table_data.index.value_counts()
        duplicated_pks = duplicated_pks[duplicated_pks > 1]

        example_duplicates = duplicated_pks.index[0 : min(duplicated_pks.shape[0], 5)]
        raise ValueError(
            f"{duplicated_pks.shape[0]} primary keys were duplicated "
            f"including {', '.join(example_duplicates)}"
        )

    # check variables
    expected_vars = set(table_schema[SCHEMA_DEFS.VARS])
    table_vars = set(list(table_data.columns))

    extra_vars = table_vars.difference(expected_vars)
    if len(extra_vars) != 0:
        logger.debug(
            f"{len(extra_vars)} extra variables were found for {table_name}: "
            f"{', '.join(extra_vars)}"
        )

    missing_vars = expected_vars.difference(table_vars)
    if len(missing_vars) != 0:
        raise ValueError(
            f"Missing {len(missing_vars)} required variables for {table_name}: "
            f"{', '.join(missing_vars)}"
        )

    # check for empty table
    if table_data.shape[0] == 0:
        raise ValueError(f"{table_name} contained no entries")


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


def _select_priority_pathway_sources(
    source_table: pd.DataFrame,
    priority_pathways: Optional[list[str]] = DEFAULT_PRIORITIZED_PATHWAYS,
) -> pd.DataFrame:
    """
    Filter the source table to only include pathways in the list. If 0 or 1 priority pathways are found, return the source table.

    Parameters
    ----------
    source_table : pd.DataFrame
        The source table to filter
    priority_pathways : Optional[list[str]], default DEFAULT_PRIORITIZED_PATHWAYS
        The list of pathways to filter to. If None, returns source_table with no filtering or warning.
        If fewer than 2 pathways are found in the source table, returns the full source table with a warning.

    Returns
    -------
    pd.DataFrame
        The filtered source table. If priority_pathways is None, returns the original source_table.
        If fewer than 2 priority pathways are found, returns the full source_table with a warning.
    """

    # If priority_pathways is explicitly None, return source_table without warning
    if priority_pathways is None:
        return source_table

    # filter to pathways of interest
    priority_source_table = _filter_to_pathways(source_table, priority_pathways)
    n_priority_pathways = priority_source_table[SOURCE_SPEC.PATHWAY_ID].nunique()

    if n_priority_pathways > 1:
        return priority_source_table
    else:
        logger.warning(
            "<2 priority pathways found, using all pathways. Set priority_pathways as None explicitly to remove this warning."
        )
        return source_table


def _summarize_ontology_cooccurrence(
    df: pd.DataFrame, stratify_by_bqb: bool = True, allow_col_multindex: bool = False
) -> pd.DataFrame:
    """
    Create a cooccurrence matrix of ontologies based entities sharing the same ontology.

    This can be used to identify ontologies which are associated with the same types of entities.

    Parameters
    ----------
    df (pd.DataFrame)
        a table generated using `sbml_dfs.get_sources`
    stratify_by_bqb (bool)
        whether to stratify by bqb
    allow_col_multindex (bool)
        whether to allow the column multindex

    Returns
    -------
    pd.DataFrame
        Square matrix with pathways as both rows and columns
    """

    # Get binarized occurrence matrix directly
    entity_ontology_matrix = _summarize_ontology_occurrence(
        df, stratify_by_bqb, allow_col_multindex, binarize=True
    )

    # Calculate co-occurrence matrix: ontologies × ontologies
    # This gives us the number of species shared between each pair of ontologies
    cooccurrences = entity_ontology_matrix.T @ entity_ontology_matrix

    return cooccurrences


def _summarize_ontology_occurrence(
    df: pd.DataFrame,
    stratify_by_bqb: bool = True,
    allow_col_multindex: bool = False,
    binarize: bool = False,
) -> pd.DataFrame:
    """
    Summarize the types of identifiers associated with each entity.

    Parameters
    ----------
    df (pd.DataFrame)
        a table generated using `sbml_dfs.get_identifiers` or `sbml_dfs.get_characteristic_species_ids`
    stratify_by_bqb (bool)
        whether to stratify by bqb
    allow_col_multindex (bool)
        whether to allow the column multindex
    binarize: bool
        whether to convert the result to binary values (0 vs 1+)

    Returns
    -------
    pd.DataFrame
        a table with entities as rows and ontologies as columns
    """

    entity_type = utils.infer_entity_type(df)
    pk = SBML_DFS_SCHEMA.SCHEMA[entity_type][SCHEMA_DEFS.PK]

    required_vars = {pk, SOURCE_SPEC.ENTRY} | IDENTIFIERS_REQUIRED_VARS
    utils.match_pd_vars(
        df, req_vars=set(required_vars), allow_series=False
    ).assert_present()

    if stratify_by_bqb:
        if allow_col_multindex:
            pivot_cols = [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.BQB]
        else:
            # combine bqb and ontology into a single column
            df["bqb_ontology"] = (
                df[IDENTIFIERS.BQB].astype(str)
                + "::"
                + df[IDENTIFIERS.ONTOLOGY].astype(str)
            )
            pivot_cols = ["bqb_ontology"]
    else:
        pivot_cols = [IDENTIFIERS.ONTOLOGY]

    result = df.pivot_table(
        index=pk,
        columns=pivot_cols,
        values=SOURCE_SPEC.ENTRY,  # Using 'entry' column as indicator
        fill_value=0,
        aggfunc="count",  # Count occurrences
    )

    if binarize:
        # Convert to binary (1 if entity has ontology, 0 otherwise)
        result = (result > 0).astype(int)

    return result


def _validate_edgelist_consistency(
    interaction_edgelist: pd.DataFrame,
    species_df: pd.DataFrame,
    compartments_df: pd.DataFrame,
    raise_on_missing: bool = True,
) -> None:
    """
    Check for missing entity references, optionally raising or warning.

    This function is used to validate the consistency of the interaction edgelist, species_df, and compartments_df.

    Parameters
    ----------
    interaction_edgelist : pd.DataFrame
        The interaction edgelist to validate
    species_df : pd.DataFrame
        The species dataframe to validate
    compartments_df : pd.DataFrame
        The compartments dataframe to validate
    raise_on_missing : bool, optional
        Whether to raise an error if missing entities are found

    Returns
    -------
    None
    """
    # Get referenced names
    upstream_species = set(
        interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM]
    )
    downstream_species = set(
        interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM]
    )
    edgelist_species = upstream_species | downstream_species

    upstream_compartments = set(
        interaction_edgelist[INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM]
    )
    downstream_compartments = set(
        interaction_edgelist[INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM]
    )
    edgelist_compartments = upstream_compartments | downstream_compartments

    # Get available names
    available_species = set(species_df[SBML_DFS.S_NAME])
    available_compartments = set(compartments_df[SBML_DFS.C_NAME])

    # Find missing
    missing_species = edgelist_species - available_species
    missing_compartments = edgelist_compartments - available_compartments

    # Handle missing compartments - always raise
    if missing_compartments:
        raise ValueError(f"Missing compartments: {missing_compartments}")

    # Handle missing species
    if missing_species:
        message = f"{len(missing_species)} species in edgelist but not in species_df: {missing_species}"
        if raise_on_missing:
            raise ValueError(f'Invalid references: ["{message}"]')
        else:
            logger.warning(message)

    return None


def _summarize_source_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a cooccurrence matrix of pathways based on the presence of entities in pathways.

    Parameters
    ----------
    df (pd.DataFrame)
        a table generated using `sbml_dfs.get_sources`

    Returns
    -------
    pd.DataFrame
        Square matrix with pathways as both rows and columns
    """

    # Get binarized occurrence matrix directly
    entity_source_matrix = _summarize_source_occurrence(df, binarize=True)

    # Calculate co-occurrence matrix: pathways × pathways
    # This gives us the number of compounds shared between each pair of pathways
    cooccurrences = entity_source_matrix.T @ entity_source_matrix

    return cooccurrences


def _summarize_source_occurrence(
    df: pd.DataFrame, binarize: bool = False
) -> pd.DataFrame:
    """
    Summarize the occurrence of entities in pathways.

    Parameters
    ----------
    df (pd.DataFrame)
        a table generated using `sbml_dfs.get_sources`
    binarize: bool
        whether to convert the result to binary values (0 vs 1+)

    Returns
    -------
    pd.DataFrame
        a table with entities as rows and pathways as columns

    """

    entity_type = utils.infer_entity_type(df)
    pk = SBML_DFS_SCHEMA.SCHEMA[entity_type][SCHEMA_DEFS.PK]

    expected_multindex = [pk, SOURCE_SPEC.ENTRY]
    if expected_multindex != df.index.names:
        raise ValueError(
            f"Expected multindex {expected_multindex} but got {df.index.names}. `df` should be generated using `sbml_dfs.get_sources`"
        )

    # Create a binary matrix: compounds × pathways
    result = df.reset_index().pivot_table(
        index=pk,
        columns=SOURCE_SPEC.PATHWAY_ID,
        values=SOURCE_SPEC.ENTRY,  # Using 'entry' column as indicator
        fill_value=0,
        aggfunc="count",  # Count occurrences
    )

    if binarize:
        # Convert to binary (1 if entity is in pathway, 0 otherwise)
        result = (result > 0).astype(int)

    return result


def _validate_non_null_values(
    df: pd.DataFrame, expected_columns: set, table_name: str
) -> None:
    """
    Validate that all required columns in a DataFrame have non-null values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate
    expected_columns : set
        Set of column names that should have non-null values
    table_name : str
        Name of the table for error messages

    Raises
    ------
    ValueError
        If any required column contains null values
    """
    for column in expected_columns:
        if column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                raise ValueError(
                    f"Column '{column}' in {table_name} contains {null_count} null values. "
                    f"All required columns must have non-null values."
                )


def _validate_matching_data(data_table: pd.DataFrame, ref_table: pd.DataFrame):
    """Validates a table against a reference

    This check if the table has the same index, no duplicates in the index
    and that all values in the index are in the reference table.

    Args:
        data_table (pd.DataFrame): a table with data that should
            match the reference
        ref_table (pd.DataFrame): a reference table

    Raises:
        ValueError: not same index name
        ValueError: index contains duplicates
        ValueError: index not subset of index of reactions table
    """
    ref_index_name = ref_table.index.name
    if data_table.index.name != ref_index_name:
        raise ValueError(
            "the index name for reaction data table was not"
            f" {ref_index_name}: {data_table.index.name}"
        )
    ids = data_table.index
    if any(ids.duplicated()):
        raise ValueError(
            "the index for reaction data table " "contained duplicate values"
        )
    if not all(ids.isin(ref_table.index)):
        raise ValueError(
            "the index for reaction data table contained values"
            " not found in the reactions table"
        )
    if not isinstance(data_table, pd.DataFrame):
        raise TypeError(
            f"The data table was type {type(data_table).__name__}"
            " but must be a pd.DataFrame"
        )


def _validate_sbo_values(sbo_series: pd.Series, validate: str = "names") -> None:
    """
    Validate SBO terms or names

    Parameters
    ----------
    sbo_series : pd.Series
        The SBO terms or names to validate.
    validate : str, optional
        Whether the values are SBO terms ("terms") or names ("names", default).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the validation type is invalid.
    TypeError
        If the invalid_counts is not a pandas DataFrame.
    ValueError
        If some reaction species have unusable SBO terms.
    """

    if validate == "terms":
        valid_values = VALID_SBO_TERMS
    elif validate == "names":
        valid_values = VALID_SBO_TERM_NAMES
    else:
        raise ValueError(f"Invalid validation type: {validate}")

    invalid_sbo_terms = sbo_series[~sbo_series.isin(valid_values)]

    if invalid_sbo_terms.shape[0] != 0:
        invalid_counts = invalid_sbo_terms.value_counts(sbo_series.name).to_frame("N")
        if not isinstance(invalid_counts, pd.DataFrame):
            raise TypeError("invalid_counts must be a pandas DataFrame")
        utils.show(invalid_counts)
        raise ValueError("Some reaction species have unusable SBO terms")
