from __future__ import annotations

import logging
import os
import random
from typing import Optional

import igraph as ig
import pandas as pd
from tqdm import tqdm

from napistu import indices, sbml_dfs_utils, source, utils
from napistu.constants import (
    BQB,
    BQB_DEFINING_ATTRS,
    ENTITIES_TO_ENTITY_DATA,
    EXPECTED_PW_INDEX_COLUMNS,
    IDENTIFIERS,
    SBML_DFS,
    SBML_DFS_METADATA,
    SBML_DFS_SCHEMA,
    SCHEMA_DEFS,
    SOURCE_SPEC,
    VALID_BQB_TERMS,
)
from napistu.identifiers import Identifiers
from napistu.ingestion import sbml
from napistu.ingestion.constants import NO_RXN_PATHWAY_IDS_DEFAULTS
from napistu.matching.mount import resolve_matches
from napistu.sbml_dfs_core import SBML_dfs

logger = logging.getLogger(__name__)


def construct_consensus_model(
    sbml_dfs_dict: dict[str, SBML_dfs],
    pw_index: indices.PWIndex,
    model_source: Optional[source.Source] = None,
    dogmatic: bool = True,
    check_mergeability: bool = True,
    no_rxn_pathway_ids: Optional[list[str]] = None,
) -> SBML_dfs:
    """
    Construct a Consensus Model by merging shared entities across pathway models.

    This function takes a dictionary of pathway models and merges shared entities (compartments, species, reactions, etc.)
    into a single consensus model, using a set of rules for entity identity and merging.

    Parameters
    ----------
    sbml_dfs_dict : dict[str, SBML_dfs]
        A dictionary of SBML_dfs objects from different models, keyed by model name.
    pw_index : indices.PWIndex
        An index of all tables being aggregated, used for cross-referencing entities.
    model_source : source.Source
        A source object for the consensus model.
    dogmatic : bool, default=True
        If True, preserve genes, transcripts, and proteins as separate species. If False, merge them when possible.
    check_mergeability : bool, default=True
        whether to check for issues which will prevent merging across models
    no_rxn_pathway_ids : list, optional
        The pathway ids for models which should not have reactions. If None, use the defaults. This can be used to
        include pathways which are just metadata like "Dogma".

    Returns
    -------
    SBML_dfs
        A consensus SBML_dfs object containing the merged model.
    """
    # Validate inputs
    logger.info("Reporting possible issues in component models")
    assert isinstance(pw_index, indices.PWIndex)
    _check_sbml_dfs_dict(sbml_dfs_dict, pw_index, check_mergeability)

    if model_source is None:
        model_source = _create_default_consensus_source(sbml_dfs_dict)
    else:
        if not isinstance(model_source, source.Source):
            raise TypeError("model_source must be a Source object or None")

    # Select valid BQB attributes based on dogmatic flag
    defining_biological_qualifiers = sbml_dfs_utils._dogmatic_to_defining_bqbs(dogmatic)

    # Step 1: Create consensus entities for all primary tables
    consensus_entities, lookup_tables = _create_consensus_entities(
        sbml_dfs_dict, pw_index, defining_biological_qualifiers, no_rxn_pathway_ids
    )

    # Step 2: Create the consensus SBML_dfs object
    sbml_dfs = SBML_dfs(consensus_entities, model_source, validate=False)  # type: ignore

    # Step 3: Add entity data from component models
    sbml_dfs = _add_entity_data(sbml_dfs, sbml_dfs_dict, lookup_tables)

    # cleanup by removing unused entities (this will generally be due to some data sources missing
    # entities which should be implied by other tables - like in the "dogma" data source there
    # are genes but not reactions.)
    sbml_dfs.remove_unused()
    sbml_dfs.validate()

    return sbml_dfs


def construct_meta_entities_fk(
    sbml_dfs_dict: dict[str, SBML_dfs],
    pw_index: pd.DataFrame,
    table: str = SBML_DFS.COMPARTMENTALIZED_SPECIES,
    fk_lookup_tables: dict = {},
    extra_defining_attrs: list = [],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct Meta Entities Defined by Foreign Keys

    Aggregating across one entity type for a set of pathway
    models merge entities which are defined by their foreign keys.

    Parameters:
    ----------
    sbml_df_dict: dict{"model": SBML_dfs}
        A dictionary of cpr.SBML_dfs
    pw_index: indices.PWIndex
        An index of all tables being aggregated
    table:
        A table/entity set from the sbml_dfs to work-with
    fk_lookup_tables: dict
        Dictionary containing lookup tables for all foreign keys used by the table
    extra_defining_attrs: list
        List of terms which uniquely define a reaction species in addition
        to the foreign keys. A common case is when a species is a modifier
        and a substrate in a reaction.

    Returns:
    ----------
    new_id_table: pd.DataFrame
        Matching the schema of one of the tables within sbml_df_dict
    lookup_table: pd.Series
        Matches the index of the aggregated entities to new_ids

    """

    if not isinstance(extra_defining_attrs, list):
        raise TypeError("extra_defining_attrs must be a list")

    # combine sbml_dfs by adding model to the index and concatinating all dfs
    agg_tbl = _unnest_SBML_df(sbml_dfs_dict, table=table)

    # since all sbml_dfs have the same schema pull out one schema for reference
    table_schema = SBML_DFS_SCHEMA.SCHEMA[table]

    # update foreign keys using provided lookup tables
    agg_tbl = _update_foreign_keys(agg_tbl, table_schema, fk_lookup_tables)

    # add nameness_score as a measure of how-readable a possible name would be
    # (this will help to select names which are more human readable after the merge)
    agg_tbl = utils._add_nameness_score_wrapper(
        agg_tbl, SCHEMA_DEFS.LABEL, table_schema
    )

    # reduce to unique elements
    induced_entities = (
        agg_tbl.reset_index(drop=True)
        .sort_values(["nameness_score"])
        .groupby(table_schema[SCHEMA_DEFS.FK] + extra_defining_attrs)
        .first()
        .drop("nameness_score", axis=1)
    )
    induced_entities["new_id"] = sbml_dfs_utils.id_formatter(
        range(induced_entities.shape[0]), table_schema[SCHEMA_DEFS.PK]
    )

    new_id_table = (
        induced_entities.reset_index()
        .rename(columns={"new_id": table_schema[SCHEMA_DEFS.PK]})
        .set_index(table_schema[SCHEMA_DEFS.PK])[table_schema[SCHEMA_DEFS.VARS]]
    )

    lookup_table = agg_tbl[table_schema[SCHEMA_DEFS.FK] + extra_defining_attrs].merge(
        induced_entities,
        left_on=table_schema[SCHEMA_DEFS.FK] + extra_defining_attrs,
        right_index=True,
    )["new_id"]

    # logging merges that occurred
    _report_consensus_merges(
        lookup_table, table_schema, agg_tbl=agg_tbl, n_example_merges=5
    )

    if SCHEMA_DEFS.SOURCE in table_schema.keys():
        # track the model(s) that each entity came from
        new_sources = _create_consensus_sources(
            agg_tbl.merge(lookup_table, left_index=True, right_index=True),
            lookup_table,
            table_schema,
            pw_index,
        )
        assert isinstance(new_sources, pd.Series)

        new_id_table = new_id_table.drop(
            table_schema[SCHEMA_DEFS.SOURCE], axis=1
        ).merge(new_sources, left_index=True, right_index=True)

    return new_id_table, lookup_table


def construct_meta_entities_identifiers(
    sbml_dfs_dict: dict[str, SBML_dfs],
    pw_index: indices.PWIndex,
    table: str,
    fk_lookup_tables: dict = {},
    defining_biological_qualifiers: list[str] = BQB_DEFINING_ATTRS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct meta-entities by merging entities across models that share identifiers.

    Aggregates a single entity type from a set of pathway models and merges entities that share identifiers
    (as defined by the provided biological qualifiers).

    Parameters
    ----------
    sbml_dfs_dict : dict[str, SBML_dfs]
        A dictionary of SBML_dfs objects from different models, keyed by model name.
    pw_index : indices.PWIndex
        An index of all tables being aggregated.
    table : str
        The name of the table/entity set to aggregate (e.g., 'species', 'compartments').
    fk_lookup_tables : dict, optional
        Dictionary containing lookup tables for all foreign keys used by the table (default: empty dict).
    defining_biological_qualifiers : list[str], optional
        List of BQB codes which define distinct entities. Defaults to BQB_DEFINING_ATTRS.

    Returns
    -------
    new_id_table : pd.DataFrame
        Table matching the schema of one of the input models, with merged entities.
    lookup_table : pd.Series
        Series mapping the index of the aggregated entities to new consensus IDs.
    """

    # combine sbml_dfs by adding model to the index and concatinating all dfs
    agg_tbl = _unnest_SBML_df(sbml_dfs_dict, table=table)

    # since all sbml_dfs have the same schema pull out one schema for reference
    table_schema = SBML_DFS_SCHEMA.SCHEMA[table]

    # update foreign keys using provided lookup tables
    if SCHEMA_DEFS.FK in table_schema.keys():
        agg_tbl = _update_foreign_keys(agg_tbl, table_schema, fk_lookup_tables)

    new_id_table, lookup_table = _reduce_to_consensus_ids(
        sbml_df=agg_tbl,
        table_schema=table_schema,
        pw_index=pw_index,
        defining_biological_qualifiers=defining_biological_qualifiers,
    )

    # logging merges that occurred
    _report_consensus_merges(
        lookup_table, table_schema, agg_tbl=agg_tbl, n_example_merges=5
    )

    return new_id_table, lookup_table


def construct_meta_entities_members(
    sbml_dfs_dict: dict[str, SBML_dfs],
    pw_index: indices.PWIndex | None,
    table: str = SBML_DFS.REACTIONS,
    defined_by: str = SBML_DFS.REACTION_SPECIES,
    defined_lookup_tables: dict = {},
    defining_attrs: list[str] = [SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct Meta Entities Defined by Membership

    Aggregating across one entity type for a set of pathway models, merge entities with the same members.

    Parameters:
    ----------
    sbml_df_dict: dict{"model": SBML_dfs}
        A dictionary of SBML_dfs
    pw_index: indices.PWIndex
        An index of all tables being aggregated
    table: str
        A table/entity set from the sbml_dfs to work-with
    defined_by: dict
        A table/entity set whose entries are members of "table"
    defined_lookup_tables: {pd.Series}
        Lookup table for updating the ids of "defined_by"
    defining_attrs: [str]
        A list of attributes which jointly define a unique entity

    Returns:
    ----------
    new_id_table: pd.DataFrame
        Matching the schema of one of the tables within sbml_df_dict
    lookup_table: pd.Series
        Matches the index of the aggregated entities to new_ids
    """
    logger.info(
        f"Merging {table} based on identical membership ({' + '.join(defining_attrs)})"
    )

    # Step 1: Get schemas for both tables
    table_schema = sbml_dfs_dict[list(sbml_dfs_dict.keys())[0]].schema[table]
    defined_by_schema = sbml_dfs_dict[list(sbml_dfs_dict.keys())[0]].schema[defined_by]

    # Step 2: Prepare the member table and validate its structure
    agg_tbl, _ = _prepare_member_table(
        sbml_dfs_dict,
        defined_by,
        defined_lookup_tables,
        table_schema,
        defined_by_schema,
        defining_attrs,
        table,
    )

    # Step 3: Create lookup table for entity membership
    membership_lookup = _create_membership_lookup(agg_tbl, table_schema)

    # Step 4: Create consensus entities and lookup table
    _, lookup_table = _create_entity_consensus(membership_lookup, table_schema)

    # Step 5: Log merger information
    _report_consensus_merges(
        lookup_table, table_schema, sbml_dfs_dict=sbml_dfs_dict, n_example_merges=5
    )

    # Step 6: Get primary entity table and merge identifiers
    agg_primary_table = _unnest_SBML_df(sbml_dfs_dict, table=table)

    logger.info(f"Merging {table} identifiers")
    updated_identifiers = _merge_entity_identifiers(
        agg_primary_table, lookup_table, table_schema
    )

    # Step 7: Create consensus table with merged entities
    new_id_table = _create_consensus_table(
        agg_primary_table, lookup_table, updated_identifiers, table_schema
    )

    # Step 8: Add source information if present
    if SCHEMA_DEFS.SOURCE in table_schema.keys():
        logger.info(f"Merging {table} sources")

        # Track the model(s) that each entity came from
        new_sources = _create_consensus_sources(
            agg_primary_table.merge(lookup_table, left_index=True, right_index=True),
            lookup_table,
            table_schema,
            pw_index,
        )

        new_id_table = new_id_table.drop(
            table_schema[SCHEMA_DEFS.SOURCE], axis=1
        ).merge(new_sources, left_index=True, right_index=True)

    return new_id_table, lookup_table


def construct_sbml_dfs_dict(
    pw_index: pd.DataFrame, strict: bool = True, verbose: bool = False
) -> dict[str, SBML_dfs]:
    """
    Construct a dictionary of SBML_dfs objects from a pathway index.

    This function converts all models in the pathway index into SBML_dfs objects and adds them to a dictionary.
    Optionally, it can skip erroneous files with a warning instead of raising an error.

    Parameters
    ----------
    pw_index : pd.DataFrame
        An index of all tables being aggregated, containing model metadata and file paths.
    strict : bool, default=True
        If True, raise an error on any file that cannot be loaded. If False, skip erroneous files with a warning.
    verbose: bool, default=False
        If True, then include detailed logs.

    Returns
    -------
    dict[str, SBML_dfs]
        A dictionary mapping model names to SBML_dfs objects.
    """

    sbml_dfs_dict = dict()
    for i in tqdm(pw_index.index.index.tolist()):
        pw_entry = pw_index.index.loc[i]
        if verbose:
            logger.info(f"processing {pw_entry[SOURCE_SPEC.NAME]}")

        sbml_path = os.path.join(pw_index.base_path, pw_entry[SOURCE_SPEC.FILE])

        # create the sbml file's model-level Source metadata
        model_source = source.Source(
            (
                pw_entry.to_frame().T.assign(
                    **{SOURCE_SPEC.MODEL: pw_entry[SOURCE_SPEC.PATHWAY_ID]}
                )
            )
        )

        try:
            sbml_obj = sbml.SBML(sbml_path)
            sbml_dfs_dict[pw_entry[SOURCE_SPEC.PATHWAY_ID]] = SBML_dfs(
                sbml_obj, model_source
            )
        except ValueError as e:
            if strict:
                raise e
            logger.warning(
                f"{pw_entry[SOURCE_SPEC.NAME]} was NOT successfully loaded:",
                exc_info=True,
            )
    return sbml_dfs_dict


def prepare_consensus_model(
    sbml_dfs_list: list[SBML_dfs],
) -> tuple[dict[str, SBML_dfs], indices.PWIndex]:
    """
    Prepare for creating a consensus model using a list of to-be-consolidated sbml_dfs objects.

    This function will extract the core source metadata from a set of SBML_dfs objects and use it to create a pathway index object. The
    pathway_id from these objects will then be used to key the the sbml_dfs_list objects to create the expected input for `construct_consensus_model`.

    Parameters
    ----------
    sbml_dfs_list : list[SBML_dfs]
        List of sbml_dfs objects to be consolidated.

    Returns
    -------
    sbml_dfs_dict : dict[str, SBML_dfs]
        Dictionary of sbml_dfs objects indexed by pathway_id.
    pw_index : indices.PWIndex
        Pathway index object.

    Raises
    ------
    ValueError
        If the sbml_dfs_list is empty.
        If the sbml_dfs_list contains sbml_dfs objects with more than one row.
        If the sbml_dfs_list contains sbml_dfs objects with missing columns.
        If the sbml_dfs_list contains sbml_dfs objects with duplicate pathway_ids.
        If the sbml_dfs_list contains sbml_dfs objects with invalid pathway_ids.
    """

    invalid_sbml_dfs = [not isinstance(x, SBML_dfs) for x in sbml_dfs_list]

    if sum(invalid_sbml_dfs) > 0:
        raise ValueError(
            f"sbml_dfs_list contains {sum(invalid_sbml_dfs)} invalid sbml_dfs objects"
        )

    if len(sbml_dfs_list) == 0:
        raise ValueError("sbml_dfs_list is empty")

    sources_dfs = [
        x.metadata[SBML_DFS_METADATA.SBML_DFS_SOURCE].source for x in sbml_dfs_list
    ]

    # check that entries have exactly 1 row
    if not all([x.shape[0] == 1 for x in sources_dfs]):
        raise ValueError("Each entry in sbml_dfs_dict must have exactly 1 row")

    source_df = pd.concat(sources_dfs)

    missing_columns = EXPECTED_PW_INDEX_COLUMNS - set(source_df.columns.tolist())
    if missing_columns:
        raise ValueError(
            f"Missing columns from sbml_dfs_source objects pulled from the metadata of sbml_dfs_dict: {missing_columns}"
        )

    # convert to pathway index object which will perform validations like ensuring that all pathway_ids are unique
    pw_index = indices.PWIndex(
        source_df[list(EXPECTED_PW_INDEX_COLUMNS)], validate_paths=False
    )

    sbml_dfs_dict = {
        x: y
        for x, y in zip(pw_index.index[SOURCE_SPEC.PATHWAY_ID].tolist(), sbml_dfs_list)
    }

    return sbml_dfs_dict, pw_index


# private functions


def _add_consensus_sources(
    new_id_table: pd.DataFrame,
    agg_table_harmonized: pd.DataFrame,
    lookup_table: pd.Series,
    table_schema: dict,
    pw_index: indices.PWIndex | None,
) -> pd.DataFrame:
    """
    Add source information to the consensus table.

    Parameters:
    ----------
    new_id_table: pd.DataFrame
        Consensus table without source information
    agg_table_harmonized: pd.DataFrame
        Original table with cluster assignments
    lookup_table: pd.Series
        Maps old IDs to new consensus IDs
    table_schema: dict
        Schema for the table
    pw_index: indices.PWIndex | None
        An index of all tables being aggregated

    Returns:
    ----------
    pd.DataFrame
        Consensus table with source information added
    """
    if type(pw_index) is not indices.PWIndex:
        raise ValueError(
            f"pw_index must be provided as a indices.PWIndex if there is a source but was type {type(pw_index)}"
        )

    # Track the model(s) that each entity came from
    new_sources = _create_consensus_sources(
        agg_table_harmonized, lookup_table, table_schema, pw_index
    )
    assert isinstance(new_sources, pd.Series)

    # Add the sources to the consensus table
    updated_table = new_id_table.drop(table_schema[SCHEMA_DEFS.SOURCE], axis=1).merge(
        new_sources, left_index=True, right_index=True
    )

    return updated_table


def _add_entity_data(
    sbml_dfs: SBML_dfs,
    sbml_dfs_dict: dict[str, SBML_dfs],
    lookup_tables: dict,
) -> SBML_dfs:
    """
    Add entity data from component models to the consensus model.

    Parameters:
    ----------
    sbml_dfs: SBML_dfs
        The consensus model being built
    sbml_dfs_dict: dict[str, SBML_dfs]
        A dictionary of SBML_dfs from different models
    lookup_tables: dict
        Dictionary of lookup tables for translating between old and new entity IDs

    Returns:
    ----------
    SBML_dfs
        The updated consensus model
    """
    # Add species data
    consensus_species_data = _merge_entity_data(
        sbml_dfs_dict,
        lookup_table=lookup_tables[SBML_DFS.SPECIES],
        table=SBML_DFS.SPECIES,
    )
    for k in consensus_species_data.keys():
        sbml_dfs.add_species_data(k, consensus_species_data[k])

    # Add reactions data
    consensus_reactions_data = _merge_entity_data(
        sbml_dfs_dict,
        lookup_table=lookup_tables[SBML_DFS.REACTIONS],
        table=SBML_DFS.REACTIONS,
    )
    for k in consensus_reactions_data.keys():
        sbml_dfs.add_reactions_data(k, consensus_reactions_data[k])

    return sbml_dfs


def _build_consensus_identifiers(
    sbml_df: pd.DataFrame,
    table_schema: dict,
    defining_biological_qualifiers: list[str] = BQB_DEFINING_ATTRS,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build consensus identifiers by clustering entities that share biological identifiers.

    This function takes a set of entities spanning multiple models and finds all unique entities
    by grouping them according to the provided biological qualifiers. It returns a mapping from
    original entities to clusters and a DataFrame of consensus identifier objects for each cluster.

    Parameters
    ----------
    sbml_df : pd.DataFrame
        Table of entities from multiple models, with model in the index (as produced by _unnest_SBML_df).
    table_schema : dict
        Schema for the table being processed.
    defining_biological_qualifiers : list[str], optional
        List of biological qualifier types to use for grouping. Defaults to BQB_DEFINING_ATTRS.

    Returns
    -------
    indexed_cluster : pd.Series
        Series mapping the index from sbml_df onto a set of clusters which define unique entities.
    cluster_consensus_identifiers_df : pd.DataFrame
        DataFrame mapping clusters to consensus identifiers (Identifiers objects).
    """
    # Step 1: Extract and validate identifiers
    logger.debug("unnesting identifiers")
    meta_identifiers = sbml_dfs_utils.unnest_identifiers(
        sbml_df, table_schema[SCHEMA_DEFS.ID]
    )
    _validate_meta_identifiers(meta_identifiers)

    # Step 2: Filter identifiers by biological qualifier type
    logger.debug("filtering identifiers by biological qualifier type")
    valid_identifiers = _filter_identifiers_by_qualifier(
        meta_identifiers, defining_biological_qualifiers
    )

    # Step 3: Handle entries that don't have identifiers
    logger.debug("handling entries that don't have identifiers")
    valid_identifiers = _handle_entries_without_identifiers(sbml_df, valid_identifiers)

    # Step 4: Prepare edgelist for clustering
    logger.debug("preparing identifier edgelist")
    id_edgelist = _prepare_identifier_edgelist(valid_identifiers, sbml_df)

    # Step 5: Cluster entities based on shared identifiers
    logger.debug("clustering entities based on shared identifiers")
    ind_clusters = utils.find_weakly_connected_subgraphs(id_edgelist)

    # Step 6: Map entity indices to clusters
    valid_identifiers_with_clusters = valid_identifiers.reset_index().merge(
        ind_clusters
    )
    indexed_cluster = valid_identifiers_with_clusters.groupby(
        sbml_df.index.names
    ).first()["cluster"]

    # Step 7: Create consensus identifiers for each cluster
    cluster_consensus_identifiers_df = _create_cluster_identifiers(
        meta_identifiers, indexed_cluster, sbml_df, ind_clusters, table_schema
    )

    return indexed_cluster, cluster_consensus_identifiers_df


def _check_sbml_dfs(
    sbml_dfs: SBML_dfs, model_label: str, N_examples: int | str = 5
) -> None:
    """Check SBML_dfs for identifiers which are associated with different entities before a merge."""

    ids = sbml_dfs.get_identifiers(SBML_DFS.SPECIES)
    defining_ids = ids[ids[IDENTIFIERS.BQB].isin(BQB_DEFINING_ATTRS)]

    defining_identifier_counts = defining_ids.value_counts(
        [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]
    )
    degenerate_defining_identities = (
        defining_identifier_counts[defining_identifier_counts > 1]
        .rename("N")
        .reset_index()
        .set_index(IDENTIFIERS.ONTOLOGY)
    )

    if degenerate_defining_identities.shape[0] > 0:
        logger.info(
            "Some defining identifiers are present multiple times "
            f"in {model_label} and will likely result in species merging "
        )

        degen_defining_id_list = list()
        for k in degenerate_defining_identities.index.unique():
            n_degen = degenerate_defining_identities.loc[k].shape[0]
            example_duplicates = utils.ensure_pd_df(
                degenerate_defining_identities.loc[k].sample(min([n_degen, N_examples]))
            )

            degen_defining_id_list.append(
                k
                + f" has {n_degen} duplicates including: "
                + ", ".join(
                    [
                        f"{x} ({y})"
                        for x, y in zip(
                            example_duplicates[IDENTIFIERS.IDENTIFIER].tolist(),
                            example_duplicates["N"].tolist(),
                        )
                    ]
                )
            )

        logger.info("\n".join(degen_defining_id_list))
    return None


def _check_sbml_dfs_dict(
    sbml_dfs_dict: dict[str, SBML_dfs],
    pw_index: indices.PWIndex,
    check_mergeability: bool = True,
) -> None:
    """Check models in SBML_dfs for problems which can be reported up-front

    Parameters
    ----------
    sbml_dfs_dict : dict[str, SBML_dfs]
        a dict of sbml_dfs models;
    pw_index : indices.PWIndex
        an index of all tables being aggregated
    check_mergeability : bool, default=True
        whether to check for issues which will prevent merging across models

    Returns
    -------
    None
        This function returns None but logs error messages if incompatible
        ontology structures are detected.
    """

    for k, v in sbml_dfs_dict.items():
        _check_sbml_dfs(sbml_dfs=v, model_label=k)

    if check_mergeability:
        _check_sbml_dfs_mergeability(sbml_dfs_dict, pw_index)

    return None


def _check_sbml_dfs_mergeability(
    sbml_dfs_dict: dict[str, SBML_dfs],
    pw_index: indices.PWIndex,
) -> None:
    """Check SBML_dfs for obvious issues which will prevent merging across models.

    Parameters
    ----------
    sbml_dfs_dict : dict[str, SBML_dfs]
        a dict of sbml_dfs models;
    pw_index : indices.PWIndex
        an index of all tables being aggregated

    Returns
    -------
    None
        This function returns None but logs error messages if incompatible
        ontology structures are detected.
    """

    logger.info("Evaluating compartment compatibility")
    _pre_consensus_compartment_check(sbml_dfs_dict, pw_index)

    logger.info("Evaluating ontology compatibility")
    _pre_consensus_ontology_check(sbml_dfs_dict, "species")

    logger.debug("Pre-consensus checks complete")

    return None


def _create_cluster_identifiers(
    meta_identifiers: pd.DataFrame,
    indexed_cluster: pd.Series,
    sbml_df: pd.DataFrame,
    ind_clusters: pd.DataFrame,
    table_schema: dict,
) -> pd.DataFrame:
    """
    Create identifier objects for each cluster.

    Parameters
    ----------
    meta_identifiers : pd.DataFrame
        All identifiers (including those filtered out by BQB)
    indexed_cluster : pd.Series
        Maps entity indices to cluster IDs
    sbml_df : pd.DataFrame
        Original table of entities
    ind_clusters : pd.DataFrame
        Cluster assignments from graph algorithm
    table_schema : dict
        Schema for the table, used to determine the correct identifier column name

    Returns
    -------
    pd.DataFrame
        Table mapping clusters to their consensus identifiers, with the identifier column named according to the schema
    """
    # Combine all identifiers with cluster assignments
    all_cluster_identifiers = meta_identifiers.reset_index().merge(
        indexed_cluster, left_on=sbml_df.index.names, right_index=True
    )

    # Create an Identifiers object for each cluster
    cluster_consensus_identifiers = {
        k: Identifiers(
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
        for k, v in all_cluster_identifiers.groupby("cluster")
    }

    # Handle clusters that don't have any identifiers
    catchup_clusters = {
        c: Identifiers(list())
        for c in set(ind_clusters["cluster"].tolist()).difference(
            cluster_consensus_identifiers
        )
    }
    cluster_consensus_identifiers = {
        **cluster_consensus_identifiers,
        **catchup_clusters,
    }

    # Convert to DataFrame with correct column name
    id_col = table_schema["id"]
    cluster_consensus_identifiers_df = pd.DataFrame(
        cluster_consensus_identifiers, index=[id_col]
    ).T
    cluster_consensus_identifiers_df.index.name = "cluster"
    return cluster_consensus_identifiers_df


def _create_consensus_sources(
    agg_tbl: pd.DataFrame,
    lookup_table: pd.Series,
    table_schema: dict,
    pw_index: indices.PWIndex | None,
) -> pd.Series:
    """
    Create Consensus Sources

    Annotate the source of to-be-merged species with the models they came from, and combine with existing annotations.

    Parameters:
    ----------
    agg_tbl: pd.DataFrame
        A table containing existing source.Source objects and a many-1
        "new_id" of their post-aggregation consensus entity
    lookup_table: pd.Series
        A series where the index are old identifiers and the values are
        post-aggregation new identifiers
    table_schema: dict
        Summary of the schema for the operant entitye type
    pw_index: indices.PWIndex
        An index of all tables being aggregated

    Returns:
    ----------
    new_sources: pd.DataFrame
        Mapping where the index is new identifiers and values are aggregated source.Source objects

    """

    logger.info("Creating source table")
    # Sources for all new entries
    new_sources = source.create_source_table(lookup_table, table_schema, pw_index)

    # create a pd.Series with an index of all new_ids (which will be rewritten as the entity primary keys)
    # and values of source.Source objects (where multiple Sources may match an index value).
    logger.info("Aggregating old sources")
    indexed_old_sources = (
        agg_tbl.reset_index(drop=True)
        .rename(columns={"new_id": table_schema[SCHEMA_DEFS.PK]})
        .groupby(table_schema[SCHEMA_DEFS.PK])[table_schema[SCHEMA_DEFS.SOURCE]]
    )

    # combine old sources into a single source.Source object per index value
    aggregated_old_sources = indexed_old_sources.agg(source.merge_sources)

    aligned_sources = new_sources.merge(
        aggregated_old_sources, left_index=True, right_index=True
    )
    assert isinstance(aligned_sources, pd.DataFrame)

    logger.info("Returning new source table")
    new_sources = aligned_sources.apply(source.merge_sources, axis=1).rename(table_schema[SCHEMA_DEFS.SOURCE])  # type: ignore
    assert isinstance(new_sources, pd.Series)

    return new_sources


def _create_consensus_entities(
    sbml_dfs_dict: dict[str, SBML_dfs],
    pw_index: indices.PWIndex,
    defining_biological_qualifiers: list[str],
    no_rxn_pathway_ids: Optional[list[str]] = None,
) -> tuple[dict, dict]:
    """
    Create consensus entities for all primary tables in the model.

    This helper function creates consensus compartments, species, compartmentalized species,
    reactions, and reaction species by finding shared entities across source models.

    Parameters:
    ----------
    sbml_dfs_dict: dict{SBML_dfs}
        A dictionary of SBML_dfs from different models
    pw_index: indices.PWIndex
        An index of all tables being aggregated
    defining_biological_qualifiers: list[str]
        Biological qualifier terms that define distinct entities
    no_rxn_pathway_ids: Optional[list[str]] = None,
        The pathway ids for models which should not have reactions. If None, use the defaults. This can be used to
        include pathways which are just metadata like "Dogma".

    Returns:
    ----------
    tuple:
        - dict of consensus entities tables
        - dict of lookup tables
    """

    no_rxn_pathway_ids = _get_no_rxn_pathway_ids(pw_index, no_rxn_pathway_ids)

    # Step 1: Compartments
    logger.info("Defining compartments based on unique ids")
    comp_consensus_entities, comp_lookup_table = construct_meta_entities_identifiers(
        sbml_dfs_dict=sbml_dfs_dict, pw_index=pw_index, table="compartments"
    )

    # Step 2: Species
    logger.info("Defining species based on unique ids")
    spec_consensus_entities, spec_lookup_table = construct_meta_entities_identifiers(
        sbml_dfs_dict=sbml_dfs_dict,
        pw_index=pw_index,
        table=SBML_DFS.SPECIES,
        defining_biological_qualifiers=defining_biological_qualifiers,
    )

    # Step 3: Compartmentalized species
    logger.info(
        "Defining compartmentalized species based on unique species x compartments"
    )
    compspec_consensus_instances, compspec_lookup_table = construct_meta_entities_fk(
        sbml_dfs_dict,
        pw_index,
        table=SBML_DFS.COMPARTMENTALIZED_SPECIES,
        fk_lookup_tables={
            SBML_DFS.C_ID: comp_lookup_table,
            SBML_DFS.S_ID: spec_lookup_table,
        },
    )

    # remove pathways which don't contribute reactions
    _remove_no_rxn_pathways(no_rxn_pathway_ids, sbml_dfs_dict, compspec_lookup_table)

    # Step 4: Reactions
    logger.info(
        "Define reactions based on membership of identical compartmentalized species"
    )
    rxn_consensus_species, rxn_lookup_table = construct_meta_entities_members(
        sbml_dfs_dict,
        pw_index,
        table=SBML_DFS.REACTIONS,
        defined_by=SBML_DFS.REACTION_SPECIES,
        defined_lookup_tables={SBML_DFS.SC_ID: compspec_lookup_table},
        defining_attrs=[SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY, SBML_DFS.SBO_TERM],
    )

    logger.info("Annotating reversibility based on merged reactions")
    rxn_consensus_species = _resolve_reversibility(
        sbml_dfs_dict, rxn_consensus_species, rxn_lookup_table
    )

    # Step 5: Reaction species
    logger.info("Define reaction species based on reactions")
    rxnspec_consensus_instances, rxnspec_lookup_table = construct_meta_entities_fk(
        sbml_dfs_dict,
        pw_index,
        table=SBML_DFS.REACTION_SPECIES,
        fk_lookup_tables={
            SBML_DFS.R_ID: rxn_lookup_table,
            SBML_DFS.SC_ID: compspec_lookup_table,
        },
        # retain species with different roles
        extra_defining_attrs=[SBML_DFS.SBO_TERM],
    )

    consensus_entities = {
        SBML_DFS.COMPARTMENTS: comp_consensus_entities,
        SBML_DFS.SPECIES: spec_consensus_entities,
        SBML_DFS.COMPARTMENTALIZED_SPECIES: compspec_consensus_instances,
        SBML_DFS.REACTIONS: rxn_consensus_species,
        SBML_DFS.REACTION_SPECIES: rxnspec_consensus_instances,
    }

    lookup_tables = {
        SBML_DFS.COMPARTMENTS: comp_lookup_table,
        SBML_DFS.SPECIES: spec_lookup_table,
        SBML_DFS.COMPARTMENTALIZED_SPECIES: compspec_lookup_table,
        SBML_DFS.REACTIONS: rxn_lookup_table,
        SBML_DFS.REACTION_SPECIES: rxnspec_lookup_table,
    }

    return consensus_entities, lookup_tables


def _create_consensus_entity_data(
    combined_entity_data: pd.DataFrame, primary_key: str
) -> pd.DataFrame:
    """
    Create consensus entity data by combining multiple rows with the same index value.

    This function takes a DataFrame that might have multiple rows for the same index value
    and combines them so there is exactly 1 row per index value using the "first" method.

    Parameters:
    -----------
    combined_entity_data : pd.DataFrame
        Input DataFrame with potentially multiple rows per index value
    primary_key : str
        The column name to use as the primary key for grouping

    Returns:
    --------
    pd.DataFrame
        DataFrame with exactly one row per unique primary key value
    """
    return combined_entity_data.reset_index().groupby(primary_key).first()


def _create_consensus_table(
    agg_primary_table: pd.DataFrame,
    lookup_table: pd.Series,
    updated_identifiers: pd.Series,
    table_schema: dict,
) -> pd.DataFrame:
    """
    Create a consensus table with merged entities.

    Parameters:
    ----------
    agg_primary_table: pd.DataFrame
        Table of entities
    lookup_table: pd.Series
        Lookup table mapping old IDs to new IDs
    updated_identifiers: pd.Series
        Series mapping new IDs to merged identifier objects
    table_schema: dict
        Schema for the table

    Returns:
    ----------
    pd.DataFrame
        Consensus table with one row per unique entity
    """
    # Add nameness scores to help select representative names
    agg_primary_table_scored = utils._add_nameness_score_wrapper(
        agg_primary_table, "label", table_schema
    )

    # Create a table with one row per consensus entity
    new_id_table = (
        agg_primary_table_scored.join(lookup_table)
        .reset_index(drop=True)
        .sort_values(["nameness_score"])
        .rename(columns={"new_id": table_schema["pk"]})
        .groupby(table_schema["pk"])
        .first()[table_schema["vars"]]
    )

    # Replace identifiers with merged versions
    new_id_table = new_id_table.drop(table_schema["id"], axis=1).merge(
        updated_identifiers, left_index=True, right_index=True
    )

    return new_id_table


def _create_default_consensus_source(
    sbml_dfs_dict: dict[str, SBML_dfs],
) -> source.Source:
    """
    A default consensus source is created when no model source object is provided.

    Parameters
    ----------
    sbml_dfs_dict : dict[str, SBML_dfs]
        A dictionary of SBML_dfs objects from different models, keyed by model name.

    Returns
    -------
    source.Source
        A default consensus source object.
    """

    return source.Source.single_entry(
        model="consensus_model",
        name=f"Consensus model merging {len(sbml_dfs_dict)} sources",
    )


def _create_entity_consensus(
    membership_lookup: pd.DataFrame, table_schema: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create consensus entities based on membership.

    Parameters:
    ----------
    membership_lookup: pd.DataFrame
        Table mapping entities to their member strings
    table_schema: dict
        Schema for the table

    Returns:
    ----------
    tuple:
        - Consensus entities DataFrame
        - Lookup table mapping old IDs to new IDs
    """
    # Group by member string to find entities with identical members
    consensus_entities = membership_lookup.groupby("member_string").first()

    # Create new IDs for the consensus entities
    consensus_entities["new_id"] = sbml_dfs_utils.id_formatter(
        range(consensus_entities.shape[0]), table_schema["pk"]
    )

    # Create lookup table mapping original entities to consensus entities
    lookup_table = membership_lookup.merge(
        consensus_entities["new_id"], left_on="member_string", right_index=True
    ).set_index([SOURCE_SPEC.MODEL, table_schema["pk"]])["new_id"]

    return consensus_entities, lookup_table


def _create_entity_lookup_table(
    agg_table_harmonized: pd.DataFrame, table_schema: dict
) -> pd.Series:
    """
    Create a lookup table mapping original entity IDs to new consensus IDs.

    Parameters:
    ----------
    agg_table_harmonized: pd.DataFrame
        Table with cluster assignments for each entity
    table_schema: dict
        Schema for the table

    Returns:
    ----------
    pd.Series
        Lookup table mapping old entity IDs to new consensus IDs
    """
    # Create a new ID based on cluster number and entity type
    agg_table_harmonized["new_id"] = sbml_dfs_utils.id_formatter(
        agg_table_harmonized["cluster"], table_schema["pk"]
    )

    # Return the lookup series
    return agg_table_harmonized["new_id"]


def _create_member_string(x: list[str]) -> str:
    x.sort()
    return "_".join(x)


def _create_membership_lookup(
    agg_tbl: pd.DataFrame, table_schema: dict
) -> pd.DataFrame:
    """
    Create a lookup table for entity membership.

    Parameters:
    ----------
    agg_tbl: pd.DataFrame
        Table with member information
    table_schema: dict
        Schema for the table

    Returns:
    ----------
    pd.DataFrame
        Lookup table mapping entity IDs to member strings
    """
    # Group members by entity
    membership_df = (
        agg_tbl.reset_index()
        .groupby(["model", table_schema["pk"]])
        .agg(membership=("member", lambda x: (list(set(x)))))
    )

    # Check for duplicated members within an entity
    for i in range(membership_df.shape[0]):
        members = membership_df["membership"].iloc[i]
        if len(members) != len(set(members)):
            raise ValueError(
                "Members were duplicated suggesting overmerging in the source"
            )

    # Convert membership lists to strings for comparison
    membership_df["member_string"] = [
        _create_member_string(x) for x in membership_df["membership"]
    ]

    return membership_df.reset_index()


def _create_vertex_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Create vertex dataframe for a specific category from a source column."""
    return (
        df.copy()
        .assign(category=category)
        .rename(columns={category: "name"})[["name", "category"]]
        .drop_duplicates()
    )


def _filter_identifiers_by_qualifier(
    meta_identifiers: pd.DataFrame, defining_biological_qualifiers: list[str]
) -> pd.DataFrame:
    """
    Filter identifiers to only include those with specific biological qualifiers.

    Parameters:
    ----------
    meta_identifiers: pd.DataFrame
        Table of identifiers
    defining_biological_qualifiers: list[str]
        List of biological qualifier types to keep

    Returns:
    ----------
    pd.DataFrame
        Filtered identifiers
    """

    invalid_bqbs = set(meta_identifiers[IDENTIFIERS.BQB]) - set(VALID_BQB_TERMS)
    if len(invalid_bqbs) > 0:
        logger.warning(f"Invalid biological qualifiers: {invalid_bqbs}")

    valid_identifiers = meta_identifiers.copy()
    return valid_identifiers[
        meta_identifiers[IDENTIFIERS.BQB].isin(defining_biological_qualifiers)
    ]


def _get_no_rxn_pathway_ids(pw_index, no_rxn_pathway_ids=None):
    """
    Get the pathway ids for models which should not have reactions.

    Parameters
    ----------
    pw_index : pd.DataFrame
        The pathway index.
    no_rxn_pathway_ids : list, optional
        The pathway ids for models which should not have reactions. If None, use the defaults.

    Returns
    -------
    no_rxn_pathway_ids : list
        The pathway ids for models which should not have reactions.
    """

    if no_rxn_pathway_ids is None:
        no_rxn_pathway_ids = [
            x
            for x in NO_RXN_PATHWAY_IDS_DEFAULTS
            if x in pw_index.index[SOURCE_SPEC.PATHWAY_ID].tolist()
        ]

    invalid_rxn_pathway_ids = [
        x
        for x in no_rxn_pathway_ids
        if x not in pw_index.index[SOURCE_SPEC.PATHWAY_ID].tolist()
    ]
    if len(invalid_rxn_pathway_ids) > 0:
        raise ValueError(
            f'The requested "no_rxn_pathway_ids" were not found in the pw_index: {invalid_rxn_pathway_ids}. Either set this to None to use defaults or set it to a list of valid pathway ids.'
        )

    return no_rxn_pathway_ids


def _handle_entries_without_identifiers(
    sbml_df: pd.DataFrame, valid_identifiers: pd.DataFrame
) -> pd.DataFrame:
    """
    Handle entities that don't have identifiers by adding dummy identifiers.

    Parameters:
    ----------
    sbml_df: pd.DataFrame
        Original table of entities
    valid_identifiers: pd.DataFrame
        Table of identifiers that passed filtering

    Returns:
    ----------
    pd.DataFrame
        Valid identifiers with dummy entries added
    """
    # Find entries which no longer have any identifiers
    filtered_entries = sbml_df.reset_index().merge(
        valid_identifiers.reset_index(),
        left_on=sbml_df.index.names,
        right_on=sbml_df.index.names,
        how="outer",
    )[sbml_df.index.names + [IDENTIFIERS.IDENTIFIER]]

    filtered_entries = filtered_entries[
        filtered_entries[IDENTIFIERS.IDENTIFIER].isnull()
    ]

    if filtered_entries.shape[0] == 0:
        return valid_identifiers

    # Add dummy identifiers to these entries
    logger.warning(
        f"{filtered_entries.shape[0]} entries didn't possess identifiers and thus cannot be merged"
    )

    filtered_entries[SOURCE_SPEC.ENTRY] = 0
    filtered_entries[IDENTIFIERS.ONTOLOGY] = "none"
    filtered_entries[IDENTIFIERS.ONTOLOGY] = [
        "dummy_value_" + str(val)
        for val in random.sample(range(1, 100000000), filtered_entries.shape[0])
    ]
    filtered_entries[IDENTIFIERS.URL] = "dummy_url"
    filtered_entries[IDENTIFIERS.BQB] = BQB.UNKNOWN

    filtered_entries = filtered_entries.set_index(
        sbml_df.index.names + [SOURCE_SPEC.ENTRY]
    )

    # Combine original valid identifiers with dummy identifiers
    # Build list of non-empty DataFrames to avoid FutureWarning

    dfs_to_concat = []
    if not valid_identifiers.empty:
        dfs_to_concat.append(valid_identifiers)
    if not filtered_entries.empty:
        dfs_to_concat.append(filtered_entries)

    if len(dfs_to_concat) == 0:
        raise ValueError("No valid identifiers found after filtering")
    elif len(dfs_to_concat) == 1:
        return dfs_to_concat[0]
    else:
        return pd.concat(dfs_to_concat)


def _merge_entity_data_create_consensus(
    entity_data_dict: dict,
    lookup_table: pd.Series,
    entity_schema: dict,
    an_entity_data_type: str,
    table: str,
) -> pd.DataFrame:
    """
    Merge Entity Data - Report Mismatches

    Report cases where a single "new" id is associated with multiple different values of entity_var

    Parameters:
    ----------
    entity_data_dict: dict
        Dictionary containing all model's "an_entity_data_type" dictionaries
    lookup_table: pd.Series
        A series where the index is an old model and primary key and the
        value is the new consensus id
    entity_schema: dict
        Schema for "table"
    an_entity_data_type: str
        data_type from species/reactions_data in entity_data_dict
    table: str
        table whose data is being consolidates (currently species or reactions)

    Returns:
    ----------
    pd.DataFrame
        Table where index is primary key of "table" and
        values are all distinct annotations from "an_entity_data_type".

    """

    models_w_entity_data_type = [
        k for k, v in entity_data_dict.items() if an_entity_data_type in v.keys()
    ]

    logger.info(
        f"Merging {len(models_w_entity_data_type)} models ({', '.join(models_w_entity_data_type)}) with a \"{an_entity_data_type}\" {ENTITIES_TO_ENTITY_DATA[table]} table"
    )

    # check that all tables have the same index and column names
    if len(models_w_entity_data_type) > 1:
        _validate_merge_entity_data_create_consensus(
            entity_data_dict, an_entity_data_type, models_w_entity_data_type
        )

    # stack all models
    combined_entity_data = pd.concat(
        {k: entity_data_dict[k][an_entity_data_type] for k in models_w_entity_data_type}
    )
    combined_entity_data.index.names = [
        SOURCE_SPEC.MODEL,
        entity_schema[SCHEMA_DEFS.PK],
    ]
    if isinstance(combined_entity_data, pd.Series):
        # enforce that atttributes should always be DataFrames
        combined_entity_data = combined_entity_data.to_frame()

    # create a table indexed by the NEW primary key containing all the entity data of type an_entity_data_type
    # right now the index may map to multiple rows if entities were consolidated
    combined_entity_data = (
        combined_entity_data.join(lookup_table)
        .reset_index(drop=True)
        .rename({"new_id": entity_schema[SCHEMA_DEFS.PK]}, axis=1)
        .set_index(entity_schema[SCHEMA_DEFS.PK])
        .sort_index()
    )

    # report cases where merges produce id-variable combinations with distinct values
    _merge_entity_data_report_mismatches(
        combined_entity_data, entity_schema, an_entity_data_type, table
    )

    # save one value for each id-variable combination
    # Prepare data for resolve_matches
    combined_entity_data_reset = combined_entity_data.reset_index()
    combined_entity_data_reset["feature_id"] = combined_entity_data_reset.index.astype(
        str
    )

    # TO DO - `resolve_matches` provides a lot of flexibility in terms of aggregating data
    # but currently, we're just taking the first value to match the old behavior
    consensus_entity_data = resolve_matches(
        combined_entity_data_reset,
        feature_id_var="feature_id",
        index_col=entity_schema["pk"],
        numeric_agg="first",
        keep_id_col=False,
    )

    if "feature_id_match_count" in consensus_entity_data.columns:
        consensus_entity_data = consensus_entity_data.drop(
            "feature_id_match_count", axis=1
        )

    return consensus_entity_data


def _merge_entity_data(
    sbml_dfs_dict: dict[str, SBML_dfs],
    lookup_table: pd.Series,
    table: str,
) -> dict:
    """
    Merge Entity Data

    Report cases where a single "new" id is associated with multiple different values of entity_var

    Parameters
    ----------
        sbml_dfs_dict : dict
            dictionary where values are to-be-merged model nnames and values
            are SBML_dfs
        lookup_table : pd.Series
            a series where the index is an old model and primary key and the
            value is the new consensus id
        table : str
            table whose data is being consolidates (currently species or reactions)

    Returns
    -------
    entity_data : dict
        dictionary containing pd.DataFrames which aggregate all of the
        individual entity_data tables in "sbml_dfs_dict"

    """

    entity_schema = sbml_dfs_dict[list(sbml_dfs_dict.keys())[0]].schema[table]
    data_table_name = ENTITIES_TO_ENTITY_DATA[table]

    entity_data_dict = {
        k: getattr(sbml_dfs_dict[k], data_table_name) for k in sbml_dfs_dict.keys()
    }

    entity_data_types = set.union(*[set(v.keys()) for v in entity_data_dict.values()])

    entity_data = {
        x: _merge_entity_data_create_consensus(
            entity_data_dict, lookup_table, entity_schema, x, table
        )
        for x in entity_data_types
    }

    return entity_data


def _merge_entity_data_report_mismatches(
    combined_entity_data: pd.DataFrame,
    entity_schema: dict,
    an_entity_data_type: str,
    table: str,
) -> None:
    """
    Merge Entity Data - Report Mismatches

    Report cases where a single "new" id is associated with multiple different values of entity_var

    Parameters
    ----------
        combined_entity_data : pd.DataFrame
            indexed by table primary key containing all
            data from "an_entity_data_type"
        entity_schema : dict
            schema for "table"
        an_entity_data_type : str
            data_type from species/reactions_data in combined_entity_data
        table : str
            table whose data is being consolidates (currently species or reactions)

    Returns
    -------
        None

    """

    data_table_name = ENTITIES_TO_ENTITY_DATA[table]

    entity_vars = combined_entity_data.columns
    for entity_var in entity_vars:
        unique_counts = (
            combined_entity_data.reset_index()
            .groupby(entity_schema[SCHEMA_DEFS.PK])
            .agg("nunique")
        )
        entities_w_imperfect_matches = unique_counts[
            unique_counts[entity_var] > 1
        ].index.tolist()

        if len(entities_w_imperfect_matches) > 0:
            N_select_entities_w_imperfect_matches = min(
                5, len(entities_w_imperfect_matches)
            )
            select_entities_w_imperfect_matches = entities_w_imperfect_matches[
                0:N_select_entities_w_imperfect_matches
            ]

            warning_msg_select = [
                x
                + ": "
                + ", ".join(
                    combined_entity_data[entity_var].loc[x].apply(str).unique().tolist()
                )
                for x in select_entities_w_imperfect_matches
            ]
            full_warning_msg = (
                f"{len(entities_w_imperfect_matches)} {table} contains multiple values for the {entity_var} variable"
                f" in the {data_table_name} table of {an_entity_data_type}: "
                + ". ".join(warning_msg_select)
            )

            logger.warning(full_warning_msg)

    return None


def _merge_entity_identifiers(
    agg_primary_table: pd.DataFrame, lookup_table: pd.Series, table_schema: dict
) -> pd.Series:
    """
    Merge identifiers from multiple entities.

    Parameters:
    ----------
    agg_primary_table: pd.DataFrame
        Table of entities
    lookup_table: pd.Series
        Lookup table mapping old IDs to new IDs
    table_schema: dict
        Schema for the table

    Returns:
    ----------
    pd.Series
        Series mapping new IDs to merged identifier objects
    """
    # Combine entities with the same consensus ID
    indexed_old_identifiers = (
        agg_primary_table.join(lookup_table)
        .reset_index(drop=True)
        .rename(columns={"new_id": table_schema[SCHEMA_DEFS.PK]})
        .groupby(table_schema[SCHEMA_DEFS.PK])[table_schema[SCHEMA_DEFS.ID]]
    )

    # Merge identifier objects
    return indexed_old_identifiers.agg(Identifiers.merge)


def _pre_consensus_compartment_check(
    sbml_dfs_dict: dict[str, SBML_dfs], pw_index: indices.PWIndex
) -> None:
    """
    Check for compartment compatibility across models before consensus building.

    This function identifies models that won't mix well in a consensus because they
    contain non-overlapping sets of compartments. It constructs a bipartite graph
    connecting models to their compartments and identifies disconnected components,
    which indicate incompatible compartment structures.

    Parameters
    ----------
    sbml_dfs_dict : dict
        Dictionary containing SBML dataframes for each model, keyed by model name.
    pw_index : pandas.DataFrame
        Pathway index dataframe containing model metadata and pathway information.

    Returns
    -------
    None
        This function returns None but logs error messages if incompatible
        compartment structures are detected.

    Notes
    -----
    The function builds a graph where:
    - Models are connected to their compartments via shared identifiers
    - Compartments are connected to their model-specific labels
    - Disconnected components indicate models with non-overlapping compartment sets

    If multiple disconnected components are found, an error is logged listing
    the incompatible compartment groups that would result in an unmixed consensus.

    Examples
    --------
    >>> sbml_dfs_dict = {"model1": sbml_dfs1, "model2": sbml_dfs2}
    >>> pw_index = pd.DataFrame({"model": ["model1", "model2"], ...})
    >>> _pre_consensus_compartment_check(sbml_dfs_dict, pw_index)
    # Logs error if models have incompatible compartment structures
    """

    _, compartments_df = construct_meta_entities_identifiers(
        sbml_dfs_dict, pw_index, table=SBML_DFS.COMPARTMENTS
    )
    models_to_compartments = (
        compartments_df.reset_index()
        .merge(
            _unnest_SBML_df(sbml_dfs_dict, SBML_DFS.COMPARTMENTS)[SBML_DFS.C_NAME],
            left_on=[SOURCE_SPEC.MODEL, SBML_DFS.C_ID],
            right_index=True,
        )
        .assign(model_c_name=lambda x: x[SOURCE_SPEC.MODEL] + ": " + x[SBML_DFS.C_NAME])
    )

    edges_df = pd.concat(
        [
            models_to_compartments[[SOURCE_SPEC.MODEL, "new_id"]].rename(
                columns={SOURCE_SPEC.MODEL: "source", "new_id": "target"}
            ),
            models_to_compartments[["new_id", "model_c_name"]].rename(
                columns={"new_id": "source", "model_c_name": "target"}
            ),
        ]
    )

    vertices_df = pd.concat(
        [
            _create_vertex_category(models_to_compartments, SOURCE_SPEC.MODEL),
            _create_vertex_category(models_to_compartments, "new_id"),
            _create_vertex_category(models_to_compartments, "model_c_name"),
        ]
    )

    g = ig.Graph.DictList(
        vertices=vertices_df.to_dict("records"), edges=edges_df.to_dict("records")
    )

    components = g.components(mode="weak")

    if len(components) > 1:

        full_label = list()
        for c in components:
            label = ", ".join(
                vertices_df.iloc[c]
                .query("category == 'model_c_name'")["name"]
                .to_list()
            )
            full_label.append(label)
        full_label = "\n".join(full_label)

        logger.error(
            f"The compartments shared across models are incompatible and will result in an-unmixed consensus model:\n{full_label}"
        )

    return None


def _pre_consensus_ontology_check(
    sbml_dfs_dict: dict[str, SBML_dfs], entity_type: str
) -> None:
    """
    Check for ontology compatibility across models before consensus building.

    This function determines whether any models possess disjoint sets of ontologies
    for a given entity type (compartments, or species). It constructs a
    bipartite graph connecting models to their ontologies and identifies disconnected
    components, which indicate models with non-overlapping ontology structures.

    Parameters
    ----------
    sbml_dfs_dict : dict[str, SBML_dfs]
        Dictionary containing SBML dataframes for each model, keyed by model name.
    entity_type : str
        The type of entity to check ontologies for. Must be one of 'compartments',
        'species', or 'reactions'.

    Returns
    -------
    None
        This function returns None but logs error messages if incompatible
        ontology structures are detected.

    Notes
    -----
    The function builds a graph where:
    - Models are connected to ontologies they contain for the specified entity type
    - Disconnected components indicate models with non-overlapping ontology sets

    If multiple disconnected components are found, an error is logged listing
    the incompatible ontology groups that would result in an unmixed consensus.

    Examples
    --------
    >>> sbml_dfs_dict = {"model1": sbml_dfs1, "model2": sbml_dfs2}
    >>> _pre_consensus_ontology_check(sbml_dfs_dict, "compartments")
    # Logs error if models have incompatible compartment ontologies
    """

    VALID_ENTITY_TYPES = [SBML_DFS.COMPARTMENTS, SBML_DFS.SPECIES]
    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(
            f"Invalid entity type: {entity_type}. Only {VALID_ENTITY_TYPES} are supported since they use identifier-based entity resolution."
        )

    ontology_counts = pd.concat(
        {
            k: (
                v._get_identifiers_table_for_ontology_occurrence(
                    entity_type
                ).value_counts([IDENTIFIERS.ONTOLOGY])
            )
            for k, v in sbml_dfs_dict.items()
        },
        axis=0,
        names=[SOURCE_SPEC.MODEL],
    ).reset_index()

    edges_df = ontology_counts[[SOURCE_SPEC.MODEL, IDENTIFIERS.ONTOLOGY]].rename(
        columns={SOURCE_SPEC.MODEL: "source", IDENTIFIERS.ONTOLOGY: "target"}
    )

    vertices_df = pd.concat(
        [
            _create_vertex_category(ontology_counts, SOURCE_SPEC.MODEL),
            _create_vertex_category(ontology_counts, IDENTIFIERS.ONTOLOGY),
        ]
    )

    g = ig.Graph.DictList(
        vertices=vertices_df.to_dict("records"), edges=edges_df.to_dict("records")
    )

    components = g.components(mode="weak")

    if len(components) > 1:

        full_label = list()
        for c in components:
            models = (
                vertices_df.iloc[c]
                .query(f"category == '{SOURCE_SPEC.MODEL}'")["name"]
                .to_list()
            )
            ontologies = (
                vertices_df.iloc[c]
                .query(f"category == '{IDENTIFIERS.ONTOLOGY}'")["name"]
                .to_list()
            )

            label = f"models: {models}\nontologies: {ontologies}"

            full_label.append(label)
        full_label = "\n\n".join(full_label)

        logger.error(
            f"The {entity_type} ontologies shared across models do not overlap and will result in an-unmixed consensus model:\n{full_label}"
        )

    return None


def _prepare_consensus_table(
    agg_table_harmonized: pd.DataFrame,
    table_schema: dict,
    cluster_consensus_identifiers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare a consensus table with one row per unique entity.

    Parameters:
    ----------
    agg_table_harmonized: pd.DataFrame
        Table with nameness scores and cluster assignments
    table_schema: dict
        Schema for the table
    cluster_consensus_identifiers: pd.DataFrame
        Consensus identifiers for each cluster

    Returns:
    ----------
    pd.DataFrame
        New consensus table with merged entities
    """
    # Sort by nameness score and keep one row per new entity ID
    agg_table_reduced = (
        agg_table_harmonized.reset_index(drop=True)
        .sort_values(["nameness_score"])
        .rename(columns={"new_id": table_schema[SCHEMA_DEFS.PK]})
        .groupby(table_schema[SCHEMA_DEFS.PK])
        .first()
        .drop("nameness_score", axis=1)
    )

    # Join in the consensus identifiers and drop the temporary cluster column
    new_id_table = (
        agg_table_reduced.drop(table_schema[SCHEMA_DEFS.ID], axis=1)
        .merge(cluster_consensus_identifiers, left_on="cluster", right_index=True)
        .drop("cluster", axis=1)
    )

    return new_id_table


def _prepare_identifier_edgelist(
    valid_identifiers: pd.DataFrame, sbml_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare an edgelist for clustering identifiers.

    Parameters:
    ----------
    valid_identifiers: pd.DataFrame
        Table of identifiers
    sbml_df: pd.DataFrame
        Original table of entities

    Returns:
    ----------
    pd.DataFrame
        Edgelist connecting entities to their identifiers
    """
    # Format identifiers as edgelist
    formatted_identifiers = utils.format_identifiers_as_edgelist(
        valid_identifiers, [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]
    )

    # Create a unique tag for each entity from the original index
    indexed_species_tags = (
        formatted_identifiers.reset_index()
        .set_index(formatted_identifiers.index.names, drop=False)[sbml_df.index.names]
        .astype(str)
        .apply("__".join, axis=1)
    )
    formatted_identifiers.loc[:, "model_spec"] = indexed_species_tags

    # Create edgelist that connects entities to identifiers
    id_edgelist = pd.concat(
        [
            formatted_identifiers[["ind", "id"]],
            # Add edges connecting model-specific instances to their identifiers
            formatted_identifiers[["model_spec", "id"]].rename(
                columns={"model_spec": "ind"}
            ),
        ]
    )

    return id_edgelist


def _prepare_member_table(
    sbml_dfs_dict: dict[str, SBML_dfs],
    defined_by: str,
    defined_lookup_tables: dict,
    table_schema: dict,
    defined_by_schema: dict,
    defining_attrs: list[str],
    table: str = SBML_DFS.REACTIONS,
) -> tuple[pd.DataFrame, str]:
    """
    Prepare a table of members and validate their structure.

    Parameters:
    ----------
    sbml_dfs_dict: dict[str, SBML_dfs]
        Dictionary of SBML_dfs from different models
    defined_by: str
        Name of the table whose entries define membership
    defined_lookup_tables: dict
        Lookup tables for updating IDs
    table_schema: dict
        Schema for the main table
    defined_by_schema: dict
        Schema for the defining table
    defining_attrs: list[str]
        Attributes that define a unique member
    table: str
        Name of the main table (default: REACTIONS)

    Returns:
    ----------
    tuple:
        - Updated aggregated table with member strings
        - Name of the foreign key
    """
    # Combine models into a single table
    agg_tbl = _unnest_SBML_df(sbml_dfs_dict, table=defined_by)

    # Update IDs using previously created lookup tables
    for k in defined_lookup_tables.keys():
        agg_tbl = (
            agg_tbl.merge(
                defined_lookup_tables[k],
                left_on=[SOURCE_SPEC.MODEL, k],
                right_index=True,
            )
            .drop(k, axis=1)
            .rename(columns={"new_id": k})
        )

    # Identify the foreign key
    defining_fk = set(defined_by_schema[SCHEMA_DEFS.FK]).difference(
        {table_schema[SCHEMA_DEFS.PK]}
    )

    if (
        len(defining_fk) != 1
        or len(defining_fk.intersection(set(defined_by_schema[SCHEMA_DEFS.FK]))) != 1
    ):
        raise ValueError(
            f"A foreign key could not be found in {defined_by} which was a primary key in {table}"
        )
    else:
        defining_fk = list(defining_fk)[0]

    # Validate defining attributes
    valid_defining_attrs = agg_tbl.columns.values.tolist()
    invalid_defining_attrs = [
        x for x in defining_attrs if x not in valid_defining_attrs
    ]

    if len(invalid_defining_attrs) != 0:
        raise ValueError(
            f"{', '.join(invalid_defining_attrs)} was not found; "
            f"valid defining_attrs are {', '.join(valid_defining_attrs)}"
        )

    # Create unique member strings
    agg_tbl["member"] = agg_tbl[defining_attrs].astype(str).apply("__".join, axis=1)

    return agg_tbl, defining_fk


def _reduce_to_consensus_ids(
    sbml_df: pd.DataFrame,
    table_schema: dict,
    pw_index: indices.PWIndex | None = None,
    defining_biological_qualifiers: list[str] = BQB_DEFINING_ATTRS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reduce a table of entities to unique entries based on consensus identifiers.

    This function clusters entities that share identifiers (as defined by the provided biological qualifiers)
    and produces a new table of unique entities, along with a lookup table mapping original entities to consensus IDs.

    Parameters
    ----------
    sbml_df : pd.DataFrame
        Table of entities from multiple models, with model in the index (as produced by _unnest_SBML_df).
    table_schema : dict
        Schema for the table being reduced.
    pw_index : indices.PWIndex, optional
        An index of all tables being aggregated (default: None).
    defining_biological_qualifiers : list[str], optional
        List of biological qualifier types which define distinct entities. Defaults to BQB_DEFINING_ATTRS.

    Returns
    -------
    new_id_table : pd.DataFrame
        Table matching the schema of one of the input models, with merged entities.
    lookup_table : pd.Series
        Series mapping the index of the aggregated entities to new consensus IDs.
    """
    # Step 1: Build consensus identifiers to create clusters of equivalent entities
    table_name = table_schema[SCHEMA_DEFS.TABLE]
    logger.debug(f"Building consensus identifiers for {table_name}")
    indexed_cluster, cluster_consensus_identifiers = _build_consensus_identifiers(
        sbml_df, table_schema, defining_biological_qualifiers
    )

    # Step 2: Join cluster information to the original table
    agg_table_harmonized = sbml_df.join(indexed_cluster)

    # Step 3: Create lookup table for entity IDs
    logger.debug(f"Creating lookup table for {table_name}")
    lookup_table = _create_entity_lookup_table(agg_table_harmonized, table_schema)

    # Step 4: Add nameness scores to help select representative names
    agg_table_harmonized = utils._add_nameness_score_wrapper(
        agg_table_harmonized, SCHEMA_DEFS.LABEL, table_schema
    )

    # Step 5: Prepare the consensus table with one row per unique entity
    logger.debug(f"Preparing consensus table for {table_name}")
    new_id_table = _prepare_consensus_table(
        agg_table_harmonized, table_schema, cluster_consensus_identifiers
    )

    # Step 6: Add source information if required
    if SCHEMA_DEFS.SOURCE in table_schema.keys():
        new_id_table = _add_consensus_sources(
            new_id_table, agg_table_harmonized, lookup_table, table_schema, pw_index
        )

    # Step 7: Validate the resulting table
    logger.debug(f"Validating consensus table for {table_name}")
    _validate_consensus_table(new_id_table, sbml_df)

    # Step 8: Validate that all primary keys in new_id_table are represented in lookup_table
    new_id_pks = set(new_id_table.index)
    lookup_new_ids = set(lookup_table.values)

    if new_id_pks != lookup_new_ids:

        missing_pks = new_id_pks - lookup_new_ids
        extra_pks = lookup_new_ids - new_id_pks

        raise ValueError(
            "The keys from the new_id_table and values from lookup_table do not match:"
            f"- There are {len(missing_pks)} keys from new_id_table are missing from lookup_table 'new_id' values"
            f"- There are {len(extra_pks)} keys from lookup_table which are missing from new_id_table"
        )

    return new_id_table, lookup_table


def _remove_no_rxn_pathways(
    no_rxn_pathway_ids, sbml_dfs_dict, compspec_lookup_table
) -> None:
    """
    Remove pathways which don't contribute reactions from the pw_index.

    Parameters
    ----------
    no_rxn_pathway_ids : list
        The pathway ids for models which should not have reactions. (i.e., models which are just species metadata like "Dogma")
    sbml_dfs_dict : dict
        The dictionary of SBML_dfs.
    compspec_lookup_table : pd.DataFrame
        The lookup table for compartmentalized species.

    Returns
    -------
    None
        Modifies objects in place.

    """

    if len(no_rxn_pathway_ids) == 0:
        return None
    else:
        logger.info(
            f"Removing {len(no_rxn_pathway_ids)} pathways which don't contribute reactions: {no_rxn_pathway_ids}"
        )

    # Remove from dictionary (will raise KeyError if missing)
    for pathway_id in no_rxn_pathway_ids:
        del sbml_dfs_dict[pathway_id]

    # Remove from DataFrame (will raise KeyError if missing)
    compspec_lookup_table.drop(
        no_rxn_pathway_ids, level=SOURCE_SPEC.MODEL, inplace=True
    )

    return None


def _report_consensus_merges(
    lookup_table: pd.Series,
    table_schema: dict,
    agg_tbl: pd.DataFrame | None = None,
    sbml_dfs_dict: dict[str, SBML_dfs] | None = None,
    n_example_merges: int = 3,
) -> None:
    """
    Report Consensus Merges

    Print a summary of merges that occurred

    Parameters:
    ----------
    lookup_table : pd.Series
        An index of "model" and the entities primary key with values of new_id
    table_schema : dict
        Schema of the table being merged
    agg_tbl : pd.DataFrame or None
        Contains the original model, primary keys and a label. Required if the primary key is not r_id (i.e., reactions)
    sbml_dfs_dict : pd.DataFrame or None
        The dict of full models across all models. Used to create reaction formulas if the primary key is r_id
    n_example_merges : int
        Number of example merges to report details on

    Returns:
    ----------
    None
    """

    entity_merge_num = lookup_table.value_counts()
    merged_entities = entity_merge_num[entity_merge_num != 1]

    if merged_entities.shape[0] == 0:
        logger.warning(f"No merging occurred for {table_schema[SCHEMA_DEFS.PK]}")
        return None

    if SCHEMA_DEFS.LABEL not in table_schema.keys():
        # we dont need to track unnamed species
        return None

    logger.info(
        f">>>> {merged_entities.sum()} {table_schema[SCHEMA_DEFS.PK]} entries merged into {merged_entities.shape[0]}"
    )

    merges_lookup = lookup_table[
        lookup_table.isin(merged_entities.index.tolist())
    ].reset_index()

    if table_schema[SCHEMA_DEFS.PK] == SBML_DFS.R_ID:
        logger.info(
            "Creating formulas for to-be-merged reactions to help with reporting merges of reactions"
            " with inconsistently named reactants"
        )
        if not isinstance(sbml_dfs_dict, dict):
            raise ValueError(
                f"sbml_dfs_dict was a {type(sbml_dfs_dict)} and must be a dict if the table_schema pk is r_id"
            )

        indexed_models = merges_lookup.set_index("model").sort_index()
        merges_dict = dict()
        for mod in indexed_models.index.unique():
            merges_dict[mod] = sbml_dfs_dict[mod].reaction_formulas(
                indexed_models.loc[mod][SBML_DFS.R_ID]
            )

        merge_labels = pd.concat(merges_dict, names=["model", SBML_DFS.R_ID]).rename(
            SCHEMA_DEFS.LABEL
        )

        # add labels to models + r_id
        merges_lookup = merges_lookup.merge(
            merge_labels, how="left", left_on=["model", SBML_DFS.R_ID], right_index=True
        )

        logger.info("Done creating reaction formulas")

    else:
        if type(agg_tbl) is not pd.DataFrame:
            raise ValueError(
                f"agg_tbl was a {type(agg_tbl)} and must be a pd.DataFrame if the table_schema pk is NOT r_id"
            )

        merges_lookup = merges_lookup.merge(
            agg_tbl[table_schema[SCHEMA_DEFS.LABEL]],
            left_on=["model", table_schema[SCHEMA_DEFS.PK]],
            right_index=True,
        ).rename(columns={table_schema[SCHEMA_DEFS.LABEL]: SCHEMA_DEFS.LABEL})

    indexed_merges_lookup = merges_lookup.set_index("new_id")

    # filter to entries with non-identical labels

    logger.info("Testing for identical formulas of to-be-merged reactions")

    index_label_counts = (
        indexed_merges_lookup["label"].drop_duplicates().index.value_counts()
    )
    inexact_merges = index_label_counts[index_label_counts > 1].index.tolist()

    if len(inexact_merges) == 0:
        logger.info("All merges names matched exactly")
    else:
        logger.warning(
            f"\n{len(inexact_merges)} merges were of entities with distinct names, including:\n"
        )

        inexact_merges_samples = random.sample(
            inexact_merges, min(len(inexact_merges), n_example_merges)
        )

        inexact_merge_collapses = (
            indexed_merges_lookup.loc[inexact_merges_samples][SCHEMA_DEFS.LABEL]
            .drop_duplicates()
            .groupby(level=0)
            .agg(" & ".join)
        )

        logger.warning("\n\n".join(inexact_merge_collapses.tolist()) + "\n")


def _resolve_reversibility(
    sbml_dfs_dict: dict[str, SBML_dfs],
    rxn_consensus_species: pd.DataFrame,
    rxn_lookup_table: pd.Series,
) -> pd.DataFrame:
    """
    For a set of merged reactions determine what their consensus reaction reversibilities are
    """

    agg_tbl = _unnest_SBML_df(sbml_dfs_dict, table=SBML_DFS.REACTIONS)

    if not all(agg_tbl[SBML_DFS.R_ISREVERSIBLE].isin([True, False])):
        invalid_levels = agg_tbl[~agg_tbl[SBML_DFS.R_ISREVERSIBLE].isin([True, False])][
            SBML_DFS.R_ISREVERSIBLE
        ].unique()
        raise ValueError(
            "One or more aggregated models included invalid values for r_isreversible in the reactions table: "
            f"{', '.join(invalid_levels)}"
        )

    # add new ids to aggregated reactions by indexes
    # map each new r_id to every distinct value of is_irreversible from reactions it originated from
    # in most cases there will only be a single level
    r_id_to_all_reversibilities = (
        agg_tbl.join(rxn_lookup_table)
        .reset_index()[["new_id", SBML_DFS.R_ISREVERSIBLE]]
        .rename({"new_id": SBML_DFS.R_ID}, axis=1)
        .drop_duplicates()
    )

    # when a reaction could be irreversible or reversible define it as reversible.
    r_id_reversibility = (
        r_id_to_all_reversibilities.sort_values(
            SBML_DFS.R_ISREVERSIBLE, ascending=False
        )
        .groupby(SBML_DFS.R_ID)
        .first()
    )

    # drop existing reversibility since it is selected arbitrarily and replace
    # with consensus reversibility which respects priorities
    rxns_w_reversibility = rxn_consensus_species.drop(
        SBML_DFS.R_ISREVERSIBLE, axis=1
    ).join(r_id_reversibility)

    if rxns_w_reversibility.shape[0] != rxn_consensus_species.shape[0]:
        raise ValueError(
            "rxns_w_reversibility and rxn_consensus_species must have the same number of rows"
        )
    if not all(rxns_w_reversibility[SBML_DFS.R_ISREVERSIBLE].isin([True, False])):
        raise ValueError(
            "All rxns_w_reversibility[R_ISREVERSIBLE] must be True or False"
        )

    return rxns_w_reversibility


def _update_foreign_keys(
    agg_tbl: pd.DataFrame, table_schema: dict, fk_lookup_tables: dict
) -> pd.DataFrame:
    """Update one or more foreign keys based on old-to-new foreign key lookup table(s)."""

    working_agg_tbl = agg_tbl.copy()

    for fk in table_schema[SCHEMA_DEFS.FK]:

        # Merge agg_tbl with lookup table for validation and FK updates
        full_merge = (
            working_agg_tbl[fk]
            .reset_index()
            .merge(
                fk_lookup_tables[fk],
                left_on=[SOURCE_SPEC.MODEL, fk],
                right_index=True,
                how="outer",
                indicator=True,
            )
        )

        # Find missing keys (present in agg_tbl but not in lookup table)
        missing_keys = full_merge[full_merge["_merge"] == "left_only"]
        if len(missing_keys) > 0:
            missing_pairs = missing_keys[[SOURCE_SPEC.MODEL, fk]].values.tolist()
            raise ValueError(
                f"{len(missing_keys)} keys from agg_tbl are missing from the {fk} lookup table: {missing_pairs[:5]}..."
            )

        # Find extra keys (present in lookup table but not used in agg_tbl)
        extra_keys = full_merge[full_merge["_merge"] == "right_only"]
        if len(extra_keys) > 0:
            extra_pairs = extra_keys[[SOURCE_SPEC.MODEL, fk]].values.tolist()
            raise ValueError(
                f"{len(extra_keys)} keys are present in the {fk} lookup table but not used in agg_tbl: {extra_pairs[:5]}..."
            )

        # Reuse the merge result for updated FKs (only successful matches, excluding right_only)
        updated_fks = (
            full_merge[full_merge["_merge"] == "both"]
            .drop([fk, "_merge"], axis=1)
            .rename(columns={"new_id": fk})
            .set_index([SOURCE_SPEC.MODEL, table_schema[SCHEMA_DEFS.PK]])
        )
        working_agg_tbl = working_agg_tbl.drop(columns=fk).join(updated_fks)

    if working_agg_tbl.shape[0] != agg_tbl.shape[0]:
        raise ValueError(
            "The output agg table had a different number of rows than the input agg table"
        )

    return working_agg_tbl


def _unnest_SBML_df(sbml_dfs_dict: dict[str, SBML_dfs], table: str) -> pd.DataFrame:
    """
    Unnest and concatenate a specific table from multiple SBML_dfs models.

    This function merges corresponding tables from a set of models into a single DataFrame,
    adding the model name as an index level.

    Parameters
    ----------
    sbml_dfs_dict : dict[str, SBML_dfs]
        A dictionary of SBML_dfs objects from different models, keyed by model name.
    table : str
        The name of the table to aggregate (e.g., 'species', 'reactions', 'compartments').

    Returns
    -------
    pd.DataFrame
        A concatenated table with a MultiIndex of model and entity ID.
    """

    # check that all sbml_dfs have the same schema
    table_schema = SBML_DFS_SCHEMA.SCHEMA[table]

    df_list = [
        getattr(sbml_dfs_dict[x], table).assign(model=x) for x in sbml_dfs_dict.keys()
    ]
    df_concat = pd.concat(df_list)

    # add model to index columns
    if df_concat.size != 0:
        df_concat_reset = df_concat.reset_index()
        df_concat = df_concat_reset.set_index(
            [SOURCE_SPEC.MODEL, table_schema[SCHEMA_DEFS.PK]]
        )

    return df_concat


def _validate_consensus_table(
    new_id_table: pd.DataFrame, sbml_df: pd.DataFrame
) -> None:
    """
    Validate that the new consensus table has the same structure as the original.

    Parameters:
    ----------
    new_id_table: pd.DataFrame
        Newly created consensus table
    sbml_df: pd.DataFrame
        Original table from which consensus was built

    Raises:
    ------
    ValueError
        If index names or columns don't match
    """
    # Check that the index names match
    if set(sbml_df.index.names).difference({SOURCE_SPEC.MODEL}) != set(
        new_id_table.index.names
    ):
        raise ValueError(
            f"The newly constructed id table's index does not match the inputs.\n"
            f"Expected index names: {sbml_df.index.names}\n"
            f"Actual index names: {new_id_table.index.names}"
        )

    # Check that the columns match
    if set(sbml_df) != set(new_id_table.columns):
        missing_in_new = set(sbml_df) - set(new_id_table.columns)
        extra_in_new = set(new_id_table.columns) - set(sbml_df)
        raise ValueError(
            "The newly constructed id table's variables do not match the inputs.\n"
            f"Expected columns: {list(sbml_df.columns)}\n"
            f"Actual columns: {list(new_id_table.columns)}\n"
            f"Missing in new: {missing_in_new}\n"
            f"Extra in new: {extra_in_new}"
        )


def _validate_merge_entity_data_create_consensus(
    entity_data_dict, an_entity_data_type, models_w_entity_data_type
):
    """
    Validate creating a consensus of entity data tables in cases where the same table is present in multiple models

    This function checks whether tables with the same entity data key can be reasonably merged (same index and column names) or whether they seem like apples-to-oranges.

    Parameters:
    ----------
    entity_data_dict: dict
        Dictionary containing all model's "an_entity_data_type" dictionaries
    an_entity_data_type: str
        The type of entity data to merge
    models_w_entity_data_type: list
        List of models with the same entity data type

    Returns:
    -------
    None

    Raises:
    ------
    ValueError:
        If the tables have different index or column names
    """

    # check that all tables have the same index and column names
    distinct_indices = {
        ", ".join(entity_data_dict[x][an_entity_data_type].index.names)
        for x in models_w_entity_data_type
    }
    if len(distinct_indices) > 1:
        raise ValueError(
            f"Multiple tables with the same {an_entity_data_type} cannot be combined"
            " because they have different index names:"
            f"{' & '.join(list(distinct_indices))}"
        )
    distinct_cols = {
        ", ".join(entity_data_dict[x][an_entity_data_type].columns.tolist())
        for x in models_w_entity_data_type
    }
    if len(distinct_cols) > 1:
        raise ValueError(
            f"Multiple tables with the same {an_entity_data_type} cannot be combined"
            " because they have different column names:"
            f"{' & '.join(list(distinct_cols))}"
        )

    return None


def _validate_meta_identifiers(meta_identifiers: pd.DataFrame) -> None:
    """Check Identifiers to make sure they aren't empty and flag cases where IDs are missing BQB terms."""

    if meta_identifiers.shape[0] == 0:
        raise ValueError(
            '"meta_identifiers" was empty; some identifiers should be present'
        )

    n_null = sum(meta_identifiers[IDENTIFIERS.BQB].isnull())
    if n_null > 0:
        msg = f"{n_null} identifiers were missing a bqb code and will not be mergeable"
        logger.warning(msg)

    return None
