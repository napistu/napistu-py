from __future__ import annotations

import logging
import os
import random

import pandas as pd
from tqdm import tqdm

from napistu import identifiers
from napistu import indices
from napistu import sbml_dfs_core
from napistu import sbml_dfs_utils
from napistu import source
from napistu import utils
from napistu.ingestion import sbml

from napistu.constants import SCHEMA_DEFS
from napistu.constants import SBML_DFS
from napistu.constants import SBML_DFS_SCHEMA
from napistu.constants import IDENTIFIERS
from napistu.constants import SOURCE_SPEC
from napistu.constants import BQB_DEFINING_ATTRS
from napistu.constants import VALID_BQB_TERMS

logger = logging.getLogger(__name__)
# set the level to show logger.info message
logging.basicConfig(level=logging.DEBUG)


def construct_consensus_model(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs],
    pw_index: indices.PWIndex,
    dogmatic: bool = True,
) -> sbml_dfs_core.SBML_dfs:
    """
    Construct a Consensus Model by merging shared entities across pathway models.

    This function takes a dictionary of pathway models and merges shared entities (compartments, species, reactions, etc.)
    into a single consensus model, using a set of rules for entity identity and merging.

    Parameters
    ----------
    sbml_dfs_dict : dict[str, sbml_dfs_core.SBML_dfs]
        A dictionary of SBML_dfs objects from different models, keyed by model name.
    pw_index : indices.PWIndex
        An index of all tables being aggregated, used for cross-referencing entities.
    dogmatic : bool, default=True
        If True, preserve genes, transcripts, and proteins as separate species. If False, merge them when possible.

    Returns
    -------
    sbml_dfs_core.SBML_dfs
        A consensus SBML_dfs object containing the merged model.
    """
    # Validate inputs
    logger.info("Reporting possible issues in component models")
    _check_sbml_dfs_dict(sbml_dfs_dict)
    assert isinstance(pw_index, indices.PWIndex)

    # Select valid BQB attributes based on dogmatic flag
    defining_biological_qualifiers = sbml_dfs_utils._dogmatic_to_defining_bqbs(dogmatic)

    # Step 1: Create consensus entities for all primary tables
    consensus_entities, lookup_tables = _create_consensus_entities(
        sbml_dfs_dict, pw_index, defining_biological_qualifiers
    )

    # Step 2: Create the consensus SBML_dfs object
    sbml_dfs = sbml_dfs_core.SBML_dfs(consensus_entities)  # type: ignore

    # Step 3: Add entity data from component models
    sbml_dfs = _add_entity_data(sbml_dfs, sbml_dfs_dict, lookup_tables)

    return sbml_dfs


def construct_sbml_dfs_dict(
    pw_index: pd.DataFrame, strict: bool = True
) -> dict[str, sbml_dfs_core.SBML_dfs]:
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

    Returns
    -------
    dict[str, sbml_dfs_core.SBML_dfs]
        A dictionary mapping model names to SBML_dfs objects.
    """

    sbml_dfs_dict = dict()
    for i in tqdm(pw_index.index.index.tolist()):
        pw_entry = pw_index.index.loc[i]
        logger.info(f"processing {pw_entry[SOURCE_SPEC.NAME]}")

        sbml_path = os.path.join(pw_index.base_path, pw_entry[SOURCE_SPEC.FILE])
        try:
            sbml_obj = sbml.SBML(sbml_path)
            sbml_dfs_dict[pw_entry[SOURCE_SPEC.PATHWAY_ID]] = sbml_dfs_core.SBML_dfs(
                sbml_obj
            )
        except ValueError as e:
            if strict:
                raise e
            logger.warning(
                f"{pw_entry[SOURCE_SPEC.NAME]} not successfully loaded:", exc_info=True
            )
    return sbml_dfs_dict


def unnest_SBML_df(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs], table: str
) -> pd.DataFrame:
    """
    Unnest and concatenate a specific table from multiple SBML_dfs models.

    This function merges corresponding tables from a set of models into a single DataFrame,
    adding the model name as an index level.

    Parameters
    ----------
    sbml_dfs_dict : dict[str, sbml_dfs_core.SBML_dfs]
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
        df_concat = df_concat.reset_index().set_index(
            [SOURCE_SPEC.MODEL, table_schema["pk"]]
        )

    return df_concat


def construct_meta_entities_identifiers(
    sbml_dfs_dict: dict,
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
    sbml_dfs_dict : dict[str, sbml_dfs_core.SBML_dfs]
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
    agg_tbl = unnest_SBML_df(sbml_dfs_dict, table=table)

    # since all sbml_dfs have the same schema pull out one schema for reference
    table_schema = SBML_DFS_SCHEMA.SCHEMA[table]

    # update foreign keys using provided lookup tables
    if "fk" in table_schema.keys():
        agg_tbl = _update_foreign_keys(agg_tbl, table_schema, fk_lookup_tables)

    new_id_table, lookup_table = reduce_to_consensus_ids(
        sbml_df=agg_tbl,
        table_schema=table_schema,
        pw_index=pw_index,
        defining_biological_qualifiers=defining_biological_qualifiers,
    )

    # logging merges that occurred
    report_consensus_merges(
        lookup_table, table_schema, agg_tbl=agg_tbl, n_example_merges=5
    )

    return new_id_table, lookup_table


def reduce_to_consensus_ids(
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
        Table of entities from multiple models, with model in the index (as produced by unnest_SBML_df).
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
    indexed_cluster, cluster_consensus_identifiers = build_consensus_identifiers(
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

    return new_id_table, lookup_table


def build_consensus_identifiers(
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
        Table of entities from multiple models, with model in the index (as produced by unnest_SBML_df).
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
    meta_identifiers = sbml_dfs_utils.unnest_identifiers(
        sbml_df, table_schema[SCHEMA_DEFS.ID]
    )
    _validate_meta_identifiers(meta_identifiers)

    # Step 2: Filter identifiers by biological qualifier type
    valid_identifiers = _filter_identifiers_by_qualifier(
        meta_identifiers, defining_biological_qualifiers
    )

    # Step 3: Handle entries that don't have identifiers
    valid_identifiers = _handle_entries_without_identifiers(sbml_df, valid_identifiers)

    # Step 4: Prepare edgelist for clustering
    id_edgelist = _prepare_identifier_edgelist(valid_identifiers, sbml_df)

    # Step 5: Cluster entities based on shared identifiers
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


def pre_consensus_ontology_check(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs], tablename: str
) -> tuple[list, pd.DataFrame]:
    """
    Check for shared ontologies across source models for a given table.

    For compartments, species, or reactions tables, this function returns the set of ontologies
    shared among all SBML_dfs in the input dictionary, as well as a DataFrame summarizing ontologies per model.

    Parameters
    ----------
    sbml_dfs_dict : dict[str, sbml_dfs_core.SBML_dfs]
        Dictionary of SBML_dfs objects from different models, keyed by model name.
    tablename : str
        Name of the table to check (should be one of 'compartments', 'species', or 'reactions').

    Returns
    -------
    shared_onto_list : list
        List of ontologies shared by all models for the specified table.
    sbml_dict_onto_df : pd.DataFrame
        DataFrame summarizing ontologies present in each model for the specified table.
    """

    # tablename: compartments/species/reactions tables with Identifiers
    # returns shared ontologies among sbml_dfs in sbml_dfs_dict for
    # compartments/species/reactions tables

    if tablename in [SBML_DFS.COMPARTMENTS, SBML_DFS.SPECIES, SBML_DFS.REACTIONS]:
        sbml_onto_lists = []
        for df_key, sbml_dfs_ind in sbml_dfs_dict.items():
            sbml_onto_df_ind = sbml_dfs_ind.get_identifiers(tablename).value_counts(
                IDENTIFIERS.ONTOLOGY
            )
            sbml_onto_lists.append(sbml_onto_df_ind.index.to_list())

        shared_onto_set = set.intersection(*map(set, sbml_onto_lists))
        shared_onto_list = list(shared_onto_set)

        sbml_name_list = list(sbml_dfs_dict.keys())
        sbml_dict_onto_df = pd.DataFrame({"single_sbml_dfs": sbml_name_list})
        sbml_dict_onto_df[IDENTIFIERS.ONTOLOGY] = sbml_onto_lists

    else:
        logger.error(
            f"{tablename} entry doesn't have identifiers and thus cannot check its ontology"
        )
        shared_onto_list = []
        sbml_dict_onto_df = []

    logger.info(
        f"Shared ontologies for {tablename} are {shared_onto_list} before building a consensus model."
    )

    return shared_onto_list, sbml_dict_onto_df


def post_consensus_species_ontology_check(sbml_dfs: sbml_dfs_core.SBML_dfs) -> set[str]:
    """
    Check and return the set of ontologies shared by different sources in a consensus model's species table.

    This function examines the species table in a consensus SBML_dfs object, determines the ontologies
    present for each source model, and returns the intersection of ontologies shared by all sources.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The consensus SBML_dfs object containing merged species from multiple models.

    Returns
    -------
    set[str]
        Set of ontology terms shared by all sources in the consensus model's species table.
    """
    # Checking the ontology in "species" shared by different sources in a consensus model
    # returns a set of shared ontologies by different sources

    consensus_sbmldf_tbl_var = sbml_dfs.get_identifiers(SBML_DFS.SPECIES)

    # get the sources of species in the consensus model
    consensus_sbmldf_tbl_var_sc = (
        source.unnest_sources(sbml_dfs.species, verbose=False)
        .reset_index()
        .sort_values([SOURCE_SPEC.NAME])
    )

    # merge columns with source info to the model's species identifiers df.
    consensus_sbmldf_tbl_var_w_sc = consensus_sbmldf_tbl_var.merge(
        consensus_sbmldf_tbl_var_sc.loc[
            :,
            [
                SBML_DFS.S_ID,
                SOURCE_SPEC.MODEL,
                SOURCE_SPEC.FILE,
                SOURCE_SPEC.PATHWAY_ID,
                SOURCE_SPEC.SOURCE,
                SOURCE_SPEC.NAME,
            ],
        ],
        on=SBML_DFS.S_ID,
    )

    # get the model/source and its ontology set to a separate df
    shared_ontology_df = (
        consensus_sbmldf_tbl_var_w_sc.groupby(SOURCE_SPEC.NAME)[IDENTIFIERS.ONTOLOGY]
        .apply(set)
        .reset_index(name="onto_expanded")
    )

    # the intersection set among ontology sets of all sources
    shared_onto_set = shared_ontology_df.onto_expanded[0]
    for i in range(1, len(shared_ontology_df.onto_expanded)):
        shared_onto_set = shared_onto_set.intersection(
            shared_ontology_df.onto_expanded[i]
        )

    logger.info(f"shared ontologies in the consesus model are: {shared_onto_set}")

    return shared_onto_set


def pre_consensus_compartment_check(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs], tablename: str
) -> tuple[list, dict]:
    """Find compartments shared across models."""

    # tablename: compartments only
    # returns shared c_name in compartments of sbml_dfs in sbml_dfs_dict for

    if tablename in [SBML_DFS.COMPARTMENTS]:
        sbml_cname_list = []
        for df_key, sbml_dfs_ind in sbml_dfs_dict.items():
            sbml_df_ind_cname = sbml_dfs_ind.get_identifiers(tablename).value_counts(
                SBML_DFS.C_NAME
            )
            sbml_cname_list.append(sbml_df_ind_cname.index.to_list())

        shared_cname_set = set.intersection(*map(set, sbml_cname_list))
        shared_cname_list = list(shared_cname_set)

        sbml_name_list = list(sbml_dfs_dict.keys())
        sbml_dict_cname_df = pd.DataFrame({"single_sbml_dfs": sbml_name_list})
        sbml_dict_cname_df["c_names"] = sbml_cname_list

    else:
        logger.error(f"{tablename} entry doesn't have c_name")

    logger.info(
        f"Shared compartments for {tablename} are {shared_cname_list} before building a consensus model."
    )

    return shared_cname_list, sbml_dict_cname_df


def post_consensus_source_check(
    sbml_dfs: sbml_dfs_core.SBML_dfs, table_name: str
) -> pd.DataFrame:
    """Provide sources of tables in a consensus model; the output df will be used to determine whether models are merged."""

    table_pk = sbml_dfs.schema[table_name][SCHEMA_DEFS.PK]

    sbml_dfs_tbl = getattr(sbml_dfs, table_name)
    sbml_dfs_tbl_pathway_source = (
        source.unnest_sources(sbml_dfs_tbl, verbose=False)
        .reset_index()
        .sort_values(["name"])
    )

    sbml_dfs_tbl_pathway_source["pathway"] = sbml_dfs_tbl_pathway_source.groupby(
        [table_pk]
    )["name"].transform(lambda x: " + ".join(set(x)))

    sbml_dfs_tbl_pathway_source = (
        sbml_dfs_tbl_pathway_source[[table_pk, "pathway"]]
        .drop_duplicates()
        .set_index(table_pk)
    )

    tbl_pathway_source_df = pd.DataFrame(
        sbml_dfs_tbl_pathway_source["pathway"].value_counts()
    )

    return tbl_pathway_source_df


def construct_meta_entities_fk(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs],
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
    sbml_df_dict: dict{"model": cpr.SBML_dfs}
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
    agg_tbl = unnest_SBML_df(sbml_dfs_dict, table=table)

    # since all sbml_dfs have the same schema pull out one schema for reference
    table_schema = sbml_dfs_dict[list(sbml_dfs_dict.keys())[0]].schema[table]

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
    report_consensus_merges(
        lookup_table, table_schema, agg_tbl=agg_tbl, n_example_merges=5
    )

    if SCHEMA_DEFS.SOURCE in table_schema.keys():
        # track the model(s) that each entity came from
        new_sources = create_consensus_sources(
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


def construct_meta_entities_members(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs],
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
    sbml_df_dict: dict{"model": cpr.SBML_dfs}
        A dictionary of cpr.SBML_dfs
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
    report_consensus_merges(
        lookup_table, table_schema, sbml_dfs_dict=sbml_dfs_dict, n_example_merges=5
    )

    # Step 6: Get primary entity table and merge identifiers
    agg_primary_table = unnest_SBML_df(sbml_dfs_dict, table=table)

    logger.info(f"Merging {table} identifiers")
    updated_identifiers = _merge_entity_identifiers(
        agg_primary_table, lookup_table, table_schema
    )

    # Step 7: Create consensus table with merged entities
    new_id_table = _create_consensus_table(
        agg_primary_table, lookup_table, updated_identifiers, table_schema
    )

    # Step 8: Add source information if present
    if "source" in table_schema.keys():
        logger.info(f"Merging {table} sources")

        # Track the model(s) that each entity came from
        new_sources = create_consensus_sources(
            agg_primary_table.merge(lookup_table, left_index=True, right_index=True),
            lookup_table,
            table_schema,
            pw_index,
        )

        new_id_table = new_id_table.drop(table_schema["source"], axis=1).merge(
            new_sources, left_index=True, right_index=True
        )

    return new_id_table, lookup_table


def create_consensus_sources(
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
        .rename(columns={"new_id": table_schema["pk"]})
        .groupby(table_schema["pk"])[table_schema["source"]]
    )

    # combine old sources into a single source.Source object per index value
    aggregated_old_sources = indexed_old_sources.agg(source.merge_sources)

    aligned_sources = new_sources.merge(
        aggregated_old_sources, left_index=True, right_index=True
    )
    assert isinstance(aligned_sources, pd.DataFrame)

    logger.info("Returning new source table")
    new_sources = aligned_sources.apply(source.merge_sources, axis=1).rename(table_schema["source"])  # type: ignore
    assert isinstance(new_sources, pd.Series)

    return new_sources


def report_consensus_merges(
    lookup_table: pd.Series,
    table_schema: dict,
    agg_tbl: pd.DataFrame | None = None,
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs] | None = None,
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
        logger.warning(f"No merging occurred for {table_schema['pk']}")
        return None

    if "label" not in table_schema.keys():
        # we dont need to track unnamed species
        return None

    logger.info(
        f">>>> {merged_entities.sum()} {table_schema['pk']} entries merged into {merged_entities.shape[0]}"
    )

    merges_lookup = lookup_table[
        lookup_table.isin(merged_entities.index.tolist())
    ].reset_index()

    if table_schema["pk"] == "r_id":
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
                indexed_models.loc[mod]["r_id"]
            )

        merge_labels = pd.concat(merges_dict, names=["model", "r_id"]).rename("label")

        # add labels to models + r_id
        merges_lookup = merges_lookup.merge(
            merge_labels, how="left", left_on=["model", "r_id"], right_index=True
        )

        logger.info("Done creating reaction formulas")

    else:
        if type(agg_tbl) is not pd.DataFrame:
            raise ValueError(
                f"agg_tbl was a {type(agg_tbl)} and must be a pd.DataFrame if the table_schema pk is NOT r_id"
            )

        merges_lookup = merges_lookup.merge(
            agg_tbl[table_schema["label"]],
            left_on=["model", table_schema["pk"]],
            right_index=True,
        ).rename(columns={table_schema["label"]: "label"})

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
            indexed_merges_lookup.loc[inexact_merges_samples]["label"]
            .drop_duplicates()
            .groupby(level=0)
            .agg(" & ".join)
        )

        logger.warning("\n\n".join(inexact_merge_collapses.tolist()) + "\n")

    logger.info("==============================\n")

    return None


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
        .rename(columns={"new_id": table_schema["pk"]})
        .groupby(table_schema["pk"])
        .first()
        .drop("nameness_score", axis=1)
    )

    # Join in the consensus identifiers and drop the temporary cluster column
    new_id_table = (
        agg_table_reduced.drop(table_schema["id"], axis=1)
        .merge(cluster_consensus_identifiers, left_on="cluster", right_index=True)
        .drop("cluster", axis=1)
    )

    return new_id_table


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
    new_sources = create_consensus_sources(
        agg_table_harmonized, lookup_table, table_schema, pw_index
    )
    assert isinstance(new_sources, pd.Series)

    # Add the sources to the consensus table
    updated_table = new_id_table.drop(table_schema[SOURCE_SPEC.SOURCE], axis=1).merge(
        new_sources, left_index=True, right_index=True
    )

    return updated_table


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


def merge_entity_data(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs],
    lookup_table: pd.Series,
    table: str,
) -> dict:
    """
    Merge Entity Data

    Report cases where a single "new" id is associated with multiple different values of entity_var

    Args
        sbml_dfs_dict (dict): dictionary where values are to-be-merged model nnames and values
          are sbml_dfs_core.SBML_dfs
        lookup_table (pd.Series): a series where the index is an old model and primary key and the
          value is the new consensus id
        table (str): table whose data is being consolidates (currently species or reactions)

    Returns:
        entity_data (dict): dictionary containing pd.DataFrames which aggregate all of the
          individual entity_data tables in "sbml_dfs_dict"

    """

    entity_schema = sbml_dfs_dict[list(sbml_dfs_dict.keys())[0]].schema[table]
    data_table_name = table + "_data"

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


def _create_consensus_entities(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs],
    pw_index: indices.PWIndex,
    defining_biological_qualifiers: list[str],
) -> tuple[dict, dict]:
    """
    Create consensus entities for all primary tables in the model.

    This helper function creates consensus compartments, species, compartmentalized species,
    reactions, and reaction species by finding shared entities across source models.

    Parameters:
    ----------
    sbml_dfs_dict: dict{cpr.SBML_dfs}
        A dictionary of SBML_dfs from different models
    pw_index: indices.PWIndex
        An index of all tables being aggregated
    defining_biological_qualifiers: list[str]
        Biological qualifier terms that define distinct entities

    Returns:
    ----------
    tuple:
        - dict of consensus entities tables
        - dict of lookup tables
    """
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
        defining_attrs=[SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY],
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


def _add_entity_data(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs],
    lookup_tables: dict,
) -> sbml_dfs_core.SBML_dfs:
    """
    Add entity data from component models to the consensus model.

    Parameters:
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        The consensus model being built
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs]
        A dictionary of SBML_dfs from different models
    lookup_tables: dict
        Dictionary of lookup tables for translating between old and new entity IDs

    Returns:
    ----------
    sbml_dfs_core.SBML_dfs
        The updated consensus model
    """
    # Add species data
    consensus_species_data = merge_entity_data(
        sbml_dfs_dict,
        lookup_table=lookup_tables[SBML_DFS.SPECIES],
        table=SBML_DFS.SPECIES,
    )
    for k in consensus_species_data.keys():
        sbml_dfs.add_species_data(k, consensus_species_data[k])

    # Add reactions data
    consensus_reactions_data = merge_entity_data(
        sbml_dfs_dict,
        lookup_table=lookup_tables[SBML_DFS.REACTIONS],
        table=SBML_DFS.REACTIONS,
    )
    for k in consensus_reactions_data.keys():
        sbml_dfs.add_reactions_data(k, consensus_reactions_data[k])

    return sbml_dfs


def _prepare_member_table(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs],
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
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs]
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
    agg_tbl = unnest_SBML_df(sbml_dfs_dict, table=defined_by)

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
    defining_fk = set(defined_by_schema["fk"]).difference({table_schema["pk"]})

    if (
        len(defining_fk) != 1
        or len(defining_fk.intersection(set(defined_by_schema["fk"]))) != 1
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
        .rename(columns={"new_id": table_schema["pk"]})
        .groupby(table_schema["pk"])[table_schema["id"]]
    )

    # Merge identifier objects
    return indexed_old_identifiers.agg(identifiers.merge_identifiers)


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
    filtered_entries[IDENTIFIERS.URL] = None
    filtered_entries[IDENTIFIERS.BQB] = None

    filtered_entries = filtered_entries.set_index(
        sbml_df.index.names + [SOURCE_SPEC.ENTRY]
    )

    # Combine original valid identifiers with dummy identifiers
    return pd.concat([valid_identifiers, filtered_entries])


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
        for k, v in all_cluster_identifiers.groupby("cluster")
    }

    # Handle clusters that don't have any identifiers
    catchup_clusters = {
        c: identifiers.Identifiers(list())
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


def _check_sbml_dfs_dict(sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs]) -> None:
    """Check models in SBML_dfs for problems which can be reported up-front

    Args:
        sbml_dfs_dict (dict(pd.DataFrame)): a dict of sbml_dfs models;
        primarily used as an input for construct_consensus_model

    Returns:
        None

    """

    for k, v in sbml_dfs_dict.items():
        _check_sbml_dfs(sbml_dfs=v, model_label=k)
    return None


def _check_sbml_dfs(
    sbml_dfs: sbml_dfs_core.SBML_dfs, model_label: str, N_examples: int | str = 5
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


def _validate_meta_identifiers(meta_identifiers: pd.DataFrame) -> None:
    """Check Identifiers to make sure they aren't empty and flag cases where IDs are missing BQB terms."""

    if meta_identifiers.shape[0] == 0:
        raise ValueError(
            '"meta_identifiers" was empty; some identifiers should be present'
        )

    n_null = sum(meta_identifiers[IDENTIFIERS.BQB].isnull())
    if n_null > 0:
        msg = f"{n_null} identifiers were missing a bqb code and will not be mergeable"
        logger.warn(msg)

    return None


def _validate_meta_identifiers(meta_identifiers: pd.DataFrame) -> None:
    """Flag cases where meta identifers are totally missing or BQB codes are not included"""

    if meta_identifiers.shape[0] == 0:
        raise ValueError(
            '"meta_identifiers" was empty; some identifiers should be present'
        )

    n_null = sum(meta_identifiers["bqb"].isnull())
    if n_null > 0:
        msg = f"{n_null} identifiers were missing a bqb code and will not be mergeable"
        logger.warn(msg)

    return None


def _update_foreign_keys(
    agg_tbl: pd.DataFrame, table_schema: dict, fk_lookup_tables: dict
) -> pd.DataFrame:
    for fk in table_schema["fk"]:
        updated_fks = (
            agg_tbl[fk]
            .reset_index()
            .merge(
                fk_lookup_tables[fk], left_on=[SOURCE_SPEC.MODEL, fk], right_index=True
            )
            .drop(fk, axis=1)
            .rename(columns={"new_id": fk})
            .set_index(["model", table_schema["pk"]])
        )
        agg_tbl = agg_tbl.drop(columns=fk).join(updated_fks)

    return agg_tbl


def _update_foreign_keys(
    agg_tbl: pd.DataFrame, table_schema: dict, fk_lookup_tables: dict
) -> pd.DataFrame:
    """Update one or more foreign keys based on old-to-new foreign key lookup table(s)."""

    for fk in table_schema["fk"]:
        updated_fks = (
            agg_tbl[fk]
            .reset_index()
            .merge(
                fk_lookup_tables[fk], left_on=[SOURCE_SPEC.MODEL, fk], right_index=True
            )
            .drop(fk, axis=1)
            .rename(columns={"new_id": fk})
            .set_index(["model", table_schema["pk"]])
        )
        agg_tbl = agg_tbl.drop(columns=fk).join(updated_fks)

    return agg_tbl


def _resolve_reversibility(
    sbml_dfs_dict: dict[str, sbml_dfs_core.SBML_dfs],
    rxn_consensus_species: pd.DataFrame,
    rxn_lookup_table: pd.Series,
) -> pd.DataFrame:
    """
    For a set of merged reactions determine what their consensus reaction reversibilities are
    """

    agg_tbl = unnest_SBML_df(sbml_dfs_dict, table=SBML_DFS.REACTIONS)

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

    Args
        entity_data_dict (dict): dictionary containing all model's "an_entity_data_type" dictionaries
        lookup_table (pd.Series): a series where the index is an old model and primary key and the
          value is the new consensus id
        entity_schema (dict): schema for "table"
        an_entity_data_type (str): data_type from species/reactions_data in entity_data_dict
        table (str): table whose data is being consolidates (currently species or reactions)

    Returns:
        consensus_entity_data (pd.DataFrame) table where index is primary key of "table" and
          values are all distinct annotations from "an_entity_data_type".

    """

    models_w_entity_data_type = [
        k for k, v in entity_data_dict.items() if an_entity_data_type in v.keys()
    ]

    logger.info(
        f"Merging {len(models_w_entity_data_type)} models with {an_entity_data_type} data in the {table} table"
    )

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

    # stack all models
    combined_entity_data = pd.concat(
        {k: entity_data_dict[k][an_entity_data_type] for k in models_w_entity_data_type}
    )
    combined_entity_data.index.names = ["model", entity_schema["pk"]]
    if isinstance(combined_entity_data, pd.Series):
        # enforce that atttributes should always be DataFrames
        combined_entity_data = combined_entity_data.to_frame()

    # create a table indexed by the NEW primary key containing all the entity data of type an_entity_data_type
    # right now the index may map to multiple rows if entities were consolidated
    combined_entity_data = (
        combined_entity_data.join(lookup_table)
        .reset_index(drop=True)
        .rename({"new_id": entity_schema["pk"]}, axis=1)
        .set_index(entity_schema["pk"])
        .sort_index()
    )

    # report cases where merges produce id-variable combinations with distinct values
    _merge_entity_data_report_mismatches(
        combined_entity_data, entity_schema, an_entity_data_type, table
    )

    # save one value for each id-variable combination
    # (this will accept the first value regardless of the above mismatches.)
    consensus_entity_data = (
        combined_entity_data.reset_index().groupby(entity_schema["pk"]).first()
    )

    return consensus_entity_data


def _merge_entity_data_report_mismatches(
    combined_entity_data: pd.DataFrame,
    entity_schema: dict,
    an_entity_data_type: str,
    table: str,
) -> None:
    """
    Merge Entity Data - Report Mismatches

    Report cases where a single "new" id is associated with multiple different values of entity_var

    Args
        combined_entity_data (pd.DataFrame): indexed by table primary key containing all
          data from "an_entity_data_type"
        entity_schema (dict): schema for "table"
        an_entity_data_type (str): data_type from species/reactions_data in combined_entity_data
        table (str): table whose data is being consolidates (currently species or reactions)

    Returns:
        None

    """

    data_table_name = table + "_data"

    entity_vars = combined_entity_data.columns
    for entity_var in entity_vars:
        unique_counts = (
            combined_entity_data.reset_index()
            .groupby(entity_schema["pk"])
            .agg("nunique")
        )
        entities_w_imperfect_matches = unique_counts[
            unique_counts[entity_var] != 1
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


def _create_member_string(x: list[str]) -> str:
    x.sort()
    return "_".join(x)
