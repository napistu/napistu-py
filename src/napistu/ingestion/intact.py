from __future__ import annotations

import datetime
import logging
import os
from typing import Dict, Set, Tuple, Union

import igraph as ig
import numpy as np
import pandas as pd
import requests

from napistu import sbml_dfs_utils, utils
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    ONTOLOGIES_LIST,
    SBML_DFS,
    SBOTERM_NAMES,
)
from napistu.identifiers import Identifiers, parse_ensembl_id
from napistu.ingestion.constants import (
    DATA_SOURCE_DESCRIPTIONS,
    DATA_SOURCES,
    DEFAULT_INTACT_RELATIVE_WEIGHTS,
    INTACT_EXPERIMENTAL_ROLES,
    INTACT_ONTOLOGY_ALIASES,
    INTACT_PUBLICATION_SCORE_THRESHOLD,
    INTACT_SCORES,
    INTACT_TERM_SCORES,
    INTERACTION_EDGELIST_DEFS,
    PSI_MI_DEFS,
    PSI_MI_INTACT_FTP_URL,
    PSI_MI_INTACT_SPECIES_TO_BASENAME,
    PSI_MI_MISSING_VALUE_STR,
    PSI_MI_ONTOLOGY_URL,
    PSI_MI_SCORED_TERMS,
    PSI_MI_STUDY_TABLES,
    PSI_MI_STUDY_TABLES_LIST,
    VALID_INTACT_EXPERIMENTAL_ROLES,  # noqa: F401
    VALID_INTACT_SECONDARY_ONTOLOGIES,
)
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator
from napistu.ontologies import renaming
from napistu.sbml_dfs_core import SBML_dfs
from napistu.source import Source

logger = logging.getLogger(__name__)


def download_intact_xmls(
    output_dir_path: str,
    organismal_species: Union[str, OrganismalSpeciesValidator],
    overwrite: bool = False,
) -> None:
    """
    Download IntAct Species

    Download the PSM-30 XML files from IntAct for a species of interest.

    Parameters
    ----------
    output_dir_path : str
        Local directory to create an unzip files into
    latin_species : str
        The species name (e.g., "Homo sapiens") to work with
    overwrite : bool, optional
        Overwrite an existing output directory, by default False

    Returns
    -------
    None
        Files are downloaded and extracted to the specified directory
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)
    intact_species_basename = _get_intact_species_basename(
        organismal_species.latin_name
    )

    intact_species_url = os.path.join(
        PSI_MI_INTACT_FTP_URL, f"{intact_species_basename}.zip"
    )

    logger.info(f"Downloading and unzipping {intact_species_url}")

    utils.download_and_extract(
        intact_species_url,
        output_dir_path=output_dir_path,
        download_method="ftp",
        overwrite=overwrite,
    )


def intact_to_sbml_dfs(
    intact_summaries: dict[str, pd.DataFrame],
    organismal_species: Union[str, OrganismalSpeciesValidator],
) -> SBML_dfs:
    """
    Convert IntAct summaries to SBML_dfs

    Parameters
    ----------
    intact_summaries : dict[str, pd.DataFrame]
        A dictionary of IntAct summaries.
    organismal_species : str | OrganismalSpeciesValidator
        The organismal species pertaining to the IntAct interactions

    Returns
    -------
    sbml_dfs : SBML_dfs
        SBML_dfs object containing the converted IntAct data

    Raises
    ------
    ValueError
        If intact_summaries does not contain the required tables
    ValueError
        If the provided species is not supported by IntAct
    ValueError
        If ontologies listed as valid secondary references are not in the Napistu controlled vocabulary
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)

    if set(intact_summaries.keys()) != set(PSI_MI_STUDY_TABLES_LIST):
        raise ValueError(
            f"IntAct summaries must contain the following tables: {PSI_MI_STUDY_TABLES_LIST}"
        )

    # format model-level metadata
    model_source = Source.single_entry(
        model=DATA_SOURCES.INTACT,
        pathway_id=DATA_SOURCES.INTACT,
        data_source=DATA_SOURCES.INTACT,
        organismal_species=organismal_species.latin_name,
        name=DATA_SOURCE_DESCRIPTIONS[DATA_SOURCES.INTACT],
        date=datetime.date.today().strftime("%Y%m%d"),
    )

    aliases = renaming.OntologySet(ontologies=INTACT_ONTOLOGY_ALIASES).ontologies
    alias_mapping = renaming._create_alias_mapping(aliases)

    valid_intact_xrefs = _filter_intact_xrefs(
        intact_summaries,
        alias_mapping,
        organismal_species,
        VALID_INTACT_SECONDARY_ONTOLOGIES,
    )

    # filter to entries with valid xrefs
    valid_interactors = intact_summaries[PSI_MI_STUDY_TABLES.SPECIES].merge(
        valid_intact_xrefs[
            [PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTOR_ID]
        ].drop_duplicates(),
        how="inner",
    )

    lookup_table, species_df = _create_species_df(
        valid_interactors, valid_intact_xrefs, organismal_species.latin_name
    )

    # turn reaction_species into a bait <-> prey edgelist
    basic_edgelist = _create_basic_edgelist(intact_summaries, lookup_table)

    # drop species that are not in the edgelist
    defined_interactors = set(
        basic_edgelist[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM]
    ) | set(basic_edgelist[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM])
    used_species_mask = species_df[SBML_DFS.S_NAME].isin(defined_interactors)

    if len(species_df[~used_species_mask]) > 0:
        logger.warning(
            f"Dropping {sum(~used_species_mask)} species that are not in the edgelist"
        )
        species_df = species_df[used_species_mask]

    # add metadata
    edgelist_w_study_metadata = basic_edgelist.merge(
        intact_summaries[PSI_MI_STUDY_TABLES.STUDY_LEVEL_DATA][
            [
                PSI_MI_DEFS.STUDY_ID,
                PSI_MI_DEFS.INTERACTION_METHOD,
                IDENTIFIERS.ONTOLOGY,
                IDENTIFIERS.IDENTIFIER,
            ]
        ]
    )

    # drop entries no study id
    invalid_study_id_mask = edgelist_w_study_metadata[IDENTIFIERS.ONTOLOGY] == ""
    if sum(invalid_study_id_mask) > 0:
        logger.warning(
            f"Dropping {sum(invalid_study_id_mask)} interactions with no study id"
        )
        edgelist_w_study_metadata = edgelist_w_study_metadata[~invalid_study_id_mask]

    # convert from specified attributes to standardized attributes with associated scores
    standardized_interaction_attrs = _standardize_interaction_attrs(
        edgelist_w_study_metadata
    )
    # deduplicate and count the number of studies with scored attributes
    scored_attribute_counts = _count_studies_with_scored_attributes(
        standardized_interaction_attrs
    )

    # create r_Identifiers objects and count citations and attributes
    interaction_edgelist_df_ids_and_counts = _define_edgelist_df_ids_and_counts(
        edgelist_w_study_metadata, alias_mapping, scored_attribute_counts
    )

    # calculate publication, method, and type scores and weight them for the composite score
    interaction_scores = _calculate_all_scores_vectorized(
        scored_attribute_counts,
        interaction_edgelist_df_ids_and_counts[INTACT_SCORES.N_PUBLICATIONS],
    )

    interactions_edgelist = (
        edgelist_w_study_metadata[
            [
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                PSI_MI_DEFS.INTERACTION_NAME,
            ]
        ]
        .rename(columns={PSI_MI_DEFS.INTERACTION_NAME: SBML_DFS.R_NAME})
        .groupby(
            [
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ]
        )
        .first()
        .reset_index()
        .merge(
            interaction_scores,
            on=[
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ],
            how="left",
        )
        .merge(
            interaction_edgelist_df_ids_and_counts,
            on=[
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ],
            how="left",
        )
    )

    sbml_dfs = SBML_dfs.from_edgelist(
        interactions_edgelist,
        species_df,
        compartments_df=sbml_dfs_utils.stub_compartments(),
        model_source=model_source,
        interaction_edgelist_defaults={
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: SBOTERM_NAMES.INTERACTOR,
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: SBOTERM_NAMES.INTERACTOR,
            SBML_DFS.R_ISREVERSIBLE: True,
        },
        keep_reactions_data=DATA_SOURCES.INTACT,
        force_edgelist_consistency=True,
    )

    return sbml_dfs


# =============================================================================
# UTILITY FUNCTIONS (alphabetically ordered)
# =============================================================================


def _build_psi_mi_ontology_graph(ontology_url: str = PSI_MI_ONTOLOGY_URL) -> ig.Graph:
    """Parse MI ontology JSON from URL and build igraph directed graph."""
    # Fetch JSON from URL
    response = requests.get(ontology_url)
    response.raise_for_status()
    ontology_json = response.json()

    # Collect all nodes and edges
    nodes = []
    edges = []

    def parse_node(node, parent_name=None):
        name = node["name"]
        nodes.append(name)

        # Add edge from parent to child (directed)
        if parent_name:
            edges.append((parent_name, name))

        # Recursively parse children
        for child in node.get("children", []):
            parse_node(child, parent_name=name)

    # Parse the root node
    parse_node(ontology_json)

    # Create igraph
    g = ig.Graph.TupleList(edges, directed=True)
    # Add any isolated nodes
    for node in set(nodes) - set(g.vs["name"]):
        g.add_vertex(node)

    return g


def _calculate_all_scores_vectorized(
    counts_df: pd.DataFrame,
    n_publications_series: pd.Series,
    max_pubs: int = INTACT_PUBLICATION_SCORE_THRESHOLD,
    weights: Dict[str, float] = DEFAULT_INTACT_RELATIVE_WEIGHTS,
) -> pd.DataFrame:
    """
    Calculate all MIscore components using vectorized operations.

    Note: This implementation follows the MIscore mathematical formulas from
    Villaveces et al. (2015), but has not been validated against published
    IntAct scores due to lack of detailed worked examples showing:
    - Specific interaction evidence (X studies of type Y using method Z)
    - The resulting component scores
    - The final MIscore

    Parameters
    ----------
    counts_df : pd.DataFrame
        DataFrame containing interaction counts and scores
    n_publications_series : pd.Series
        Series containing publication counts for each interaction
    max_pubs : int, optional
        Maximum publication threshold for scoring, by default INTACT_PUBLICATION_SCORE_THRESHOLD
    weights : dict[str, float], optional
        Dictionary of weights for different score components, by default DEFAULT_INTACT_RELATIVE_WEIGHTS

    Returns
    -------
    pd.DataFrame
        DataFrame containing all calculated scores for each interaction

    Raises
    ------
    ValueError
        If the weights dictionary does not contain the expected keys
    """

    expected_weighted_keys = DEFAULT_INTACT_RELATIVE_WEIGHTS.keys()
    if set(weights.keys()) != set(expected_weighted_keys):
        raise ValueError(
            f"The weights dictionary must contain the following keys: {expected_weighted_keys}"
        )

    # Get all unique interactions
    interactions = counts_df[
        [
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
        ]
    ].drop_duplicates()

    # Calculate publication scores from the Series
    pub_scores = n_publications_series.to_frame(
        INTACT_SCORES.N_PUBLICATIONS
    ).reset_index()
    pub_scores[INTACT_SCORES.PUBLICATION_SCORE] = np.where(
        pub_scores[INTACT_SCORES.N_PUBLICATIONS] > max_pubs,
        1.0,
        np.log(pub_scores[INTACT_SCORES.N_PUBLICATIONS] + 1) / np.log(max_pubs + 1),
    )
    pub_scores = pub_scores.drop(columns=[INTACT_SCORES.N_PUBLICATIONS])

    # Calculate method scores
    method_scores = _calculate_category_scores_vectorized(
        counts_df, PSI_MI_DEFS.INTERACTION_METHOD
    )

    # Calculate type scores
    type_scores = _calculate_category_scores_vectorized(
        counts_df, PSI_MI_DEFS.INTERACTION_TYPE
    )

    # Merge all scores together
    result = (
        interactions.merge(
            pub_scores,
            on=[
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ],
            how="left",
        )
        .merge(
            method_scores,
            on=[
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ],
            how="left",
        )
        .merge(
            type_scores,
            on=[
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ],
            how="left",
        )
        # Fill missing scores with 0
        .fillna(0.0)
    )

    # Calculate final MIscore
    total_weight = sum(weights.values())
    result[INTACT_SCORES.MI_SCORE] = (
        weights[INTACT_SCORES.PUBLICATION_SCORE]
        * result[INTACT_SCORES.PUBLICATION_SCORE]
        + weights[INTACT_SCORES.INTERACTION_METHOD_SCORE]
        * result[INTACT_SCORES.INTERACTION_METHOD_SCORE]
        + weights[INTACT_SCORES.INTERACTION_TYPE_SCORE]
        * result[INTACT_SCORES.INTERACTION_TYPE_SCORE]
    ) / total_weight

    return result


def _calculate_category_scores_vectorized(
    counts_df: pd.DataFrame, attribute_type: str
) -> pd.DataFrame:
    """
    Calculate method or type scores for all interactions using vectorized operations.

    Parameters
    ----------
    counts_df : pd.DataFrame
        DataFrame containing interaction data
    attribute_type : str
        Type of attribute to calculate scores for (e.g., 'interaction_method', 'interaction_type')

    Returns
    -------
    pd.DataFrame
        DataFrame containing category scores for each interaction
    """
    # Filter to specific attribute type
    filtered_df = counts_df[
        counts_df[INTACT_SCORES.ATTRIBUTE_TYPE] == attribute_type
    ].copy()

    if len(filtered_df) == 0:
        # Return empty dataframe with expected structure
        return pd.DataFrame(
            columns=[
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                f"{attribute_type}_score",
            ]
        )

    # Calculate a = sum(score * count) for each interaction
    filtered_df["score_x_count"] = (
        filtered_df[INTACT_SCORES.RAW_SCORE] * filtered_df["count"]
    )
    a_values = (
        filtered_df.groupby(
            [
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ]
        )["score_x_count"]
        .sum()
        .reset_index()
        .rename(columns={"score_x_count": "a"})
    )

    # Calculate max count by score group for each interaction
    max_by_score_group = (
        filtered_df.groupby(
            [
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                INTACT_SCORES.RAW_SCORE,
            ]
        )["count"]
        .max()
        .reset_index()
        .groupby(
            [
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            ]
        )["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "max_sum"})
    )

    # Merge a and max_sum
    merged = a_values.merge(
        max_by_score_group,
        on=[
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
        ],
    )
    merged["b"] = merged["a"] + merged["max_sum"]

    # Calculate category score: log(a+1) / log(b+1)
    merged[f"{attribute_type}_score"] = np.where(
        merged["a"] == 0, 0.0, np.log(merged["a"] + 1) / np.log(merged["b"] + 1)
    )

    return merged[
        [
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            f"{attribute_type}_score",
        ]
    ]


def _count_studies_with_scored_attributes(
    standardized_interaction_attrs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count the number of studies which report an interaction based on scored attributes.

    Parameters
    ----------
    standardized_interaction_attrs : pd.DataFrame
        Long-form dataframe with columns: upstream_name, downstream_name,
        study_id, attribute_type, scored_term, score

    Returns
    -------
    scored_attribute_counts : pd.DataFrame
        The number of studies and score for each interaction-attribute_type-scored_term combination.
    """

    scored_attribute_counts = (
        standardized_interaction_attrs[
            [
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                PSI_MI_DEFS.STUDY_ID,
                INTACT_SCORES.ATTRIBUTE_TYPE,
                INTACT_SCORES.SCORED_TERM,
                INTACT_SCORES.RAW_SCORE,
            ]
        ]
        .drop_duplicates()
        # Count studies for each interaction-attribute_type-scored_term combination
        .groupby(
            [
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                INTACT_SCORES.ATTRIBUTE_TYPE,
                INTACT_SCORES.SCORED_TERM,
                INTACT_SCORES.RAW_SCORE,  # Include score in groupby to preserve it
            ]
        )
        .size()
        .reset_index(name="count")
    )

    return scored_attribute_counts


def _create_basic_edgelist(
    intact_summaries: Dict[str, pd.DataFrame], lookup_table: pd.Series
) -> pd.DataFrame:
    """
    Create a basic edgelist from the IntAct summaries and lookup table.

    The edgelist is created by merging the IntAct summaries and lookup table on the study id and interaction id.
    The edgelist is then filtered to only include interactions where the bait is present and the prey is present.
    The edgelist is then pivoted based on the hub and spoke model used by IntAct, where a single bait is connected
    to one or more prey.

    Parameters
    ----------
    intact_summaries : Dict[str, pd.DataFrame]
        A dictionary of IntAct summaries, keyed by study id.
    lookup_table : pd.Series
        A lookup table of interaction ids and their corresponding interaction names.

    Returns
    -------
    edgelist_df : pd.DataFrame
        A dataframe of the edgelist with the following columns:
        - upstream_name : the name of the upstream node
        - downstream_name : the name of the downstream node
        - interaction_name : the name of the interaction
        - study_id : the id of the study
        - interaction_type : the type of interaction

    Notes
    -----
    The convention of associating each bait to many prey follows conventions set by
    yeast 2 hybrid screens but it is applied across the board even for technologies
    when bait-prey relationships are not appropriate (e.g., purifying a whole complex).
    In these cases IntAct chooses a random component to serve as the prey. This will make
    it more closely related in a network sense than its interactors than they would be
    to one another. This could be addressed by expanding interactions but this would  be
    quite tricky because some interactions have 100s of prey and (N choose 2) would be
    cumbersome. This could be done for just certain types of annotations but it seems
    like a big headache for very little practical gain.
    """

    interactions = (
        intact_summaries[PSI_MI_STUDY_TABLES.REACTION_SPECIES]
        .merge(
            lookup_table,
            on=[PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTOR_ID],
            how="inner",
        )
        .query("experimental_role in @VALID_INTACT_EXPERIMENTAL_ROLES")
    )

    # the hub and spoke model connects a single bait to 1 or more prey
    # following the conventions yeast 2 hybrid screens
    valid_interactions_structure = (
        interactions.value_counts(
            [
                PSI_MI_DEFS.STUDY_ID,
                PSI_MI_DEFS.INTERACTION_NAME,
                PSI_MI_DEFS.EXPERIMENTAL_ROLE,
            ]
        )
        .to_frame()
        .pivot_table(
            index=[PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTION_NAME],
            columns=PSI_MI_DEFS.EXPERIMENTAL_ROLE,
            values="count",  # or whatever your count column is called
            fill_value=0,  # fill missing values with 0
        )
        .query(
            f"{INTACT_EXPERIMENTAL_ROLES.PREY} >= 1 & {INTACT_EXPERIMENTAL_ROLES.BAIT} == 1"
        )
    )

    valid_reaction_species = interactions.merge(
        valid_interactions_structure,
        left_on=[PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTION_NAME],
        right_index=True,
    ).drop(columns=[INTACT_EXPERIMENTAL_ROLES.BAIT, INTACT_EXPERIMENTAL_ROLES.PREY])

    all_prey = (
        valid_reaction_species.query(
            "experimental_role == @INTACT_EXPERIMENTAL_ROLES.PREY"
        ).rename(columns={SBML_DFS.S_NAME: INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM})
    )[
        [
            PSI_MI_DEFS.STUDY_ID,
            PSI_MI_DEFS.INTERACTION_NAME,
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
        ]
    ]

    all_bait = (
        valid_reaction_species.query(
            "experimental_role == @INTACT_EXPERIMENTAL_ROLES.BAIT"
        ).rename(columns={SBML_DFS.S_NAME: INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM})
    )[
        [
            PSI_MI_DEFS.STUDY_ID,
            PSI_MI_DEFS.INTERACTION_NAME,
            PSI_MI_DEFS.INTERACTION_TYPE,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
        ]
    ]

    edgelist_df = all_prey.merge(
        all_bait, on=[PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTION_NAME], how="left"
    )
    edgelist_df = edgelist_df[
        [
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            PSI_MI_DEFS.INTERACTION_NAME,
            PSI_MI_DEFS.STUDY_ID,
            PSI_MI_DEFS.INTERACTION_TYPE,
        ]
    ]

    # flip upstream and downstream names if needed so
    # upstream name alphanumerically comes before downstream name

    from_orig = edgelist_df[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM].copy()
    to_orig = edgelist_df[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM].copy()

    # Create mask and assign
    needs_flip = from_orig > to_orig
    edgelist_df[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM] = np.where(
        needs_flip, to_orig, from_orig
    )
    edgelist_df[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM] = np.where(
        needs_flip, from_orig, to_orig
    )

    return edgelist_df


def _create_r_identifiers(group_data: pd.DataFrame) -> Identifiers:
    """
    Create a list of identifiers for an experiment's interactions.

    Parameters
    ----------
    group_data : pd.DataFrame
        A dataframe with the study metadata.

    Returns
    -------
    Identifiers
        An Identifiers object containing the interaction identifiers
    """

    interaction_dict = [
        {
            IDENTIFIERS.ONTOLOGY: x[IDENTIFIERS.ONTOLOGY],
            IDENTIFIERS.IDENTIFIER: x[IDENTIFIERS.IDENTIFIER],
            IDENTIFIERS.BQB: BQB.IS_DESCRIBED_BY,
        }
        for x in group_data.to_dict("records")
    ]

    return Identifiers(interaction_dict)


def _create_species_df(
    raw_species_df: pd.DataFrame,
    raw_species_identifiers_df: pd.DataFrame,
    organismal_species: Union[str, OrganismalSpeciesValidator],
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create a species dataframe from the raw species dataframe and the raw species identifiers dataframe.

    Parameters
    ----------
    raw_species_df : pd.DataFrame
        The raw species dataframe.
    raw_species_identifiers_df : pd.DataFrame
        The raw species identifiers dataframe.
    organismal_species : str | OrganismalSpeciesValidator
        The organismal species pertaining to the IntAct interactions

    Returns
    -------
    lookup_table : pd.Series
        A lookup table mapping study_id and interactor_id to the molecular species name.
    species_df : pd.DataFrame
        The molecular species dataframe.

    Raises
    ------
    ValueError
        If the provided species is not supported by IntAct
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)
    intact_species_basename = _get_intact_species_basename(
        organismal_species.latin_name
    )

    raw_species_df = raw_species_df.copy()
    raw_species_df["species_match"] = raw_species_df[
        PSI_MI_DEFS.INTERACTOR_LABEL
    ].str.endswith(intact_species_basename)

    all_species_annotations = raw_species_identifiers_df.merge(
        raw_species_df[
            [
                PSI_MI_DEFS.INTERACTOR_ID,
                PSI_MI_DEFS.STUDY_ID,
                PSI_MI_DEFS.INTERACTOR_LABEL,
                "species_match",
            ]
        ],
        on=[PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTOR_ID],
        how="inner",
    )[
        [
            PSI_MI_DEFS.INTERACTOR_LABEL,
            IDENTIFIERS.ONTOLOGY,
            IDENTIFIERS.IDENTIFIER,
            IDENTIFIERS.BQB,
            "species_match",
            "white_listed",
        ]
    ].drop_duplicates()

    valid_species_annotations = (
        all_species_annotations
        # remove names which are not associated with at least 1 species match by name or a white-listed
        # annotation like a ChEBI id or a species-matched ensembl id
        .groupby("interactor_label")
        .filter(lambda group: (group["species_match"] | group["white_listed"]).any())
        .drop(columns=["species_match", "white_listed"])
    )

    # aggregate annotations so there is just 1 row per interactor label
    species_df = (
        valid_species_annotations
        # group by interactor label and apply the Identifiers constructor
        .groupby([PSI_MI_DEFS.INTERACTOR_LABEL])
        .apply(lambda x: Identifiers(x.to_dict(orient="records")), include_groups=False)
        .rename(SBML_DFS.S_IDENTIFIERS)
        .rename_axis(SBML_DFS.S_NAME)
        .reset_index()
    )

    lookup_table = (
        raw_species_df.loc[
            raw_species_df[PSI_MI_DEFS.INTERACTOR_LABEL].isin(
                species_df[SBML_DFS.S_NAME]
            )
        ]
        .rename(columns={PSI_MI_DEFS.INTERACTOR_LABEL: SBML_DFS.S_NAME})
        .set_index([PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTOR_ID])[SBML_DFS.S_NAME]
    )

    return lookup_table, species_df


def _define_edgelist_df_ids_and_counts(
    edgelist_w_study_metadata: pd.DataFrame,
    alias_mapping: Dict[str, str],
    scored_attribute_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add attributes to the edgelist.

    Parameters
    ----------
    edgelist_w_study_metadata : pd.DataFrame
        The edgelist with study metadata.
    alias_mapping : dict[str, str]
        A dictionary mapping from ontology aliases to the Napistu controlled vocabulary.
    scored_attribute_counts : pd.DataFrame
        A dataframe with the number of studies with each attribute.

    Returns
    -------
    pd.DataFrame
        A dataframe with the edgelist, citation counts, and identifiers.
    """

    # rename ontologies
    tmp_edgelist_w_study_metadata = edgelist_w_study_metadata.copy()
    tmp_edgelist_w_study_metadata[IDENTIFIERS.ONTOLOGY] = tmp_edgelist_w_study_metadata[
        IDENTIFIERS.ONTOLOGY
    ].replace(alias_mapping)

    unrecognized_ontologies = set(
        tmp_edgelist_w_study_metadata[IDENTIFIERS.ONTOLOGY]
    ) - set(ONTOLOGIES_LIST)

    if len(unrecognized_ontologies) > 0:
        logger.warning(
            f"The following ontologies were primary references for interactions but they are not part of the  Napistu controlled vocabulary (ONTOLOGIES): {unrecognized_ontologies}"
        )

    track_citations = edgelist_w_study_metadata[
        [
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            PSI_MI_DEFS.STUDY_ID,
            IDENTIFIERS.ONTOLOGY,
            IDENTIFIERS.IDENTIFIER,
        ]
    ].drop_duplicates()

    edgelist_groups = track_citations.groupby(
        [
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
        ]
    )

    # count the # of citations
    edgelist_citation_counts = edgelist_groups.size().rename(
        INTACT_SCORES.N_PUBLICATIONS
    )
    edgelist_identifiers = edgelist_groups.apply(_create_r_identifiers).rename(
        SBML_DFS.R_IDENTIFIERS
    )

    # create a wide output with one column per interaction attribute
    tmp_scored_attribute_counts = scored_attribute_counts.copy()
    tmp_scored_attribute_counts["composite_attribute"] = (
        tmp_scored_attribute_counts["attribute_type"]
        + "_"
        + tmp_scored_attribute_counts["scored_term"]
    )
    wide_attribute_counts = tmp_scored_attribute_counts.pivot_table(
        index=[
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
        ],
        columns="composite_attribute",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    return pd.concat(
        [edgelist_citation_counts, edgelist_identifiers, wide_attribute_counts], axis=1
    )


def _filter_intact_xrefs(
    intact_summaries: Dict[str, pd.DataFrame],
    alias_mapping: Dict[str, str],
    organismal_species: Union[str, OrganismalSpeciesValidator],
    valid_secondary_ontologies: Set[str] = VALID_INTACT_SECONDARY_ONTOLOGIES,
) -> pd.DataFrame:
    """
    Filter IntAct species identifiers to only those which should be added as s_Identifiers.

    Parameters
    ----------
    intact_summaries : Dict[str, pd.DataFrame]
        The IntAct summaries table.
    alias_mapping : Dict[str, str]
        A dictionary mapping from ontology aliases to the Napistu controlled vocabulary.
    organismal_species : str | OrganismalSpeciesValidator
        The organismal species pertaining to the IntAct interactions
    valid_secondary_ontologies : Set[str], optional
        A set of ontologies which are valid secondary references, by default VALID_INTACT_SECONDARY_ONTOLOGIES

    Returns
    -------
    pd.DataFrame
        A DataFrame of IntAct species identifiers which should be added as s_Identifiers.

    Raises
    ------
    ValueError
        If ontologies listed as valid secondary references are not in the Napistu controlled vocabulary
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)

    invalid_valid_secondary_ontologies = set(valid_secondary_ontologies) - set(
        ONTOLOGIES_LIST
    )
    if len(invalid_valid_secondary_ontologies) > 0:
        raise ValueError(
            f"The following ontologies are listed as valid secondary references but they are not in the Napistu controlled vocabulary (ONTOLOGIES_LIST): {invalid_valid_secondary_ontologies}"
        )

    # pull out the raw species identifiers table
    standardized_intact_xrefs = intact_summaries[
        PSI_MI_STUDY_TABLES.SPECIES_IDENTIFIERS
    ]

    # drop entries which are missing either their ontology or identifiers
    valid_xref_mask = (
        standardized_intact_xrefs[IDENTIFIERS.ONTOLOGY] != PSI_MI_MISSING_VALUE_STR
    ) & (standardized_intact_xrefs[IDENTIFIERS.IDENTIFIER] != PSI_MI_MISSING_VALUE_STR)

    if sum(~valid_xref_mask) > 0:
        logger.warning(
            f"Dropping {sum(~valid_xref_mask)} species identifiers which are missing either their ontology or identifier."
        )

    standardized_intact_xrefs = standardized_intact_xrefs[valid_xref_mask]

    # convert from general "ensembl" ontology to ensembl gene, protein, etc. and remove version numbers
    sanitized_intact_xrefs = _sanitize_identifiers(
        standardized_intact_xrefs, organismal_species
    )

    # convert ontologies into the Napistu controlled vocabulary
    sanitized_intact_xrefs[IDENTIFIERS.ONTOLOGY] = sanitized_intact_xrefs[
        IDENTIFIERS.ONTOLOGY
    ].map(lambda x: alias_mapping.get(x, x))

    # format primary references
    primary_refs = sanitized_intact_xrefs[
        sanitized_intact_xrefs[PSI_MI_DEFS.REF_TYPE] == PSI_MI_DEFS.PRIMARY
    ]
    # all entries in primary refs should have a valid ontology
    invalid_ontologies = set(primary_refs[IDENTIFIERS.ONTOLOGY]) - set(ONTOLOGIES_LIST)
    if len(invalid_ontologies) > 0:
        _log_invalid_primary_refs(primary_refs, invalid_ontologies)
        primary_refs = primary_refs[
            ~primary_refs[IDENTIFIERS.ONTOLOGY].isin(invalid_ontologies)
        ]

    # only look at a subset of the secondary references
    # including additional ontologies to this set results in (rediculous) overmerging of species
    # when creating consensus models where species are defined by their s_Identifiers
    secondary_refs = sanitized_intact_xrefs[
        sanitized_intact_xrefs[PSI_MI_DEFS.REF_TYPE] == PSI_MI_DEFS.SECONDARY
    ].query("ontology in @VALID_INTACT_SECONDARY_ONTOLOGIES")
    # drop entries which are missing their primary ref
    secondary_refs = secondary_refs.merge(
        primary_refs[[PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTOR_ID]], how="inner"
    )

    valid_xrefs = pd.concat([primary_refs, secondary_refs]).assign(
        **{IDENTIFIERS.BQB: BQB.IS}
    )

    return valid_xrefs


def _get_intact_scored_term(term_name: str, term_lookup: Dict[str, str]) -> str:
    """Get the scored ancestor term name for a given term."""
    return term_lookup.get(term_name, PSI_MI_SCORED_TERMS.UNKNOWN)


def _get_intact_species_basename(latin_species: str) -> str:

    if latin_species not in PSI_MI_INTACT_SPECIES_TO_BASENAME.keys():
        raise ValueError(
            f"The provided species {latin_species} did not match any of the species in INTACT_SPECIES_TO_BASENAME: "
            f"{', '.join(PSI_MI_INTACT_SPECIES_TO_BASENAME.keys())}"
            "If this is a species supported by IntAct please add the species to the PSI_MI_INTACT_SPECIES_TO_BASENAME dictionary."
        )

    return PSI_MI_INTACT_SPECIES_TO_BASENAME[latin_species]


def _get_intact_term_with_score(
    ontology_graph: ig.Graph, scored_terms: Dict[str, float]
) -> Dict[str, str]:
    """Build lookup mapping all terms to their scored ancestor names."""
    term_lookup = {}

    for vertex in ontology_graph.vs:
        term_name = vertex["name"]

        # Check if term has explicit score
        if term_name in scored_terms:
            term_lookup[term_name] = term_name  # Maps to itself
        else:
            # Get all ancestors (nodes reachable via incoming edges)
            ancestors_idx = ontology_graph.subcomponent(vertex.index, mode="in")
            ancestors = [
                ontology_graph.vs[idx]["name"]
                for idx in ancestors_idx
                if idx != vertex.index
            ]

            # Find highest scoring ancestor
            scored_ancestors = [
                (ancestor, scored_terms[ancestor])
                for ancestor in ancestors
                if ancestor in scored_terms
            ]

            if scored_ancestors:
                # Map to the name of the highest scoring ancestor
                best_ancestor = max(scored_ancestors, key=lambda x: x[1])[0]
                term_lookup[term_name] = best_ancestor

    return term_lookup


def _log_invalid_primary_refs(
    primary_refs: pd.DataFrame, invalid_ontologies: Set[str]
) -> None:

    invalid_ontologies_counts = primary_refs[
        primary_refs[IDENTIFIERS.ONTOLOGY].isin(invalid_ontologies)
    ].value_counts(IDENTIFIERS.ONTOLOGY)

    invalid_ontologies_str = ", ".join(
        [
            f"{ontology} (N={count})"
            for ontology, count in invalid_ontologies_counts.items()
        ]
    )

    logger.warning(
        "The following ontologies are listed as primary species references but they are not\n"
        "in the Napistu controlled vocabulary (ONTOLOGIES_LIST). These species and their interactions will be ignored:\n"
        f"{invalid_ontologies_str}"
    )


def _process_ensembl_ids(ensembl_primary_xrefs, latin_species: str) -> pd.DataFrame:
    """
    Process ensembl IntAct references to convert them from the meta ensembl ontology to the Napistu controlled vocabulary.

    Parameters
    ----------
    ensembl_primary_xrefs : pd.DataFrame
        The standardized IntAct references filtered to the "ensembl" ontology.
    latin_species : str
        The latin species name to filter ensembl ids to.

    Returns
    -------
    pd.DataFrame
        The processed ensembl IntAct references.
    """

    def _safe_parse_ensembl_id(x):
        try:
            return parse_ensembl_id(x)
        except (TypeError, AttributeError):
            return (None, None, None)

    distinct_ensembl_ids = (
        ensembl_primary_xrefs[IDENTIFIERS.IDENTIFIER]
        .to_frame()
        .drop_duplicates()
        .rename(columns={IDENTIFIERS.IDENTIFIER: "original_identifier"})
    )

    distinct_ensembl_ids[
        [IDENTIFIERS.IDENTIFIER, IDENTIFIERS.ONTOLOGY, "latin_species"]
    ] = distinct_ensembl_ids["original_identifier"].apply(
        lambda x: pd.Series(_safe_parse_ensembl_id(x))
    )

    invalid_ensembl_ids = distinct_ensembl_ids[
        distinct_ensembl_ids[IDENTIFIERS.IDENTIFIER].isna()
    ]
    if len(invalid_ensembl_ids) > 0:
        logger.warning(
            f"Dropped {len(invalid_ensembl_ids)} ensembl ids which could not be formatted by identifiers.parse_ensembl_id: {invalid_ensembl_ids['original_identifier'].tolist()}"
        )

    distinct_ensembl_ids = distinct_ensembl_ids.dropna(subset=[IDENTIFIERS.IDENTIFIER])

    # filter by latin name
    correct_species_mask = distinct_ensembl_ids["latin_species"] == latin_species
    if sum(~correct_species_mask) > 0:
        logger.warning(
            f"Dropped {sum(~correct_species_mask)} ensembl ids which did not match the specified latin species name {latin_species} and will be ignored."
        )

    distinct_ensembl_ids = distinct_ensembl_ids[correct_species_mask]

    processed_ensembl_ids = (
        ensembl_primary_xrefs.rename(
            columns={IDENTIFIERS.IDENTIFIER: "original_identifier"}
        )
        .drop(columns=[IDENTIFIERS.ONTOLOGY])
        .merge(distinct_ensembl_ids, on="original_identifier", how="inner")
        .drop(columns=["original_identifier", "latin_species"])
    )

    return processed_ensembl_ids


def _sanitize_identifiers(
    standardized_intact_xrefs,
    organismal_species: str | OrganismalSpeciesValidator,
    ensembl_ontology_name: str = "ensembl",
    chebi_ontology_name: str = ONTOLOGIES.CHEBI,
    rna_central_ontology_name: str = ONTOLOGIES.RNACENTRAL,
) -> pd.DataFrame:
    """
    Sanitizes the identifiers in the standardized IntAct references.

    This functions applies ontology-specific manipulations and white lists non-genic molecular species like metabolites so they aren't filtered downstream.

    Parameters
    ----------
    standardized_intact_xrefs : pd.DataFrame
        The standardized IntAct references.
    organismal_species : str
        The organismal species to filter ensembl ids to.
    ensembl_ontology_name : str
        The name of the ontology to convert from.
    chebi_ontology_name : str
        The name of the ontology to convert from.
    rna_central_ontology_name : str
        The name of the ontology to convert from.

    Returns
    -------
    pd.DataFrame
        The sanitized IntAct references.
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)

    # split ontologies needing special treatmnet
    ensembl_primary_xrefs_mask = (
        standardized_intact_xrefs[PSI_MI_DEFS.REF_TYPE] == PSI_MI_DEFS.PRIMARY
    ) & (standardized_intact_xrefs[IDENTIFIERS.ONTOLOGY] == ensembl_ontology_name)
    if sum(ensembl_primary_xrefs_mask) == 0:
        logger.warning(
            "No ensembl primary references found in the standardized IntAct references. This may indicate a change in IntAct notation."
        )

    chebi_xrefs_mask = (
        standardized_intact_xrefs[IDENTIFIERS.ONTOLOGY] == chebi_ontology_name
    )
    if sum(chebi_xrefs_mask) == 0:
        logger.warning(
            "No CHEBI references found in the standardized IntAct references. This may indicate a change in IntAct notation."
        )

    rna_central_mask = (
        standardized_intact_xrefs[IDENTIFIERS.ONTOLOGY] == rna_central_ontology_name
    )
    if sum(rna_central_mask) == 0:
        logger.warning(
            "No RNA Central references found in the standardized IntAct references. This may indicate a change in IntAct notation."
        )

    other_ontologies_mask = ~(
        ensembl_primary_xrefs_mask | chebi_xrefs_mask | rna_central_mask
    )

    # format special ontologies
    ensembl_ids = _process_ensembl_ids(
        standardized_intact_xrefs.loc[ensembl_primary_xrefs_mask],
        organismal_species.latin_name,
    )
    chebi_ids = standardized_intact_xrefs.loc[chebi_xrefs_mask]
    # remove the "CHEBI:" prefix
    chebi_ids.loc[:, IDENTIFIERS.IDENTIFIER] = chebi_ids[
        IDENTIFIERS.IDENTIFIER
    ].str.replace("CHEBI:", "")

    white_listed_ids = pd.concat(
        [chebi_ids, ensembl_ids, standardized_intact_xrefs.loc[rna_central_mask]]
    ).assign(white_listed=True)

    return pd.concat(
        [
            white_listed_ids,
            standardized_intact_xrefs.loc[other_ontologies_mask].assign(
                white_listed=False
            ),
        ]
    )


def _standardize_interaction_attrs(
    edgelist_w_study_metadata: pd.DataFrame,
    ontology_url: str = PSI_MI_ONTOLOGY_URL,
) -> pd.DataFrame:

    # read ontology hierarchy from the IntAct github
    psi_mi_ontology_graph = _build_psi_mi_ontology_graph(ontology_url)
    # map from recognized terms in the ontology to terms with an explicit score
    # (other entries will be assigned unknown and receive its score)
    intact_term_lookup = _get_intact_term_with_score(
        psi_mi_ontology_graph, INTACT_TERM_SCORES
    )

    tall_interaction_attrs = pd.melt(
        edgelist_w_study_metadata,
        id_vars=[
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            PSI_MI_DEFS.INTERACTION_NAME,
            PSI_MI_DEFS.STUDY_ID,
        ],
        value_vars=[PSI_MI_DEFS.INTERACTION_TYPE, PSI_MI_DEFS.INTERACTION_METHOD],
        var_name=INTACT_SCORES.ATTRIBUTE_TYPE,
        value_name=INTACT_SCORES.ATTRIBUTE_VALUE,
    )

    attribute_scores = (
        tall_interaction_attrs[INTACT_SCORES.ATTRIBUTE_VALUE]
        .drop_duplicates()
        .reset_index(drop=True)
        .to_frame()
    )

    # map to an appropraite term with a score
    attribute_scores[INTACT_SCORES.SCORED_TERM] = attribute_scores[
        INTACT_SCORES.ATTRIBUTE_VALUE
    ].apply(lambda x: _get_intact_scored_term(x, intact_term_lookup))
    attribute_scores[INTACT_SCORES.RAW_SCORE] = attribute_scores[
        INTACT_SCORES.SCORED_TERM
    ].apply(lambda x: INTACT_TERM_SCORES[x])

    tall_standardized_interaction_attrs = tall_interaction_attrs.merge(
        attribute_scores, on=INTACT_SCORES.ATTRIBUTE_VALUE, how="left"
    )

    return tall_standardized_interaction_attrs
