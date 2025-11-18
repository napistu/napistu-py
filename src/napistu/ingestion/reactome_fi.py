import datetime
import logging

import pandas as pd

from napistu import identifiers, sbml_dfs_core, sbml_dfs_utils, utils
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    SBML_DFS,
    SBOTERM_NAMES,
)
from napistu.ingestion.constants import (
    DATA_SOURCE_DESCRIPTIONS,
    DATA_SOURCES,
    INTERACTION_EDGELIST_DEFS,
    LATIN_SPECIES_NAMES,
    REACTOME_FI,
    REACTOME_FI_RULES_FORWARD,
    REACTOME_FI_RULES_REVERSE,
    REACTOME_FI_URL,
    VALID_REACTOME_FI_DIRECTIONS,
)
from napistu.ontologies.constants import GENODEXITO_DEFS
from napistu.ontologies.genodexito import Genodexito
from napistu.source import Source

logger = logging.getLogger(__name__)


def download_reactome_fi(target_uri: str, url: str = REACTOME_FI_URL) -> None:
    """
    Download the Reactome Functional Interactions (FI) dataset as a TSV file.

    Parameters
    ----------
    target_uri : str
        The URI where the Reactome FI data should be saved. Should end with .tsv
    url : str, optional
        URL to download the zipped Reactome functional interactions TSV from.
        Defaults to REACTOME_FI_URL.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If target_uri does not end with .tsv
    """

    if not target_uri.endswith(".tsv"):
        raise ValueError(f"Target URI must end with .tsv, got {target_uri}")

    file_ext = url.split(".")[-1]
    target_filename = url.split("/")[-1].split(f".{file_ext}")[0]
    logger.info("Start downloading Reactome FI %s to %s", url, target_uri)
    # target_filename is the name of the file in the zip file which will be renamed to target_uri
    utils.download_wget(url, target_uri, target_filename=target_filename)

    return None


def convert_reactome_fi_to_sbml_dfs(
    interactions: pd.DataFrame,
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
) -> sbml_dfs_core.SBML_dfs:
    """
    Format the Reactome FI interactions DataFrame as an edgelist for network analysis.

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame containing Reactome FI interactions.
    preferred_method : str
        The preferred Genodexito method to use for identifier mapping.
    allow_fallback : bool
        Whether to allow fallback to other Genodexito methods for identifier mapping.

    Returns
    -------
    sbml_dfs_core.SBML_dfs:
        A SBML_dfs object containing the species, compartments, and reactions data.
    """

    # define metadata for the Reactome-level model
    model_source = Source.single_entry(
        model=DATA_SOURCES.REACTOME_FI,
        pathway_id=DATA_SOURCES.REACTOME_FI,
        data_source=DATA_SOURCES.REACTOME_FI,
        organismal_species=LATIN_SPECIES_NAMES.HOMO_SAPIENS,
        name=DATA_SOURCE_DESCRIPTIONS[DATA_SOURCES.REACTOME_FI],
        date=datetime.date.today().strftime("%Y%m%d"),
    )

    species_df = create_species_df(interactions, preferred_method, allow_fallback)
    interaction_edgelist = create_interaction_edgelist(interactions)
    compartments_df = sbml_dfs_utils.stub_compartments()

    sbml_dfs = sbml_dfs_core.SBML_dfs.from_edgelist(
        interaction_edgelist=interaction_edgelist,
        species_df=species_df,
        compartments_df=compartments_df,
        model_source=model_source,
        keep_reactions_data=DATA_SOURCES.REACTOME_FI,
    )

    return sbml_dfs


def create_species_df(
    interactions: pd.DataFrame,
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """
    Create a species DataFrame from a set of interactions.

    Parameters
    ----------
    interactions : pd.DataFrame
        The interactions to create the species DataFrame from.
    preferred_method : str
        The preferred method to use for identifier mapping.
    allow_fallback : bool
        Whether to allow fallback to other methods for identifier mapping.

    Returns
    -------
    pd.DataFrame
        The species DataFrame containing the species names and their identifiers:
        - SBML_DFS.S_NAME : The species' name
        - SBML_DFS.S_IDENTIFIERS : The identifiers for the species
    """

    all_gene_names = set(interactions[REACTOME_FI.GENE1]) | set(
        interactions[REACTOME_FI.GENE2]
    )
    species_systematic_ids = _get_reactome_fi_species_systematic_ids(
        all_gene_names, preferred_method, allow_fallback
    )

    # create Identifiers objects
    id_table = species_systematic_ids.rename(
        columns={
            ONTOLOGIES.SYMBOL: SBML_DFS.S_ID,
            ONTOLOGIES.NCBI_ENTREZ_GENE: IDENTIFIERS.IDENTIFIER,
        }
    ).assign(ontology=ONTOLOGIES.NCBI_ENTREZ_GENE, bqb=BQB.IS)

    species_identifiers = (
        identifiers.df_to_identifiers(id_table)
        .reset_index()
        .rename(columns={SBML_DFS.S_ID: SBML_DFS.S_NAME})
    )

    # fill missing entries with empty identifiers
    missing_species_names = list(
        all_gene_names - set(species_identifiers[SBML_DFS.S_NAME])
    )
    missing_species = pd.DataFrame({SBML_DFS.S_NAME: missing_species_names}).assign(
        **{SBML_DFS.S_IDENTIFIERS: identifiers.Identifiers([])}
    )
    species_identifiers = pd.concat([species_identifiers, missing_species]).reset_index(
        drop=True
    )

    return species_identifiers


def create_interaction_edgelist(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Create an edgelist of Reactome functional interactions

    Parameters
    ----------
    interactions : pd.DataFrame
        A DataFrame of Reactome functional interactions

    Returns
    -------
    pd.DataFrame
        Functional interactions with the following columns:
        - upstream_name: the name of the upstream species
        - downstream_name: the name of the downstream species
        - upstream_sbo_term_name: the sbo term name for the upstream species
        - downstream_sbo_term_name: the sbo term name for the downstream species
        - r_Identifiers: an identifiers.Identifiers object
        - r_name: a human readable name for the interaction
    """

    formatted_annotations = _parse_reactome_fi_annotations(interactions)

    # this join will expand some rows to 2 since the bidirectional relationships are captured as separate edges in Napistu
    annotated_interactions = interactions.merge(
        formatted_annotations,
        on=[REACTOME_FI.ANNOTATION, REACTOME_FI.DIRECTION_RAW],
        how="left",
    )

    # flip reverse entries so all relationships are forward or undirected
    formatted_interactions = (
        pd.concat(
            [
                annotated_interactions.query(
                    f"{REACTOME_FI.DIRECTION} == '{REACTOME_FI.FORWARD}'"
                ),
                (
                    annotated_interactions.query(
                        f"{REACTOME_FI.DIRECTION} == '{REACTOME_FI.REVERSE}'"
                    ).rename(
                        columns={
                            REACTOME_FI.GENE1: REACTOME_FI.GENE2,
                            REACTOME_FI.GENE2: REACTOME_FI.GENE1,
                        }
                    )
                ),
            ]
        )[
            [
                REACTOME_FI.GENE1,
                REACTOME_FI.GENE2,
                INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM,
                REACTOME_FI.SCORE,
            ]
        ]
        # looks like they were already unique edges
        .sort_values(REACTOME_FI.SCORE, ascending=False)
        .groupby([REACTOME_FI.GENE1, REACTOME_FI.GENE2])
        .first()
    ).rename(columns={REACTOME_FI.SCORE: REACTOME_FI.FI_REACTION_DATA_SCORE})

    fi_edgelist = (
        formatted_interactions.reset_index()
        .rename(
            columns={
                REACTOME_FI.GENE1: INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                REACTOME_FI.GENE2: INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            }
        )
        .assign(r_Identifiers=identifiers.Identifiers([]))
    )
    fi_edgelist[SBML_DFS.R_NAME] = fi_edgelist.apply(
        lambda row: sbml_dfs_utils._name_interaction(
            row[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM],
            row[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM],
            row[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM],
        ),
        axis=1,
    )

    return fi_edgelist


def _parse_reactome_fi_annotations(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and annotate Reactome FI interaction types and directions using regex-based rules.

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame containing Reactome FI interactions, with annotation and direction columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with annotation, direction, SBO term name, and polarity for each unique annotation/direction pair.

    Raises
    ------
    ValueError
        If an annotation/direction pair cannot be matched to a rule or if invalid directions are found.
    """

    distinct_annotations = (
        interactions[[REACTOME_FI.ANNOTATION, REACTOME_FI.DIRECTION_RAW]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    invalid_directions = distinct_annotations.loc[
        ~distinct_annotations[REACTOME_FI.DIRECTION_RAW].isin(
            VALID_REACTOME_FI_DIRECTIONS
        ),
        REACTOME_FI.DIRECTION_RAW,
    ]
    if len(invalid_directions) > 0:
        raise ValueError(f"Invalid directions: {invalid_directions}")

    annotations = list()
    for _, vals in distinct_annotations.iterrows():
        annot, direction = vals

        forward_match = utils.match_regex_dict(
            annot, REACTOME_FI_RULES_FORWARD.NAME_RULES
        )
        if not forward_match:
            if direction in REACTOME_FI_RULES_FORWARD.DIRECTION_RULES:
                forward_match = REACTOME_FI_RULES_FORWARD.DIRECTION_RULES[direction]

        reverse_match = utils.match_regex_dict(
            annot, REACTOME_FI_RULES_REVERSE.NAME_RULES
        )
        if not reverse_match:
            if direction in REACTOME_FI_RULES_REVERSE.DIRECTION_RULES:
                reverse_match = REACTOME_FI_RULES_REVERSE.DIRECTION_RULES[direction]

        if not (forward_match or reverse_match):
            raise ValueError(f"No match found for {annot} with direction {direction}")

        if forward_match:
            annotations.append(
                {
                    REACTOME_FI.ANNOTATION: annot,
                    REACTOME_FI.DIRECTION_RAW: direction,
                    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: forward_match,
                    REACTOME_FI.DIRECTION: REACTOME_FI.FORWARD,
                }
            )

        if reverse_match:
            annotations.append(
                {
                    REACTOME_FI.ANNOTATION: annot,
                    REACTOME_FI.DIRECTION_RAW: direction,
                    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: reverse_match,
                    REACTOME_FI.DIRECTION: REACTOME_FI.REVERSE,
                }
            )

    formatted_annotations = pd.DataFrame(annotations)

    # set downstream sbo term and reversibility based sbo term (interactors are unidrected and symmetrical)
    formatted_annotations[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM] = [
        (
            SBOTERM_NAMES.INTERACTOR
            if x == SBOTERM_NAMES.INTERACTOR
            else SBOTERM_NAMES.MODIFIED
        )
        for x in formatted_annotations[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM]
    ]

    formatted_annotations[SBML_DFS.R_ISREVERSIBLE] = [
        True if x == SBOTERM_NAMES.INTERACTOR else False
        for x in formatted_annotations[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM]
    ]

    return formatted_annotations


def _get_reactome_fi_species_systematic_ids(
    all_gene_names: set[str],
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """
    Get the species systematic IDs for the genes in the interactions.

    Parameters
    ----------
    all_gene_names: set[str]
        The gene names to get the species systematic IDs for.
    preferred_method: str
        The preferred Genodexito method to use for identifier mapping.
    allow_fallback: bool
        Whether to allow fallback to other Genodexito methods if the preferred method fails.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for the gene symbol and NCBI Entrez Gene ID.
    """

    symbol_to_entrez = _get_human_symbol_to_entrez_mapping(
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    )

    missing_annotations = all_gene_names - set(symbol_to_entrez[ONTOLOGIES.SYMBOL])
    if missing_annotations:
        if len(missing_annotations) == len(all_gene_names):
            raise ValueError("No annotations found for any of the gene names")
        else:
            example_gene_names = list(missing_annotations)[:5]
            logger.info(
                f"No Entrez gene IDs found for {len(missing_annotations)} of {len(all_gene_names)} gene names, including: {example_gene_names}"
            )

    return symbol_to_entrez[symbol_to_entrez[ONTOLOGIES.SYMBOL].isin(all_gene_names)]


def _get_human_symbol_to_entrez_mapping(
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """
    Get a mapping of human gene symbols to NCBI Entrez Gene IDs.

    Parameters
    ----------
    preferred_method: str
        The preferred method to use for the mapping.
    allow_fallback: bool
        Whether to allow fallback to other methods if the preferred method fails.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for the gene symbol and NCBI Entrez Gene ID.
    """

    genodexito = Genodexito(
        organismal_species="Homo sapiens",
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    )

    genodexito.create_mapping_tables(
        mappings={ONTOLOGIES.SYMBOL, ONTOLOGIES.NCBI_ENTREZ_GENE}
    )

    genodexito.merge_mappings([ONTOLOGIES.SYMBOL, ONTOLOGIES.NCBI_ENTREZ_GENE])

    return genodexito.merged_mappings
