"""Ingestion and formatting utilities for Perturb-seq datasets."""

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    SBML_DFS,
)
from napistu.ingestion.constants import (
    GOF_PERTURBATION_TYPES,
    HARMONIZOME_DATASET_FILES,
    HARMONIZOME_DATASET_SHORTNAMES,
    HARMONIZOME_DEFS,
    LOF_PERTURBATION_TYPES,
    PERTURBSEQ_DEFS,
    PERTURBSEQ_DIRECTIONS,
    PERTURBSEQ_PERTURBATION_TYPES,
    REPLOGLE_DEFS,
)
from napistu.ingestion.harmonizome import (
    load_harmonizome_datasets,
    process_harmonizome_datasets,
)
from napistu.matching.constants import FEATURE_ID_VAR_DEFAULT
from napistu.matching.species import features_to_pathway_species
from napistu.utils import download_wget

logger = logging.getLogger(__name__)


def assign_predicted_direction(
    df,
    perturbation_type_col=HARMONIZOME_DEFS.PERTURBATION_TYPE,
    standardized_value_col=HARMONIZOME_DEFS.STANDARDIZED_VALUE,
    threshold_value_col=HARMONIZOME_DEFS.THRESHOLDED_VALUE,
):
    """
    Assign predicted direction categories based on perturbation type and fold-change.

    For OE (overexpression):
        - standardized_value > threshold: strong activation
        - 0 < standardized_value <= threshold: weak activation
        - -threshold <= standardized_value < 0: weak repression
        - standardized_value < -threshold: strong repression

    For KD/KO (knockdown/knockout) - directions are flipped:
        - standardized_value > threshold: strong repression
        - 0 < standardized_value <= threshold: weak repression
        - -threshold <= standardized_value < 0: weak activation
        - standardized_value < -threshold: strong activation

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with perturbation data
    perturbation_type_col : str
        Column name for perturbation type (should contain 'KD', 'KO', or 'OE')
    standardized_value_col : str
        Column name for standardized fold-change values
    threshold_value_col : str
        Column name for threshold values (absolute value)

    Returns
    -------
    pd.Series
        Series with predicted direction categories
    """

    if perturbation_type_col not in df.columns:
        raise ValueError(
            f"Perturbation type column {perturbation_type_col} not found in dataframe"
        )
    if standardized_value_col not in df.columns:
        raise ValueError(
            f"Standardized value column {standardized_value_col} not found in dataframe"
        )
    if threshold_value_col not in df.columns:
        raise ValueError(
            f"Threshold value column {threshold_value_col} not found in dataframe"
        )

    return df.apply(
        _categorize_perturbseq_row,
        axis=1,
        perturbation_type_col=perturbation_type_col,
        standardized_value_col=standardized_value_col,
        threshold_value_col=threshold_value_col,
    )


def ingest_replogle_pvalues(target_uri: str) -> None:
    """
    Ingest Replogle et al. Perturb-seq p-values.

    Parameters
    ----------
    target_uri: str
        Target URI to download the Replogle et al. Perturb-seq p-values to.

    Returns
    -------
    None
    """

    logger.info(
        f"Downloading Replogle PerturbSeq data from {REPLOGLE_DEFS.REPLOGLE_PERTURBSEQ_URI} to {target_uri}"
    )
    download_wget(
        url=REPLOGLE_DEFS.REPLOGLE_PERTURBSEQ_URI,
        path=target_uri,
        target_filename=REPLOGLE_DEFS.REPLOGLE_PERTURBSEQ_ZIPPED_FILENAME,
    )

    return None


def load_harmonizome_perturbseq_datasets(
    harmonizome_data_dir: str,
    species_identifiers: pd.DataFrame,
    datasets_w_formatters: Optional[Dict[str, Callable]] = None,
    return_distinct_interactions: bool = False,
) -> pd.DataFrame:
    """
    Load aggregated perturbseq data with species IDs.

    Parameters
    ----------
    harmonizome_data_dir: str
        Directory containing harmonizome data.
    species_identifiers: pd.DataFrame
        Species identifiers dataframe.
    datasets_w_formatters: Optional[Dict[str, Callable]]
        Dictionary mapping dataset shortnames to formatters. By default, uses the human perturbseq datasets to formatters.
    return_distinct_interactions: bool
        Whether to return distinct interactions. Default is False.

    Returns
    -------
    pd.DataFrame
        Aggregated perturbseq data with the following columns:
        - perturbed_species_id: the species id of the perturbed gene
        - target_species_id: the species id of the target gene
        - perturbation_type: the type of perturbation (for perturbatlas, e.g., KO for knockout)
        - perturbation_study: the study that reported the perturbation (for perturbatlas, a study code)
        - standardized_value: the standardized value of the perturbation
        - thresholded_value: the thresholded value of the perturbation
        - dataset_shortname: the shortname of the dataset
    """

    if datasets_w_formatters is None:
        datasets_w_formatters = HUMAN_PERTURBSEQ_DATASETS_TO_FORMATTERS

    _ = process_harmonizome_datasets(datasets_w_formatters.keys(), harmonizome_data_dir)
    perturbseq_data = load_harmonizome_datasets(
        datasets_w_formatters.keys(), harmonizome_data_dir
    )

    perturbseq_interactions_with_species_ids = {
        dataset_shortname: datasets_w_formatters[dataset_shortname](
            perturbseq_data[dataset_shortname][HARMONIZOME_DATASET_FILES.INTERACTIONS],
            species_identifiers,
        ).assign(dataset_shortname=dataset_shortname)
        for dataset_shortname in datasets_w_formatters.keys()
    }

    # add a perturbation_type and perturbation_study columns to tables where they don't exist
    for dataset_shortname in datasets_w_formatters.keys():
        if (
            HARMONIZOME_DEFS.PERTURBATION_TYPE
            not in perturbseq_interactions_with_species_ids[dataset_shortname].columns
        ):
            perturbseq_interactions_with_species_ids[dataset_shortname][
                HARMONIZOME_DEFS.PERTURBATION_TYPE
            ] = PERTURBSEQ_PERTURBATION_TYPES.KD
        if (
            HARMONIZOME_DEFS.PERTURBATION_STUDY
            not in perturbseq_interactions_with_species_ids[dataset_shortname].columns
        ):
            perturbseq_interactions_with_species_ids[dataset_shortname][
                HARMONIZOME_DEFS.PERTURBATION_STUDY
            ] = dataset_shortname

    harmonizome_perturbseq_interactions = pd.concat(
        perturbseq_interactions_with_species_ids.values()
    )

    if return_distinct_interactions:
        return _get_distinct_harmonizome_perturbseq_interactions(
            harmonizome_perturbseq_interactions
        )
    else:
        return harmonizome_perturbseq_interactions


def load_replogle_pvalues_with_species_ids(
    path_to_wide_replogle_pvalues: Union[str, Path],
    species_identifiers: pd.DataFrame,
    return_distinct_interactions: bool = False,
) -> pd.DataFrame:
    """
    Load Replogle et al. Perturb-seq p-values with species IDs.

    Parameters
    ----------
    path_to_wide_replogle_pvalues: Union[str, Path]
        Path to the wide Replogle et al. Perturb-seq p-values file.
    species_identifiers: pd.DataFrame
        Species identifiers dataframe.
    return_distinct_interactions: bool
        Whether to return distinct interactions. Default is False.

    Returns
    -------
    pd.DataFrame
        Replogle et al. Perturb-seq p-values with species IDs.
    """

    wide_replogle_pvalues = pd.read_csv(path_to_wide_replogle_pvalues, index_col=0)
    wide_replogle_pvalues.index.name = PERTURBSEQ_DEFS.TARGET_ENSEMBL_GENE

    perturbations_to_ensg = pd.DataFrame(
        {"perturbed_gene_id": wide_replogle_pvalues.columns}
    ).assign(
        perturbed_ensembl_gene=lambda df: df["perturbed_gene_id"].str.extract(
            r"(ENS[A-Z]+\d+)$"
        )
    )

    long_replogle_pvalues = (
        wide_replogle_pvalues.reset_index()
        .melt(
            id_vars=PERTURBSEQ_DEFS.TARGET_ENSEMBL_GENE,
            var_name="perturbed_gene_id",
            value_name="pvalue",
        )
        .merge(perturbations_to_ensg, on="perturbed_gene_id")
    )

    replogle_gene_ids = pd.DataFrame(
        {
            ONTOLOGIES.ENSEMBL_GENE: list(
                set(long_replogle_pvalues[PERTURBSEQ_DEFS.PERTURBED_ENSEMBL_GENE])
                | set(long_replogle_pvalues[PERTURBSEQ_DEFS.TARGET_ENSEMBL_GENE])
            )
        }
    ).assign(**{FEATURE_ID_VAR_DEFAULT: lambda x: np.arange(len(x))})

    replogle_genes_to_species = features_to_pathway_species(
        feature_identifiers=replogle_gene_ids,
        species_identifiers=species_identifiers,
        ontologies={ONTOLOGIES.ENSEMBL_GENE},
        feature_identifiers_var=ONTOLOGIES.ENSEMBL_GENE,
    ).query(f"{IDENTIFIERS.BQB} != '{BQB.HAS_PART}'")

    replogle_pvalues_with_species_ids = long_replogle_pvalues.merge(
        replogle_genes_to_species.rename(
            columns={
                ONTOLOGIES.ENSEMBL_GENE: PERTURBSEQ_DEFS.TARGET_ENSEMBL_GENE,
                SBML_DFS.S_ID: PERTURBSEQ_DEFS.TARGET_SPECIES_ID,
            }
        )[[PERTURBSEQ_DEFS.TARGET_ENSEMBL_GENE, PERTURBSEQ_DEFS.TARGET_SPECIES_ID]],
        how="left",
    ).merge(
        replogle_genes_to_species.rename(
            columns={
                ONTOLOGIES.ENSEMBL_GENE: PERTURBSEQ_DEFS.PERTURBED_ENSEMBL_GENE,
                SBML_DFS.S_ID: PERTURBSEQ_DEFS.PERTURBED_SPECIES_ID,
            }
        )[
            [
                PERTURBSEQ_DEFS.PERTURBED_ENSEMBL_GENE,
                PERTURBSEQ_DEFS.PERTURBED_SPECIES_ID,
            ]
        ],
        how="left",
    )

    if return_distinct_interactions:
        return _get_distinct_replogle_pvalues(replogle_pvalues_with_species_ids)
    else:
        return replogle_pvalues_with_species_ids


# private functions


def _categorize_perturbseq_row(
    row: pd.Series,
    perturbation_type_col: str,
    standardized_value_col: str,
    threshold_value_col: str,
) -> str:
    """
    Categorize a row of perturbseq data into a direction category.

    Parameters
    ----------
    row : pd.Series
        Row of perturbseq data
    perturbation_type_col : str
        Column name for perturbation type
    standardized_value_col : str
        Column name for standardized value
    threshold_value_col : str
        Column name for threshold value

    Returns
    -------
    pd.Series
        Series with direction category

    Examples
    --------
    df = pd.Series({
        'perturbation_type': 'OE',
        'standardized_value': 1.0,
        'threshold_value': 0.5
    })
    _categorize_perturbseq_row(df, 'perturbation_type', 'standardized_value', 'threshold_value')
    """

    perturbation_type = row[perturbation_type_col]
    value = row[standardized_value_col]
    threshold = abs(row[threshold_value_col])  # Ensure threshold is positive

    # Determine if it's a loss-of-function perturbation
    is_gof = perturbation_type in GOF_PERTURBATION_TYPES
    is_lof = perturbation_type in LOF_PERTURBATION_TYPES

    # Calculate categories based on value and threshold
    if value > threshold:
        # Strong positive change
        if is_gof:
            return PERTURBSEQ_DIRECTIONS.STRONG_ACTIVATION
        elif is_lof:
            return PERTURBSEQ_DIRECTIONS.STRONG_REPRESSION
        else:
            return PERTURBSEQ_DIRECTIONS.STRONG_CHANGE
    elif value > 0:
        # Weak positive change
        if is_gof:
            return PERTURBSEQ_DIRECTIONS.WEAK_ACTIVATION
        elif is_lof:
            return PERTURBSEQ_DIRECTIONS.WEAK_REPRESSION
        else:
            return PERTURBSEQ_DIRECTIONS.WEAK_CHANGE
    elif value >= -threshold:
        # Weak negative change
        if is_gof:
            return PERTURBSEQ_DIRECTIONS.WEAK_REPRESSION
        elif is_lof:
            return PERTURBSEQ_DIRECTIONS.WEAK_ACTIVATION
        else:
            return PERTURBSEQ_DIRECTIONS.WEAK_CHANGE
    else:
        # Strong negative change
        if is_gof:
            return PERTURBSEQ_DIRECTIONS.STRONG_REPRESSION
        elif is_lof:
            return PERTURBSEQ_DIRECTIONS.STRONG_ACTIVATION
        else:
            return PERTURBSEQ_DIRECTIONS.STRONG_CHANGE


def _format_harmonizome_replogle_with_species_ids(
    harmonizome_replogle_interactions: pd.DataFrame, species_identifiers: pd.DataFrame
) -> pd.DataFrame:
    """
    Format Replogle interactions from Harmonizome with species IDs.

    Parameters
    ----------
    harmonizome_replogle_interactions: pd.DataFrame
        Harmonizome's Replogle interactions dataframe.
    species_identifiers: pd.DataFrame
        Species identifiers dataframe.

    Returns
    -------
    pd.DataFrame
        Replogle interactions with species IDs.

    Examples
    --------
    datasets = [HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_K562_ESSENTIAL]
    _ = process_harmonizome_datasets(datasets, "/tmp/harmonizome_data")
    perturbseq_data = load_harmonizome_datasets(datasets, "/tmp/harmonizome_data")
    harmonizome_replogle_interactions_with_species_ids = format_harmonizome_replogle_with_species_ids(perturbseq_data[datasets[0]]["interactions"], species_identifiers)
    """

    # target genes are tracked with NCBI Entrez gene IDs
    harmonizome_replogle_target_identifiers = (
        harmonizome_replogle_interactions[[ONTOLOGIES.NCBI_ENTREZ_GENE]]
        .drop_duplicates()
        .assign(**{FEATURE_ID_VAR_DEFAULT: lambda x: np.arange(len(x))})
    )
    harmonizome_replogle_target_species = (
        features_to_pathway_species(
            feature_identifiers=harmonizome_replogle_target_identifiers,
            species_identifiers=species_identifiers,
            ontologies={ONTOLOGIES.NCBI_ENTREZ_GENE},
            feature_identifiers_var=ONTOLOGIES.NCBI_ENTREZ_GENE,
        )
        .query(f"{IDENTIFIERS.BQB} != '{BQB.HAS_PART}'")
        .rename(
            columns={
                ONTOLOGIES.NCBI_ENTREZ_GENE: PERTURBSEQ_DEFS.TARGET_NCBI_ENTREZ_GENE,
                SBML_DFS.S_ID: PERTURBSEQ_DEFS.TARGET_SPECIES_ID,
            }
        )[[PERTURBSEQ_DEFS.TARGET_NCBI_ENTREZ_GENE, PERTURBSEQ_DEFS.TARGET_SPECIES_ID]]
    )

    # perturbed genes are tracked with Ensembl gene IDs
    harmonizome_replogle_perturbed_identifiers = (
        harmonizome_replogle_interactions[[HARMONIZOME_DEFS.PERTURBED_ENSEMBL_GENE]]
        .drop_duplicates()
        .assign(**{FEATURE_ID_VAR_DEFAULT: lambda x: np.arange(len(x))})
    )
    harmonizome_replogle_perturbed_species = (
        features_to_pathway_species(
            feature_identifiers=harmonizome_replogle_perturbed_identifiers,
            species_identifiers=species_identifiers,
            ontologies={ONTOLOGIES.ENSEMBL_GENE},
            feature_identifiers_var=HARMONIZOME_DEFS.PERTURBED_ENSEMBL_GENE,
        )
        .query(f"{IDENTIFIERS.BQB} != '{BQB.HAS_PART}'")
        .rename(
            columns={
                ONTOLOGIES.ENSEMBL_GENE: PERTURBSEQ_DEFS.PERTURBED_ENSEMBL_GENE,
                SBML_DFS.S_ID: PERTURBSEQ_DEFS.PERTURBED_SPECIES_ID,
            }
        )[
            [
                PERTURBSEQ_DEFS.PERTURBED_ENSEMBL_GENE,
                PERTURBSEQ_DEFS.PERTURBED_SPECIES_ID,
            ]
        ]
    )

    harmonizome_replogle_interactions_with_species_ids = (
        harmonizome_replogle_interactions.rename(
            columns={
                ONTOLOGIES.NCBI_ENTREZ_GENE: PERTURBSEQ_DEFS.TARGET_NCBI_ENTREZ_GENE
            }
        )
        .merge(
            harmonizome_replogle_target_species,
            how="left",
            on=PERTURBSEQ_DEFS.TARGET_NCBI_ENTREZ_GENE,
        )
        .merge(
            harmonizome_replogle_perturbed_species,
            how="left",
            on=PERTURBSEQ_DEFS.PERTURBED_ENSEMBL_GENE,
        )
    )

    return harmonizome_replogle_interactions_with_species_ids[
        [
            PERTURBSEQ_DEFS.PERTURBED_SPECIES_ID,
            PERTURBSEQ_DEFS.TARGET_SPECIES_ID,
            HARMONIZOME_DEFS.STANDARDIZED_VALUE,
            HARMONIZOME_DEFS.THRESHOLDED_VALUE,
        ]
    ]


def _format_perturbatlas_with_species_ids(
    perturbatlas_interactions: pd.DataFrame, species_identifiers: pd.DataFrame
) -> pd.DataFrame:
    """
    Format PerturbAtlas interactions with species IDs.

    Parameters
    ----------
    perturbatlas_interactions: pd.DataFrame
        PerturbAtlas interactions dataframe.
    species_identifiers: pd.DataFrame
        Species identifiers dataframe.

    Returns
    -------
    pd.DataFrame
        PerturbAtlas interactions with species IDs.

    Examples
    --------
    datasets = [HARMONIZOME_DATASET_SHORTNAMES.PERTURB_ATLAS_MOUSE]
    _ = process_harmonizome_datasets(datasets, "/tmp/harmonizome_data")
    perturbseq_data = load_harmonizome_datasets(datasets, "/tmp/harmonizome_data")
    perturbatlas_interactions_with_species_ids = format_perturbatlas_with_species_ids(perturbseq_data[datasets[0]]["interactions"], species_identifiers)
    """

    perturbatlas_gene_ids = pd.DataFrame(
        {
            ONTOLOGIES.NCBI_ENTREZ_GENE: list(
                set(perturbatlas_interactions[ONTOLOGIES.NCBI_ENTREZ_GENE])
                | set(
                    perturbatlas_interactions[
                        HARMONIZOME_DEFS.PERTURBED_NCBI_ENTREZ_GENE
                    ]
                )
            )
        }
    ).assign(**{FEATURE_ID_VAR_DEFAULT: lambda x: np.arange(len(x))})

    perturbatlas_genes_to_species = features_to_pathway_species(
        feature_identifiers=perturbatlas_gene_ids,
        species_identifiers=species_identifiers,
        ontologies={ONTOLOGIES.NCBI_ENTREZ_GENE},
        feature_identifiers_var=ONTOLOGIES.NCBI_ENTREZ_GENE,
    ).query(f"{IDENTIFIERS.BQB} != '{BQB.HAS_PART}'")

    perturbatlas_interactions_with_species_ids = (
        perturbatlas_interactions.rename(
            columns={
                ONTOLOGIES.NCBI_ENTREZ_GENE: PERTURBSEQ_DEFS.TARGET_NCBI_ENTREZ_GENE
            }
        )
        .merge(
            perturbatlas_genes_to_species.rename(
                columns={
                    ONTOLOGIES.NCBI_ENTREZ_GENE: PERTURBSEQ_DEFS.TARGET_NCBI_ENTREZ_GENE,
                    SBML_DFS.S_ID: PERTURBSEQ_DEFS.TARGET_SPECIES_ID,
                }
            )[
                [
                    PERTURBSEQ_DEFS.TARGET_NCBI_ENTREZ_GENE,
                    PERTURBSEQ_DEFS.TARGET_SPECIES_ID,
                ]
            ],
            how="left",
            on=PERTURBSEQ_DEFS.TARGET_NCBI_ENTREZ_GENE,
        )
        .merge(
            perturbatlas_genes_to_species.rename(
                columns={
                    ONTOLOGIES.NCBI_ENTREZ_GENE: PERTURBSEQ_DEFS.PERTURBED_NCBI_ENTREZ_GENE,
                    SBML_DFS.S_ID: PERTURBSEQ_DEFS.PERTURBED_SPECIES_ID,
                }
            )[
                [
                    PERTURBSEQ_DEFS.PERTURBED_NCBI_ENTREZ_GENE,
                    PERTURBSEQ_DEFS.PERTURBED_SPECIES_ID,
                ]
            ],
            how="left",
            on=PERTURBSEQ_DEFS.PERTURBED_NCBI_ENTREZ_GENE,
        )
    )

    return perturbatlas_interactions_with_species_ids[
        [
            PERTURBSEQ_DEFS.PERTURBED_SPECIES_ID,
            PERTURBSEQ_DEFS.TARGET_SPECIES_ID,
            HARMONIZOME_DEFS.PERTURBATION_TYPE,
            HARMONIZOME_DEFS.PERTURBATION_STUDY,
            HARMONIZOME_DEFS.STANDARDIZED_VALUE,
            HARMONIZOME_DEFS.THRESHOLDED_VALUE,
        ]
    ]


def _get_distinct_harmonizome_perturbseq_interactions(
    aggregated_perturbseq_data_with_species_ids: pd.DataFrame,
) -> pd.DataFrame:
    """Reduce the harmonizome perturbseq data to a single entry per study-type-perturbed-target pair."""

    grouped = (
        aggregated_perturbseq_data_with_species_ids.assign(
            **{"absolute std value": lambda df: df["Standardized Value"].abs()}
        )
        .sort_values("absolute std value", ascending=False)
        .groupby(
            [
                "dataset_shortname",
                "perturbation_study",
                "perturbation_type",
                "perturbed_species_id",
                "target_species_id",
            ]
        )
    )

    distinct_harmonizome_perturbseq_interactions = (
        grouped.agg(
            {
                "Standardized Value": "first",
                "Threshold Value": "first",
            }
        )
        .assign(n_interactions=grouped.size())
        .reset_index()
    )

    return distinct_harmonizome_perturbseq_interactions


def _get_distinct_replogle_pvalues(
    replogle_pvalues_with_species_ids: pd.DataFrame,
) -> pd.DataFrame:
    """Reduce the Replogle reported significance to a single entry per perturbed-target pair."""

    grouped = replogle_pvalues_with_species_ids.groupby(
        ["perturbed_species_id", "target_species_id"]
    )
    distinct_replogle_pvalues = (
        grouped.agg({"pvalue": "min"}).assign(n=grouped.size()).reset_index()
    )
    return distinct_replogle_pvalues


# human perturbseq datasets to formatters


HUMAN_PERTURBSEQ_DATASETS_TO_FORMATTERS = {
    HARMONIZOME_DATASET_SHORTNAMES.PERTURB_ATLAS: _format_perturbatlas_with_species_ids,
    HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_K562_ESSENTIAL: _format_harmonizome_replogle_with_species_ids,
    HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_K562_GENOME_WIDE: _format_harmonizome_replogle_with_species_ids,
    HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_RPE1_ESSENTIAL: _format_harmonizome_replogle_with_species_ids,
}
