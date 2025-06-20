from __future__ import annotations

import logging
import pandas as pd
from fs import open_fs
from napistu import utils

from napistu.constants import ONTOLOGIES
from napistu.ingestion.constants import GTEX_DEFS, GTEX_RNASEQ_EXPRESSION_URL

logger = logging.getLogger(__name__)


def download_gtex_rnaseq(
    target_uri: str, url: str = GTEX_RNASEQ_EXPRESSION_URL
) -> None:
    """Download GTEx RNA-seq expression data.

    Parameters
    ----------
    target_uri : str
        The URI where the GTEx data should be saved
    url : str, optional
        URL to download the GTEx RNA-seq expression data from.
        Defaults to GTEX_RNASEQ_EXPRESSION_URL.

    Returns
    -------
    None

    Notes
    -----
    Downloads GTEx RNA-seq expression data (median TPM per gene per tissue) from the
    specified URL and saves it to the target URI. By default, downloads from GTEx
    Analysis V8 data (dbGaP Accession phs000424.v8.p2).
    """
    logger.info("Start downloading gtex %s to %s", url, target_uri)
    utils.download_wget(url, target_uri)


def load_and_clean_gtex_data(gtex_data_path: str) -> pd.DataFrame:
    """Load and format GTEx tissue specific expression data.

    This function loads tissue-specific expression data from GTEx (median value per gene per tissue).

    Parameters
    ----------
    gtex_data_path : str
        Path to GTEx tissue specific expression data (medians)

    Returns
    -------
    pd.DataFrame
        DataFrame containing all the information from the GTEx file with standardized column names:
        - ensembl_gene_id: Ensembl gene ID without version number
        - ensembl_geneTranscript_id: Original GTEx hybrid gene/transcript ID
        - Description: Gene description/symbol
        - Multiple tissue columns with median TPM values

    Notes
    -----
    The function:
    1. Skips the first 2 lines of the GTEx file (header info)
    2. Creates clean Ensembl gene IDs by removing version numbers
    3. Renames columns for clarity
    4. Reorders columns to put ID and description columns first

    Raises
    ------
    FileNotFoundError
        If the input file does not exist
    """
    # Check file exists
    base_path, file_name = utils.get_source_base_and_path(gtex_data_path)

    logger.info("Loading GTEx tissue specific expression data")

    # Read the TSV file using pandas, skipping first 2 lines
    with open_fs(base_path) as base_fs:
        with base_fs.open(file_name, "rb") as f:
            gtex_expression_data = pd.read_csv(
                f, sep="\t", skiprows=2, dtype=str, na_values=[""], keep_default_na=True
            )

    # Create ensembl_gene_id by removing version numbers from Name column
    gtex_expression_data[ONTOLOGIES.ENSEMBL_GENE] = gtex_expression_data[
        GTEX_DEFS.NAME
    ].str.replace(r"\.[0-9]+$", "", regex=True)

    # Rename Name column to be more informative
    gtex_expression_data = gtex_expression_data.rename(
        columns={
            GTEX_DEFS.NAME: ONTOLOGIES.ENSEMBL_GENE_VERSION,
            GTEX_DEFS.DESCRIPTION: ONTOLOGIES.SYMBOL,
        }
    )

    # Reorder columns to put ID and description columns first
    first_cols = [
        ONTOLOGIES.ENSEMBL_GENE,
        ONTOLOGIES.ENSEMBL_GENE_VERSION,
        ONTOLOGIES.SYMBOL,
    ]
    other_cols = [col for col in gtex_expression_data.columns if col not in first_cols]
    gtex_expression_data = gtex_expression_data[first_cols + other_cols]

    # Convert tissue columns to numeric
    numeric_cols = [col for col in other_cols if col not in first_cols]
    gtex_expression_data[numeric_cols] = gtex_expression_data[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    return gtex_expression_data
