from __future__ import annotations

import logging
import pandas as pd
from napistu import utils
from fs import open_fs
from napistu.constants import ONTOLOGIES
from napistu.ingestion.constants import PROTEINATLAS_SUBCELL_LOC_URL, PROTEINATLAS_DEFS


logger = logging.getLogger(__name__)


def download_hpa_data(target_uri: str, url: str = PROTEINATLAS_SUBCELL_LOC_URL) -> None:
    """Download protein localization data from the Human Protein Atlas.

    Parameters
    ----------
    target_uri : str
        The URI where the HPA data should be saved. Should end with .tsv
    url : str, optional
        URL to download the zipped protein atlas subcellular localization tsv from.
        Defaults to PROTEINATLAS_SUBCELL_LOC_URL.

    Returns
    -------
    None

    Notes
    -----
    Downloads the subcellular localization data from the Human Protein Atlas and saves
    it to the specified target URI. The data is downloaded from the official HPA website
    as a ZIP file and automatically unzipped to extract the TSV.

    Raises
    ------
    ValueError
        If target_uri does not end with .tsv
    """
    if not target_uri.endswith(".tsv"):
        raise ValueError(f"Target URI must end with .tsv, got {target_uri}")

    file_ext = url.split(".")[-1]
    target_filename = url.split("/")[-1].split(f".{file_ext}")[0]
    logger.info("Start downloading proteinatlas %s to %s", url, target_uri)
    # target_filename is the name of the file in the zip file which will be renamed to target_uri
    utils.download_wget(url, target_uri, target_filename=target_filename)

    return None


def load_and_clean_hpa_data(hpa_data_path: str) -> pd.DataFrame:
    """Load and format Human Protein Atlas subcellular localization data.

    Parameters
    ----------
    hpa_data_path : str
        Path to HPA subcellular localization data TSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with genes as rows and GO terms as columns. Each cell
        is a binary value (0 or 1) indicating whether that gene (row) is found in that
        compartment (column). Genes with no compartment annotations are filtered out.

    Notes
    -----
    This function loads subcellular localization data from the Human Protein Atlas
    and creates a binary matrix where rows are genes and columns are GO terms,
    with 1 indicating that a gene is localized to that compartment and 0 indicating
    it is not.

    The function filters out genes that have no compartment annotations and logs
    information about the number of genes filtered and the final matrix dimensions.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist
    ValueError
        If no gene-compartment associations are found in the data
    """
    # Check file exists
    base_path, file_name = utils.get_source_base_and_path(hpa_data_path)

    logger.info("Loading Human Protein Atlas subcellular localization data")

    # Read the TSV file using pandas
    with open_fs(base_path) as base_fs:
        with base_fs.open(file_name, "rb") as f:
            protein_subcellular_localizations = pd.read_csv(
                f, sep="\t", dtype=str, na_values=[""], keep_default_na=True
            )

    # Rename Gene column to be more informative
    protein_subcellular_localizations = protein_subcellular_localizations.rename(
        columns={PROTEINATLAS_DEFS.GENE: ONTOLOGIES.ENSEMBL_GENE}
    )

    # Convert GO id column to lists
    def _split_go_terms(go_terms):
        if pd.isna(go_terms):
            return []
        return go_terms.split(";")

    # Create a list of all gene-GO term pairs
    gene_go_pairs = []
    for _, row in protein_subcellular_localizations.iterrows():
        go_terms = _split_go_terms(row[PROTEINATLAS_DEFS.GO_ID])
        for term in go_terms:
            gene_go_pairs.append(
                {
                    ONTOLOGIES.ENSEMBL_GENE: row[ONTOLOGIES.ENSEMBL_GENE],
                    ONTOLOGIES.GO: term,
                }
            )

    # Convert to DataFrame and pivot to create binary matrix
    gene_go_df = pd.DataFrame(gene_go_pairs)
    if len(gene_go_df) == 0:
        raise ValueError("No gene-compartment associations found in the data")

    localization_matrix = pd.crosstab(
        gene_go_df[ONTOLOGIES.ENSEMBL_GENE], gene_go_df[ONTOLOGIES.GO]
    ).astype(int)

    # Log number of genes without compartments that were filtered
    n_total_genes = len(
        protein_subcellular_localizations[ONTOLOGIES.ENSEMBL_GENE].unique()
    )
    n_genes_with_compartments = len(localization_matrix)
    n_filtered = n_total_genes - n_genes_with_compartments
    if n_filtered > 0:
        logger.debug(
            "Filtered out %d genes with no compartment annotations (from %d total genes)",
            n_filtered,
            n_total_genes,
        )

    logger.info(
        "Created localization matrix with shape %d genes x %d compartments",
        localization_matrix.shape[0],
        localization_matrix.shape[1],
    )

    return localization_matrix
