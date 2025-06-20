from __future__ import annotations

import logging

import pandas as pd
from napistu.rpy2 import (
    require_rpy2,
    report_r_exceptions,
)

from napistu.rpy2.callr import bioconductor_org_r_function, r_dataframe_to_pandas

from napistu.constants import ONTOLOGIES
from napistu.rpy2.constants import BIOC_VALID_EXPANDED_SPECIES_ONTOLOGIES
from napistu.rpy2.constants import BIOC_ONTOLOGY_MAPPING
from napistu.rpy2.constants import BIOC_NOMENCLATURE

logger = logging.getLogger(__name__)


@require_rpy2
@report_r_exceptions
def create_bioconductor_mapping_tables(
    mappings: set[str], species: str, r_paths: str | None = None
) -> dict[str, pd.DataFrame]:
    """Create Bioconductor Mapping Tables.

    Creating a dictionary of mappings between entrez and other ontologies.

    Parameters
    ----------
    mappings : set[str]
        A set of ontologies to work with. The valid ontologies are:
        "ensembl_gene", "ensembl_transcript", and "uniprot".
    species : str
        The organismal species that we are working with (e.g., Homo sapiens).
    r_paths : str | None, optional
        Optional path to a library of R packages.

    Returns
    -------
    dict[str, pd.DataFrame]
        A table of entrez ids, and tables mapping from each ontology in "mappings" to entrez.

    Raises
    ------
    ValueError
        If any of the requested mappings are not supported
    """
    logger.info(
        "Creating mapping tables from entrez genes to/from %s", ", ".join(mappings)
    )

    invalid_mappings = set(mappings).difference(BIOC_VALID_EXPANDED_SPECIES_ONTOLOGIES)
    if len(invalid_mappings) > 0:
        raise ValueError(
            f"{len(invalid_mappings)} mappings could not be created: {', '.join(invalid_mappings)}.\n"
            f"The valid mappings are {', '.join(BIOC_VALID_EXPANDED_SPECIES_ONTOLOGIES)}"
        )

    mappings_dict = {}

    # Create mapping tables for each requested ontology
    for ontology in mappings:
        mappings_dict[ontology] = _create_single_mapping(ontology, species, r_paths)

    return mappings_dict


def _create_single_mapping(
    ontology: str, species: str, r_paths: str | None = None
) -> pd.DataFrame:
    """Create a single mapping table for a given ontology.

    Parameters
    ----------
    ontology : str
        The ontology to map (e.g. ENSEMBL_GENE, UNIPROT)
    species : str
        The organismal species to map
    r_paths : str | None, optional
        Optional path to R packages directory

    Returns
    -------
    pd.DataFrame
        DataFrame containing the mapping between entrez and the target ontology
    """

    if ontology not in BIOC_ONTOLOGY_MAPPING:
        raise ValueError(f"Unsupported ontology: {ontology}")

    table_name, column_name = BIOC_ONTOLOGY_MAPPING[ontology]

    df = r_dataframe_to_pandas(
        bioconductor_org_r_function(table_name, species, r_paths=r_paths)
    )

    # Drop chromosome column if this is the chromosome table
    # this was only introduced so we had a table with 1 row per unique entrez id
    if table_name == BIOC_NOMENCLATURE.CHR_TBL:
        df = df.drop(BIOC_NOMENCLATURE.CHROMOSOME, axis=1)

    # Rename columns and set index
    df = df.rename(
        columns={
            BIOC_NOMENCLATURE.NCBI_ENTREZ_GENE: ONTOLOGIES.NCBI_ENTREZ_GENE,
            column_name: ontology,
        }
    ).set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)

    return df
