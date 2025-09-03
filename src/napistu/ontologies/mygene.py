import itertools
import logging
from types import GeneratorType
from typing import Dict, List, Set, Union

import mygene
import pandas as pd

from napistu.constants import ONTOLOGIES
from napistu.ontologies.constants import (
    INTERCONVERTIBLE_GENIC_ONTOLOGIES,
    MYGENE_DEFAULT_QUERIES,
    MYGENE_DEFS,
    MYGENE_QUERY_DEFS_LIST,
    NAPISTU_FROM_MYGENE_FIELDS,
    NAPISTU_TO_MYGENE_FIELDS,
    SPECIES_TO_TAXID,
)

# Configure logging to suppress biothings warnings
logging.getLogger("biothings.client").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def create_python_mapping_tables(
    mappings: Set[str], species: str = "Homo sapiens", test_mode: bool = False
) -> Dict[str, pd.DataFrame]:
    """Create genome-wide mapping tables between Entrez and other gene identifiers.

    Python equivalent of create_bioconductor_mapping_tables using MyGene.info API.

    Parameters
    ----------
    mappings : Set[str]
        Set of ontologies to create mappings for. Must be valid ontologies from
        INTERCONVERTIBLE_GENIC_ONTOLOGIES.
    species : str, default "Homo sapiens"
        Species name (e.g., "Homo sapiens", "Mus musculus"). Must be a key in
        SPECIES_TO_TAXID or a valid NCBI taxonomy ID.
    test_mode : bool, default False
        If True, only fetch the first 1000 genes for testing purposes.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with ontology names as keys and DataFrames as values.
        Each DataFrame has Entrez gene IDs as index and mapped identifiers as values.

    Raises
    ------
    ValueError
        If any requested mappings are invalid or species is not recognized.
    ImportError
        If mygene package is not available.

    Notes
    -----
    The function uses MyGene.info API to fetch gene annotations and creates mapping
    tables between different gene identifier systems. It supports various ontologies
    like Ensembl genes/transcripts/proteins, UniProt, gene symbols, etc.

    Examples
    --------
    >>> mappings = {'ensembl_gene', 'symbol', 'uniprot'}
    >>> tables = create_python_mapping_tables(mappings, 'Homo sapiens')
    >>> print(tables['symbol'].head())
    """

    mygene_fields = _format_mygene_fields(mappings)

    # Convert species name to taxonomy ID
    taxa_id = _format_mygene_species(species)

    # Initialize MyGene client
    mg = mygene.MyGeneInfo()

    # Fetch comprehensive gene data
    logger.info("Fetching genome-wide gene data from MyGene...")
    all_genes_df = _fetch_mygene_data_all_queries(
        mg=mg, taxa_id=taxa_id, fields=mygene_fields, test_mode=test_mode
    )

    if all_genes_df.empty:
        raise ValueError(f"No gene data retrieved for species: {species}")

    logger.info(f"Retrieved {len(all_genes_df)} genes and RNAs")
    mapping_tables = _create_mygene_mapping_tables(all_genes_df, mygene_fields)

    return mapping_tables


def _fetch_mygene_data_all_queries(
    mg: mygene.MyGeneInfo,
    taxa_id: int,
    fields: List[str],
    query_strategies: List[str] = MYGENE_DEFAULT_QUERIES,
    test_mode: bool = False,
) -> pd.DataFrame:
    """Fetch comprehensive gene data from MyGene using multiple query strategies.

    Parameters
    ----------
    mg : mygene.MyGeneInfo
        Initialized MyGene.info client
    taxa_id : int
        NCBI taxonomy ID for the species
    fields : List[str]
        List of MyGene.info fields to retrieve
    query_strategies : List[str], default MYGENE_DEFAULT_QUERIES
        List of query strategies to use from MYGENE_QUERY_DEFS_LIST
    test_mode : bool, default False
        If True, only fetch first 1000 genes

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with gene data from all queries

    Raises
    ------
    ValueError
        If any query strategies are invalid
    """

    all_results = []

    # Validate queries
    invalid_queries = set(query_strategies) - set(MYGENE_QUERY_DEFS_LIST)
    if invalid_queries:
        raise ValueError(
            f"Invalid queries: {', '.join(invalid_queries)}. "
            f"Valid queries are: {', '.join(MYGENE_QUERY_DEFS_LIST)}"
        )

    for query in query_strategies:
        results_df = _fetch_mygene_data(
            mg=mg, query=query, taxa_id=taxa_id, fields=fields, test_mode=test_mode
        )

        all_results.append(results_df)

    return pd.concat(all_results)


def _format_mygene_fields(mappings: Set[str]) -> Set[str]:
    """Format and validate ontology mappings for MyGene.info queries.

    Parameters
    ----------
    mappings : Set[str]
        Set of ontologies to validate and convert to MyGene.info field names

    Returns
    -------
    Set[str]
        Set of valid MyGene.info field names including NCBI_ENTREZ_GENE

    Raises
    ------
    ValueError
        If any mappings are invalid
    """
    # Validate inputs
    invalid_mappings = mappings - INTERCONVERTIBLE_GENIC_ONTOLOGIES
    if invalid_mappings:
        raise ValueError(
            f"Invalid mappings: {', '.join(invalid_mappings)}. "
            f"Valid options are: {', '.join(INTERCONVERTIBLE_GENIC_ONTOLOGIES)}"
        )

    logger.info(
        f"Creating mapping tables from entrez genes to/from {', '.join(mappings)}"
    )

    # Convert ontologies to MyGene fields and ensure NCBI_ENTREZ_GENE is included
    mygene_fields = {NAPISTU_TO_MYGENE_FIELDS[ontology] for ontology in mappings}
    mygene_fields.add(MYGENE_DEFS.NCBI_ENTREZ_GENE)

    return mygene_fields


def _format_mygene_species(species: Union[str, int]) -> int:
    """Convert species name or taxonomy ID to NCBI taxonomy ID.

    Parameters
    ----------
    species : Union[str, int]
        Species name (e.g. "Homo sapiens") or NCBI taxonomy ID

    Returns
    -------
    int
        NCBI taxonomy ID

    Raises
    ------
    ValueError
        If species name is not recognized
    """
    if isinstance(species, int):
        logger.debug(f"Using taxonomy ID: {species}")
        return species
    else:
        if species not in SPECIES_TO_TAXID:
            raise ValueError(
                f"Invalid species: {species}. Please use a species name in "
                "SPECIES_TO_TAXID or directly pass the NCBI Taxonomy ID."
            )

        taxid = SPECIES_TO_TAXID[species]
        logger.debug(f"Using species name: {species}; taxid: {taxid}")

        return taxid


def _fetch_mygene_data(
    mg: mygene.MyGeneInfo,
    query: str,
    taxa_id: int,
    fields: List[str],
    test_mode: bool = False,
) -> pd.DataFrame:
    """Fetch gene data from MyGene.info for a single query.

    Parameters
    ----------
    mg : mygene.MyGeneInfo
        Initialized MyGene.info client
    query : str
        Query string to search for genes
    taxa_id : int
        NCBI taxonomy ID for the species
    fields : List[str]
        List of MyGene.info fields to retrieve
    test_mode : bool, default False
        If True, only fetch first 1000 genes

    Returns
    -------
    pd.DataFrame
        DataFrame containing gene data from the query

    Raises
    ------
    ValueError
        If query results are not in expected format
    """
    logger.debug(f"Querying: {query}")

    result = mg.query(query, species=taxa_id, fields=",".join(fields), fetch_all=True)

    # Validate result is a generator
    if isinstance(result, GeneratorType):
        all_hits = []

        if test_mode:
            # Only look at first 1000 genes in test mode
            result = itertools.islice(result, 1000)

        for i, gene in enumerate(result):
            all_hits.append(gene)

    else:
        raise ValueError("The query results are not a generator")

    results_df = pd.DataFrame(all_hits).assign(query_type=query)

    if results_df.empty:
        logger.warning(
            f"No results found for {query} of species taxa id: {taxa_id} "
            f"and fields: {', '.join(fields)}"
        )
        return pd.DataFrame()
    else:
        logger.info(f"Retrieved {results_df.shape[0]} genes from {query}")
        return results_df


def unnest_mygene_ontology(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Unnest a column containing list of dicts in MyGene.info results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing MyGene.info results
    field : str
        Field name to unnest, must contain a period to indicate nesting

    Returns
    -------
    pd.DataFrame
        DataFrame with unnested values, containing columns for entrez ID and the
        unnested field value

    Raises
    ------
    ValueError
        If field format is invalid or data structure is unexpected
    """
    if "." in field:
        # Extract nested ontology field
        col_name, key_name = field.split(".")
    else:
        raise ValueError(
            f"This function should only be called on a nested mygene ontology "
            f"field; but you passed: {field} (the period indicates nesting)"
        )

    valid_df = df.dropna()
    rows = []
    for i, row in valid_df.iterrows():
        entrez = row[MYGENE_DEFS.NCBI_ENTREZ_GENE]

        if isinstance(row[col_name], list):
            for item in row[col_name]:
                rows.append([entrez, item[key_name]])
        elif isinstance(row[col_name], dict):
            rows.append([entrez, row[col_name][key_name]])
        else:
            raise ValueError(f"Unexpected type: {type(row[col_name])} for row {i}")

    return pd.DataFrame(rows, columns=[MYGENE_DEFS.NCBI_ENTREZ_GENE, field])


def _create_mygene_mapping_tables(
    mygene_results_df: pd.DataFrame, mygene_fields: Set[str]
) -> Dict[str, pd.DataFrame]:
    """Create mapping tables from MyGene.info query results.

    Parameters
    ----------
    mygene_results_df : pd.DataFrame
        DataFrame containing MyGene.info query results
    mygene_fields : Set[str]
        Set of MyGene.info fields that were queried

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping ontology names to DataFrames containing identifier mappings
    """
    mapping_tables = {}
    for field in mygene_fields:
        logger.info(f"Processing field: {field}")

        # Select entrezgene + query field
        if field == MYGENE_DEFS.NCBI_ENTREZ_GENE:
            tbl = mygene_results_df.loc[:, [MYGENE_DEFS.NCBI_ENTREZ_GENE]]
        elif "." in field:
            ontology, entity = field.split(".")
            tbl = unnest_mygene_ontology(
                mygene_results_df.loc[:, [MYGENE_DEFS.NCBI_ENTREZ_GENE, ontology]],
                field,
            )
        else:
            tbl = mygene_results_df.loc[:, [MYGENE_DEFS.NCBI_ENTREZ_GENE, field]]

        mapping_tables[NAPISTU_FROM_MYGENE_FIELDS[field]] = (
            # Rename records
            tbl.rename(columns={c: NAPISTU_FROM_MYGENE_FIELDS[c] for c in tbl.columns})
            # Force all records to be strings
            .astype(str)
            # Remove duplicates
            .drop_duplicates()
            # Set index
            .set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)
        )

    return mapping_tables
