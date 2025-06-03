import logging
from typing import Dict, List, Set, Union
from types import GeneratorType

import mygene
import pandas as pd

from napistu.constants import ONTOLOGIES
from napistu.ontologies.constants import (
    MYGENE_DEFS,
    NAPISTU_FROM_MYGENE_FIELDS,
    NAPISTU_TO_MYGENE_FIELDS,
    INTERCONVERTIBLE_GENIC_ONTOLOGIES,
    MYGENE_QUERY_DEFS_LIST,
    MYGENE_DEFAULT_QUERIES,
    SPECIES_TO_TAXID,
)

logger = logging.getLogger(__name__)


def create_python_mapping_tables(
    mappings: Set[str],
    species: str = "Homo sapiens",
    batch_size: int = 1000
) -> Dict[str, pd.DataFrame]:
    """
    Create genome-wide mapping tables between Entrez and other gene identifiers.
    
    Python equivalent of create_bioconductor_mapping_tables using MyGene.info
    
    Parameters
    ----------
    mappings : set[str]
        Set of ontologies to create mappings for. Valid options:
        "ensembl_gene", "ensembl_transcript", "ensembl_protein", 
        "uniprot", "symbol", "gene_name"
    species : str, default "homo sapiens"
        Species name (e.g., "homo sapiens", "mus musculus")
    batch_size : int, default 1000
        Number of genes to fetch per batch from MyGene
        
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with ontology names as keys and DataFrames as values.
        Each DataFrame has entrez gene IDs as index and the mapped identifiers.
        
    Raises
    ------
    ValueError
        If any requested mappings are invalid
    ImportError
        If mygene package is not available
        
    Examples
    --------
    >>> mappings = {'ensembl_gene', 'symbol', 'uniprot'}
    >>> tables = create_python_mapping_tables(mappings, 'homo sapiens')
    >>> print(tables['symbol'].head())
    """
    
    mygene_fields = _format_mygene_fields(mappings)

    # Convert species name
    taxa_id = _format_mygene_species(species)
        
    # Initialize MyGene client
    mg = mygene.MyGeneInfo()
    
    # Fetch comprehensive gene data
    logger.info("Fetching genome-wide gene data from MyGene...")
    all_genes_df = _fetch_mygene_data_all_queries(
        mg = mg, 
        taxa_id = taxa_id, 
        fields = mygene_fields, 
        batch_size = batch_size
    )
    
    if all_genes_df.empty:
        raise ValueError(f"No gene data retrieved for species: {species}")
    
    logger.info(f"Retrieved {len(all_genes_df)} genes and RNAs")
    mapping_tables = _create_mygene_mapping_tables(all_genes_df, mygene_fields)
    
    return mapping_tables


def _fetch_mygene_data_all_queries(
    mg, 
    taxa_id: int, 
    fields: List[str], 
    query_strategies: List[str] = MYGENE_DEFAULT_QUERIES,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Fetch comprehensive gene data from MyGene using multiple query strategies
    """
    
    all_results = []
    
    # validate that all queries are in MYGENE_QUERY_DEFS
    invalid_queries = set(query_strategies) - set(MYGENE_QUERY_DEFS_LIST)
    if invalid_queries:
        raise ValueError(f"Invalid queries: {', '.join(invalid_queries)}. Valid queries are: {', '.join(MYGENE_QUERY_DEFS_LIST)}")

    for query in query_strategies:
        results_df = _fetch_mygene_data(
            mg = mg, 
            query = query, 
            taxa_id = taxa_id, 
            fields = fields, 
            batch_size = batch_size
        )

        all_results.append(results_df)
    
    return pd.concat(all_results) 


def _format_mygene_fields(mappings: Set[str]) -> List[str]:

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

    # Get all requested MyGene fields + entrez whether requested or not
    mygene_fields = {MYGENE_DEFS.NCBI_ENTREZ_GENE} | {
        NAPISTU_TO_MYGENE_FIELDS[ontology] for ontology in mappings
    }

    return mygene_fields

def _format_mygene_species(species: Union[str, int]) -> int:

    if isinstance(species, int):
        logger.debug(f"Using taxonomy ID: {species}")
        return species
    else: 

        if species not in SPECIES_TO_TAXID.keys():
            raise ValueError(f"Invalid species: {species}. Please use a species name in SPECIES_TO_TAXID or directly pass the NCBI Taxonomy ID of the species you are interested in.")

        taxid = SPECIES_TO_TAXID[species]
        logger.debug(f"Using species name: {species}; taxid: {taxid}")

        return taxid

def _fetch_mygene_data(mg: mygene.MyGeneInfo, query: str, taxa_id: int, fields: List[str], batch_size: int = 1000) -> dict:
        
    """
    Fetch comprehensive gene data from MyGene using multiple query strategies
    """

    logger.debug(f"Querying: {query}")
    
    result = mg.query(
        query,
        species=taxa_id,
        fields=','.join(fields),
        fetch_all=True
    )

    # Simple check: is it a generator?
    if isinstance(result, GeneratorType):
        all_hits = []
        for i, gene in enumerate(result):
            all_hits.append(gene)

    else:
        ValueError("The query results are not a generator")

    results_df = pd.DataFrame(all_hits).assign(query_type = query)

    if results_df.empty:
        logger.warning(f"No results found for {query} of species taxa id: {taxa_id} and fields: {', '.join(fields)}")
        return pd.DataFrame()
    else:
        logger.info(f"Retrieved {results_df.shape[0]} genes from {query}")
        return results_df    
    
def unnest_mygene_ontology(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Unnest a column containing list of dicts"""
    
    if "." in field:
        # extract a nested ontology field
        col_name, key_name = field.split('.')
    else:
        ValueError("This functions should only be called on a nested mygeneontology field; but you passed: {field} (the period indicated nesting)")

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
    mygene_results_df: pd.DataFrame, mygene_fields: List[str]) -> Dict[str, pd.DataFrame]:

    mapping_tables = {}
    for field in mygene_fields:

        logger.info(f"Processing field: {field}")

        # select entrezgene + the query field
        if field == MYGENE_DEFS.NCBI_ENTREZ_GENE:
            tbl = mygene_results_df.loc[:, [MYGENE_DEFS.NCBI_ENTREZ_GENE]]
        elif "." in field:
            ontology, entity = field.split('.')
            
            tbl = unnest_mygene_ontology(mygene_results_df.loc[:, [MYGENE_DEFS.NCBI_ENTREZ_GENE, ontology]], field)
        else:
            tbl = mygene_results_df.loc[:, [MYGENE_DEFS.NCBI_ENTREZ_GENE, field]]

        mapping_tables[NAPISTU_FROM_MYGENE_FIELDS[field]] = (
            # rename records
            tbl.rename(columns={c: NAPISTU_FROM_MYGENE_FIELDS[c] for c in tbl.columns})
            # force all rows to be strs
            .astype(str)
            # set the index to be the entrez gene id
            .set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)
        )

    return mapping_tables
