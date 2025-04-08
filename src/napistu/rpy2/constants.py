"""Module for Rpy2 module-specific constants"""

# Contextualization
# Proteinatlas
from __future__ import annotations

from types import SimpleNamespace

from napistu.constants import ONTOLOGIES
from napistu.constants import MINI_SBO_FROM_NAME

# available ontologies for mapping via bioconductor "org" packages as part of rpy2.rids
# ontologies which are valid to map to and/or from when adding annotations to an SBML_dfs model.
BIOC_VALID_EXPANDED_SPECIES_ONTOLOGIES = {
    ONTOLOGIES.ENSEMBL_GENE,
    ONTOLOGIES.ENSEMBL_TRANSCRIPT,
    ONTOLOGIES.ENSEMBL_PROTEIN,
    ONTOLOGIES.NCBI_ENTREZ_GENE,
    ONTOLOGIES.UNIPROT,
    ONTOLOGIES.GENE_NAME,
    ONTOLOGIES.SYMBOL,
}

# bioc ontologies used for linking systematic identifiers
# (entrez is not part of this list because it forms the gene index)
BIOC_DOGMATIC_MAPPING_ONTOLOGIES = {
    ONTOLOGIES.ENSEMBL_GENE,
    ONTOLOGIES.ENSEMBL_TRANSCRIPT,
    ONTOLOGIES.ENSEMBL_PROTEIN,
    ONTOLOGIES.UNIPROT,
    ONTOLOGIES.GENE_NAME,
    ONTOLOGIES.SYMBOL,
}
BIOC_PROTEIN_ONTOLOGIES = [ONTOLOGIES.UNIPROT, ONTOLOGIES.ENSEMBL_PROTEIN]
BIOC_GENE_ONTOLOGIES = [
    ONTOLOGIES.NCBI_ENTREZ_GENE,
    ONTOLOGIES.ENSEMBL_GENE,
    ONTOLOGIES.ENSEMBL_TRANSCRIPT,
]
BIOC_NAME_ONTOLOGIES = {
    ONTOLOGIES.GENE_NAME: 0,
    ONTOLOGIES.SYMBOL: 1,
    ONTOLOGIES.UNIPROT: 2,
    ONTOLOGIES.ENSEMBL_PROTEIN: 3,
}

# prefixes for bioconductor mapping tables
BIOC_NOMENCLATURE = SimpleNamespace(
    CHR_TBL="CHR",
    ENSG_TBL="ENSEMBL",
    ENST_TBL="ENSEMBLTRANS",
    ENSP_TBL="ENSEMBLPROT",
    UNIPROT_TBL="UNIPROT",
    NAME_TBL="GENENAME",
    SYMBOL_TBL="SYMBOL",
    CHROMOSOME="chromosome",
    NCBI_ENTREZ_GENE="gene_id",
    ENSEMBL_GENE="ensembl_id",
    ENSEMBL_TRANSCRIPT="trans_id",
    ENSEMBL_PROTEIN="prot_id",
    UNIPROT="uniprot_id",
    GENE_NAME="gene_name",
    SYMBOL="symbol",
)

# netcontextr constants

COL_GENE = "gene"
COL_PROTEIN_1 = "protein1"
COL_PROTEIN_2 = "protein2"

FIELD_INTERACTIONS = "interactions"
FIELD_GENES = "genes"
FIELD_REACTIONS = "reactions"

# Netcontextr reactions
COL_ROLE = "role"
COL_REACTION_ID = "reaction_id"
COL_STOICHIOMETRY = "stoi"

SBO_TERM_MAP = {
    "reactant": "substrate",
    "product": "product",
    "catalyst": "catalyst",
    "interactor": "interactor",
    "stimulator": "activator",
    "inhibitor": "inhibitor",
}

NETCONTEXTR_ONTOLOGY = "ensembl_gene"


def _map_sbo_identifiers() -> dict[str, str]:
    """Map sbo identifiers to netcontextr identifiers"""

    sbo_map = {MINI_SBO_FROM_NAME[k]: v for k, v in SBO_TERM_MAP.items()}

    return sbo_map


NETCONTEXTR_SBO_MAP = _map_sbo_identifiers()
