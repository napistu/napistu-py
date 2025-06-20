import logging

from typing import Dict
from types import SimpleNamespace
from napistu.constants import ONTOLOGIES

logger = logging.getLogger(__name__)

# Valid ontologies that can be interconverted
INTERCONVERTIBLE_GENIC_ONTOLOGIES = {
    ONTOLOGIES.ENSEMBL_GENE,
    ONTOLOGIES.ENSEMBL_TRANSCRIPT,
    ONTOLOGIES.ENSEMBL_PROTEIN,
    ONTOLOGIES.NCBI_ENTREZ_GENE,
    ONTOLOGIES.UNIPROT,
    ONTOLOGIES.GENE_NAME,
    ONTOLOGIES.SYMBOL,
}

GENODEXITO_DEFS = SimpleNamespace(
    BIOCONDUCTOR="bioconductor",
    PYTHON="python",
)
GENODEXITO_MAPPERS = {GENODEXITO_DEFS.BIOCONDUCTOR, GENODEXITO_DEFS.PYTHON}

# Mapping from our ontology names to MyGene field names
MYGENE_DEFS = SimpleNamespace(
    ENSEMBL_GENE="ensembl.gene",
    ENSEMBL_TRANSCRIPT="ensembl.transcript",
    ENSEMBL_PROTEIN="ensembl.protein",
    UNIPROT="uniprot.Swiss-Prot",
    SYMBOL="symbol",
    GENE_NAME="name",
    NCBI_ENTREZ_GENE="entrezgene",
)

NAPISTU_TO_MYGENE_FIELDS = {
    ONTOLOGIES.ENSEMBL_GENE: MYGENE_DEFS.ENSEMBL_GENE,
    ONTOLOGIES.ENSEMBL_TRANSCRIPT: MYGENE_DEFS.ENSEMBL_TRANSCRIPT,
    ONTOLOGIES.ENSEMBL_PROTEIN: MYGENE_DEFS.ENSEMBL_PROTEIN,
    ONTOLOGIES.UNIPROT: MYGENE_DEFS.UNIPROT,
    ONTOLOGIES.SYMBOL: MYGENE_DEFS.SYMBOL,
    ONTOLOGIES.GENE_NAME: MYGENE_DEFS.GENE_NAME,
    ONTOLOGIES.NCBI_ENTREZ_GENE: MYGENE_DEFS.NCBI_ENTREZ_GENE,
}

NAPISTU_FROM_MYGENE_FIELDS = {
    MYGENE_DEFS.ENSEMBL_GENE: ONTOLOGIES.ENSEMBL_GENE,
    MYGENE_DEFS.ENSEMBL_TRANSCRIPT: ONTOLOGIES.ENSEMBL_TRANSCRIPT,
    MYGENE_DEFS.ENSEMBL_PROTEIN: ONTOLOGIES.ENSEMBL_PROTEIN,
    MYGENE_DEFS.UNIPROT: ONTOLOGIES.UNIPROT,
    MYGENE_DEFS.SYMBOL: ONTOLOGIES.SYMBOL,
    MYGENE_DEFS.GENE_NAME: ONTOLOGIES.GENE_NAME,
    MYGENE_DEFS.NCBI_ENTREZ_GENE: ONTOLOGIES.NCBI_ENTREZ_GENE,
}


SPECIES_TO_TAXID: Dict[str, int] = {
    # MyGene.info supported common species (9 species with common names)
    "Homo sapiens": 9606,  # human
    "Mus musculus": 10090,  # mouse
    "Rattus norvegicus": 10116,  # rat
    "Drosophila melanogaster": 7227,  # fruitfly
    "Caenorhabditis elegans": 6239,  # nematode
    "Danio rerio": 7955,  # zebrafish
    "Arabidopsis thaliana": 3702,  # thale-cress
    "Xenopus tropicalis": 8364,  # frog
    "Xenopus laevis": 8355,  # frog (alternative species)
    "Sus scrofa": 9823,  # pig
    # Additional commonly used model organisms
    "Saccharomyces cerevisiae": 4932,  # yeast
    "Schizosaccharomyces pombe": 4896,  # fission yeast
    "Gallus gallus": 9031,  # chicken
    "Bos taurus": 9913,  # cow/cattle
    "Canis familiaris": 9615,  # dog
    "Macaca mulatta": 9544,  # rhesus monkey/macaque
    "Pan troglodytes": 9598,  # chimpanzee
    "Escherichia coli": 511145,  # E. coli (K-12 MG1655)
    # Additional species that might be encountered
    "Anopheles gambiae": 7165,  # malaria mosquito
    "Oryza sativa": 4530,  # rice
    "Neurospora crassa": 5141,  # bread mold
    "Kluyveromyces lactis": 28985,  # yeast species
    "Magnaporthe oryzae": 318829,  # rice blast fungus
    "Eremothecium gossypii": 33169,  # cotton fungus
}

MYGENE_QUERY_DEFS = SimpleNamespace(
    BIOLOGICAL_REGION="type_of_gene:biological-region",
    NCRNA="type_of_gene:ncrna",
    PROTEIN_CODING="type_of_gene:protein-coding",
    PSEUDO="type_of_gene:pseudo",
    SNORNA="type_of_gene:snorna",
    UNKNOWN="type_of_gene:unknown",
    OTHER="type_of_gene:other",
    RRNA="type_of_gene:rrna",
    TRNA="type_of_gene:trna",
    SNRNA="type_of_gene:snrna",
)

MYGENE_QUERY_DEFS_LIST = [
    MYGENE_QUERY_DEFS.BIOLOGICAL_REGION,
    MYGENE_QUERY_DEFS.NCRNA,
    MYGENE_QUERY_DEFS.PROTEIN_CODING,
    MYGENE_QUERY_DEFS.PSEUDO,
    MYGENE_QUERY_DEFS.SNORNA,
    MYGENE_QUERY_DEFS.UNKNOWN,
    MYGENE_QUERY_DEFS.OTHER,
    MYGENE_QUERY_DEFS.RRNA,
    MYGENE_QUERY_DEFS.TRNA,
    MYGENE_QUERY_DEFS.SNRNA,
]

MYGENE_DEFAULT_QUERIES = [MYGENE_QUERY_DEFS.PROTEIN_CODING, MYGENE_QUERY_DEFS.NCRNA]

# bioc ontologies used for linking systematic identifiers
# (entrez is not part of this list because it forms the gene index)
PROTEIN_ONTOLOGIES = [ONTOLOGIES.UNIPROT, ONTOLOGIES.ENSEMBL_PROTEIN]
GENE_ONTOLOGIES = [
    ONTOLOGIES.NCBI_ENTREZ_GENE,
    ONTOLOGIES.ENSEMBL_GENE,
    ONTOLOGIES.ENSEMBL_TRANSCRIPT,
]
NAME_ONTOLOGIES = {
    ONTOLOGIES.GENE_NAME: 0,
    ONTOLOGIES.SYMBOL: 1,
    ONTOLOGIES.UNIPROT: 2,
    ONTOLOGIES.ENSEMBL_PROTEIN: 3,
}
