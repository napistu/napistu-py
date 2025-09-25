import logging
from types import SimpleNamespace
from typing import Dict

from napistu.constants import ONTOLOGIES
from napistu.ontologies._validation import SpeciesTypeOntologyMapping

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

MYGENE_QUERY_DEFS_LIST = list(MYGENE_QUERY_DEFS.__dict__.values())

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

# PubChem constants
PUBCHEM_ID_ENTRYPOINT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/property/IUPACName,Title,IsomericSMILES,SMILES/JSON"

PUBCHEM_DEFS = SimpleNamespace(NAME="name", SMILES="smiles", PUBCHEM_ID="pubchem_id")

PUBCHEM_PROPERTIES = SimpleNamespace(
    CID="CID",
    IUPAC_NAME="IUPACName",
    TITLE="Title",
    ISOMERIC_SMILES="IsomericSMILES",
    SMILES="SMILES",
    PROPERTY_TABLE="PropertyTable",
    PROPERTIES="Properties",
)

# miRBase constants

MIRBASE_TABLES = SimpleNamespace(
    MATURE_DATABASE_LINKS="mature_database_links",
    MATURE_DATABASE_URL="mature_database_url",
    URL="url",
    HEADER="header",
    DATABASE_ENTRY="database_entry",
    DATABASE="database",
    URL_TEMPLATE="url_template",  # unused
    RNA_ID="rna_id",  # distinct molecules
    PRIMARY_ID="primary_id",
    SECONDARY_ID="secondary_id",
)

MIRBASE_TABLE_SPECS = {
    MIRBASE_TABLES.MATURE_DATABASE_LINKS: {
        MIRBASE_TABLES.URL: "https://mirbase.org/download/CURRENT/database_files/mature_database_links.txt",
        MIRBASE_TABLES.HEADER: [
            MIRBASE_TABLES.RNA_ID,
            MIRBASE_TABLES.DATABASE_ENTRY,
            MIRBASE_TABLES.PRIMARY_ID,
            MIRBASE_TABLES.SECONDARY_ID,
        ],
    },
    MIRBASE_TABLES.MATURE_DATABASE_URL: {
        MIRBASE_TABLES.URL: "https://mirbase.org/download/CURRENT/database_files/mature_database_url.txt",
        MIRBASE_TABLES.HEADER: [
            MIRBASE_TABLES.DATABASE_ENTRY,
            MIRBASE_TABLES.DATABASE,
            MIRBASE_TABLES.URL_TEMPLATE,
            "unknown",
        ],
    },
}

# Add your species type mappings

SPECIES_TYPES = SimpleNamespace(
    COMPLEX="complex",
    DRUG="drug",
    METABOLITE="metabolite",
    PROTEIN="protein",
    REGULATORY_RNA="regulatory_rna",
    OTHER="other",
    UNKNOWN="unknown",
)

SPECIES_TYPE_PLURAL = {
    SPECIES_TYPES.COMPLEX: "complexes",
    SPECIES_TYPES.DRUG: "drugs",
    SPECIES_TYPES.METABOLITE: "metabolites",
    SPECIES_TYPES.PROTEIN: "proteins",
    SPECIES_TYPES.REGULATORY_RNA: "regulatory RNAs",
    SPECIES_TYPES.OTHER: "other",
    SPECIES_TYPES.UNKNOWN: "unknowns",
}

SPECIES_TYPE_ONTOLOGIES = {
    SPECIES_TYPES.COMPLEX: [ONTOLOGIES.CORUM],
    SPECIES_TYPES.DRUG: [ONTOLOGIES.DRUGBANK, ONTOLOGIES.KEGG_DRUG],
    SPECIES_TYPES.METABOLITE: [
        ONTOLOGIES.BIGG_METABOLITE,
        ONTOLOGIES.CHEBI,
        ONTOLOGIES.KEGG,
        ONTOLOGIES.PUBCHEM,
        ONTOLOGIES.SMILES,
    ],
    SPECIES_TYPES.PROTEIN: [
        ONTOLOGIES.ENSEMBL_GENE,
        ONTOLOGIES.ENSEMBL_TRANSCRIPT,
        ONTOLOGIES.ENSEMBL_PROTEIN,
        ONTOLOGIES.NCBI_ENTREZ_GENE,
        ONTOLOGIES.UNIPROT,
        ONTOLOGIES.SYMBOL,
        ONTOLOGIES.GENE_NAME,
    ],
    SPECIES_TYPES.REGULATORY_RNA: [ONTOLOGIES.MIRBASE, ONTOLOGIES.RNACENTRAL],
}

# if the ontology's associated with these categories are seen then other categories are ignored
PRIORITIZED_SPECIES_TYPES = {SPECIES_TYPES.DRUG, SPECIES_TYPES.COMPLEX}

# Validate the mapping and create the flattened lookup at module load time
validated_mapping = SpeciesTypeOntologyMapping(mappings=SPECIES_TYPE_ONTOLOGIES)
ONTOLOGY_TO_SPECIES_TYPE = validated_mapping.create_ontology_to_species_type_mapping()
