import logging
from types import SimpleNamespace
from typing import Dict

from pandas import DataFrame

logger = logging.getLogger(__name__)


ONTOLOGIES = SimpleNamespace(
    BIGG_METABOLITE="bigg_metabolite",
    BIORXIV="biorxiv",
    CHEBI="chebi",
    CHEMSPIDER="chemspider",
    CORUM="corum",  # protein complexes
    DRUGBANK="drugbank",
    DOI="doi",  # doi.org digital object identifiers
    DX_DOI="dx_doi",  # dx.doi.org digital object identifiers
    EC_CODE="ec-code",  # enzyme commission codes
    EBI_REFSEQ="ebi_refseq",  # ebi DNA reference sequences
    ENSEMBL_GENE="ensembl_gene",
    ENSEMBL_GENE_VERSION="ensembl_gene_version",
    ENSEMBL_TRANSCRIPT="ensembl_transcript",
    ENSEMBL_TRANSCRIPT_VERSION="ensembl_transcript_version",
    ENSEMBL_PROTEIN="ensembl_protein",
    ENSEMBL_PROTEIN_VERSION="ensembl_protein_version",
    ENVIPATH="envipath",  # microbial environmental transformation pathways
    GENE_NAME="gene_name",
    GENOME_NET="genome_net",
    GO="go",  # gene ontology
    GUIDETOPHARMACOLOGY="guidetopharmacology",
    INCHIKEY="inchikey",  # hashed chemical structure
    INTACT="intact",  # intact protein interactions
    KEGG="kegg",
    KEGG_DRUG="kegg.drug",
    MIRBASE="mirbase",  # microRNAs
    MATRIXDB_BIOMOLECULE="matrixdb_biomolecule",  # extracellular matrix molecules and interactions
    MATRIXDB_MOLECULE_CLASS="matrixdb_molecule_class",  # extracellular matrix molecules and interactions
    MDPI="mdpi",  # mdpi journal articles
    NCBI_BOOKS="ncbi_books",
    NCBI_ENTREZ_GENE="ncbi_entrez_gene",
    NCBI_ENTREZ_PCCOMPOUND="ncbi_entrez_pccompound",  # pubchem inchikeys
    NCBI_REFSEQ="ncbi_refseq",  # ncbi reference sequences
    NCI_THESAURUS="nci_thesaurus",
    OLS="ols",  # ontology of ontologies
    PHAROS="pharos",  # pharos gene summaries
    PHOSPHOSITE="phosphosite",  # phosphosite.org kinase and ligand interactions
    PUBCHEM="pubchem",
    PUBMED="pubmed",
    REACTOME="reactome",
    RHEA="rhea",  # curated metabolic reactions
    RNACENTRAL="rnacentral",
    SGC="sgc",  # structural genomics consortium
    SGD="sgd",  # saccharomyces genome database
    SIGNOR="signor",  # signaling pathways
    SMILES="smiles",  # molecular structure
    SYMBOL="symbol",
    URL="url",
    UNIPROT="uniprot",
    WIKIPATHWAYS="wikipathways",
)

ONTOLOGIES_LIST = list(ONTOLOGIES.__dict__.values())

ONTOLOGY_SPECIES_ALIASES = {
    ONTOLOGIES.NCBI_ENTREZ_GENE: {"ncbigene", "ncbi_gene"},
    ONTOLOGIES.ENSEMBL_GENE: {"ensembl_gene_id"},
    ONTOLOGIES.UNIPROT: {"Uniprot"},
    ONTOLOGIES.CORUM: {"CORUM"},
    ONTOLOGIES.SIGNOR: {"SIGNOR"},
}

# rules for specific ontologies

# refere to ontology by name rather than using the IDENTIFIERS namespace to avoid circular imports
ONTOLOGY_PRIORITIES = DataFrame(
    [
        {"ontology": ONTOLOGIES.REACTOME, "ontology_rank": 1},
        {"ontology": ONTOLOGIES.ENSEMBL_GENE, "ontology_rank": 2},
        {"ontology": ONTOLOGIES.CHEBI, "ontology_rank": 3},
        {"ontology": ONTOLOGIES.UNIPROT, "ontology_rank": 4},
        {"ontology": ONTOLOGIES.GO, "ontology_rank": 5},
    ]
)

ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY = {
    "G": ONTOLOGIES.ENSEMBL_GENE,
    "T": ONTOLOGIES.ENSEMBL_TRANSCRIPT,
    "P": ONTOLOGIES.ENSEMBL_PROTEIN,
}

ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY = {
    ONTOLOGIES.ENSEMBL_GENE: "G",
    ONTOLOGIES.ENSEMBL_TRANSCRIPT: "T",
    ONTOLOGIES.ENSEMBL_PROTEIN: "P",
}

ENSEMBL_SPECIES_FROM_CODE = {"MUS": "Mus musculus"}
ENSEMBL_SPECIES_TO_CODE = {"Mus musculus": "MUS"}

ENSEMBL_PREFIX_TO_ONTOLOGY = {
    "ENSG": ONTOLOGIES.ENSEMBL_GENE,
    "ENST": ONTOLOGIES.ENSEMBL_TRANSCRIPT,
    "ENSP": ONTOLOGIES.ENSEMBL_PROTEIN,
}


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

# Ontology to URL map
ONTOLOGY_MAP = SimpleNamespace(URL="url", ID_REGEX="id_regex")

ONTOLOGY_TO_URL_MAP = {
    ONTOLOGIES.BIGG_METABOLITE: {
        ONTOLOGY_MAP.URL: "http://identifiers.org/bigg.metabolite/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.CHEBI: {
        ONTOLOGY_MAP.URL: "https://www.ebi.ac.uk/chebi/CHEBI:{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.CHEMSPIDER: {
        ONTOLOGY_MAP.URL: "https://www.chemspider.com/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.DOI: {
        ONTOLOGY_MAP.URL: "https://doi.org/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.DX_DOI: {
        ONTOLOGY_MAP.URL: "https://dx.doi.org/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: r"^[0-9]+\.[0-9]+$",
    },
    ONTOLOGIES.EC_CODE: {
        ONTOLOGY_MAP.URL: "https://identifiers.org/ec-code/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+\\.[0-9]+\\.[0-9]+(\\.[0-9]+)?$",
    },
    ONTOLOGIES.ENVIPATH: {
        ONTOLOGY_MAP.URL: "http://identifiers.org/envipath/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.GO: {
        ONTOLOGY_MAP.URL: "https://www.ebi.ac.uk/QuickGO/term/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^GO:[0-9]{7}$",
    },
    ONTOLOGIES.MATRIXDB_BIOMOLECULE: {
        ONTOLOGY_MAP.URL: "http://matrixdb.univ-lyon1.fr/cgi-bin/current/newPort?type=biomolecule&value={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9A-Za-z]+$",
    },
    ONTOLOGIES.MATRIXDB_MOLECULE_CLASS: {
        ONTOLOGY_MAP.URL: "http://matrixdb.univ-lyon1.fr/cgi-bin/current/newPort?type=biomolecule&value={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9A-Za-z]+$",
    },
    ONTOLOGIES.MDPI: {
        ONTOLOGY_MAP.URL: "https://www.mdpi.com/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.NCBI_BOOKS: {
        ONTOLOGY_MAP.URL: "http://www.ncbi.nlm.nih.gov/books/{identifier}/",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9A-Z]+$",
    },
    ONTOLOGIES.NCBI_ENTREZ_GENE: {
        ONTOLOGY_MAP.URL: "https://www.ncbi.nlm.nih.gov/gene/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.NCBI_ENTREZ_PCCOMPOUND: {
        ONTOLOGY_MAP.URL: "http://www.ncbi.nlm.nih.gov/sites/entrez?cmd=search&db=pccompound&term={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[A-Z]{14}\\-[A-Z]{10}\\-[A-Z]{1}$",
    },
    ONTOLOGIES.NCI_THESAURUS: {
        ONTOLOGY_MAP.URL: "https://ncithesaurus.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&code={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[A-Z][0-9]+$",
    },
    ONTOLOGIES.PHOSPHOSITE: {
        ONTOLOGY_MAP.URL: "https://www.phosphosite.org/siteAction.action?id={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.PUBCHEM: {
        ONTOLOGY_MAP.URL: "http://pubchem.ncbi.nlm.nih.gov/compound/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.PUBMED: {
        ONTOLOGY_MAP.URL: "http://www.ncbi.nlm.nih.gov/pubmed/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.REACTOME: {
        ONTOLOGY_MAP.URL: "https://reactome.org/content/detail/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^R\\-[A-Z]{3}\\-[0-9]{7}$",
    },
    ONTOLOGIES.RNACENTRAL: {
        ONTOLOGY_MAP.URL: "https://rnacentral.org/rna/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.SGC: {
        ONTOLOGY_MAP.URL: "https://www.thesgc.org/structures/structure_description/{identifier}/",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9A-Z]+$",
    },
    ONTOLOGIES.UNIPROT: {
        ONTOLOGY_MAP.URL: "https://purl.uniprot.org/uniprot/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[A-Z0-9]+$",
    },
    ONTOLOGIES.URL: {ONTOLOGY_MAP.URL: "{identifier}", ONTOLOGY_MAP.ID_REGEX: None},
}
