# Ingestion constants
from __future__ import annotations

from types import SimpleNamespace


SPECIES_FULL_NAME_HUMAN = "Homo sapiens"
SPECIES_FULL_NAME_MOUSE = "Mus musculus"
SPECIES_FULL_NAME_YEAST = "Saccharomyces cerevisiae"
SPECIES_FULL_NAME_RAT = "Rattus norvegicus"
SPECIES_FULL_NAME_WORM = "Caenorhabditis elegans"

PROTEINATLAS_SUBCELL_LOC_URL = (
    "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip"
)

PROTEINATLAS_DEFS = SimpleNamespace(
    GO_ID="GO id",
    GENE="Gene",
)


# GTEx
GTEX_RNASEQ_EXPRESSION_URL = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"

GTEX_DEFS = SimpleNamespace(
    NAME="Name",
    DESCRIPTION="Description",
)

# BIGG
BIGG_MODEL_URLS = {
    SPECIES_FULL_NAME_HUMAN: "http://bigg.ucsd.edu/static/models/Recon3D.xml",
    SPECIES_FULL_NAME_MOUSE: "http://bigg.ucsd.edu/static/models/iMM1415.xml",
    SPECIES_FULL_NAME_YEAST: "http://bigg.ucsd.edu/static/models/iMM904.xml",
}

BIGG_MODEL_FIELD_URL = "url"
BIGG_MODEL_FIELD_SPECIES = "species"

BIGG_MODEL_KEYS = {
    SPECIES_FULL_NAME_HUMAN: "recon3D",
    SPECIES_FULL_NAME_MOUSE: "iMM1415",
    SPECIES_FULL_NAME_YEAST: "iMM904",
}
BIGG_RECON3D_FIELD_ID = "id"
BIGG_RECON3D_FIELD_TYPE = "type"
BIGG_RECON3D_FIELD_URI = "uri"

# IDENTIFIERS ETL
IDENTIFIERS_ETL_YEAST_URL = "https://www.uniprot.org/docs/yeast.txt"
IDENTIFIERS_ETL_SBO_URL = (
    "https://raw.githubusercontent.com/EBI-BioModels/SBO/master/SBO_OBO.obo"
)
IDENTIFIERS_ETL_YEAST_FIELDS = (
    "common",
    "common_all",
    "OLN",
    "SwissProt_acc",
    "SwissProt_entry",
    "SGD",
    "size",
    "3d",
    "chromosome",
)

# OBO
OBO_GO_BASIC_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
OBO_GO_BASIC_LOCAL_TMP = "/tmp/go-basic.obo"


# PSI MI
PSI_MI_INTACT_FTP_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/intact/current/psi30/species"
)
PSI_MI_INTACT_DEFAULT_OUTPUT_DIR = "/tmp/intact_tmp"
PSI_MI_INTACT_XML_NAMESPACE = "{http://psi.hupo.org/mi/mif300}"

PSI_MI_INTACT_SPECIES_TO_BASENAME = {
    SPECIES_FULL_NAME_YEAST: "yeast",
    SPECIES_FULL_NAME_HUMAN: "human",
    SPECIES_FULL_NAME_MOUSE: "mouse",
    SPECIES_FULL_NAME_RAT: "rat",
    SPECIES_FULL_NAME_WORM: "caeel",
}


# REACTOME
REACTOME_SMBL_URL = "https://reactome.org/download/current/all_species.3.1.sbml.tgz"
REACTOME_PATHWAYS_URL = "https://reactome.org/download/current/ReactomePathways.txt"
REACTOME_PATHWAY_INDEX_COLUMNS = ["file", "source", "species", "pathway_id", "name"]
REACTOME_PATHWAY_LIST_COLUMNS = ["pathway_id", "name", "species"]

# SBML
SMBL_ERROR_NUMBER = "error_number"
SMBL_ERROR_CATEGORY = "category"
SMBL_ERROR_SEVERITY = "severity"
SMBL_ERROR_DESCRIPTION = "description"
SMBL_ERROR_MESSAGE = "message"

SMBL_SUMMARY_PATHWAY_NAME = "Pathway Name"
SMBL_SUMMARY_PATHWAY_ID = "Pathway ID"
SMBL_SUMMARY_N_SPECIES = "# of Species"
SMBL_SUMMARY_N_REACTIONS = "# of Reactions"
SMBL_SUMMARY_COMPARTMENTS = "Compartments"

SMBL_REACTION_DICT_ID = "r_id"
SMBL_REACTION_DICT_NAME = "r_name"
SMBL_REACTION_DICT_IDENTIFIERS = "r_Identifiers"
SMBL_REACTION_DICT_SOURCE = "r_Source"
SMBL_REACTION_DICT_IS_REVERSIBLE = "r_isreversible"

SMBL_REACTION_SPEC_RSC_ID = "rsc_id"
SMBL_REACTION_SPEC_SC_ID = "sc_id"
SMBL_REACTION_SPEC_STOICHIOMETRY = "stoichiometry"
SMBL_REACTION_SPEC_SBO_TERM = "sbo_term"

SBML_COMPARTMENT_DICT_ID = "c_id"
SBML_COMPARTMENT_DICT_NAME = "c_name"
SBML_COMPARTMENT_DICT_IDENTIFIERS = "c_Identifiers"
SBML_COMPARTMENT_DICT_SOURCE = "c_Source"

SBML_SPECIES_DICT_ID = "s_id"
SBML_SPECIES_DICT_NAME = "s_name"
SBML_SPECIES_DICT_IDENTIFIERS = "s_Identifiers"

SBML_COMPARTMENTALIZED_SPECIES_DICT_NAME = "sc_name"
SBML_COMPARTMENTALIZED_SPECIES_DICT_SOURCE = "sc_Source"

SBML_REACTION_ATTR_GET_GENE_PRODUCT = "getGeneProduct"

# STRING
STRING_URL_EXPRESSIONS = {
    "interactions": "https://stringdb-static.org/download/protein.links.full.v{version}/{taxid}.protein.links.full.v{version}.txt.gz",
    "aliases": "https://stringdb-static.org/download/protein.aliases.v{version}/{taxid}.protein.aliases.v{version}.txt.gz",
}
STRING_PROTEIN_ID_RAW = "#string_protein_id"
STRING_PROTEIN_ID = "string_protein_id"
STRING_SOURCE = "protein1"
STRING_TARGET = "protein2"

STRING_VERSION = 11.5

STRING_TAX_IDS = {
    SPECIES_FULL_NAME_WORM: 6239,
    SPECIES_FULL_NAME_HUMAN: 9606,
    SPECIES_FULL_NAME_MOUSE: 10090,
    SPECIES_FULL_NAME_RAT: 10116,
    SPECIES_FULL_NAME_YEAST: 4932,
}

STRING_UPSTREAM_COMPARTMENT = "upstream_compartment"
STRING_DOWNSTREAM_COMPARTMENT = "downstream_compartment"
STRING_UPSTREAM_NAME = "upstream_name"
STRING_DOWNSTREAM_NAME = "downstream_name"


# TRRUST
TTRUST_URL_RAW_DATA_HUMAN = (
    "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
)
TRRUST_SYMBOL = "symbol"
TRRUST_UNIPROT = "uniprot"
TRRUST_UNIPROT_ID = "uniprot_id"

TRRUST_COMPARTMENT_NUCLEOPLASM = "nucleoplasm"
TRRUST_COMPARTMENT_NUCLEOPLASM_GO_ID = "GO:0005654"

TRRUST_SIGNS = SimpleNamespace(ACTIVATION="Activation", REPRESSION="Repression")

# YEAST IDEA
# https://idea.research.calicolabs.com/data
YEAST_IDEA_KINETICS_URL = "https://storage.googleapis.com/calico-website-pin-public-bucket/datasets/idea_kinetics.zip"
YEAST_IDEA_SOURCE = "TF"
YEAST_IDEA_TARGET = "GeneName"
YEAST_IDEA_PUBMED_ID = "32181581"  # ids are characters by convention

# Identifiers ETL

IDENTIFIERS_ETL_YEAST_HEADER_REGEX = "__________"
