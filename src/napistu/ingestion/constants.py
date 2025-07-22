# Ingestion constants
from __future__ import annotations

from types import SimpleNamespace

from napistu.constants import (
    ONTOLOGIES,
    SBML_DFS,
    SBOTERM_NAMES,
)

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
PSI_MI_INTACT_XML_NAMESPACE = "{http://psi.hupo.org/mi/mif300}"

PSI_MI_INTACT_SPECIES_TO_BASENAME = {
    SPECIES_FULL_NAME_YEAST: "yeast",
    SPECIES_FULL_NAME_HUMAN: "human",
    SPECIES_FULL_NAME_MOUSE: "mouse",
    SPECIES_FULL_NAME_RAT: "rat",
    SPECIES_FULL_NAME_WORM: "caeel",
}

PSI_MI_RAW_ATTRS = SimpleNamespace(
    ABSTRACT_INTERACTION="abstractInteraction",
    DB="db",
    ENTRY="entry",
    ID="id",
    INTERACTION="interaction",
    INTERACTIONS_LIST="interactionList",
    INTERACTOR="interactor",
    PARTICIPANT="participant",
    PARTICIPANT_LIST="participantList",
    PRIMARY_REF="primaryRef",
    SECONDARY_REF="secondaryRef",
    TAG="tag",
)

PSI_MI_DEFS = SimpleNamespace(
    BIOLOGICAL_ROLE="biological_role",
    EXPERIMENT="experiment",
    EXPERIMENT_NAME="experiment_name",
    EXPERIMENTAL_ROLE="experimental_role",
    FULL_NAME="full_name",
    INTERACTOR_ALIASES="interactor_aliases",
    INTERACTOR_ID="interactor_id",
    INTERACTOR_LABEL="interactor_label",
    INTERACTOR_LIST="interactor_list",
    INTERACTOR_NAME="interactor_name",
    INTERACTOR_XREFS="interactor_xrefs",
    INTERACTION_METHOD="interaction_method",
    INTERACTION_NAME="interaction_name",
    INTERACTION_TYPE="interaction_type",
    INTERACTIONS_LIST="interactions_list",
    INTERACTORS="interactors",
    PARTICIPANT_ID="participant_id",
    PRIMARY="primary",
    REF_TYPE="ref_type",
    SECONDARY="secondary",
    SOURCE="source",
    SHORT_LABEL="short_label",
    STUDY_ID="study_id",
)

PSI_MI_REFS = SimpleNamespace(
    PRIMARY_REF_DB="primary_ref_db",
    PRIMARY_REF_ID="primary_ref_id",
)

INTACT_ONTOLOGY_CV_LOOKUP = {
    "uniprotkb": ONTOLOGIES.UNIPROT,
    # we actually don't want these GO IDs
    ONTOLOGIES.GO: "ignored_go",
}

INTACT_ONTOLOGY_ENSEMBL_VAGUE = "ensembl"

# REACTOME
REACTOME_SMBL_URL = "https://reactome.org/download/current/all_species.3.1.sbml.tgz"
REACTOME_PATHWAYS_URL = "https://reactome.org/download/current/ReactomePathways.txt"
REACTOME_PATHWAY_INDEX_COLUMNS = ["file", "source", "species", "pathway_id", "name"]
REACTOME_PATHWAY_LIST_COLUMNS = ["pathway_id", "name", "species"]

# REACTOME FI
REACTOME_FI_URL = "http://cpws.reactome.org/caBigR3WebApp2025/FIsInGene_04142025_with_annotations.txt.zip"

REACTOME_FI = SimpleNamespace(
    GENE1="Gene1",
    GENE2="Gene2",
    ANNOTATION="Annotation",
    DIRECTION_RAW="Direction",
    # teasing out directionality
    DIRECTION="direction",
    FORWARD="forward",
    REVERSE="reverse",
    # scores
    SCORE="Score",  # raw name
    FI_REACTION_DATA_NAME="reactome_fi",
    FI_REACTION_DATA_SCORE="fi_score",
)

REACTOME_FI_DIRECTIONS = SimpleNamespace(
    UNDIRECTED="-",
    STIMULATED_BY="<-",
    STIMULATES="->",
    STIMULATES_AND_STIMULATED_BY="<->",
    INHIBITED_BY="|-",
    INHIBITS="-|",
    INHIBITS_AND_INHIBITED_BY="|-|",
    STIMULATES_AND_INHIBITED_BY="|->",
    INHIBITS_AND_STIMULATED_BY="<-|",
)

VALID_REACTOME_FI_DIRECTIONS = REACTOME_FI_DIRECTIONS.__dict__.values()

REACTOME_FI_RULES_REVERSE = SimpleNamespace(
    NAME_RULES={"catalyzed by": SBOTERM_NAMES.CATALYST},
    DIRECTION_RULES={
        REACTOME_FI_DIRECTIONS.STIMULATED_BY: SBOTERM_NAMES.STIMULATOR,
        REACTOME_FI_DIRECTIONS.STIMULATES_AND_STIMULATED_BY: SBOTERM_NAMES.STIMULATOR,
        REACTOME_FI_DIRECTIONS.INHIBITED_BY: SBOTERM_NAMES.INHIBITOR,
        REACTOME_FI_DIRECTIONS.INHIBITS_AND_INHIBITED_BY: SBOTERM_NAMES.INHIBITOR,
        REACTOME_FI_DIRECTIONS.STIMULATES_AND_INHIBITED_BY: SBOTERM_NAMES.INHIBITOR,
        REACTOME_FI_DIRECTIONS.UNDIRECTED: SBOTERM_NAMES.INTERACTOR,
    },
)

REACTOME_FI_RULES_FORWARD = SimpleNamespace(
    NAME_RULES={"catalyze(;$)": SBOTERM_NAMES.CATALYST},
    DIRECTION_RULES={
        REACTOME_FI_DIRECTIONS.STIMULATES: SBOTERM_NAMES.STIMULATOR,
        REACTOME_FI_DIRECTIONS.STIMULATES_AND_STIMULATED_BY: SBOTERM_NAMES.STIMULATOR,
        REACTOME_FI_DIRECTIONS.STIMULATES_AND_INHIBITED_BY: SBOTERM_NAMES.STIMULATOR,
        REACTOME_FI_DIRECTIONS.INHIBITS: SBOTERM_NAMES.INHIBITOR,
        REACTOME_FI_DIRECTIONS.INHIBITS_AND_INHIBITED_BY: SBOTERM_NAMES.INHIBITOR,
        REACTOME_FI_DIRECTIONS.INHIBITS_AND_STIMULATED_BY: SBOTERM_NAMES.INHIBITOR,
        REACTOME_FI_DIRECTIONS.UNDIRECTED: SBOTERM_NAMES.INTERACTOR,
    },
)

# SBML
SBML_DEFS = SimpleNamespace(
    ERROR_NUMBER="error_number",
    ERROR_CATEGORY="category",
    ERROR_SEVERITY="severity",
    ERROR_DESCRIPTION="description",
    ERROR_MESSAGE="message",
    SUMMARY_PATHWAY_NAME="Pathway Name",
    SUMMARY_PATHWAY_ID="Pathway ID",
    SUMMARY_N_SPECIES="# of Species",
    SUMMARY_N_REACTIONS="# of Reactions",
    SUMMARY_COMPARTMENTS="Compartments",
    REACTION_ATTR_GET_GENE_PRODUCT="getGeneProduct",
)

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

COMPARTMENTS = SimpleNamespace(
    NUCLEOPLASM="nucleoplasm",
    CYTOPLASM="cytoplasm",
    CELLULAR_COMPONENT="cellular_component",
    CYTOSOL="cytosol",
    MITOCHONDRIA="mitochondria",
    MITOMEMBRANE="mitochondrial membrane",
    INNERMITOCHONDRIA="inner mitochondria",
    MITOMATRIX="mitochondrial matrix",
    ENDOPLASMICRETICULUM="endoplasmic reticulum",
    ERMEMBRANE="endoplasmic reticulum membrane",
    ERLUMEN="endoplasmic reticulum lumen",
    GOLGIAPPARATUS="golgi apparatus",
    GOLGIMEMBRANE="golgi membrane",
    NUCLEUS="nucleus",
    NUCLEARLUMEN="nuclear lumen",
    NUCLEOLUS="nucleolus",
    LYSOSOME="lysosome",
    PEROXISOME="peroxisome",
    EXTRACELLULAR="extracellular",
)

GENERIC_COMPARTMENT = COMPARTMENTS.CELLULAR_COMPONENT
EXCHANGE_COMPARTMENT = COMPARTMENTS.CYTOSOL
VALID_COMPARTMENTS = list(COMPARTMENTS.__dict__.values())

COMPARTMENT_ALIASES = {
    COMPARTMENTS.NUCLEOPLASM: ["nucleoplasm", "Nucleoplasm"],
    COMPARTMENTS.CYTOPLASM: ["cytoplasm", "Cytoplasm"],
    COMPARTMENTS.CELLULAR_COMPONENT: ["cellular_component", "Cellular_component"],
    COMPARTMENTS.CYTOSOL: ["cytosol", "Cytosol"],
    COMPARTMENTS.MITOCHONDRIA: ["mitochondria", "Mitochondria"],
    COMPARTMENTS.MITOMEMBRANE: ["mitochondrial membrane", "Mitochondrial membrane"],
    COMPARTMENTS.INNERMITOCHONDRIA: [
        "inner mitochondria",
        "Inner mitochondria",
        "inner mitochondrial compartment",
    ],
    COMPARTMENTS.MITOMATRIX: [
        "mitochondrial matrix",
        "Mitochondrial matrix",
        "mitochondrial lumen",
        "Mitochondrial lumen",
    ],
    COMPARTMENTS.ENDOPLASMICRETICULUM: [
        "endoplasmic reticulum",
        "Endoplasmic reticulum",
    ],
    COMPARTMENTS.ERMEMBRANE: [
        "endoplasmic reticulum membrane",
        "Endoplasmic reticulum membrane",
    ],
    COMPARTMENTS.ERLUMEN: [
        "endoplasmic reticulum lumen",
        "Endoplasmic reticulum lumen",
    ],
    COMPARTMENTS.GOLGIAPPARATUS: ["golgi apparatus", "Golgi apparatus"],
    COMPARTMENTS.GOLGIMEMBRANE: ["Golgi membrane", "golgi membrane"],
    COMPARTMENTS.NUCLEUS: ["nucleus", "Nucleus"],
    COMPARTMENTS.NUCLEARLUMEN: ["nuclear lumen", "Nuclear lumen"],
    COMPARTMENTS.NUCLEOLUS: ["nucleolus", "Nucleolus"],
    COMPARTMENTS.LYSOSOME: ["lysosome", "Lysosome"],
    COMPARTMENTS.PEROXISOME: ["peroxisome", "Peroxisome", "peroxisome/glyoxysome"],
    COMPARTMENTS.EXTRACELLULAR: [
        "extracellular",
        "Extracellular",
        "extracellular space",
        "Extracellular space",
    ],
}

COMPARTMENTS_GO_TERMS = {
    COMPARTMENTS.NUCLEOPLASM: "GO:0005654",
    COMPARTMENTS.CELLULAR_COMPONENT: "GO:0005575",
    COMPARTMENTS.CYTOPLASM: "GO:0005737",
    COMPARTMENTS.CYTOSOL: "GO:0005829",
    COMPARTMENTS.MITOCHONDRIA: "GO:0005739",
    COMPARTMENTS.MITOMEMBRANE: "GO:0031966",
    COMPARTMENTS.INNERMITOCHONDRIA: "GO:0005743",
    COMPARTMENTS.MITOMATRIX: "GO:0005759",
    COMPARTMENTS.ENDOPLASMICRETICULUM: "GO:0005783",
    COMPARTMENTS.ERMEMBRANE: "GO:0005789",
    COMPARTMENTS.ERLUMEN: "GO:0005788",
    COMPARTMENTS.GOLGIAPPARATUS: "GO:0005794",
    COMPARTMENTS.GOLGIMEMBRANE: "GO:0000139",
    COMPARTMENTS.NUCLEUS: "GO:0005634",
    COMPARTMENTS.NUCLEARLUMEN: "GO:0031981",
    COMPARTMENTS.NUCLEOLUS: "GO:0005730",
    COMPARTMENTS.LYSOSOME: "GO:0005764",
    COMPARTMENTS.PEROXISOME: "GO:0005777",
    COMPARTMENTS.EXTRACELLULAR: "GO:0005615",
}

INTERACTION_EDGELIST_DEFS = SimpleNamespace(
    UPSTREAM_NAME="upstream_name",
    DOWNSTREAM_NAME="downstream_name",
    UPSTREAM_COMPARTMENT="upstream_compartment",
    DOWNSTREAM_COMPARTMENT="downstream_compartment",
    UPSTREAM_SBO_TERM_NAME="upstream_sbo_term_name",
    DOWNSTREAM_SBO_TERM_NAME="downstream_sbo_term_name",
    UPSTREAM_STOICHIOMETRY="upstream_stoichiometry",
    DOWNSTREAM_STOICHIOMETRY="downstream_stoichiometry",
)

# terms which should be defined for every interaction either as a default or an explicit value
INTERACTION_EDGELIST_EXPECTED_VARS = {
    INTERACTION_EDGELIST_DEFS.UPSTREAM_NAME,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_NAME,
    SBML_DFS.R_NAME,
    SBML_DFS.R_IDENTIFIERS,
    SBML_DFS.R_ISREVERSIBLE,
    INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT,
    INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_SBO_TERM_NAME,
    INTERACTION_EDGELIST_DEFS.UPSTREAM_STOICHIOMETRY,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_STOICHIOMETRY,
}

# terms which can be defined at the interaction-level or globally
INTERACTION_EDGELIST_OPTIONAL_VARS = {
    INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT,
    INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_SBO_TERM_NAME,
    INTERACTION_EDGELIST_DEFS.UPSTREAM_STOICHIOMETRY,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_STOICHIOMETRY,
    SBML_DFS.R_ISREVERSIBLE,
}

INTERACTION_EDGELIST_DEFAULTS = {
    INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT: GENERIC_COMPARTMENT,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT: GENERIC_COMPARTMENT,
    INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME: SBOTERM_NAMES.MODIFIER,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_SBO_TERM_NAME: SBOTERM_NAMES.MODIFIED,
    INTERACTION_EDGELIST_DEFS.UPSTREAM_STOICHIOMETRY: 0,
    INTERACTION_EDGELIST_DEFS.DOWNSTREAM_STOICHIOMETRY: 0,
    SBML_DFS.R_ISREVERSIBLE: False,
}
