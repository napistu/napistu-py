# Ingestion constants
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

from napistu.constants import (
    ONTOLOGIES,
    SBML_DFS,
    SBOTERM_NAMES,
    SOURCE_SPEC,
)

# aliases and descriptions for major data sources

# high-level sources
DATA_SOURCES = SimpleNamespace(
    BIGG="BiGG",
    DOGMA="Dogma",
    IDEA_YEAST="IDEA",
    INTACT="IntAct",
    OMNIPATH="OmniPath",
    REACTOME="Reactome",
    REACTOME_FI="Reactome-FI",
    STRING="STRING",
    TRRUST="TRRUST",
)

DATA_SOURCE_DESCRIPTIONS = {
    DATA_SOURCES.BIGG: "UCSD genome-scale metabolic models",
    DATA_SOURCES.DOGMA: "Napistu gene, transcript, and protein annotations",
    DATA_SOURCES.IDEA_YEAST: "Induction Dynamics Expression Atlas",
    DATA_SOURCES.INTACT: "IntAct protein-protein interaction database",
    DATA_SOURCES.OMNIPATH: "Intra- & intercellular signaling knowledge",
    DATA_SOURCES.REACTOME: "Reactome pathway database",
    DATA_SOURCES.REACTOME_FI: "Reactome functional interactions",
    DATA_SOURCES.STRING: "STRING protein-protein interaction database",
    DATA_SOURCES.TRRUST: "Transcriptional regulatory interactions database",
}

# names for specific models within sources
MODEL_SOURCES = SimpleNamespace(
    RECON3D="Recon3D",
    IMM1415="iMM1415",
    IMM904="iMM904",
)

MODEL_SOURCE_DESCRIPTIONS = {
    MODEL_SOURCES.RECON3D: "The Recon3D human metabolic model",
    MODEL_SOURCES.IMM1415: "The iMM1415 mouse metabolic model",
    MODEL_SOURCES.IMM904: "The iMM904 yeast metabolic model",
}

DEFAULT_PRIORITIZED_PATHWAYS = [
    *DATA_SOURCES.__dict__.values(),
    *MODEL_SOURCES.__dict__.values(),
]

NO_RXN_PATHWAY_IDS_DEFAULTS = [DATA_SOURCES.DOGMA]

# standardization - species
LATIN_SPECIES_NAMES = SimpleNamespace(
    HOMO_SAPIENS="Homo sapiens",
    MUS_MUSCULUS="Mus musculus",
    SACCHAROMYCES_CEREVISIAE="Saccharomyces cerevisiae",
    RATTUS_NORVEGICUS="Rattus norvegicus",
    CAENORHABDITIS_ELEGANS="Caenorhabditis elegans",
    DROSOPHILIA_MELANOGASTER="Drosophila melanogaster",
)

LATIN_TO_COMMON_SPECIES_NAMES: Dict[str, str] = {
    # Latin name -> common name
    LATIN_SPECIES_NAMES.HOMO_SAPIENS: "human",
    LATIN_SPECIES_NAMES.MUS_MUSCULUS: "mouse",
    LATIN_SPECIES_NAMES.SACCHAROMYCES_CEREVISIAE: "yeast",
    LATIN_SPECIES_NAMES.RATTUS_NORVEGICUS: "rat",
    LATIN_SPECIES_NAMES.CAENORHABDITIS_ELEGANS: "worm",
    LATIN_SPECIES_NAMES.DROSOPHILIA_MELANOGASTER: "fly",
}

# standardization - compartments

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

# ingesting interaction edgelists to sbml_dfs

INTERACTION_EDGELIST_DEFS = SimpleNamespace(
    COMPARTMENT_DOWNSTREAM="compartment_downstream",
    COMPARTMENT_UPSTREAM="compartment_upstream",
    NAME_DOWNSTREAM="name_downstream",
    NAME_UPSTREAM="name_upstream",
    SBO_TERM_NAME_DOWNSTREAM="sbo_term_name_downstream",
    SBO_TERM_NAME_UPSTREAM="sbo_term_name_upstream",
    STOICHIOMETRY_DOWNSTREAM="stoichiometry_downstream",
    STOICHIOMETRY_UPSTREAM="stoichiometry_upstream",
)

# terms which should be defined for every interaction either as a default or an explicit value
INTERACTION_EDGELIST_EXPECTED_VARS = {
    INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
    INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
    SBML_DFS.R_NAME,
    SBML_DFS.R_IDENTIFIERS,
    SBML_DFS.R_ISREVERSIBLE,
    INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM,
    INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM,
    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM,
    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM,
    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM,
    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM,
}

# terms which can be defined at the interaction-level or globally
INTERACTION_EDGELIST_OPTIONAL_VARS = {
    INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM,
    INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM,
    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM,
    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM,
    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM,
    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM,
    SBML_DFS.R_ISREVERSIBLE,
}

INTERACTION_EDGELIST_DEFAULTS = {
    INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: GENERIC_COMPARTMENT,
    INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: GENERIC_COMPARTMENT,
    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: SBOTERM_NAMES.MODIFIED,
    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: SBOTERM_NAMES.MODIFIER,
    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM: 0,
    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM: 0,
    SBML_DFS.R_ISREVERSIBLE: False,
}

# ETLing specific sources

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
    LATIN_SPECIES_NAMES.HOMO_SAPIENS: "http://bigg.ucsd.edu/static/models/Recon3D.xml",
    LATIN_SPECIES_NAMES.MUS_MUSCULUS: "http://bigg.ucsd.edu/static/models/iMM1415.xml",
    LATIN_SPECIES_NAMES.SACCHAROMYCES_CEREVISIAE: "http://bigg.ucsd.edu/static/models/iMM904.xml",
}

BIGG_MODEL_KEYS = {
    LATIN_SPECIES_NAMES.HOMO_SAPIENS: MODEL_SOURCES.RECON3D,
    LATIN_SPECIES_NAMES.MUS_MUSCULUS: MODEL_SOURCES.IMM1415,
    LATIN_SPECIES_NAMES.SACCHAROMYCES_CEREVISIAE: MODEL_SOURCES.IMM904,
}

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
    LATIN_SPECIES_NAMES.SACCHAROMYCES_CEREVISIAE: "yeast",
    LATIN_SPECIES_NAMES.HOMO_SAPIENS: "human",
    LATIN_SPECIES_NAMES.MUS_MUSCULUS: "mouse",
    LATIN_SPECIES_NAMES.RATTUS_NORVEGICUS: "rat",
    LATIN_SPECIES_NAMES.CAENORHABDITIS_ELEGANS: "caeel",
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

PSI_MI_STUDY_TABLES = SimpleNamespace(
    REACTION_SPECIES="reaction_species",
    SPECIES="species",
    SPECIES_IDENTIFIERS="species_identifiers",
    STUDY_LEVEL_DATA="study_level_data",
)

PSI_MI_STUDY_TABLES_LIST = PSI_MI_STUDY_TABLES.__dict__.values()

PSI_MI_MISSING_VALUE_STR = ""

INTACT_ONTOLOGY_ALIASES = {ONTOLOGIES.UNIPROT: {"uniprotkb"}}

VALID_INTACT_SECONDARY_ONTOLOGIES = {ONTOLOGIES.INTACT}

INTACT_EXPERIMENTAL_ROLES = SimpleNamespace(BAIT="bait", PREY="prey")

VALID_INTACT_EXPERIMENTAL_ROLES = {
    INTACT_EXPERIMENTAL_ROLES.BAIT,
    INTACT_EXPERIMENTAL_ROLES.PREY,
}

# adding scores and consolidating terms for IntAct
INTACT_SCORES = SimpleNamespace(
    ATTRIBUTE_TYPE="attribute_type",
    ATTRIBUTE_VALUE="attribute_value",
    SCORED_TERM="scored_term",
    RAW_SCORE="raw_score",
    N_PUBLICATIONS="n_publications",
    PUBLICATION_SCORE="publication_score",
    INTERACTION_METHOD_SCORE="interaction_method_score",
    INTERACTION_TYPE_SCORE="interaction_type_score",
    MI_SCORE="miscore",
)

DEFAULT_INTACT_RELATIVE_WEIGHTS = {
    INTACT_SCORES.PUBLICATION_SCORE: 1.0,
    INTACT_SCORES.INTERACTION_METHOD_SCORE: 1.0,
    INTACT_SCORES.INTERACTION_TYPE_SCORE: 1.0,
}

INTACT_PUBLICATION_SCORE_THRESHOLD = 7

PSI_MI_ONTOLOGY_URL = "https://raw.githubusercontent.com/MICommunity/miscore/refs/heads/master/miscore/src/main/resources/psimiOntology.json"

PSI_MI_SCORED_TERMS = SimpleNamespace(
    # Interaction types
    GENETIC_INTERACTION="genetic interaction",
    COLOCALIZATION="colocalization",
    ASSOCIATION="association",
    PHYSICAL_ASSOCIATION="physical association",
    DIRECT_INTERACTION="direct interaction",
    # Detection methods
    BIOPHYSICAL="biophysical",
    PROTEIN_COMPLEMENTATION_ASSAY="protein complementation assay",
    GENETIC_INTERFERENCE="genetic interference",
    POST_TRANSCRIPTIONAL_INTERFERENCE="post transcriptional interference",
    BIOCHEMICAL="biochemical",
    IMAGING_TECHNIQUE="imaging technique",
    # other
    UNKNOWN="unknown",
)

# from https://github.com/MICommunity/miscore/blob/master/miscore/src/main/resources/scoreCategories.properties
INTACT_TERM_SCORES = {
    PSI_MI_SCORED_TERMS.GENETIC_INTERACTION: 0.10,
    PSI_MI_SCORED_TERMS.COLOCALIZATION: 0.33,
    PSI_MI_SCORED_TERMS.ASSOCIATION: 0.33,
    PSI_MI_SCORED_TERMS.PHYSICAL_ASSOCIATION: 0.66,
    PSI_MI_SCORED_TERMS.DIRECT_INTERACTION: 1.00,
    PSI_MI_SCORED_TERMS.BIOPHYSICAL: 1.00,
    PSI_MI_SCORED_TERMS.PROTEIN_COMPLEMENTATION_ASSAY: 0.66,
    PSI_MI_SCORED_TERMS.GENETIC_INTERFERENCE: 0.10,
    PSI_MI_SCORED_TERMS.POST_TRANSCRIPTIONAL_INTERFERENCE: 0.10,
    PSI_MI_SCORED_TERMS.BIOCHEMICAL: 1.00,
    PSI_MI_SCORED_TERMS.IMAGING_TECHNIQUE: 0.33,
    PSI_MI_SCORED_TERMS.UNKNOWN: 0.05,
}

# omnipath

VALID_OMNIPATH_SPECIES = {
    LATIN_SPECIES_NAMES.HOMO_SAPIENS,
    LATIN_SPECIES_NAMES.MUS_MUSCULUS,
    LATIN_SPECIES_NAMES.RATTUS_NORVEGICUS,
}

OMNIPATH_ANNOTATIONS = SimpleNamespace(
    NAME="name",
    ANNOTATION="annotation",
    ANNOTATION_STR="annotation_str",
)

OMNIPATH_COMPLEXES = SimpleNamespace(
    COMPONENTS="components",
    NAME="name",
    STOICHIOMETRY="stoichiometry",
    IDENTIFIERS="identifiers",
    COMPLEX_FSTRING="COMPLEX:{x}",
)

OMNIPATH_INTERACTIONS = SimpleNamespace(
    INTERACTOR_ID="interactor_id",
    SOURCE="source",
    TARGET="target",
    IS_DIRECTED="is_directed",
    IS_STIMULATION="is_stimulation",
    IS_INHIBITION="is_inhibition",
    CONSENSUS_DIRECTION="consensus_direction",
    CONSENSUS_STIMULATION="consensus_stimulation",
    CONSENSUS_INHIBITION="consensus_inhibition",
    CURATION_EFFORT="curation_effort",
    REFERENCES="references",
    SOURCES="sources",
    N_SOURCES="n_sources",
    N_PRIMARY_SOURCES="n_primary_sources",
    N_REFERENCES="n_references",
    REFERENCES_STRIPPED="references_stripped",
)

OMNIPATH_ONTOLOGY_ALIASES = {ONTOLOGIES.CORUM: {"CORUM"}, ONTOLOGIES.SIGNOR: {"SIGNOR"}}

# REACTOME
REACTOME_SMBL_URL = "https://reactome.org/download/current/all_species.3.1.sbml.tgz"
REACTOME_PATHWAYS_URL = "https://reactome.org/download/current/ReactomePathways.txt"
REACTOME_PATHWAY_LIST_COLUMNS = [
    SOURCE_SPEC.PATHWAY_ID,
    SOURCE_SPEC.NAME,
    SOURCE_SPEC.ORGANISMAL_SPECIES,
]

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
    LATIN_SPECIES_NAMES.CAENORHABDITIS_ELEGANS: 6239,
    LATIN_SPECIES_NAMES.HOMO_SAPIENS: 9606,
    LATIN_SPECIES_NAMES.MUS_MUSCULUS: 10090,
    LATIN_SPECIES_NAMES.RATTUS_NORVEGICUS: 10116,
    LATIN_SPECIES_NAMES.SACCHAROMYCES_CEREVISIAE: 4932,
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

# IDEA YEAST TF -> targets
# https://idea.research.calicolabs.com/data
IDEA_YEAST = SimpleNamespace(
    KINETICS_URL="https://storage.googleapis.com/calico-website-pin-public-bucket/datasets/idea_kinetics.zip",
    KINETICS_FILE="idea_kinetics.tsv",
    SOURCE="TF",
    TARGET="GeneName",
    PUBMED_ID="32181581",
)

# Identifiers ETL

IDENTIFIERS_ETL_YEAST_HEADER_REGEX = "__________"

# Harmonizome

HARMONIZOME_DATASET_DEFS = SimpleNamespace(
    NAME="name",
    SHORT_NAME="short_name",
    DESCRIPTION="description",
    DOWNLOAD_FILES="download_files",
    CUSTOM_URLS="custom_urls",
    CUSTOM_LOADERS="custom_loaders",
)

HARMONIZOME_DATASET_FILES = SimpleNamespace(
    INTERACTIONS="interactions",
    GENES="genes",
    ATTRIBUTES="attributes",
)

# these are downloaded by default for all Harmonizome datasets
HARMONIZOME_STANDARD_FILES = [
    HARMONIZOME_DATASET_FILES.INTERACTIONS,
    HARMONIZOME_DATASET_FILES.GENES,
    HARMONIZOME_DATASET_FILES.ATTRIBUTES,
]

# names used internally by Harmonizome - defines URLs
HARMONIZOME_DATASET_SHORTNAMES = SimpleNamespace(
    ACHILLES="achilles",
    CLINVAR="clinvar25",
    DBGAP="dbgap",
    DRUG_BANK="drugbank",
    GENETIC_ASSOCIATION_DATABASE="gad",
    PERTURB_ATLAS="perturbatlas",
    PERTURB_ATLAS_MOUSE="perturbatlasmouse",
    REPLOGLE_K562_ESSENTIAL="reploglek562essential",
    REPLOGLE_K562_GENOME_WIDE="reploglek562genomewide",
    REPLOGLE_RPE1_ESSENTIAL="reploglerpe1essential",
    # TAHOE_100M="tahoe100m",
)

HARMONIZOME_URL_TEMPLATES = SimpleNamespace(
    BASE_URL="https://maayanlab.cloud/static/hdfs/harmonizome/data/{short_name}",
    INTERACTIONS="{base_url}/gene_attribute_edges.txt.gz",
    GENES="{base_url}/gene_list_terms.txt.gz",
    ATTRIBUTES="{base_url}/attribute_list_entries.txt.gz",
)

HARMONIZOME_DATASETS = {
    HARMONIZOME_DATASET_SHORTNAMES.ACHILLES: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.ACHILLES,
        HARMONIZOME_DATASET_DEFS.NAME: "Achilles Cell Line Gene Essentiality Profiles",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: f"CRISPR knockout essentiality - called {-1,1} fitnesses for individual cell lines.",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: HARMONIZOME_STANDARD_FILES,
    },
    HARMONIZOME_DATASET_SHORTNAMES.CLINVAR: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.CLINVAR,
        HARMONIZOME_DATASET_DEFS.NAME: "ClinVar",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: "SNP-phenotype associations curated by ClinVar users from various sources",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: HARMONIZOME_STANDARD_FILES,
    },
    HARMONIZOME_DATASET_SHORTNAMES.DBGAP: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.DBGAP,
        HARMONIZOME_DATASET_DEFS.NAME: "Database of Genotypes and Phenotypes",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: "Database of gene-trait associations curated from genetic association studies",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: HARMONIZOME_STANDARD_FILES,
    },
    HARMONIZOME_DATASET_SHORTNAMES.DRUG_BANK: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.DRUG_BANK,
        HARMONIZOME_DATASET_DEFS.NAME: "DrugBank",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: "Protein-drug associations by manual literature curation",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: HARMONIZOME_STANDARD_FILES,
    },
    HARMONIZOME_DATASET_SHORTNAMES.GENETIC_ASSOCIATION_DATABASE: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.GENETIC_ASSOCIATION_DATABASE,
        HARMONIZOME_DATASET_DEFS.NAME: "Genetic Association Database (GAD)",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: "Gene-disease associations curated from genetic association studies",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: HARMONIZOME_STANDARD_FILES,
    },
    HARMONIZOME_DATASET_SHORTNAMES.PERTURB_ATLAS: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.PERTURB_ATLAS,
        HARMONIZOME_DATASET_DEFS.NAME: "PerturbAtlas",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: "Gene expression profiles for cell lines, cell types, tissues, and models following genetic perturbation ",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: [
            HARMONIZOME_DATASET_FILES.INTERACTIONS,
            HARMONIZOME_DATASET_FILES.GENES,
        ],  # attributes are not available
    },
    HARMONIZOME_DATASET_SHORTNAMES.PERTURB_ATLAS_MOUSE: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.PERTURB_ATLAS_MOUSE,
        HARMONIZOME_DATASET_DEFS.NAME: "PerturbAtlas Mouse",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: "Gene expression profiles for mouse cell lines, cell types, tissues, and models following genetic perturbation",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: HARMONIZOME_STANDARD_FILES,
    },
    HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_K562_ESSENTIAL: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_K562_ESSENTIAL,
        HARMONIZOME_DATASET_DEFS.NAME: "Replogle et al., Cell 2022: K562 Essential Perturb-seq Gene Perturbation Signatures",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: "K562 CRISPRi essential genes",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: HARMONIZOME_STANDARD_FILES,
    },
    HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_K562_GENOME_WIDE: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_K562_GENOME_WIDE,
        HARMONIZOME_DATASET_DEFS.NAME: "Replogle et al., Cell 2022: K562 Genome-wide Perturb-seq Gene Perturbation Signatures",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: "K562 CRISPRi gene perturbation signatures",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: HARMONIZOME_STANDARD_FILES,
    },
    HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_RPE1_ESSENTIAL: {
        HARMONIZOME_DATASET_DEFS.SHORT_NAME: HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_RPE1_ESSENTIAL,
        HARMONIZOME_DATASET_DEFS.NAME: "Replogle et al., Cell 2022: RPE1 Essential Perturb-seq Gene Perturbation Signatures",
        HARMONIZOME_DATASET_DEFS.DESCRIPTION: "RPE1 CRISPRi essential genes",
        HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES: HARMONIZOME_STANDARD_FILES,
    },
    # the interactions aren't available for Tahoe100M right now
    # HARMONIZOME_DATASET_SHORTNAMES.TAHOE_100M: {
    #    HARMONIZOME_DATASET_DEFS.SHORT_NAME : HARMONIZOME_DATASET_SHORTNAMES.TAHOE_100M,
    #    HARMONIZOME_DATASET_DEFS.NAME : "Tahoe 100M signatures based on drug-perturbation and scRNAseq",
    #    HARMONIZOME_DATASET_DEFS.DESCRIPTION : "Single-cell expression data from drug perturbation across cell lines in over 95 million cells",
    #    HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES : HARMONIZOME_STANDARD_FILES
    # },
}

HARMONIZOME_DEFS = SimpleNamespace(
    # standardized fields
    PERTURBED_SYMBOL=f"perturbed_{ONTOLOGIES.SYMBOL}",
    PERTURBED_NCBI_ENTREZ_GENE=f"perturbed_{ONTOLOGIES.NCBI_ENTREZ_GENE}",
    PERTURBED_ENSEMBL_GENE=f"perturbed_{ONTOLOGIES.ENSEMBL_GENE}",
    PERTURBATION_TYPE="perturbation_type",
    PERTURBATION_STUDY="perturbation_study",
    # Source file column names
    GENE_SYM="GeneSym",
    GENE_ID="GeneID",
    GENE="Gene",
    GENE_ID_SPACED="Gene ID",
    GENE_PERTURBATION_ID="Gene Perturbation ID",
    DISEASE="Disease",
    DISEASE_CLASS="DiseaseClass",
    TRAIT="Trait",
    UNIPROT_ACC="UniprotAcc",
    DRUGBANK_ID="DrugBankID",
    PERTURBATION="Perturbation",
    # interaction columns
    STANDARDIZED_VALUE="Standardized Value",
    THRESHOLDED_VALUE="Threshold Value",
)

# perturbseq

PERTURBSEQ_DEFS = SimpleNamespace(
    PERTURBED_SYMBOL=f"perturbed_{ONTOLOGIES.SYMBOL}",
    PERTURBED_NCBI_ENTREZ_GENE=f"perturbed_{ONTOLOGIES.NCBI_ENTREZ_GENE}",
    PERTURBED_ENSEMBL_GENE=f"perturbed_{ONTOLOGIES.ENSEMBL_GENE}",
    PERTURBED_SPECIES_ID="perturbed_species_id",
    TARGET_SYMBOL=f"target_{ONTOLOGIES.SYMBOL}",
    TARGET_NCBI_ENTREZ_GENE=f"target_{ONTOLOGIES.NCBI_ENTREZ_GENE}",
    TARGET_ENSEMBL_GENE=f"target_{ONTOLOGIES.ENSEMBL_GENE}",
    TARGET_SPECIES_ID="target_species_id",
)

PERTURBSEQ_DIRECTIONS = SimpleNamespace(
    STRONG_ACTIVATION="strong activation",
    STRONG_CHANGE="strong change",
    STRONG_REPRESSION="strong repression",
    WEAK_ACTIVATION="weak activation",
    WEAK_CHANGE="weak change",
    WEAK_REPRESSION="weak repression",
)

VALID_PERTURBSEQ_DIRECTIONS = PERTURBSEQ_DIRECTIONS.__dict__.values()

PERTURBSEQ_PERTURBATION_TYPES = SimpleNamespace(
    OE="OE",  # overexpression
    KD="KD",  # knockdown
    KI="KI",  # knockin
    KO="KO",  # knockout
    MUT="MUT",  # mutation
)

SIGNED_PERTURBATION_TYPES = [
    PERTURBSEQ_PERTURBATION_TYPES.OE,
    PERTURBSEQ_PERTURBATION_TYPES.KD,
    PERTURBSEQ_PERTURBATION_TYPES.KO,
]
GOF_PERTURBATION_TYPES = [PERTURBSEQ_PERTURBATION_TYPES.OE]
LOF_PERTURBATION_TYPES = [
    PERTURBSEQ_PERTURBATION_TYPES.KD,
    PERTURBSEQ_PERTURBATION_TYPES.KO,
]

ORDERED_SIGNED_PERTURBSEQ_DIRECTIONS = [
    PERTURBSEQ_DIRECTIONS.STRONG_REPRESSION,
    PERTURBSEQ_DIRECTIONS.WEAK_REPRESSION,
    PERTURBSEQ_DIRECTIONS.WEAK_ACTIVATION,
    PERTURBSEQ_DIRECTIONS.STRONG_ACTIVATION,
]
STRONG_ORDERED_SIGNED_PERTURBSEQ_DIRECTIONS = [
    PERTURBSEQ_DIRECTIONS.STRONG_REPRESSION,
    PERTURBSEQ_DIRECTIONS.STRONG_ACTIVATION,
]

REPLOGLE_DEFS = SimpleNamespace(
    REPLOGLE_PERTURBSEQ_URI="https://plus.figshare.com/ndownloader/files/38349308",
    REPLOGLE_PERTURBSEQ_ZIPPED_FILENAME="anderson-darling p-values, BH-corrected.csv.gz",
)
