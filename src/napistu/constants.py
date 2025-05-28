"""Module to contain all constants for CPR"""

from __future__ import annotations

import libsbml

from types import SimpleNamespace
import pandas as pd


PACKAGE_DEFS = SimpleNamespace(
    NAPISTU="napistu",
    GITHUB_OWNER="napistu",
    GITHUB_PROJECT_REPO="napistu",
    GITHUB_NAPISTU_PY="napistu-py",
    GITHUB_NAPISTU_R="napistu-r",
    TUTORIALS_URL="https://github.com/napistu/napistu/wiki",
    # User-facing functionality should use a user-defined directory but
    # for convenience, we provide a default cache directory for dev-facing
    # workflows
    CACHE_DIR="napistu_data",
)

PROTEINATLAS_SUBCELL_LOC_URL = (
    "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip"
)

# GTEx
GTEX_RNASEQ_EXPRESSION_URL = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"

# Gencode
GENCODE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.transcripts.fa.gz"

FILE_EXT_ZIP = "zip"
FILE_EXT_GZ = "gz"

# SBML_dfs

SBML_DFS = SimpleNamespace(
    COMPARTMENTS="compartments",
    SPECIES="species",
    COMPARTMENTALIZED_SPECIES="compartmentalized_species",
    REACTIONS="reactions",
    REACTION_SPECIES="reaction_species",
    SPECIES_DATA="species_data",
    REACTIONS_DATA="reactions_data",
    C_ID="c_id",
    C_NAME="c_name",
    C_IDENTIFIERS="c_Identifiers",
    C_SOURCE="c_Source",
    S_ID="s_id",
    S_NAME="s_name",
    S_IDENTIFIERS="s_Identifiers",
    S_SOURCE="s_Source",
    SC_ID="sc_id",
    SC_NAME="sc_name",
    SC_SOURCE="sc_Source",
    R_ID="r_id",
    R_NAME="r_name",
    R_IDENTIFIERS="r_Identifiers",
    R_SOURCE="r_Source",
    R_ISREVERSIBLE="r_isreversible",
    RSC_ID="rsc_id",
    STOICHIOMETRY="stoichiometry",
    SBO_TERM="sbo_term",
)

SBML_DFS_SCHEMA = SimpleNamespace(
    SCHEMA={
        SBML_DFS.COMPARTMENTS: {
            "pk": SBML_DFS.C_ID,
            "label": SBML_DFS.C_NAME,
            "id": SBML_DFS.C_IDENTIFIERS,
            "source": SBML_DFS.C_SOURCE,
            "vars": [SBML_DFS.C_NAME, SBML_DFS.C_IDENTIFIERS, SBML_DFS.C_SOURCE],
        },
        SBML_DFS.SPECIES: {
            "pk": SBML_DFS.S_ID,
            "label": SBML_DFS.S_NAME,
            "id": SBML_DFS.S_IDENTIFIERS,
            "source": SBML_DFS.S_SOURCE,
            "vars": [SBML_DFS.S_NAME, SBML_DFS.S_IDENTIFIERS, SBML_DFS.S_SOURCE],
        },
        SBML_DFS.COMPARTMENTALIZED_SPECIES: {
            "pk": SBML_DFS.SC_ID,
            "label": SBML_DFS.SC_NAME,
            "fk": [SBML_DFS.S_ID, SBML_DFS.C_ID],
            "source": SBML_DFS.SC_SOURCE,
            "vars": [
                SBML_DFS.SC_NAME,
                SBML_DFS.S_ID,
                SBML_DFS.C_ID,
                SBML_DFS.SC_SOURCE,
            ],
        },
        SBML_DFS.REACTIONS: {
            "pk": SBML_DFS.R_ID,
            "label": SBML_DFS.R_NAME,
            "id": SBML_DFS.R_IDENTIFIERS,
            "source": SBML_DFS.R_SOURCE,
            "vars": [
                SBML_DFS.R_NAME,
                SBML_DFS.R_IDENTIFIERS,
                SBML_DFS.R_SOURCE,
                SBML_DFS.R_ISREVERSIBLE,
            ],
        },
        SBML_DFS.REACTION_SPECIES: {
            "pk": SBML_DFS.RSC_ID,
            "fk": [SBML_DFS.R_ID, SBML_DFS.SC_ID],
            "vars": [
                SBML_DFS.R_ID,
                SBML_DFS.SC_ID,
                SBML_DFS.STOICHIOMETRY,
                SBML_DFS.SBO_TERM,
            ],
        },
    },
    REQUIRED_ENTITIES={
        SBML_DFS.COMPARTMENTS,
        SBML_DFS.SPECIES,
        SBML_DFS.COMPARTMENTALIZED_SPECIES,
        SBML_DFS.REACTIONS,
        SBML_DFS.REACTION_SPECIES,
    },
    OPTIONAL_ENTITIES={
        SBML_DFS.SPECIES_DATA,
        SBML_DFS.REACTIONS_DATA,
    },
)

ENTITIES_W_DATA = {SBML_DFS.SPECIES, SBML_DFS.REACTIONS}

REQUIRED_REACTION_FROMEDGELIST_COLUMNS = [
    "sc_id_up",
    "sc_id_down",
    "sbo_term",
    "r_name",
    "r_Identifiers",
    "r_isreversible",
]

CPR_STANDARD_OUTPUTS = SimpleNamespace(
    SPECIES_IDENTIFIERS="species_identifiers.tsv",
    SPECIES="species.json",
    REACTIONS="reactions.json",
    REACTION_SPECIES="reaction_species.json",
    COMPARTMENTS="compartments.json",
    COMPARTMENTALIZED_SPECIES="compartmentalized_species.json",
)

INTERACTION_EDGELIST_EXPECTED_VARS = {
    "upstream_name",
    "downstream_name",
    "upstream_compartment",
    "downstream_compartment",
    "r_name",
    "sbo_term",
    "r_Identifiers",
    "r_isreversible",
}

BQB_PRIORITIES = pd.DataFrame(
    [{"bqb": "BQB_IS", "bqb_rank": 1}, {"bqb": "BQB_HAS_PART", "bqb_rank": 2}]
)

ONTOLOGY_PRIORITIES = pd.DataFrame(
    [
        {"ontology": "reactome", "ontology_rank": 1},
        {"ontology": "ensembl_gene", "ontology_rank": 2},
        {"ontology": "chebi", "ontology_rank": 3},
        {"ontology": "uniprot", "ontology_rank": 4},
        {"ontology": "go", "ontology_rank": 5},
    ]
)

# SBML
# Biological qualifiers
# Biomodels qualifiers
BQB = SimpleNamespace(
    IS="BQB_IS",
    HAS_PART="BQB_HAS_PART",
    IS_PART_OF="BQB_IS_PART_OF",
    IS_VERSION_OF="BQB_IS_VERSION_OF",
    HAS_VERSION="BQB_HAS_VERSION",
    IS_HOMOLOG_TO="BQB_IS_HOMOLOG_TO",
    IS_DESCRIBED_BY="BQB_IS_DESCRIBED_BY",
    IS_ENCODED_BY="BQB_IS_ENCODED_BY",
    ENCODES="BQB_ENCODES",
    OCCURS_IN="BQB_OCCURS_IN",
    HAS_PROPERTY="BQB_HAS_PROPERTY",
    IS_PROPERTY_OF="BQB_IS_PROPERTY_OF",
    HAS_TAXON="BQB_HAS_TAXON",
    UNKNOWN="BQB_UNKNOWN",
)

# molecules are distinctly defined by these BQB terms
BQB_DEFINING_ATTRS = ["BQB_IS", "IS_HOMOLOG_TO"]

# a looser convention which will aggregate genes, transcripts, and proteins
# if they are linked with the appropriate bioqualifiers
BQB_DEFINING_ATTRS_LOOSE = [
    "BQB_IS",
    "IS_HOMOLOG_TO",
    "BQB_IS_ENCODED_BY",
    "BQB_ENCODES",
]

# identifiers
IDENTIFIERS = SimpleNamespace(
    ONTOLOGY="ontology", IDENTIFIER="identifier", BQB="bqb", URL="url"
)

SPECIES_IDENTIFIERS_REQUIRED_VARS = {
    SBML_DFS.S_ID,
    IDENTIFIERS.ONTOLOGY,
    IDENTIFIERS.IDENTIFIER,
    IDENTIFIERS.BQB,
    SBML_DFS.S_NAME,
}

BIOLOGICAL_QUALIFIERS = [
    "BQB_IS",
    "BQB_HAS_PART",
    "BQB_IS_PART_OF",
    "BQB_IS_VERSION_OF",
    "BQB_HAS_VERSION",
    "BQB_IS_HOMOLOG_TO",
    "BQB_IS_DESCRIBED_BY",
    "BQB_IS_ENCODED_BY",
    "BQB_ENCODES",
    "BQB_OCCURS_IN",
    "BQB_HAS_PROPERTY",
    "BQB_IS_PROPERTY_OF",
    "BQB_HAS_TAXON",
    "BQB_UNKNOWN",
]


def get_biological_qualifier_codes():
    bio_qualifier_codes = {getattr(libsbml, bqb): bqb for bqb in BIOLOGICAL_QUALIFIERS}

    return bio_qualifier_codes


BIOLOGICAL_QUALIFIER_CODES = get_biological_qualifier_codes()

# Systems biology ontology
SBOTERM_NAMES = SimpleNamespace(
    REACTANT="reactant",
    PRODUCT="product",
    CATALYST="catalyst",
    INHIBITOR="inhibitor",
    STIMULATOR="stimulator",
    MODIFIER="modifier",
    INTERACTOR="interactor",
)

MINI_SBO_TO_NAME = {
    "SBO:0000010": SBOTERM_NAMES.REACTANT,
    "SBO:0000011": SBOTERM_NAMES.PRODUCT,
    "SBO:0000013": SBOTERM_NAMES.CATALYST,
    "SBO:0000020": SBOTERM_NAMES.INHIBITOR,
    "SBO:0000459": SBOTERM_NAMES.STIMULATOR,
    "SBO:0000019": SBOTERM_NAMES.MODIFIER,
    "SBO:0000336": SBOTERM_NAMES.INTERACTOR,
}

MINI_SBO_FROM_NAME = {
    SBOTERM_NAMES.REACTANT: "SBO:0000010",
    SBOTERM_NAMES.PRODUCT: "SBO:0000011",
    SBOTERM_NAMES.CATALYST: "SBO:0000013",
    SBOTERM_NAMES.INHIBITOR: "SBO:0000020",
    SBOTERM_NAMES.STIMULATOR: "SBO:0000459",
    SBOTERM_NAMES.MODIFIER: "SBO:0000019",  # parent category of inhibitor and stimulator (i.e., activator)
    SBOTERM_NAMES.INTERACTOR: "SBO:0000336",  # entity participating in a physical or functional interaction
}

SBO_MODIFIER_NAMES = {
    SBOTERM_NAMES.INHIBITOR,
    SBOTERM_NAMES.STIMULATOR,
    SBOTERM_NAMES.MODIFIER,
}

MINI_SBO_NAME_TO_POLARITY = {
    SBOTERM_NAMES.REACTANT: "activation",
    SBOTERM_NAMES.PRODUCT: "activation",
    SBOTERM_NAMES.CATALYST: "activation",
    SBOTERM_NAMES.INHIBITOR: "inhibition",
    SBOTERM_NAMES.STIMULATOR: "activation",
    SBOTERM_NAMES.MODIFIER: "ambiguous",
    SBOTERM_NAMES.INTERACTOR: "ambiguous",
}

# how does changing a reactions' membership
# affect whether a reaction can occur
# for example, if I remove any substrate a reaction won't occur
# but I would have to remove all catalysts for it to not occur
SBO_NAME_TO_ROLE = {
    SBOTERM_NAMES.REACTANT: "DEFINING",
    SBOTERM_NAMES.PRODUCT: "DEFINING",
    SBOTERM_NAMES.INTERACTOR: "DEFINING",
    SBOTERM_NAMES.CATALYST: "REQUIRED",
    SBOTERM_NAMES.INHIBITOR: "OPTIONAL",
    SBOTERM_NAMES.STIMULATOR: "OPTIONAL",
    SBOTERM_NAMES.MODIFIER: "OPTIONAL",
}

# see also https://github.com/calico/netcontextr/blob/main/R/reactionTrimmingFunctions.R
VALID_SBO_ROLES = (
    # there is a direct correspondence between the set of defining entries and the identity of a reaction
    # e.g., the stoichiometery of a metabolic reaction or the members of a protein-protein interaction
    "DEFINING",
    # 1+ entries are needed if entries were initially defined. i.e., reactions which require a catalyst
    # would no longer exist if the catalyst was removed, but many reactions do not require a catalyst.
    "REQUIRED",
    # 0+ entries. optional species can be added or removed to a reaction without changing its identity
    "OPTIONAL",
)

# required variables for the edgelist formats used by mechanism_matching
CPR_EDGELIST = SimpleNamespace(
    S_ID_UPSTREAM="s_id_upstream",
    S_ID_DOWNSTREAM="s_id_downstream",
    SC_ID_UPSTREAM="sc_id_upstream",
    SC_ID_DOWNSTREAM="sc_id_downstream",
    IDENTIFIER_UPSTREAM="identifier_upstream",
    IDENTIFIER_DOWNSTREAM="identifier_downstream",
    S_NAME_UPSTREAM="s_name_upstream",
    S_NAME_DOWNSTREAM="s_name_downstream",
    SC_ID_ORIGIN="sc_id_origin",
    SC_ID_DEST="sc_id_dest",
)

IDENTIFIER_EDGELIST_REQ_VARS = {
    CPR_EDGELIST.IDENTIFIER_UPSTREAM,
    CPR_EDGELIST.IDENTIFIER_DOWNSTREAM,
}

CPR_EDGELIST_REQ_VARS = {
    CPR_EDGELIST.S_ID_UPSTREAM,
    CPR_EDGELIST.S_ID_DOWNSTREAM,
    CPR_EDGELIST.SC_ID_UPSTREAM,
    CPR_EDGELIST.SC_ID_DOWNSTREAM,
}

CPR_PATH_REQ_VARS = {CPR_EDGELIST.SC_ID_ORIGIN, CPR_EDGELIST.SC_ID_DEST}

FEATURE_ID_VAR_DEFAULT = "feature_id"

RESOLVE_MATCHES_AGGREGATORS = SimpleNamespace(
    WEIGHTED_MEAN="weighted_mean", MEAN="mean", FIRST="first", MAX="max"
)

RESOLVE_MATCHES_TMP_WEIGHT_COL = "__tmp_weight_for_aggregation__"

# specifying weighting schemes schema

DEFAULT_WT_TRANS = "identity"

DEFINED_WEIGHT_TRANSFORMATION = {
    DEFAULT_WT_TRANS: "_wt_transformation_identity",
    "string": "_wt_transformation_string",
    "string_inv": "_wt_transformation_string_inv",
}

SCORE_CALIBRATION_POINTS_DICT = {
    "weights": {"strong": 3, "good": 7, "okay": 20, "weak": 40},
    "string_wt": {"strong": 950, "good": 400, "okay": 230, "weak": 150},
}

SOURCE_VARS_DICT = {"string_wt": 10}

# source
SOURCE_SPEC = SimpleNamespace(
    PATHWAY_ID="pathway_id",
    MODEL="model",
    SOURCE="source",
    SPECIES="species",
    NAME="name",
    ENTRY="entry",
    N_COLLAPSED_PATHWAYS="n_collapsed_pathways",
    INDEX_NAME="entry",
    FILE="file",
    DATE="date",
)

EXPECTED_PW_INDEX_COLUMNS = {
    SOURCE_SPEC.FILE,
    SOURCE_SPEC.PATHWAY_ID,
    SOURCE_SPEC.SOURCE,
    SOURCE_SPEC.SPECIES,
    SOURCE_SPEC.NAME,
    SOURCE_SPEC.DATE,
}

# rules for specific ontologies

ONTOLOGIES = SimpleNamespace(
    CHEBI="chebi",
    ENSEMBL_GENE="ensembl_gene",
    ENSEMBL_TRANSCRIPT="ensembl_transcript",
    ENSEMBL_PROTEIN="ensembl_protein",
    GENE_NAME="gene_name",
    GO="go",
    MIRBASE="mirbase",
    NCBI_ENTREZ_GENE="ncbi_entrez_gene",
    PHAROS="pharos",
    REACTOME="reactome",
    SYMBOL="symbol",
    UNIPROT="uniprot",
)

ONTOLOGIES_LIST = list(ONTOLOGIES.__dict__.values())

CHARACTERISTIC_COMPLEX_ONTOLOGIES = [
    ONTOLOGIES.ENSEMBL_GENE,
    ONTOLOGIES.NCBI_ENTREZ_GENE,
    ONTOLOGIES.MIRBASE,
]

ONTOLOGY_ALIASES = SimpleNamespace(NCBI_ENTREZ_GENE={"ncbigene", "ncbi_gene"})

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

COMPARTMENTS = {
    "NUCLEOPLASM": "nucleoplasm",
    "CYTOPLASM": "cytoplasm",
    "CELLULAR_COMPONENT": "cellular_component",
    "CYTOSOL": "cytosol",
    "MITOCHONDRIA": "mitochondria",
    "MITOMEMBRANE": "mitochondrial membrane",
    "INNERMITOCHONDRIA": "inner mitochondria",
    "MITOMATRIX": "mitochondrial matrix",
    "ENDOPLASMICRETICULUM": "endoplasmic reticulum",
    "ERMEMBRANE": "endoplasmic reticulum membrane",
    "ERLUMEN": "endoplasmic reticulum lumen",
    "GOLGIAPPARATUS": "golgi apparatus",
    "GOLGIMEMBRANE": "golgi membrane",
    "NUCLEUS": "nucleus",
    "NUCLEARLUMEN": "nuclear lumen",
    "NUCLEOLUS": "nucleolus",
    "LYSOSOME": "lysosome",
    "PEROXISOME": "peroxisome",
    "EXTRACELLULAR": "extracellular",
}

COMPARTMENT_ALIASES = {
    "NUCLEOPLASM": ["nucleoplasm", "Nucleoplasm"],
    "CYTOPLASM": ["cytoplasm", "Cytoplasm"],
    "CELLULAR_COMPONENT": ["cellular_component", "Cellular_component"],
    "CYTOSOL": ["cytosol", "Cytosol"],
    "MITOCHONDRIA": ["mitochondria", "Mitochondria"],
    "MITOMEMBRANE": ["mitochondrial membrane", "Mitochondrial membrane"],
    "INNERMITOCHONDRIA": [
        "inner mitochondria",
        "Inner mitochondria",
        "inner mitochondrial compartment",
    ],
    "MITOMATRIX": [
        "mitochondrial matrix",
        "Mitochondrial matrix",
        "mitochondrial lumen",
        "Mitochondrial lumen",
    ],
    "ENDOPLASMICRETICULUM": ["endoplasmic reticulum", "Endoplasmic reticulum"],
    "ERMEMBRANE": ["endoplasmic reticulum membrane", "Endoplasmic reticulum membrane"],
    "ERLUMEN": ["endoplasmic reticulum lumen", "Endoplasmic reticulum lumen"],
    "GOLGIAPPARATUS": ["golgi apparatus", "Golgi apparatus"],
    "GOLGIMEMBRANE": ["Golgi membrane", "golgi membrane"],
    "NUCLEUS": ["nucleus", "Nucleus"],
    "NUCLEARLUMEN": ["nuclear lumen", "Nuclear lumen"],
    "NUCLEOLUS": ["nucleolus", "Nucleolus"],
    "LYSOSOME": ["lysosome", "Lysosome"],
    "PEROXISOME": ["peroxisome", "Peroxisome", "peroxisome/glyoxysome"],
    "EXTRACELLULAR": [
        "extracellular",
        "Extracellular",
        "extracellular space",
        "Extracellular space",
    ],
}

COMPARTMENTS_GO_TERMS = {
    "NUCLEOPLASM": "GO:0005654",
    "CELLULAR_COMPONENT": "GO:0005575",
    "CYTOPLASM": "GO:0005737",
    "CYTOSOL": "GO:0005829",
    "MITOCHONDRIA": "GO:0005739",
    "MITOMEMBRANE": "GO:0031966",
    "INNERMITOCHONDRIA": "GO:0005743",
    "MITOMATRIX": "GO:0005759",
    "ENDOPLASMICRETICULUM": "GO:0005783",
    "ERMEMBRANE": "GO:0005789",
    "ERLUMEN": "GO:0005788",
    "GOLGIAPPARATUS": "GO:0005794",
    "GOLGIMEMBRANE": "GO:0000139",
    "NUCLEUS": "GO:0005634",
    "NUCLEARLUMEN": "GO:0031981",
    "NUCLEOLUS": "GO:0005730",
    "LYSOSOME": "GO:0005764",
    "PEROXISOME": "GO:0005777",
    "EXTRACELLULAR": "GO:0005615",
}
