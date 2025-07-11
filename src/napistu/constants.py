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

SCHEMA_DEFS = SimpleNamespace(
    TABLE="table",
    PK="pk",
    FK="fk",
    LABEL="label",
    ID="id",
    SOURCE="source",
    VARS="vars",
)

SBML_DFS_SCHEMA = SimpleNamespace(
    SCHEMA={
        SBML_DFS.COMPARTMENTS: {
            SCHEMA_DEFS.TABLE: SBML_DFS.COMPARTMENTS,
            SCHEMA_DEFS.PK: SBML_DFS.C_ID,
            SCHEMA_DEFS.LABEL: SBML_DFS.C_NAME,
            SCHEMA_DEFS.ID: SBML_DFS.C_IDENTIFIERS,
            SCHEMA_DEFS.SOURCE: SBML_DFS.C_SOURCE,
            SCHEMA_DEFS.VARS: [
                SBML_DFS.C_NAME,
                SBML_DFS.C_IDENTIFIERS,
                SBML_DFS.C_SOURCE,
            ],
        },
        SBML_DFS.SPECIES: {
            SCHEMA_DEFS.TABLE: SBML_DFS.SPECIES,
            SCHEMA_DEFS.PK: SBML_DFS.S_ID,
            SCHEMA_DEFS.LABEL: SBML_DFS.S_NAME,
            SCHEMA_DEFS.ID: SBML_DFS.S_IDENTIFIERS,
            SCHEMA_DEFS.SOURCE: SBML_DFS.S_SOURCE,
            SCHEMA_DEFS.VARS: [
                SBML_DFS.S_NAME,
                SBML_DFS.S_IDENTIFIERS,
                SBML_DFS.S_SOURCE,
            ],
        },
        SBML_DFS.COMPARTMENTALIZED_SPECIES: {
            SCHEMA_DEFS.TABLE: SBML_DFS.COMPARTMENTALIZED_SPECIES,
            SCHEMA_DEFS.PK: SBML_DFS.SC_ID,
            SCHEMA_DEFS.LABEL: SBML_DFS.SC_NAME,
            SCHEMA_DEFS.FK: [SBML_DFS.S_ID, SBML_DFS.C_ID],
            SCHEMA_DEFS.SOURCE: SBML_DFS.SC_SOURCE,
            SCHEMA_DEFS.VARS: [
                SBML_DFS.SC_NAME,
                SBML_DFS.S_ID,
                SBML_DFS.C_ID,
                SBML_DFS.SC_SOURCE,
            ],
        },
        SBML_DFS.REACTIONS: {
            SCHEMA_DEFS.TABLE: SBML_DFS.REACTIONS,
            SCHEMA_DEFS.PK: SBML_DFS.R_ID,
            SCHEMA_DEFS.LABEL: SBML_DFS.R_NAME,
            SCHEMA_DEFS.ID: SBML_DFS.R_IDENTIFIERS,
            SCHEMA_DEFS.SOURCE: SBML_DFS.R_SOURCE,
            SCHEMA_DEFS.VARS: [
                SBML_DFS.R_NAME,
                SBML_DFS.R_IDENTIFIERS,
                SBML_DFS.R_SOURCE,
                SBML_DFS.R_ISREVERSIBLE,
            ],
        },
        SBML_DFS.REACTION_SPECIES: {
            SCHEMA_DEFS.TABLE: SBML_DFS.REACTION_SPECIES,
            SCHEMA_DEFS.PK: SBML_DFS.RSC_ID,
            SCHEMA_DEFS.FK: [SBML_DFS.R_ID, SBML_DFS.SC_ID],
            SCHEMA_DEFS.VARS: [
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

ENTITIES_TO_ENTITY_DATA = {
    SBML_DFS.SPECIES: SBML_DFS.SPECIES_DATA,
    SBML_DFS.REACTIONS: SBML_DFS.REACTIONS_DATA,
}

REQUIRED_REACTION_FROMEDGELIST_COLUMNS = [
    "sc_id_up",
    "sc_id_down",
    SBML_DFS.SBO_TERM,
    SBML_DFS.R_NAME,
    SBML_DFS.R_IDENTIFIERS,
    SBML_DFS.R_ISREVERSIBLE,
]

NAPISTU_STANDARD_OUTPUTS = SimpleNamespace(
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
    SBML_DFS.R_NAME,
    SBML_DFS.SBO_TERM,
    SBML_DFS.R_IDENTIFIERS,
    SBML_DFS.R_ISREVERSIBLE,
}

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

VALID_BQB_TERMS = list(BQB.__dict__.values())

# molecules are distinctly defined by these BQB terms
BQB_DEFINING_ATTRS = [BQB.IS, BQB.IS_HOMOLOG_TO]

# a looser convention which will aggregate genes, transcripts, and proteins
# if they are linked with the appropriate bioqualifiers
BQB_DEFINING_ATTRS_LOOSE = [
    BQB.IS,
    BQB.IS_HOMOLOG_TO,
    BQB.IS_ENCODED_BY,
    BQB.ENCODES,
]

# identifiers
IDENTIFIERS = SimpleNamespace(
    ONTOLOGY="ontology", IDENTIFIER="identifier", BQB="bqb", URL="url"
)

BQB_PRIORITIES = pd.DataFrame(
    [
        {IDENTIFIERS.BQB: BQB.IS, "bqb_rank": 1},
        {IDENTIFIERS.BQB: BQB.HAS_PART, "bqb_rank": 2},
    ]
)

IDENTIFIERS_REQUIRED_VARS = {
    IDENTIFIERS.ONTOLOGY,
    IDENTIFIERS.IDENTIFIER,
    IDENTIFIERS.BQB,
}

SPECIES_IDENTIFIERS_REQUIRED_VARS = IDENTIFIERS_REQUIRED_VARS | {
    SBML_DFS.S_ID,
    SBML_DFS.S_NAME,
}


def get_biological_qualifier_codes():
    bio_qualifier_codes = {getattr(libsbml, bqb): bqb for bqb in VALID_BQB_TERMS}

    return bio_qualifier_codes


BIOLOGICAL_QUALIFIER_CODES = get_biological_qualifier_codes()

# Systems biology ontology
SBOTERM_NAMES = SimpleNamespace(
    REACTANT="reactant",
    PRODUCT="product",
    CATALYST="catalyst",
    INHIBITOR="inhibitor",
    STIMULATOR="stimulator",
    MODIFIED="modified",
    MODIFIER="modifier",
    INTERACTOR="interactor",
)

MINI_SBO_TO_NAME = {
    "SBO:0000010": SBOTERM_NAMES.REACTANT,
    "SBO:0000011": SBOTERM_NAMES.PRODUCT,
    "SBO:0000013": SBOTERM_NAMES.CATALYST,
    "SBO:0000019": SBOTERM_NAMES.MODIFIER,
    "SBO:0000020": SBOTERM_NAMES.INHIBITOR,
    "SBO:0000336": SBOTERM_NAMES.INTERACTOR,
    "SBO:0000459": SBOTERM_NAMES.STIMULATOR,
    "SBO:0000644": SBOTERM_NAMES.MODIFIED,
}

MINI_SBO_FROM_NAME = {
    SBOTERM_NAMES.CATALYST: "SBO:0000013",
    SBOTERM_NAMES.INHIBITOR: "SBO:0000020",
    SBOTERM_NAMES.INTERACTOR: "SBO:0000336",  # entity participating in a physical or functional interaction
    SBOTERM_NAMES.MODIFIED: "SBO:0000644",
    SBOTERM_NAMES.MODIFIER: "SBO:0000019",  # parent category of inhibitor and stimulator (i.e., activator)
    SBOTERM_NAMES.PRODUCT: "SBO:0000011",
    SBOTERM_NAMES.REACTANT: "SBO:0000010",  # aka substrate
    SBOTERM_NAMES.STIMULATOR: "SBO:0000459",  # aka activator
}

VALID_SBO_TERM_NAMES = list(SBOTERM_NAMES.__dict__.values())
VALID_SBO_TERMS = list(MINI_SBO_FROM_NAME.values())

SBO_MODIFIER_NAMES = {
    SBOTERM_NAMES.INHIBITOR,
    SBOTERM_NAMES.STIMULATOR,
    SBOTERM_NAMES.MODIFIER,
}

MINI_SBO_NAME_TO_POLARITY = {
    SBOTERM_NAMES.CATALYST: "activation",
    SBOTERM_NAMES.INHIBITOR: "inhibition",
    SBOTERM_NAMES.INTERACTOR: "ambiguous",
    SBOTERM_NAMES.MODIFIED: "ambiguous",
    SBOTERM_NAMES.MODIFIER: "ambiguous",
    SBOTERM_NAMES.PRODUCT: "activation",
    SBOTERM_NAMES.REACTANT: "ambiguous",
    SBOTERM_NAMES.STIMULATOR: "activation",
}

# how does changing a reactions' membership
# affect whether a reaction can occur
# for example, if I remove any substrate a reaction won't occur
# but I would have to remove all catalysts for it to not occur
SBO_ROLES_DEFS = SimpleNamespace(
    DEFINING="DEFINING", REQUIRED="REQUIRED", OPTIONAL="OPTIONAL", SBO_ROLE="sbo_role"
)

SBO_NAME_TO_ROLE = {
    SBOTERM_NAMES.REACTANT: SBO_ROLES_DEFS.DEFINING,
    SBOTERM_NAMES.PRODUCT: SBO_ROLES_DEFS.DEFINING,
    SBOTERM_NAMES.INTERACTOR: SBO_ROLES_DEFS.DEFINING,
    SBOTERM_NAMES.CATALYST: SBO_ROLES_DEFS.REQUIRED,
    SBOTERM_NAMES.INHIBITOR: SBO_ROLES_DEFS.OPTIONAL,
    SBOTERM_NAMES.STIMULATOR: SBO_ROLES_DEFS.OPTIONAL,
    SBOTERM_NAMES.MODIFIED: SBO_ROLES_DEFS.DEFINING,
    SBOTERM_NAMES.MODIFIER: SBO_ROLES_DEFS.OPTIONAL,
}

# see also https://github.com/calico/netcontextr/blob/main/R/reactionTrimmingFunctions.R
VALID_SBO_ROLES = (
    # there is a direct correspondence between the set of defining entries and the identity of a reaction
    # e.g., the stoichiometery of a metabolic reaction or the members of a protein-protein interaction
    SBO_ROLES_DEFS.DEFINING,
    # 1+ entries are needed if entries were initially defined. i.e., reactions which require a catalyst
    # would no longer exist if the catalyst was removed, but many reactions do not require a catalyst.
    SBO_ROLES_DEFS.REQUIRED,
    # 0+ entries. optional species can be added or removed to a reaction without changing its identity
    SBO_ROLES_DEFS.OPTIONAL,
)

# required variables for the edgelist formats used by the matching subpackage
# also used in some network modules
NAPISTU_EDGELIST = SimpleNamespace(
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
    NAPISTU_EDGELIST.IDENTIFIER_UPSTREAM,
    NAPISTU_EDGELIST.IDENTIFIER_DOWNSTREAM,
}

NAPISTU_EDGELIST_REQ_VARS = {
    NAPISTU_EDGELIST.S_ID_UPSTREAM,
    NAPISTU_EDGELIST.S_ID_DOWNSTREAM,
    NAPISTU_EDGELIST.SC_ID_UPSTREAM,
    NAPISTU_EDGELIST.SC_ID_DOWNSTREAM,
}

NAPISTU_PATH_REQ_VARS = {NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST}

FEATURE_ID_VAR_DEFAULT = "feature_id"

RESOLVE_MATCHES_AGGREGATORS = SimpleNamespace(
    WEIGHTED_MEAN="weighted_mean", MEAN="mean", FIRST="first", MAX="max"
)

RESOLVE_MATCHES_TMP_WEIGHT_COL = "__tmp_weight_for_aggregation__"

# source information

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
    ENSEMBL_GENE_VERSION="ensembl_gene_version",
    ENSEMBL_TRANSCRIPT="ensembl_transcript",
    ENSEMBL_TRANSCRIPT_VERSION="ensembl_transcript_version",
    ENSEMBL_PROTEIN="ensembl_protein",
    ENSEMBL_PROTEIN_VERSION="ensembl_protein_version",
    GENE_NAME="gene_name",
    GO="go",
    KEGG="kegg",
    MIRBASE="mirbase",
    NCBI_ENTREZ_GENE="ncbi_entrez_gene",
    PHAROS="pharos",
    REACTOME="reactome",
    SYMBOL="symbol",
    UNIPROT="uniprot",
    WIKIPATHWAYS="wikipathways",
)

ONTOLOGIES_LIST = list(ONTOLOGIES.__dict__.values())

ONTOLOGY_SPECIES_ALIASES = {
    ONTOLOGIES.NCBI_ENTREZ_GENE: {"ncbigene", "ncbi_gene"},
    ONTOLOGIES.ENSEMBL_GENE: {"ensembl_gene_id"},
    ONTOLOGIES.UNIPROT: {"Uniprot"},
}

ONTOLOGY_PRIORITIES = pd.DataFrame(
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
