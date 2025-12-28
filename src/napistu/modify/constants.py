"""Module to contain constants for the modify submodule"""

from __future__ import annotations

from types import SimpleNamespace

from napistu.constants import IDENTIFIERS, ONTOLOGIES

# if_all defines reactions species which must all be present for a filter to occur
# except_any defines reaction species which will override "if_all"
# as_substrates defines reaction species which must be present as a substrate for filtering to occur

COFACTORS = SimpleNamespace(
    ACETYL_COA="acetyl-CoA",
    ADP="ADP",
    AMP="AMP",
    ATP="ATP",
    CL="Cl-",
    CO2="CO2",
    COA="CoA",
    FAD="FAD",
    FADH2="FADH2",
    GDP="GDP",
    GLU="Glu",
    GLN="Gln",
    GSH="GSH",
    GSSG="GSSG",
    GTP="GTP",
    H2CO3="H2CO3",
    HCO3="HCO3",
    H_PLUS="H+",
    NAD_PLUS="NAD+",
    NADH="NADH",
    NADP_PLUS="NADP+",
    NADPH="NADPH",
    NA_PLUS="Na+",
    O2="O2",
    PO4="PO4",
    PPI="PPi",
    SAH="SAH",
    SAM="SAM",
    UDP="UDP",
    WATER="water",
)

COFACTOR_DEFS = SimpleNamespace(
    # schema
    EXCEPT_ANY="except_any",
    AS_SUBSTRATE="as_substrate",
    IF_ALL="if_all",
    # variables
    COFACTOR="cofactor",
    FILTER_REASON="filter_reason",
)

COFACTOR_SCHEMA = {
    "ATP PO4 donation": {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP],
        COFACTOR_DEFS.EXCEPT_ANY: [COFACTORS.AMP],
    },
    "GTP PO4 donation": {COFACTOR_DEFS.IF_ALL: [COFACTORS.GTP, COFACTORS.GDP]},
    "ATP PPi donation": {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.AMP],
        COFACTOR_DEFS.EXCEPT_ANY: [COFACTORS.ADP],
    },
    "NADH H- donation": {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.NADH, COFACTORS.NAD_PLUS],
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.NADH],
    },
    "NADPH H- donation": {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.NADPH, COFACTORS.NADP_PLUS],
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.NADPH],
    },
    "SAH methyltransferase": {COFACTOR_DEFS.IF_ALL: [COFACTORS.SAH, COFACTORS.SAM]},
    "Glutathione oxidation": {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.GSSG, COFACTORS.GSH],
        COFACTOR_DEFS.EXCEPT_ANY: [COFACTORS.NADPH],
    },
    # "Glutamine aminotransferase" :
    #    {"if_all" : [COFACTORS.GLN, COFACTORS.GLU],
    #     "except_any" : [COFACTORS.ATP]},
    "Water": {COFACTOR_DEFS.IF_ALL: [COFACTORS.WATER]},
    "PO4": {COFACTOR_DEFS.IF_ALL: [COFACTORS.PO4]},
    "PPi": {COFACTOR_DEFS.IF_ALL: [COFACTORS.PPI]},
    "H+": {COFACTOR_DEFS.IF_ALL: [COFACTORS.H_PLUS]},
    "O2": {COFACTOR_DEFS.IF_ALL: [COFACTORS.O2]},
    "CO2": {COFACTOR_DEFS.IF_ALL: [COFACTORS.CO2]},
    "Na+": {COFACTOR_DEFS.IF_ALL: [COFACTORS.NA_PLUS]},
    "Cl-": {COFACTOR_DEFS.IF_ALL: [COFACTORS.CL]},
    "CoA": {COFACTOR_DEFS.IF_ALL: [COFACTORS.COA]},
    "HCO3-": {COFACTOR_DEFS.IF_ALL: [COFACTORS.HCO3]},
}

COFACTOR_CHEBI_IDS = {
    COFACTORS.ACETYL_COA: [15351],
    COFACTORS.ADP: [456216, 16761],  # ADP(3−)
    COFACTORS.AMP: [16027],
    COFACTORS.ATP: [30616, 15422],  # ATP(4-)
    COFACTORS.CL: [29311],
    COFACTORS.CO2: [16526],
    COFACTORS.COA: [1146900, 57287],
    COFACTORS.FAD: [16238],
    COFACTORS.FADH2: [17877],
    COFACTORS.GDP: [17552],
    COFACTORS.GLU: [29985],
    COFACTORS.GLN: [58359],
    COFACTORS.GSH: [16856],
    COFACTORS.GSSG: [17858],
    COFACTORS.GTP: [15996],
    COFACTORS.H2CO3: [28976],
    COFACTORS.HCO3: [17544],
    COFACTORS.H_PLUS: [15378, 24636],
    COFACTORS.NAD_PLUS: [57540, 15846],  # NAD(1-), NAD(+)
    COFACTORS.NADH: [57945, 16908],  # NADH(2−), NADH
    COFACTORS.NADP_PLUS: [18009, 58349],  # NADP(3−)
    COFACTORS.NADPH: [16474],
    COFACTORS.NA_PLUS: [29101],
    COFACTORS.O2: [15379],
    COFACTORS.PO4: [18367],
    COFACTORS.PPI: [29888, 18361],  # H2PO4, PPi4-
    COFACTORS.SAH: [16680],
    COFACTORS.SAM: [15414],
    COFACTORS.UDP: [17659],
    COFACTORS.WATER: [15377, 16234],  # HO-
}

NEO4J_MEMBERS_RAW = SimpleNamespace(
    SET_NAME="set_name",
    SET_ID="set_id",
    MEMBER_NAME="member_name",
    MEMBER_ID="member_id",
    IDENTIFIER=IDENTIFIERS.IDENTIFIER,
    ONTOLOGY=IDENTIFIERS.ONTOLOGY,
)

NEO4_MEMBERS_SET = {
    NEO4J_MEMBERS_RAW.SET_NAME,
    NEO4J_MEMBERS_RAW.SET_ID,
    NEO4J_MEMBERS_RAW.MEMBER_NAME,
    NEO4J_MEMBERS_RAW.MEMBER_ID,
    NEO4J_MEMBERS_RAW.IDENTIFIER,
    NEO4J_MEMBERS_RAW.ONTOLOGY,
}

REACTOME_CROSSREF_RAW = SimpleNamespace(
    MEMBER_NAME="member_name",
    REACTOME_ID="reactome_id",
    UNIPROT=ONTOLOGIES.UNIPROT,
    IDENTIFIER=IDENTIFIERS.IDENTIFIER,
    ONTOLOGY=IDENTIFIERS.ONTOLOGY,
    URL=IDENTIFIERS.URL,
)

REACTOME_CROSSREF_SET = {
    REACTOME_CROSSREF_RAW.MEMBER_NAME,
    REACTOME_CROSSREF_RAW.REACTOME_ID,
    REACTOME_CROSSREF_RAW.UNIPROT,
    REACTOME_CROSSREF_RAW.IDENTIFIER,
    REACTOME_CROSSREF_RAW.ONTOLOGY,
    REACTOME_CROSSREF_RAW.URL,
}

CURATION_DEFS = SimpleNamespace(
    # Entity types
    SPECIES="species",
    REACTIONS="reactions",
    COMPARTMENTS="compartments",
    COMPARTMENTALIZED_SPECIES="compartmentalized_species",
    REACTION_SPECIES="reaction_species",
    REMOVE="remove",
    FOCI="foci",
    # Field names
    CURATOR="curator",
    EVIDENCE="evidence",
    URI="uri",
    SBO_TERM_NAME="sbo_term_name",
    # Default values
    UNKNOWN="unknown",
    # Remove table columns
    TABLE="table",
    VARIABLE="variable",
)

VALID_ANNOTATION_TYPES = [
    CURATION_DEFS.FOCI,
    CURATION_DEFS.REACTIONS,
    CURATION_DEFS.SPECIES,
    CURATION_DEFS.COMPARTMENTS,
    CURATION_DEFS.COMPARTMENTALIZED_SPECIES,
    CURATION_DEFS.REACTION_SPECIES,
    CURATION_DEFS.REMOVE,
]
