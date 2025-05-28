"""Module to contain constants for the modify submodule"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from napistu.constants import IDENTIFIERS
from napistu.constants import ONTOLOGIES

VALID_ANNOTATION_TYPES = [
    "foci",
    "reactions",
    "species",
    "compartments",
    "compartmentalized_species",
    "reaction_species",
    "remove",
]

# if_all defines reactions species which must all be present for a filter to occur
# except_any defines reaction species which will override "if_all"
# as_substrates defines reaction species which must be present as a substrate for filtering to occur
COFACTOR_SCHEMA = {
    "ATP PO4 donation": {"if_all": ["ATP", "ADP"], "except_any": ["AMP"]},
    "GTP PO4 donation": {"if_all": ["GTP", "GDP"]},
    "ATP PPi donation": {"if_all": ["ATP", "AMP"], "except_any": ["ADP"]},
    "NADH H- donation": {"if_all": ["NADH", "NAD+"], "as_substrate": ["NADH"]},
    "NADPH H- donation": {"if_all": ["NADPH", "NADP+"], "as_substrate": ["NADPH"]},
    "SAH methyltransferase": {"if_all": ["SAH", "SAM"]},
    "Glutathione oxidation": {"if_all": ["GSSG", "GSH"], "except_any": ["NADPH"]},
    # "Glutamine aminotransferase" :
    #    {"if_all" : ["Gln", "Glu"],
    #     "except_any" : ["ATP"]},
    "Water": {"if_all": ["water"]},
    "PO4": {"if_all": ["PO4"]},
    "PPi": {"if_all": ["PPi"]},
    "H+": {"if_all": ["H+"]},
    "O2": {"if_all": ["O2"]},
    "CO2": {"if_all": ["CO2"]},
    "Na+": {"if_all": ["Na+"]},
    "Cl-": {"if_all": ["Cl-"]},
    "CoA": {"if_all": ["CoA"]},
    "HCO3-": {"if_all": ["HCO3"]},
}

COFACTOR_CHEBI_IDS = pd.DataFrame(
    [
        ("ADP", 456216),  # ADP(3−)
        ("ADP", 16761),
        ("AMP", 16027),
        ("ATP", 30616),  # ATP(4-)
        ("ATP", 15422),
        ("CO2", 16526),
        ("HCO3", 17544),
        ("H2CO3", 28976),
        ("GDP", 17552),
        ("GSH", 16856),
        ("GSSG", 17858),
        ("GTP", 15996),
        ("Glu", 29985),
        ("Gln", 58359),
        ("H+", 15378),
        ("H+", 24636),
        ("O2", 15379),
        ("NADH", 57945),  # NADH(2−)
        ("NADH", 16908),  # NADH
        ("NAD+", 57540),  # NAD(1-)
        ("NAD+", 15846),  # NAD(+)
        ("NADPH", 16474),
        ("NADP+", 18009),
        ("NADP+", 58349),  # NADP(3−)
        ("PO4", 18367),
        ("PPi", 29888),  # H2PO4
        ("PPi", 18361),  # PPi4-
        ("SAH", 16680),
        ("SAM", 15414),
        ("water", 15377),
        ("water", 16234),  # HO-
        ("Na+", 29101),
        ("Cl-", 29311),
        ("CoA", 1146900),
        ("CoA", 57287),
        ("acetyl-CoA", 15351),
        ("FAD", 16238),
        ("FADH2", 17877),
        ("UDP", 17659),
    ],
    columns=["cofactor", "chebi"],
)

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
