"""Module to contain constants for the modify submodule"""

from __future__ import annotations

import pandas as pd

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
