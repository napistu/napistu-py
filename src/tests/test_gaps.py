import copy
import pytest
import types
import pandas as pd
import numpy as np

import pandas as pd
from napistu.sbml_dfs_core import SBML_dfs
from napistu.constants import SBML_DFS
from napistu.constants import MINI_SBO_FROM_NAME

from napistu.modify.gaps import add_transportation_reactions

# Minimal compartments table
compartments = pd.DataFrame({
    SBML_DFS.C_NAME: ["mitochondria", "nucleus", "cytosol"],
    SBML_DFS.C_IDENTIFIERS: [None],
    SBML_DFS.C_SOURCE: [None]
}, index=["c_mito", "c_nucl", "c_cytosol"]).rename_axis(SBML_DFS.C_ID)

# Minimal species table
species = pd.DataFrame({
    SBML_DFS.S_NAME: ["A"],
    SBML_DFS.S_IDENTIFIERS: [None],
    SBML_DFS.S_SOURCE: [None]
}, index=["s_A"]).rename_axis(SBML_DFS.S_ID)

# Minimal compartmentalized_species table
compartmentalized_species = pd.DataFrame({
    SBML_DFS.SC_NAME: ["A [mitochondria]", "A [nucleus]"],
    SBML_DFS.S_ID: ["s_A", "s_A"],
    SBML_DFS.C_ID: ["c_mito", "c_nucl"],
    SBML_DFS.SC_SOURCE: [None]
}, index=["sc_A_mito", "sc_A_nucl"]).rename_axis(SBML_DFS.SC_ID)

# Minimal reactions table
reactions = pd.DataFrame({
    SBML_DFS.R_NAME: ["A [mito] -> A [mito]", "A [nucl] -> A [nucl]"],
    SBML_DFS.R_IDENTIFIERS: [None],
    SBML_DFS.R_SOURCE: [None],
    SBML_DFS.R_ISREVERSIBLE: [True, True]
}, index=["r_A_mito", "r_A_nucl"]).rename_axis(SBML_DFS.R_ID)

# Minimal reaction_species table
reaction_species = pd.DataFrame({
    SBML_DFS.R_ID: ["r_A_mito", "r_A_mito", "r_A_nucl", "r_A_nucl"],
    SBML_DFS.SC_ID: ["sc_A_mito", "sc_A_mito", "sc_A_nucl", "sc_A_nucl"],
    SBML_DFS.STOICHIOMETRY: [-1, 1, -1, 1],
    SBML_DFS.SBO_TERM: [MINI_SBO_FROM_NAME["reactant"], MINI_SBO_FROM_NAME["product"], MINI_SBO_FROM_NAME["reactant"], MINI_SBO_FROM_NAME["product"]]
}, index=["rsc_A_mito_sub", "rsc_A_mito_prod", "rsc_A_nucl_sub", "rsc_A_nucl_prod"]).rename_axis(SBML_DFS.RSC_ID)

# Assemble the dict
sbml_dict = {
    SBML_DFS.COMPARTMENTS: compartments,
    SBML_DFS.SPECIES: species,
    SBML_DFS.COMPARTMENTALIZED_SPECIES: compartmentalized_species,
    SBML_DFS.REACTIONS: reactions,
    SBML_DFS.REACTION_SPECIES: reaction_species,
}

# Create the SBML_dfs object
sbml_dfs = SBML_dfs(sbml_dict)

def test_add_transportation_reactions(sbml_dfs):
    sbml_dfs_w_transport = add_transportation_reactions(sbml_dfs, exchange_compartment="cytosol")
    assert sbml_dfs_w_transport.reactions.shape[0] == 3, "Should add 1 reactions"