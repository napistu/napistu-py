from __future__ import annotations

import os

from napistu import sbml_dfs_core
from napistu.ingestion import sbml
from napistu.modify import uncompartmentalize

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
sbml_path = os.path.join(test_path, "test_data", "reactome_glucose_metabolism.sbml")

if not os.path.isfile(sbml_path):
    raise ValueError(f"{sbml_path} not found")


def test_uncompartmentalize():
    sbml_model = sbml.SBML(sbml_path)
    sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)
    sbml_dfs.validate()

    assert sbml_dfs.compartmentalized_species.shape[0] == 107
    assert sbml_dfs.reactions.shape[0] == 50
    assert sbml_dfs.reaction_species.shape[0] == 250

    uncomp_sbml_dfs = uncompartmentalize.uncompartmentalize_sbml_dfs(sbml_dfs)
    uncomp_sbml_dfs.validate()

    assert uncomp_sbml_dfs.compartments.shape[0] == 1
    # assert uncomp_sbml_dfs.species.shape[0] == sbml_dfs.species.shape[0]
    assert uncomp_sbml_dfs.compartmentalized_species.shape[0] == 80
    assert uncomp_sbml_dfs.reactions.shape[0] == 47
    assert uncomp_sbml_dfs.reaction_species.shape[0] == 217


################################################
# __main__
################################################

if __name__ == "__main__":
    test_uncompartmentalize()
