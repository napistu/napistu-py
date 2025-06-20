from __future__ import annotations

import pandas as pd
from napistu import sbml_dfs_core
from napistu.ingestion import sbml


def test_sbml_dfs(sbml_path):
    sbml_model = sbml.SBML(sbml_path)
    _ = sbml_model.model

    dfs = sbml_dfs_core.SBML_dfs(sbml_model)
    dfs.validate()

    assert type(dfs.get_cspecies_features()) is pd.DataFrame
    assert type(dfs.get_species_features()) is pd.DataFrame
    assert type(dfs.get_identifiers("species")) is pd.DataFrame
