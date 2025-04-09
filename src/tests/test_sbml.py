from __future__ import annotations

import os
import pytest

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


@pytest.mark.skip_on_windows
def test_adding_sbml_annotations(sbml_model):
    annotations = pd.DataFrame(
        [
            {
                "id": "compartment_12045",
                "type": "compartment",
                "uri": "http://identifiers.org/chebi/CHEBI:00000",
            },
            {
                "id": "species_9033251",
                "type": "species",
                "uri": "http://identifiers.org/bigg.metabolite/fakemet",
            },
            {
                "id": "species_9033251",
                "type": "species",
                "uri": "http://identifiers.org/chebi/CHEBI:00000",
            },
        ]
    )

    outpath = "/tmp/tmp_write_test.sbml"

    sbml.add_sbml_annotations(sbml_model, annotations, save_path=outpath)
    assert os.path.isfile(outpath) is True
