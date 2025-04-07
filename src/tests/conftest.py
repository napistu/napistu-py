from __future__ import annotations

import os

from cpr import consensus
from cpr import indices
from cpr import sbml_dfs_core
from cpr.ingestion import sbml
from pytest import fixture


@fixture
def sbml_path():
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    sbml_path = os.path.join(test_path, "test_data", "R-HSA-1237044.sbml")

    if not os.path.isfile(sbml_path):
        raise ValueError(f"{sbml_path} not found")
    return sbml_path


@fixture
def sbml_model(sbml_path):
    sbml_model = sbml.SBML(sbml_path)
    return sbml_model


@fixture
def sbml_dfs(sbml_model):
    sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)
    return sbml_dfs


@fixture
def sbml_dfs_metabolism():
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    test_data = os.path.join(test_path, "test_data")

    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index_metabolism.tsv"))
    sbml_dfs_dict = consensus.construct_sbml_dfs_dict(pw_index)
    sbml_dfs = consensus.construct_consensus_model(sbml_dfs_dict, pw_index)

    return sbml_dfs


@fixture
def sbml_dfs_glucose_metabolism():
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    test_data = os.path.join(test_path, "test_data")
    sbml_path = os.path.join(test_data, "reactome_glucose_metabolism.sbml")

    sbml_model = sbml.SBML(sbml_path).model
    sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)

    return sbml_dfs
