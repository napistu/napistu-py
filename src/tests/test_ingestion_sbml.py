from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from napistu import sbml_dfs_core
from napistu.ingestion import sbml


def test_sbml_dfs(sbml_path, model_source_stub):
    sbml_model = sbml.SBML(sbml_path)
    _ = sbml_dfs_core.SBML_dfs(sbml_model, model_source_stub)


def test_sbml_dfs_reactome_sce(model_source_stub):
    """SBML_dfs(SBML(x)) succeeds on newer Reactome R-SCE-9696264.sbml.

    This file uses updated Reactome conventions; the test ensures the pipeline
    returns an SBML_dfs when those are supported.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    sbml_path = os.path.join(test_dir, "test_data", "R-SCE-9696264.sbml")
    if not os.path.isfile(sbml_path):
        pytest.skip(f"Test data not found: {sbml_path}")

    sbml_model = sbml.SBML(sbml_path)
    result = sbml_dfs_core.SBML_dfs(sbml_model, model_source_stub, verbose=True)

    assert result is not None
    assert isinstance(result, sbml_dfs_core.SBML_dfs)


def test_compartment_aliases_validation_positive():
    """
    Tests that a valid compartment aliases dictionary passes validation.
    """
    valid_aliases = {
        "extracellular": ["ECM", "extracellular space"],
        "cytosol": ["cytoplasm"],
    }
    # This should not raise an exception
    sbml.CompartmentAliasesValidator.model_validate(valid_aliases)


def test_compartment_aliases_validation_negative():
    """
    Tests that an invalid compartment aliases dictionary raises a ValidationError.
    """
    invalid_aliases = {
        "extracellular": ["ECM"],
        "not_a_real_compartment": ["fake"],
    }
    with pytest.raises(ValidationError):
        sbml.CompartmentAliasesValidator.model_validate(invalid_aliases)


def test_compartment_aliases_validation_bad_type():
    """
    Tests that a validation error is raised for incorrect data types.
    """
    # Test with a non-dict input
    with pytest.raises(ValidationError):
        sbml.CompartmentAliasesValidator.model_validate(["extracellular"])

    # Test with incorrect value types in the dictionary
    with pytest.raises(ValidationError):
        sbml.CompartmentAliasesValidator.model_validate({"extracellular": "ECM"})
