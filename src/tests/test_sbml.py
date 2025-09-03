from __future__ import annotations

import pytest
from pydantic import ValidationError

from napistu import sbml_dfs_core
from napistu.ingestion import sbml


def test_sbml_dfs(sbml_path, model_source_stub):
    sbml_model = sbml.SBML(sbml_path)
    _ = sbml_dfs_core.SBML_dfs(sbml_model, model_source_stub)


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
