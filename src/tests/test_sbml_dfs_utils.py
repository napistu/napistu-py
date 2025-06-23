from __future__ import annotations

import pandas as pd

from napistu import sbml_dfs_utils
from napistu.constants import BQB, BQB_DEFINING_ATTRS, BQB_DEFINING_ATTRS_LOOSE


def test_id_formatter():
    input_vals = range(50, 100)

    # create standard IDs
    ids = sbml_dfs_utils.id_formatter(input_vals, "s_id", id_len=8)
    # invert standard IDs
    inv_ids = sbml_dfs_utils.id_formatter_inv(ids)

    assert list(input_vals) == inv_ids


def test_get_characteristic_species_ids():
    """
    Test get_characteristic_species_ids function with both dogmatic and non-dogmatic cases.
    """
    # Create mock species identifiers data
    mock_species_ids = pd.DataFrame(
        {
            "s_id": ["s1", "s2", "s3", "s4", "s5"],
            "identifier": ["P12345", "CHEBI:15377", "GO:12345", "P67890", "P67890"],
            "ontology": ["uniprot", "chebi", "go", "uniprot", "chebi"],
            "bqb": [
                "BQB_IS",
                "BQB_IS",
                "BQB_HAS_PART",
                "BQB_HAS_VERSION",
                "BQB_ENCODES",
            ],
        }
    )

    # Create mock SBML_dfs object
    class MockSBML_dfs:
        def get_identifiers(self, entity_type):
            return mock_species_ids

    mock_sbml = MockSBML_dfs()

    # Test dogmatic case (default)
    expected_bqbs = BQB_DEFINING_ATTRS + [BQB.HAS_PART]  # noqa: F841
    dogmatic_result = sbml_dfs_utils.get_characteristic_species_ids(mock_sbml)
    expected_dogmatic = mock_species_ids.query("bqb in @expected_bqbs")

    pd.testing.assert_frame_equal(dogmatic_result, expected_dogmatic, check_like=True)

    # Test non-dogmatic case
    expected_bqbs = BQB_DEFINING_ATTRS_LOOSE + [BQB.HAS_PART]  # noqa: F841
    non_dogmatic_result = sbml_dfs_utils.get_characteristic_species_ids(
        mock_sbml, dogmatic=False
    )
    expected_non_dogmatic = mock_species_ids.query("bqb in @expected_bqbs")

    pd.testing.assert_frame_equal(
        non_dogmatic_result, expected_non_dogmatic, check_like=True
    )


def test_formula(sbml_dfs):
    # create a formula string

    an_r_id = sbml_dfs.reactions.index[0]

    reaction_species_df = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species["r_id"] == an_r_id
    ].merge(sbml_dfs.compartmentalized_species, left_on="sc_id", right_index=True)

    formula_str = sbml_dfs_utils.construct_formula_string(
        reaction_species_df, sbml_dfs.reactions, name_var="sc_name"
    )

    assert isinstance(formula_str, str)
    assert (
        formula_str
        == "CO2 [extracellular region] -> CO2 [cytosol] ---- modifiers: AQP1 tetramer [plasma membrane]]"
    )
