import warnings

import numpy as np
import pandas as pd
import pytest

from napistu.constants import BQB, IDENTIFIERS, MINI_SBO_FROM_NAME, ONTOLOGIES, SBML_DFS
from napistu.identifiers import Identifiers
from napistu.ingestion.constants import COMPARTMENTS, EXCHANGE_COMPARTMENT
from napistu.modify import gaps
from napistu.sbml_dfs_core import SBML_dfs
from napistu.source import Source


@pytest.fixture
def sbml_dfs_missing_transport_rxns(model_source_stub):
    """Create a minimal SBML_dfs object missing transport reactions."""

    blank_id = Identifiers([])
    blank_source = Source.empty()

    compartments = pd.DataFrame(
        {
            SBML_DFS.C_NAME: [
                COMPARTMENTS.MITOCHONDRIA,
                COMPARTMENTS.NUCLEUS,
                EXCHANGE_COMPARTMENT,
            ],
            SBML_DFS.C_IDENTIFIERS: [blank_id],
            SBML_DFS.C_SOURCE: [blank_source],
        },
        index=["c_mito", "c_nucl", "c_exchange"],
    ).rename_axis(SBML_DFS.C_ID)

    # Minimal species table
    species = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["A", "B"],
            SBML_DFS.S_IDENTIFIERS: [
                Identifiers(
                    [
                        {
                            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.UNIPROT,
                            IDENTIFIERS.IDENTIFIER: "PFAKE1",
                            IDENTIFIERS.BQB: BQB.IS,
                            IDENTIFIERS.URL: None,
                        },
                        {
                            IDENTIFIERS.ONTOLOGY: ONTOLOGIES.UNIPROT,
                            IDENTIFIERS.IDENTIFIER: "PFAKE2",
                            IDENTIFIERS.BQB: BQB.IS,
                            IDENTIFIERS.URL: None,
                        },
                    ]
                )
            ],
            SBML_DFS.S_SOURCE: [blank_source],
        },
        index=["s_A", "s_B"],
    ).rename_axis(SBML_DFS.S_ID)

    # Minimal compartmentalized_species table
    compartmentalized_species = pd.DataFrame(
        {
            SBML_DFS.SC_NAME: ["A [mitochondria]", "A [nucleus]", "B [exchange]"],
            SBML_DFS.S_ID: ["s_A", "s_A", "s_B"],
            SBML_DFS.C_ID: ["c_mito", "c_nucl", "c_exchange"],
            SBML_DFS.SC_SOURCE: [blank_source],
        },
        index=["sc_A_mito", "sc_A_nucl", "sc_B_exchange"],
    ).rename_axis(SBML_DFS.SC_ID)

    # Minimal reactions table
    reactions = pd.DataFrame(
        {
            SBML_DFS.R_NAME: [
                "A [mito] -> A [mito]",
                "A [nucl] -> A [nucl]",
                "B [exchange] -> B [exchange]",
            ],
            SBML_DFS.R_IDENTIFIERS: [blank_id],
            SBML_DFS.R_SOURCE: [blank_source],
            SBML_DFS.R_ISREVERSIBLE: [True],
        },
        index=["r_A_mito", "r_A_nucl", "r_B_self"],
    ).rename_axis(SBML_DFS.R_ID)

    # Minimal reaction_species table
    reaction_species = pd.DataFrame(
        {
            SBML_DFS.R_ID: [
                "r_A_mito",
                "r_A_mito",
                "r_A_nucl",
                "r_A_nucl",
                "r_B_self",
                "r_B_self",
            ],
            SBML_DFS.SC_ID: [
                "sc_A_mito",
                "sc_A_mito",
                "sc_A_nucl",
                "sc_A_nucl",
                "sc_B_exchange",
                "sc_B_exchange",
            ],
            SBML_DFS.STOICHIOMETRY: [-1, 1, -1, 1, 0, 0],
            SBML_DFS.SBO_TERM: [
                MINI_SBO_FROM_NAME["reactant"],
                MINI_SBO_FROM_NAME["product"],
                MINI_SBO_FROM_NAME["reactant"],
                MINI_SBO_FROM_NAME["product"],
                MINI_SBO_FROM_NAME["modifier"],
                MINI_SBO_FROM_NAME["modified"],
            ],
        },
        index=[
            "rsc_A_mito_sub",
            "rsc_A_mito_prod",
            "rsc_A_nucl_sub",
            "rsc_A_nucl_prod",
            "rsc_B_self_modr",
            "rsc_B_self_modd",
        ],
    ).rename_axis(SBML_DFS.RSC_ID)

    # Assemble the dict
    sbml_dict = {
        SBML_DFS.COMPARTMENTS: compartments,
        SBML_DFS.SPECIES: species,
        SBML_DFS.COMPARTMENTALIZED_SPECIES: compartmentalized_species,
        SBML_DFS.REACTIONS: reactions,
        SBML_DFS.REACTION_SPECIES: reaction_species,
    }

    # Create the SBML_dfs object
    sbml_dfs = SBML_dfs(sbml_dict, model_source_stub)

    return sbml_dfs


def test_add_transportation_reactions(sbml_dfs_missing_transport_rxns):

    sbml_dfs_w_transport = gaps.update_sbml_df_with_exchange(
        np.array(["s_A"]),
        sbml_dfs_missing_transport_rxns,
        exchange_compartment=EXCHANGE_COMPARTMENT,
    )
    assert sbml_dfs_w_transport.reactions.shape[0] == 5, "Should add 2 reactions"
    assert sbml_dfs_w_transport.reactions[
        SBML_DFS.R_ISREVERSIBLE
    ].all(), "Should be reversible"


def test_identify_species_needing_transport_reactions(
    sbml_dfs, sbml_dfs_missing_transport_rxns
):
    result = gaps._identify_species_needing_transport_reactions(sbml_dfs)
    assert isinstance(result, np.ndarray)
    assert result.size == 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = gaps._identify_species_needing_transport_reactions(
            sbml_dfs_missing_transport_rxns
        )
    assert isinstance(result, np.ndarray)
    assert result.size == 1
    assert result[0] == "s_A"
