from __future__ import annotations

import pandas as pd
import pytest
from napistu import sbml_dfs_core
from napistu.identifiers import Identifiers
from napistu.rpy2 import netcontextr
from napistu.source import Source


@pytest.fixture
def sbml_dfs_one_reaction():
    """An sbml_dfs with one reaction and one annotated reactant"""
    interaction_edgelist = pd.DataFrame(
        {
            "upstream_name": ["a"],
            "downstream_name": ["b"],
            "upstream_compartment": ["nucleoplasm"],
            "downstream_compartment": ["nucleoplasm"],
            "r_name": ["a b of a"],
            "sbo_term": ["SBO:0000010"],
            "r_Identifiers": Identifiers([]),
            "r_isreversible": False,
        }
    )
    species = pd.DataFrame(
        {
            "s_name": ["a", "b"],
            "s_Identifiers": [
                Identifiers([{"ontology": "ensembl_gene", "identifier": "test"}]),
                Identifiers([]),
            ],
        }
    )
    compartments = pd.DataFrame(
        {"c_name": ["nucleoplasm"], "c_Identifiers": Identifiers([])}
    )
    interaction_source = Source(init=True)
    sbml_dfs = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist, species, compartments, interaction_source
    )
    return sbml_dfs


@pytest.fixture
def sbml_dfs_one_reaction_duplicated_identifiers():
    """
    An sbml_dfs with one reactions and one annotated reactant
    that has two identifiers
    """
    interaction_edgelist = pd.DataFrame(
        {
            "upstream_name": ["a"],
            "downstream_name": ["b"],
            "upstream_compartment": ["nucleoplasm"],
            "downstream_compartment": ["nucleoplasm"],
            "r_name": ["a b of a"],
            "sbo_term": ["SBO:0000010"],
            "r_Identifiers": Identifiers([]),
            "r_isreversible": False,
        }
    )
    species = pd.DataFrame(
        {
            "s_name": ["a", "b"],
            "s_Identifiers": [
                Identifiers(
                    [
                        {"ontology": "ensembl_gene", "identifier": "test"},
                        {"ontology": "ensembl_gene", "identifier": "test2"},
                    ]
                ),
                Identifiers([]),
            ],
        }
    )
    compartments = pd.DataFrame(
        {"c_name": ["nucleoplasm"], "c_Identifiers": Identifiers([])}
    )
    interaction_source = Source(init=True)
    sbml_dfs = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist, species, compartments, interaction_source
    )
    return sbml_dfs


def test_get_reactions_one_reaction(sbml_dfs_one_reaction):
    reactions = netcontextr._get_reactions(sbml_dfs_one_reaction)
    assert not reactions[netcontextr.COL_GENE].isna().any()
    assert reactions.shape[0] == 1


def test_get_reactions_outcols(sbml_dfs_one_reaction):
    reactions = netcontextr._get_reactions(sbml_dfs_one_reaction)
    assert netcontextr.COL_GENE in reactions.columns
    assert netcontextr.COL_REACTION_ID in reactions.columns
    assert netcontextr.COL_ROLE in reactions.columns


def test_get_reactions_one_reaction_duplicated_ids(
    sbml_dfs_one_reaction_duplicated_identifiers,
):
    reactions = netcontextr._get_reactions(sbml_dfs_one_reaction_duplicated_identifiers)
    assert not reactions[netcontextr.COL_GENE].isna().any()
    assert reactions.shape[0] == 2
