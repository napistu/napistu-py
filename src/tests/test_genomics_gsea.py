"""Tests for gene set enrichment analysis (GSEA) functionality."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from napistu.constants import BQB, IDENTIFIERS, ONTOLOGIES, SBML_DFS
from napistu.genomics.gsea import (
    GenesetCollection,
    GmtsConfig,
    _calculate_geneset_edge_counts,
    _filter_genesets_to_universe,
)
from napistu.identifiers import Identifiers, construct_cspecies_identifiers
from napistu.ingestion.constants import COMPARTMENTS
from napistu.network.constants import IGRAPH_DEFS
from napistu.source import Source
from napistu.utils.optional import import_gseapy

try:
    gp = import_gseapy()
except ImportError:
    pytest.skip("gseapy is not available", allow_module_level=True)


@pytest.fixture
def sbml_dfs_w_entrez_ids(model_source_stub):
    """Create a minimal sbml_dfs with Entrez IDs extracted from a gene set collection."""
    from napistu.sbml_dfs_core import SBML_dfs

    # Create a gene set collection and extract some Entrez IDs
    collection = GenesetCollection(organismal_species="Homo sapiens")
    config = GmtsConfig(
        engine=gp.msigdb.Msigdb,
        categories=["h.all"],
        dbver="2023.2.Hs",
    )
    collection.add_gmts(gmts_config=config)

    # Extract a sample of Entrez IDs from the collection
    # Get the first gene set and take first 5-10 Entrez IDs
    if len(collection.gmt) == 0:
        raise ValueError("No gene sets found in collection")

    first_geneset = next(iter(collection.gmt.keys()))
    entrez_ids = collection.gmt[first_geneset][:10]  # Take first 10 Entrez IDs

    # Create species with Entrez IDs
    species_list = []
    for i, entrez_id in enumerate(entrez_ids):
        s_id = f"S{i+1:05d}"
        species_identifiers = Identifiers(
            [
                {
                    IDENTIFIERS.ONTOLOGY: ONTOLOGIES.NCBI_ENTREZ_GENE,
                    IDENTIFIERS.IDENTIFIER: entrez_id,
                    IDENTIFIERS.BQB: BQB.IS,
                }
            ]
        )
        species_list.append(
            {
                SBML_DFS.S_NAME: f"Gene_{entrez_id}",
                SBML_DFS.S_IDENTIFIERS: species_identifiers,
                SBML_DFS.S_SOURCE: Source.empty(),
            }
        )

    species_df = pd.DataFrame(species_list)
    species_df.index = [f"S{i+1:05d}" for i in range(len(entrez_ids))]
    species_df.index.name = SBML_DFS.S_ID

    # Create compartments
    compartments_df = pd.DataFrame(
        {
            SBML_DFS.C_NAME: [COMPARTMENTS.CYTOPLASM],
            SBML_DFS.C_IDENTIFIERS: [Identifiers([])],
            SBML_DFS.C_SOURCE: [Source.empty()],
        }
    )
    compartments_df.index = ["C00001"]
    compartments_df.index.name = SBML_DFS.C_ID

    # Create compartmentalized species
    compartmentalized_species_list = []
    for i, s_id in enumerate(species_df.index):
        compartmentalized_species_list.append(
            {
                SBML_DFS.SC_NAME: f"{species_df.loc[s_id, SBML_DFS.S_NAME]} [cytoplasm]",
                SBML_DFS.S_ID: s_id,
                SBML_DFS.C_ID: "C00001",
                SBML_DFS.SC_SOURCE: Source.empty(),
            }
        )

    compartmentalized_species_df = pd.DataFrame(compartmentalized_species_list)
    compartmentalized_species_df.index = [
        f"SC{i+1:05d}" for i in range(len(entrez_ids))
    ]
    compartmentalized_species_df.index.name = SBML_DFS.SC_ID

    # Create minimal reactions and reaction_species
    reactions_df = pd.DataFrame(
        {
            SBML_DFS.R_NAME: ["test_reaction"],
            SBML_DFS.R_IDENTIFIERS: [Identifiers([])],
            SBML_DFS.R_SOURCE: [Source.empty()],
            SBML_DFS.R_ISREVERSIBLE: [False],
        }
    )
    reactions_df.index = ["R00001"]
    reactions_df.index.name = SBML_DFS.R_ID

    reaction_species_df = pd.DataFrame(
        {
            SBML_DFS.R_ID: ["R00001"],
            SBML_DFS.SC_ID: ["SC00001"],
            SBML_DFS.STOICHIOMETRY: [1.0],
            SBML_DFS.SBO_TERM: ["SBO:0000011"],
        }
    )
    reaction_species_df.index = ["RSC00001"]
    reaction_species_df.index.name = SBML_DFS.RSC_ID

    sbml_dict = {
        SBML_DFS.COMPARTMENTS: compartments_df,
        SBML_DFS.SPECIES: species_df,
        SBML_DFS.COMPARTMENTALIZED_SPECIES: compartmentalized_species_df,
        SBML_DFS.REACTIONS: reactions_df,
        SBML_DFS.REACTION_SPECIES: reaction_species_df,
    }

    return SBML_dfs(sbml_dict, model_source_stub, validate=False)


def test_add_gmts_h_all():
    """Test adding the 'h.all' gene set to a GenesetCollection."""
    collection = GenesetCollection(organismal_species="Homo sapiens")

    config = GmtsConfig(
        engine=gp.msigdb.Msigdb,
        categories=["h.all"],
        dbver="2023.2.Hs",
    )

    collection.add_gmts(gmts_config=config)

    assert "h.all" in collection.gmts
    assert isinstance(collection.gmts["h.all"], dict)
    assert len(collection.gmts["h.all"]) > 0


def test_get_gmt_w_napistu_ids_species_and_cspecies(sbml_dfs_w_entrez_ids):
    """Test get_gmt_w_napistu_ids with both species_identifiers and cspecies_identifiers."""
    # Create a gene set collection
    collection = GenesetCollection(organismal_species="Homo sapiens")

    config = GmtsConfig(
        engine=gp.msigdb.Msigdb,
        categories=["h.all"],
        dbver="2023.2.Hs",
    )
    collection.add_gmts(gmts_config=config)

    # Get species identifiers from sbml_dfs_w_entrez_ids
    species_identifiers = sbml_dfs_w_entrez_ids.get_characteristic_species_ids(
        dogmatic=True
    )

    # Test with species_identifiers (s_id)
    result_s_id = collection.get_gmt_w_napistu_ids(
        species_identifiers=species_identifiers,
        id_type=SBML_DFS.S_ID,
    )

    # Verify result structure
    assert isinstance(result_s_id, dict)
    assert len(result_s_id) > 0
    # Check that values are lists of s_ids
    for geneset, s_ids in result_s_id.items():
        assert isinstance(s_ids, list)
        assert len(s_ids) > 0
        # Verify all s_ids are strings
        assert all(isinstance(s_id, str) for s_id in s_ids)

    # Create cspecies_identifiers using construct_cspecies_identifiers
    cspecies_identifiers = construct_cspecies_identifiers(
        species_identifiers=species_identifiers,
        cspecies_references=sbml_dfs_w_entrez_ids,
    )

    # Test with cspecies_identifiers (sc_id)
    result_sc_id = collection.get_gmt_w_napistu_ids(
        species_identifiers=cspecies_identifiers,
        id_type=SBML_DFS.SC_ID,
    )

    # Verify result structure
    assert isinstance(result_sc_id, dict)
    assert len(result_sc_id) > 0
    # Check that values are lists of sc_ids
    for geneset, sc_ids in result_sc_id.items():
        assert isinstance(sc_ids, list)
        assert len(sc_ids) > 0
        # Verify all sc_ids are strings
        assert all(isinstance(sc_id, str) for sc_id in sc_ids)

    # Verify that both results have the same gene sets
    assert set(result_s_id.keys()) == set(result_sc_id.keys())

    # Verify that sc_id results have at least as many IDs per gene set as s_id
    # (since one s_id can map to multiple sc_ids)
    for geneset in result_s_id.keys():
        assert len(result_sc_id[geneset]) >= len(result_s_id[geneset]), (
            f"Gene set {geneset} should have at least as many sc_ids as s_ids, "
            f"since one s_id can map to multiple sc_ids"
        )


def test_prerank_with_napistu_genesets(sbml_dfs_w_entrez_ids):
    """Test that napistu-centric geneset formats can be used with gseapy.prerank."""
    # Create a gene set collection
    collection = GenesetCollection(organismal_species="Homo sapiens")

    config = GmtsConfig(
        engine=gp.msigdb.Msigdb,
        categories=["h.all"],
        dbver="2023.2.Hs",
    )
    collection.add_gmts(gmts_config=config)

    # Get species identifiers from sbml_dfs_w_entrez_ids
    species_identifiers = sbml_dfs_w_entrez_ids.get_characteristic_species_ids(
        dogmatic=True
    )

    # Get napistu geneset with s_id
    napistu_geneset = collection.get_gmt_w_napistu_ids(
        species_identifiers=species_identifiers,
        id_type=SBML_DFS.S_ID,
    )

    # Create a value series indexed by s_id (similar to sandbox example)
    example_data = sbml_dfs_w_entrez_ids.species.copy()
    # Add a column with random normal values
    example_data["values"] = np.random.randn(example_data.shape[0])
    # Select the series of values indexed by s_id
    value_series = example_data["values"]

    # Verify that prerank can be called with the napistu geneset format
    # Convert Series to DataFrame format that gseapy expects (gene_name, prerank columns)
    # This avoids the gseapy bug with single-element Series
    value_df = pd.DataFrame(
        {"gene_name": value_series.index, "prerank": value_series.values}
    )

    gsea_results = gp.prerank(rnk=value_df, gene_sets=napistu_geneset, min_size=1)

    # Verify that results were returned
    assert gsea_results is not None
    # gseapy.prerank returns a results object with a .res2d attribute containing the results DataFrame
    assert hasattr(gsea_results, "res2d")
    assert isinstance(gsea_results.res2d, pd.DataFrame)
    assert len(gsea_results.res2d) > 0


def test_filter_genesets_to_universe(napistu_graph):
    names = napistu_graph.vs[IGRAPH_DEFS.NAME][:3]
    collection = SimpleNamespace(
        gmt={"ok": names + ["NOT_IN_UNIVERSE"], "bad": ["ALSO_NOT_IN_UNIVERSE"]}
    )
    filtered, df = _filter_genesets_to_universe(
        napistu_graph, collection.gmt, min_set_size=1
    )
    assert set(filtered) == {"ok"} and set(filtered["ok"]) == set(names)
    assert set(df["geneset"]) == {"ok"} and set(df["vertex_name"]) == set(names)


def test_calculate_geneset_edge_counts(napistu_graph):
    e0 = napistu_graph.es[0]
    src = napistu_graph.vs[e0.source][IGRAPH_DEFS.NAME]
    tgt = napistu_graph.vs[e0.target][IGRAPH_DEFS.NAME]
    observed = pd.DataFrame({IGRAPH_DEFS.SOURCE: [src], IGRAPH_DEFS.TARGET: [tgt]})
    genesets = SimpleNamespace(gmt={"gs1": [src, tgt], "gs2": [tgt]}).gmt
    out = _calculate_geneset_edge_counts(
        observed, genesets, napistu_graph, min_set_size=1, directed=True
    )
    ab = out.query("source_geneset == 'gs1' and target_geneset == 'gs2'")[
        "observed_edges"
    ].iloc[0]
    ba = out.query("source_geneset == 'gs2' and target_geneset == 'gs1'")[
        "observed_edges"
    ].iloc[0]
    assert ab == 1 and ba == 0
