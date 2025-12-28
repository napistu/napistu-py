# GCS constants
from __future__ import annotations

from types import SimpleNamespace

from napistu.constants import (
    NAPISTU_DEFS,
    NAPISTU_STANDARD_OUTPUTS,
)

GCS_ASSETS_NAMES = SimpleNamespace(
    TEST_PATHWAY="test_pathway",
    HUMAN_CONSENSUS="human_consensus",
    HUMAN_CONSENSUS_W_DISTANCES="human_consensus_w_distances",
    HUMAN_CONSENSUS_NO_RXNS="human_consensus_no_rxns",
    REACTOME_MEMBERS="reactome_members",
    REACTOME_XREFS="reactome_xrefs",
)

GCS_SUBASSET_NAMES = SimpleNamespace(
    SBML_DFS=NAPISTU_DEFS.SBML_DFS,
    NAPISTU_GRAPH=NAPISTU_DEFS.NAPISTU_GRAPH,
    SPECIES_IDENTIFIERS="species_identifiers",
    REACTIONS_SOURCE_TOTAL_COUNTS="reactions_source_total_counts",
    PRECOMPUTED_DISTANCES="precomputed_distances",
)

GCS_FILETYPES = SimpleNamespace(
    SBML_DFS="sbml_dfs.pkl",
    NAPISTU_GRAPH="napistu_graph.pkl",
    SPECIES_IDENTIFIERS=NAPISTU_STANDARD_OUTPUTS.SPECIES_IDENTIFIERS,
    REACTIONS_SOURCE_TOTAL_COUNTS=NAPISTU_STANDARD_OUTPUTS.REACTIONS_SOURCE_TOTAL_COUNTS,
    PRECOMPUTED_DISTANCES="precomputed_distances.parquet",
)

GCS_ASSETS_DEFS = SimpleNamespace(
    PROJECT="project",
    BUCKET="bucket",
    ASSETS="assets",
)

GCS_ASSET_DEFS = SimpleNamespace(
    FILE="file",
    SUBASSETS="subassets",
    PUBLIC_URL="public_url",
    VERSIONS="versions",
)

GCS_ASSETS = SimpleNamespace(
    PROJECT="shackett",
    BUCKET="shackett-napistu-public",
    ASSETS={
        GCS_ASSETS_NAMES.TEST_PATHWAY: {
            GCS_ASSET_DEFS.FILE: f"{GCS_ASSETS_NAMES.TEST_PATHWAY}.tar.gz",
            GCS_ASSET_DEFS.SUBASSETS: {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
                GCS_SUBASSET_NAMES.PRECOMPUTED_DISTANCES: GCS_FILETYPES.PRECOMPUTED_DISTANCES,
                GCS_SUBASSET_NAMES.REACTIONS_SOURCE_TOTAL_COUNTS: GCS_FILETYPES.REACTIONS_SOURCE_TOTAL_COUNTS,
            },
            GCS_ASSET_DEFS.PUBLIC_URL: f"https://storage.googleapis.com/shackett-napistu-public/{GCS_ASSETS_NAMES.TEST_PATHWAY}.tar.gz",
            GCS_ASSET_DEFS.VERSIONS: {
                "20250901": f"https://storage.googleapis.com/shackett-napistu-public/{GCS_ASSETS_NAMES.TEST_PATHWAY}_20250901.tar.gz"
            },
        },
        GCS_ASSETS_NAMES.HUMAN_CONSENSUS: {
            GCS_ASSET_DEFS.FILE: f"{GCS_ASSETS_NAMES.HUMAN_CONSENSUS}.tar.gz",
            GCS_ASSET_DEFS.SUBASSETS: {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
                GCS_SUBASSET_NAMES.REACTIONS_SOURCE_TOTAL_COUNTS: GCS_FILETYPES.REACTIONS_SOURCE_TOTAL_COUNTS,
            },
            GCS_ASSET_DEFS.PUBLIC_URL: f"https://storage.googleapis.com/shackett-napistu-public/{GCS_ASSETS_NAMES.HUMAN_CONSENSUS}.tar.gz",
            GCS_ASSET_DEFS.VERSIONS: {
                "20250901": f"https://storage.googleapis.com/shackett-napistu-public/{GCS_ASSETS_NAMES.HUMAN_CONSENSUS}_20250901.tar.gz",
                "20250923": f"https://storage.googleapis.com/shackett-napistu-public/{GCS_ASSETS_NAMES.HUMAN_CONSENSUS}_20250923.tar.gz",
            },
        },
        GCS_ASSETS_NAMES.HUMAN_CONSENSUS_W_DISTANCES: {
            GCS_ASSET_DEFS.FILE: f"{GCS_ASSETS_NAMES.HUMAN_CONSENSUS_W_DISTANCES}.tar.gz",
            GCS_ASSET_DEFS.SUBASSETS: {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
                GCS_SUBASSET_NAMES.PRECOMPUTED_DISTANCES: GCS_FILETYPES.PRECOMPUTED_DISTANCES,
                GCS_SUBASSET_NAMES.REACTIONS_SOURCE_TOTAL_COUNTS: GCS_FILETYPES.REACTIONS_SOURCE_TOTAL_COUNTS,
            },
            GCS_ASSET_DEFS.PUBLIC_URL: f"https://storage.googleapis.com/shackett-napistu-public/{GCS_ASSETS_NAMES.HUMAN_CONSENSUS_W_DISTANCES}.tar.gz",
            GCS_ASSET_DEFS.VERSIONS: {
                "20250901": f"https://storage.googleapis.com/shackett-napistu-public/{GCS_ASSETS_NAMES.HUMAN_CONSENSUS_W_DISTANCES}_20250901.tar.gz"
            },
        },
        GCS_ASSETS_NAMES.HUMAN_CONSENSUS_NO_RXNS: {
            GCS_ASSET_DEFS.FILE: f"{GCS_ASSETS_NAMES.HUMAN_CONSENSUS_NO_RXNS}.tar.gz",
            GCS_ASSET_DEFS.SUBASSETS: {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
            },
            GCS_ASSET_DEFS.PUBLIC_URL: f"https://storage.googleapis.com/shackett-napistu-public/{GCS_ASSETS_NAMES.HUMAN_CONSENSUS_NO_RXNS}.tar.gz",
            GCS_ASSET_DEFS.VERSIONS: {
                "20251218": f"https://storage.googleapis.com/shackett-napistu-public/{GCS_ASSETS_NAMES.HUMAN_CONSENSUS_NO_RXNS}_20251218.tar.gz"
            },
        },
        GCS_ASSETS_NAMES.REACTOME_MEMBERS: {
            GCS_ASSET_DEFS.FILE: "external_pathways/external_pathways_reactome_neo4j_members.csv",
            GCS_ASSET_DEFS.SUBASSETS: None,
            GCS_ASSET_DEFS.PUBLIC_URL: "https://storage.googleapis.com/calico-cpr-public/external_pathways/external_pathways_reactome_neo4j_members.csv",
            GCS_ASSET_DEFS.VERSIONS: None,
        },
        GCS_ASSETS_NAMES.REACTOME_XREFS: {
            GCS_ASSET_DEFS.FILE: "external_pathways/external_pathways_reactome_neo4j_crossref.csv",
            GCS_ASSET_DEFS.SUBASSETS: None,
            GCS_ASSET_DEFS.PUBLIC_URL: "https://storage.googleapis.com/calico-cpr-public/external_pathways/external_pathways_reactome_neo4j_crossref.csv",
            GCS_ASSET_DEFS.VERSIONS: None,
        },
    },
)

INIT_DATA_DIR_MSG = "The `data_dir` {data_dir} does not exist."
