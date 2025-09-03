# GCS constants
from __future__ import annotations

from types import SimpleNamespace

from napistu.constants import NAPISTU_STANDARD_OUTPUTS

GCS_ASSETS_NAMES = SimpleNamespace(
    TEST_PATHWAY="test_pathway",
    HUMAN_CONSENSUS="human_consensus",
    HUMAN_CONSENSUS_W_DISTANCES="human_consensus_w_distances",
    REACTOME_MEMBERS="reactome_members",
    REACTOME_XREFS="reactome_xrefs",
)

GCS_SUBASSET_NAMES = SimpleNamespace(
    SBML_DFS="sbml_dfs",
    NAPISTU_GRAPH="napistu_graph",
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
    FILE="file",
    SUBASSETS="subassets",
    PUBLIC_URL="public_url",
    VERSIONS="versions",
)

GCS_ASSETS = SimpleNamespace(
    PROJECT="calico-public-data",
    BUCKET="calico-cpr-public",
    ASSETS={
        GCS_ASSETS_NAMES.TEST_PATHWAY: {
            GCS_ASSETS_DEFS.FILE: "test_pathway.tar.gz",
            GCS_ASSETS_DEFS.SUBASSETS: {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
                GCS_SUBASSET_NAMES.PRECOMPUTED_DISTANCES: GCS_FILETYPES.PRECOMPUTED_DISTANCES,
                GCS_SUBASSET_NAMES.REACTIONS_SOURCE_TOTAL_COUNTS: GCS_FILETYPES.REACTIONS_SOURCE_TOTAL_COUNTS,
            },
            GCS_ASSETS_DEFS.PUBLIC_URL: "https://storage.googleapis.com/shackett-napistu-public/test_pathway.tar.gz",
            GCS_ASSETS_DEFS.VERSIONS: {
                "20250901": "https://storage.googleapis.com/shackett-napistu-public/test_pathway_20250901.tar.gz"
            },
        },
        GCS_ASSETS_NAMES.HUMAN_CONSENSUS: {
            GCS_ASSETS_DEFS.FILE: "human_consensus.tar.gz",
            GCS_ASSETS_DEFS.SUBASSETS: {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
                GCS_SUBASSET_NAMES.REACTIONS_SOURCE_TOTAL_COUNTS: GCS_FILETYPES.REACTIONS_SOURCE_TOTAL_COUNTS,
            },
            GCS_ASSETS_DEFS.PUBLIC_URL: "https://storage.googleapis.com/shackett-napistu-public/human_consensus.tar.gz",
            GCS_ASSETS_DEFS.VERSIONS: {
                "20250901": "https://storage.googleapis.com/shackett-napistu-public/human_consensus_20250901.tar.gz"
            },
        },
        GCS_ASSETS_NAMES.HUMAN_CONSENSUS_W_DISTANCES: {
            GCS_ASSETS_DEFS.FILE: "human_consensus_w_distances.tar.gz",
            GCS_ASSETS_DEFS.SUBASSETS: {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
                GCS_SUBASSET_NAMES.PRECOMPUTED_DISTANCES: GCS_FILETYPES.PRECOMPUTED_DISTANCES,
                GCS_SUBASSET_NAMES.REACTIONS_SOURCE_TOTAL_COUNTS: GCS_FILETYPES.REACTIONS_SOURCE_TOTAL_COUNTS,
            },
            GCS_ASSETS_DEFS.PUBLIC_URL: "https://storage.googleapis.com/shackett-napistu-public/human_consensus_w_distances.tar.gz",
            GCS_ASSETS_DEFS.VERSIONS: {
                "20250901": "https://storage.googleapis.com/shackett-napistu-public/human_consensus_w_distances_20250901.tar.gz"
            },
        },
        GCS_ASSETS_NAMES.REACTOME_MEMBERS: {
            GCS_ASSETS_DEFS.FILE: "external_pathways/external_pathways_reactome_neo4j_members.csv",
            GCS_ASSETS_DEFS.SUBASSETS: None,
            GCS_ASSETS_DEFS.PUBLIC_URL: "https://storage.googleapis.com/calico-cpr-public/external_pathways/external_pathways_reactome_neo4j_members.csv",
            GCS_ASSETS_DEFS.VERSIONS: None,
        },
        GCS_ASSETS_NAMES.REACTOME_XREFS: {
            GCS_ASSETS_DEFS.FILE: "external_pathways/external_pathways_reactome_neo4j_crossref.csv",
            GCS_ASSETS_DEFS.SUBASSETS: None,
            GCS_ASSETS_DEFS.PUBLIC_URL: "https://storage.googleapis.com/calico-cpr-public/external_pathways/external_pathways_reactome_neo4j_crossref.csv",
            GCS_ASSETS_DEFS.VERSIONS: None,
        },
    },
)

INIT_DATA_DIR_MSG = "The `data_dir` {data_dir} does not exist."
