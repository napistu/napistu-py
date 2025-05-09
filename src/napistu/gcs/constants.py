# GCS constants
from __future__ import annotations

from types import SimpleNamespace

GCS_SUBASSET_NAMES = SimpleNamespace(
    SBML_DFS="sbml_dfs",
    IDENTIFIERS="identifiers",
    REGULATORY_GRAPH="regulatory_graph",
    REGULATORY_DISTANCES="regulatory_distances",
)


GCS_FILETYPES = SimpleNamespace(
    SBML_DFS="sbml_dfs.pkl",
    IDENTIFIERS="identifiers.tsv",
    REGULATORY_GRAPH="regulatory_graph.pkl",
    REGULATORY_DISTANCES="regulatory_distances.json",
)


GCS_ASSETS = SimpleNamespace(
    PROJECT="calico-public-data",
    BUCKET="calico-cpr-public",
    ASSETS={
        "test_pathway": {
            "file": "test_pathway.tar.gz",
            "subassets": {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.IDENTIFIERS: GCS_FILETYPES.IDENTIFIERS,
                GCS_SUBASSET_NAMES.REGULATORY_GRAPH: GCS_FILETYPES.REGULATORY_GRAPH,
                GCS_SUBASSET_NAMES.REGULATORY_DISTANCES: GCS_FILETYPES.REGULATORY_DISTANCES,
            },
            "public_url": "https://storage.googleapis.com/shackett-napistu-public/test_pathway.tar.gz",
        },
        "human_consensus": {
            "file": "human_consensus.tar.gz",
            "subassets": {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.IDENTIFIERS: GCS_FILETYPES.IDENTIFIERS,
                GCS_SUBASSET_NAMES.REGULATORY_GRAPH: GCS_FILETYPES.REGULATORY_GRAPH,
            },
            "public_url": "https://storage.googleapis.com/shackett-napistu-public/human_consensus.tar.gz",
        },
        "human_consensus_w_distances": {
            "file": "human_consensus_w_distances.tar.gz",
            "subassets": {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.IDENTIFIERS: GCS_FILETYPES.IDENTIFIERS,
                GCS_SUBASSET_NAMES.REGULATORY_GRAPH: GCS_FILETYPES.REGULATORY_GRAPH,
                GCS_SUBASSET_NAMES.REGULATORY_DISTANCES: GCS_FILETYPES.REGULATORY_DISTANCES,
            },
            "public_url": "https://storage.googleapis.com/calico-cpr-public/human_consensus_w_distances.tar.gz",
        },
        "reactome_members": {
            "file": "external_pathways/external_pathways_reactome_neo4j_members.csv",
            "subassets": None,
            "public_url": "https://storage.googleapis.com/calico-cpr-public/external_pathways/external_pathways_reactome_neo4j_members.csv",
        },
        "reactome_xrefs": {
            "file": "external_pathways/external_pathways_reactome_neo4j_crossref.csv",
            "subassets": None,
            "public_url": "https://storage.googleapis.com/calico-cpr-public/external_pathways/external_pathways_reactome_neo4j_crossref.csv",
        },
    },
)


INIT_DATA_DIR_MSG = "The `data_dir` {data_dir} does not exist."
