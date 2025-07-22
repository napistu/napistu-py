# GCS constants
from __future__ import annotations

from types import SimpleNamespace

from napistu.constants import NAPISTU_STANDARD_OUTPUTS

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


GCS_ASSETS = SimpleNamespace(
    PROJECT="calico-public-data",
    BUCKET="calico-cpr-public",
    ASSETS={
        "test_pathway": {
            "file": "test_pathway.tar.gz",
            "subassets": {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
                GCS_SUBASSET_NAMES.PRECOMPUTED_DISTANCES: GCS_FILETYPES.PRECOMPUTED_DISTANCES,
                GCS_SUBASSET_NAMES.REACTIONS_SOURCE_TOTAL_COUNTS: GCS_FILETYPES.REACTIONS_SOURCE_TOTAL_COUNTS,
            },
            "public_url": "https://storage.googleapis.com/shackett-napistu-public/test_pathway.tar.gz",
        },
        "human_consensus": {
            "file": "human_consensus.tar.gz",
            "subassets": {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
                GCS_SUBASSET_NAMES.REACTIONS_SOURCE_TOTAL_COUNTS: GCS_FILETYPES.REACTIONS_SOURCE_TOTAL_COUNTS,
            },
            "public_url": "https://storage.googleapis.com/shackett-napistu-public/human_consensus.tar.gz",
        },
        "human_consensus_w_distances": {
            "file": "human_consensus_w_distances.tar.gz",
            "subassets": {
                GCS_SUBASSET_NAMES.SBML_DFS: GCS_FILETYPES.SBML_DFS,
                GCS_SUBASSET_NAMES.NAPISTU_GRAPH: GCS_FILETYPES.NAPISTU_GRAPH,
                GCS_SUBASSET_NAMES.SPECIES_IDENTIFIERS: GCS_FILETYPES.SPECIES_IDENTIFIERS,
                GCS_SUBASSET_NAMES.PRECOMPUTED_DISTANCES: GCS_FILETYPES.PRECOMPUTED_DISTANCES,
                GCS_SUBASSET_NAMES.REACTIONS_SOURCE_TOTAL_COUNTS: GCS_FILETYPES.REACTIONS_SOURCE_TOTAL_COUNTS,
            },
            "public_url": "https://storage.googleapis.com/shackett-napistu-public/human_consensus_w_distances.tar.gz",
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
