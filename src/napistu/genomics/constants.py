"""Constants for the genomics module."""

from types import SimpleNamespace

# scverse constants

ADATA = SimpleNamespace(
    LAYERS="layers",
    OBS="obs",
    OBSM="obsm",
    OBSP="obsp",
    VAR="var",
    VARM="varm",
    VARP="varp",
    X="X",
)

ADATA_DICTLIKE_ATTRS = {ADATA.LAYERS, ADATA.OBSM, ADATA.OBSP, ADATA.VARM, ADATA.VARP}
ADATA_IDENTITY_ATTRS = {ADATA.OBS, ADATA.VAR, ADATA.X}
ADATA_FEATURELEVEL_ATTRS = {ADATA.LAYERS, ADATA.VAR, ADATA.VARM, ADATA.X}
ADATA_ARRAY_ATTRS = {
    ADATA.LAYERS,
    ADATA.OBSM,
    ADATA.OBSP,
    ADATA.VARM,
    ADATA.VARP,
    ADATA.X,
}

SCVERSE_DEFS = SimpleNamespace(ADATA="adata", MDATA="mdata")

VALID_MUDATA_LEVELS = {SCVERSE_DEFS.ADATA, SCVERSE_DEFS.MDATA}

# gsea constants

GENESET_COLLECTION_DEFS = SimpleNamespace(
    GENESET="geneset",  # category
    GENESET_NAME="gene_set_name",  # category name
    IDENTIFIER="identifier",  # identifier (e.g., Entrez ID) for a gene in a gene set
    ONTOLOGY_NAME="ontology_name",  # ontology name
    DEEP_NAME="deep_name",  # the name of a gene set within an ontology
    SHALLOW_NAME="shallow_name",  # the name of a gene set within a collection of ontologies
)

GMTS_CONFIG_FIELDS = SimpleNamespace(
    ENGINE="engine",
    CATEGORIES="categories",
    DBVER="dbver",
)

# predefined gene sets

GENESET_SOURCES = SimpleNamespace(
    MSIGDB="msigdb",
)

GENESET_COLLECTIONS = SimpleNamespace(
    H_ALL="h.all",
    C2_CP_KEGG_LEGACY="c2.cp.kegg_legacy",
    C5_GO_BP="c5.go.bp",
    C2_CP_WIKIPATHWAYS="c2.cp.wikipathways",
)

GENESET_SOURCE_VERSIONS = SimpleNamespace(
    HS_2023_2="2023.2.Hs",
)

GENESET_DEFAULT_CONFIG_NAMES = SimpleNamespace(
    BP_KEGG_HALLMARKS="bp_kegg_hallmarks",
    HALLMARKS="hallmarks",
    WIKIPATHWAYS="wikipathways",
)

VALID_GENESET_DEFAULT_CONFIG_NAMES = list(
    GENESET_DEFAULT_CONFIG_NAMES.__dict__.values()
)
