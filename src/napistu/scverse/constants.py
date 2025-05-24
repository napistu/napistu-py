from types import SimpleNamespace

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

ADATA_DICTLIKE_ATTRS = [ADATA.LAYERS, ADATA.OBSM, ADATA.OBSP, ADATA.VARM, ADATA.VARP]
ADATA_IDENTITY_ATTRS = [ADATA.OBS, ADATA.VAR, ADATA.X]