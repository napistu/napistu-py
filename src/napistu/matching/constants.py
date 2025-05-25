from types import SimpleNamespace

FEATURE_ID_VAR_DEFAULT = "feature_id"

RESOLVE_MATCHES_AGGREGATORS = SimpleNamespace(
    WEIGHTED_MEAN = "weighted_mean",
    MEAN = "mean",
    FIRST = "first",
    MAX = "max"
)

RESOLVE_MATCHES_TMP_WEIGHT_COL = "__tmp_weight_for_aggregation__"