from types import SimpleNamespace

CONTINGENCY_TABLE = SimpleNamespace(
    OBSERVED_MEMBERS="observed_members",  # in query (e.g., observed data) and reference set (e.g., a pathway)
    MISSING_MEMBERS="missing_members",  # in reference set but not query (e.g., in a pathway but not observed data)
    OBSERVED_NONMEMBERS="observed_nonmembers",  # in query but not reference set (e.g., observed data but not in a pathway)
    NONOBSERVED_NONMEMBERS="nonobserved_nonmembers",  # in neither query nor reference set (e.g., not observed data and not in a pathway)
    TOTAL_COUNTS="total_counts",  # total size of the reference sets
    N_TOTAL_ENTITIES="n_total_entities",  # the universe of possible entries in the query (e.g., the total number of reactions in the model)
)
