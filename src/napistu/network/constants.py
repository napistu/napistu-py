"""Module to contain all constants used for representing and working with networks"""

from __future__ import annotations

from types import SimpleNamespace


from napistu.constants import SBML_DFS
from napistu.constants import SBOTERM_NAMES

NAPISTU_GRAPH = SimpleNamespace(VERTICES="vertices", EDGES="edges", METADATA="metadata")

GRAPH_DIRECTEDNESS = SimpleNamespace(DIRECTED="directed", UNDIRECTED="undirected")

GRAPH_RELATIONSHIPS = SimpleNamespace(
    ANCESTORS="ancestors",
    CHILDREN="children",
    DESCENDANTS="descendants",
    FOCAL="focal",
    PARENTS="parents",
)

NAPISTU_GRAPH_VERTICES = SimpleNamespace(
    NAME="name",  # internal name
    NODE_NAME="node_name",  # human readable name
    NODE_TYPE="node_type",  # type of node (species or reaction)
)

NAPISTU_GRAPH_EDGES = SimpleNamespace(
    DIRECTED="directed",
    DIRECTION="direction",
    FROM="from",
    R_ID=SBML_DFS.R_ID,
    R_ISREVERSIBLE=SBML_DFS.R_ISREVERSIBLE,
    SBO_TERM=SBML_DFS.SBO_TERM,
    SBO_NAME="sbo_name",
    SC_DEGREE="sc_degree",
    SC_PARENTS="sc_parents",
    SC_CHILDREN="sc_children",
    SPECIES_TYPE="species_type",
    STOICHIOMETRY=SBML_DFS.STOICHIOMETRY,
    TO="to",
    UPSTREAM_WEIGHT="upstream_weight",
    WEIGHT="weight",
)

NAPISTU_GRAPH_REQUIRED_EDGE_VARS = {
    NAPISTU_GRAPH_EDGES.FROM,
    NAPISTU_GRAPH_EDGES.TO,
    NAPISTU_GRAPH_EDGES.DIRECTION,
}

NAPISTU_GRAPH_NODE_TYPES = SimpleNamespace(SPECIES="species", REACTION="reaction")

VALID_NAPISTU_GRAPH_NODE_TYPES = [
    NAPISTU_GRAPH_NODE_TYPES.REACTION,
    NAPISTU_GRAPH_NODE_TYPES.SPECIES,
]

# translating an SBML_dfs -> NapistuGraph

GRAPH_WIRING_APPROACHES = SimpleNamespace(
    BIPARTITE="bipartite", REGULATORY="regulatory", SURROGATE="surrogate"
)

VALID_GRAPH_WIRING_APPROACHES = list(GRAPH_WIRING_APPROACHES.__dict__.values())

GRAPH_WIRING_HIERARCHIES = {
    # three tiers with reactions in the middle
    # in a bipartite networks molecules are connected to reactions but not other molecules
    GRAPH_WIRING_APPROACHES.BIPARTITE: [
        [
            SBOTERM_NAMES.CATALYST,
            SBOTERM_NAMES.INHIBITOR,
            SBOTERM_NAMES.INTERACTOR,
            SBOTERM_NAMES.MODIFIER,
            SBOTERM_NAMES.REACTANT,
            SBOTERM_NAMES.STIMULATOR,
        ],
        [NAPISTU_GRAPH_NODE_TYPES.REACTION],
        [SBOTERM_NAMES.MODIFIED, SBOTERM_NAMES.PRODUCT],
    ],
    # the regulatory graph defines a hierarchy of upstream and downstream
    # entities in a reaction
    # modifier/stimulator/inhibitor -> catalyst -> reactant -> reaction -> product
    GRAPH_WIRING_APPROACHES.REGULATORY: [
        [SBOTERM_NAMES.INHIBITOR, SBOTERM_NAMES.MODIFIER, SBOTERM_NAMES.STIMULATOR],
        [SBOTERM_NAMES.CATALYST],
        [SBOTERM_NAMES.INTERACTOR, SBOTERM_NAMES.REACTANT],
        [NAPISTU_GRAPH_NODE_TYPES.REACTION],
        [SBOTERM_NAMES.MODIFIED, SBOTERM_NAMES.PRODUCT],
    ],
    # an alternative layout to regulatory where enyzmes are downstream of substrates.
    # this doesn't make much sense from a regulatory perspective because
    # enzymes modify substrates not the other way around. but, its what one might
    # expect if catalysts are a surrogate for reactions as is the case for metabolic
    # network layouts
    GRAPH_WIRING_APPROACHES.SURROGATE: [
        [SBOTERM_NAMES.INHIBITOR, SBOTERM_NAMES.MODIFIER, SBOTERM_NAMES.STIMULATOR],
        [SBOTERM_NAMES.INTERACTOR, SBOTERM_NAMES.REACTANT],
        [SBOTERM_NAMES.CATALYST],
        [NAPISTU_GRAPH_NODE_TYPES.REACTION],
        [SBOTERM_NAMES.MODIFIED, SBOTERM_NAMES.PRODUCT],
    ],
}

# when should reaction vertices be excluded from the graph?

DROP_REACTIONS_WHEN = SimpleNamespace(
    ALWAYS="always",
    # if there are 2 participants
    EDGELIST="edgelist",
    # if there are 2 participants which are both "interactor"
    SAME_TIER="same_tier",
)

VALID_DROP_REACTIONS_WHEN = list(DROP_REACTIONS_WHEN.__dict__.values())

# adding weights to NapistuGraph

NAPISTU_WEIGHTING_STRATEGIES = SimpleNamespace(
    CALIBRATED="calibrated", MIXED="mixed", TOPOLOGY="topology", UNWEIGHTED="unweighted"
)

VALID_WEIGHTING_STRATEGIES = [
    NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED,
    NAPISTU_WEIGHTING_STRATEGIES.TOPOLOGY,
    NAPISTU_WEIGHTING_STRATEGIES.MIXED,
    NAPISTU_WEIGHTING_STRATEGIES.CALIBRATED,
]

# edge reversal

NAPISTU_GRAPH_EDGE_DIRECTIONS = SimpleNamespace(
    FORWARD="forward", REVERSE="reverse", UNDIRECTED="undirected"
)

EDGE_REVERSAL_ATTRIBUTE_MAPPING = {
    NAPISTU_GRAPH_EDGES.FROM: NAPISTU_GRAPH_EDGES.TO,
    NAPISTU_GRAPH_EDGES.TO: NAPISTU_GRAPH_EDGES.FROM,
    NAPISTU_GRAPH_EDGES.SC_PARENTS: NAPISTU_GRAPH_EDGES.SC_CHILDREN,
    NAPISTU_GRAPH_EDGES.SC_CHILDREN: NAPISTU_GRAPH_EDGES.SC_PARENTS,
    NAPISTU_GRAPH_EDGES.WEIGHT: NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT,
    NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT: NAPISTU_GRAPH_EDGES.WEIGHT,
    # Note: stoichiometry requires special handling (* -1)
}

# Direction enum values
EDGE_DIRECTION_MAPPING = {
    NAPISTU_GRAPH_EDGE_DIRECTIONS.FORWARD: NAPISTU_GRAPH_EDGE_DIRECTIONS.REVERSE,
    NAPISTU_GRAPH_EDGE_DIRECTIONS.REVERSE: NAPISTU_GRAPH_EDGE_DIRECTIONS.FORWARD,
    NAPISTU_GRAPH_EDGE_DIRECTIONS.UNDIRECTED: NAPISTU_GRAPH_EDGE_DIRECTIONS.UNDIRECTED,  # unchanged
}

# Net edge direction
NET_POLARITY = SimpleNamespace(
    LINK_POLARITY="link_polarity",
    NET_POLARITY="net_polarity",
    ACTIVATION="activation",
    INHIBITION="inhibition",
    AMBIGUOUS="ambiguous",
    AMBIGUOUS_ACTIVATION="ambiguous activation",
    AMBIGUOUS_INHIBITION="ambiguous inhibition",
)

VALID_LINK_POLARITIES = [
    NET_POLARITY.ACTIVATION,
    NET_POLARITY.INHIBITION,
    NET_POLARITY.AMBIGUOUS,
]

VALID_NET_POLARITIES = [
    NET_POLARITY.ACTIVATION,
    NET_POLARITY.INHIBITION,
    NET_POLARITY.AMBIGUOUS,
    NET_POLARITY.AMBIGUOUS_ACTIVATION,
    NET_POLARITY.AMBIGUOUS_INHIBITION,
]

NEIGHBORHOOD_NETWORK_TYPES = SimpleNamespace(
    DOWNSTREAM="downstream", HOURGLASS="hourglass", UPSTREAM="upstream"
)

VALID_NEIGHBORHOOD_NETWORK_TYPES = [
    NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
    NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM,
]

# weighting networks and transforming attributes

WEIGHTING_SPEC = SimpleNamespace(
    TABLE="table",
    VARIABLE="variable",
    TRANSFORMATION="trans",
)

DEFAULT_WT_TRANS = "identity"

DEFINED_WEIGHT_TRANSFORMATION = {
    DEFAULT_WT_TRANS: "_wt_transformation_identity",
    "string": "_wt_transformation_string",
    "string_inv": "_wt_transformation_string_inv",
}

SCORE_CALIBRATION_POINTS_DICT = {
    "weights": {"strong": 3, "good": 7, "okay": 20, "weak": 40},
    "string_wt": {"strong": 950, "good": 400, "okay": 230, "weak": 150},
}

SOURCE_VARS_DICT = {"string_wt": 10}

# network propagation
NET_PROPAGATION_DEFS = SimpleNamespace(PERSONALIZED_PAGERANK="personalized_pagerank")

# null distributions
NULL_STRATEGIES = SimpleNamespace(
    UNIFORM="uniform",
    PARAMETRIC="parametric",
    VERTEX_PERMUTATION="vertex_permutation",
    EDGE_PERMUTATION="edge_permutation",
)

VALID_NULL_STRATEGIES = NULL_STRATEGIES.__dict__.values()

PARAMETRIC_NULL_DEFAULT_DISTRIBUTION = "norm"

# masks

MASK_KEYWORDS = SimpleNamespace(
    ATTR="attr",
)

NEIGHBORHOOD_DICT_KEYS = SimpleNamespace(
    GRAPH="graph",
    VERTICES="vertices",
    EDGES="edges",
    REACTION_SOURCES="reaction_sources",
    NEIGHBORHOOD_PATH_ENTITIES="neighborhood_path_entities",
)

DISTANCES = SimpleNamespace(
    # core attributes of precomputed distances
    SC_ID_ORIGIN="sc_id_origin",
    SC_ID_DEST="sc_id_dest",
    PATH_LENGTH="path_length",
    PATH_UPSTREAM_WEIGHT="path_upstream_weight",
    PATH_WEIGHT="path_weight",
    # other attributes associated with paths/distances
    FINAL_FROM="final_from",
    FINAL_TO="final_to",
)
