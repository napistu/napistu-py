"""Module to contain all constants used for representing and working with networks"""

from __future__ import annotations

from types import SimpleNamespace

from napistu.constants import SBML_DFS, SBML_DFS_METHOD_DEFS, SBOTERM_NAMES

IGRAPH_DEFS = SimpleNamespace(
    VERTICES="vertices",
    EDGES="edges",
    NAME="name",
    INDEX="index",
    SOURCE="source",
    TARGET="target",
)

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
    SPECIES_TYPE=SBML_DFS_METHOD_DEFS.SPECIES_TYPE,
)

CORE_NAPISTU_GRAPH_VERTICES_VARS = {
    NAPISTU_GRAPH_VERTICES.NAME,
    NAPISTU_GRAPH_VERTICES.NODE_NAME,
    NAPISTU_GRAPH_VERTICES.NODE_TYPE,
    # added during creation
    NAPISTU_GRAPH_VERTICES.SPECIES_TYPE,
}

NAPISTU_GRAPH_EDGES = SimpleNamespace(
    DIRECTED="directed",
    DIRECTION="direction",
    FROM="from",
    R_ID=SBML_DFS.R_ID,
    R_ISREVERSIBLE=SBML_DFS.R_ISREVERSIBLE,
    SBO_NAME="sbo_name",
    SBO_TERM_DOWNSTREAM="sbo_term_downstream",
    SBO_TERM_UPSTREAM="sbo_term_upstream",
    SC_DEGREE="sc_degree",
    SC_PARENTS="sc_parents",
    SC_CHILDREN="sc_children",
    SPECIES_TYPE="species_type",
    STOICHIOMETRY_DOWNSTREAM="stoichiometry_downstream",
    STOICHIOMETRY_UPSTREAM="stoichiometry_upstream",
    TO="to",
    WEIGHT_UPSTREAM="weight_upstream",
    WEIGHT="weight",
    SOURCE_WT="source_wt",
)

SINGULAR_GRAPH_ENTITIES = {
    NAPISTU_GRAPH.EDGES: "edge",
    NAPISTU_GRAPH.VERTICES: "vertex",
}

# added during graph wiring
NAPISTU_GRAPH_EDGES_FROM_WIRING_VARS = [
    NAPISTU_GRAPH_EDGES.FROM,
    NAPISTU_GRAPH_EDGES.TO,
    NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM,
    NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM,
    NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM,
    NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM,
    NAPISTU_GRAPH_EDGES.R_ID,
]

NAPISTU_GRAPH_EDGES_FROM_AUGMENTATION_VARS = {
    NAPISTU_GRAPH_EDGES.SPECIES_TYPE,
    NAPISTU_GRAPH_EDGES.R_ISREVERSIBLE,
}

OPTIONAL_NAPISTU_GRAPH_EDGES_FROM_ADD_DEGREE = {
    NAPISTU_GRAPH_EDGES.SC_DEGREE,
    NAPISTU_GRAPH_EDGES.SC_CHILDREN,
    NAPISTU_GRAPH_EDGES.SC_PARENTS,
}

NAPISTU_GRAPH_NODE_TYPES = SimpleNamespace(SPECIES="species", REACTION="reaction")

VALID_NAPISTU_GRAPH_NODE_TYPES = [
    NAPISTU_GRAPH_NODE_TYPES.REACTION,
    NAPISTU_GRAPH_NODE_TYPES.SPECIES,
]

NODE_TYPES_TO_ENTITY_TABLES = {
    NAPISTU_GRAPH_NODE_TYPES.REACTION: SBML_DFS.REACTIONS,
    NAPISTU_GRAPH_NODE_TYPES.SPECIES: SBML_DFS.COMPARTMENTALIZED_SPECIES,
}

NAPISTU_METADATA_KEYS = SimpleNamespace(
    REACTION_ATTRS="reaction_attrs",
    SPECIES_ATTRS="species_attrs",
    TRANSFORMATIONS_APPLIED="transformations_applied",
    RAW_ATTRIBUTES="raw_attributes",
    IS_REVERSED="is_reversed",
    WIRING_APPROACH="wiring_approach",
    WEIGHTING_STRATEGY="weighting_strategy",
    WEIGHT_BY="weight_by",
    CREATION_PARAMS="creation_params",
)

ENTITIES_TO_ATTRS = {
    SBML_DFS.REACTIONS: NAPISTU_METADATA_KEYS.REACTION_ATTRS,
    SBML_DFS.SPECIES: NAPISTU_METADATA_KEYS.SPECIES_ATTRS,
}

NAPISTU_GRAPH_EDGE_ENDPOINT_ATTRIBUTES = SimpleNamespace(
    ATTRIBUTE_NAME="attribute",
    ENDPOINT="endpoint",
)

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
    MIXED="mixed", TOPOLOGY="topology", UNWEIGHTED="unweighted"
)

VALID_WEIGHTING_STRATEGIES = NAPISTU_WEIGHTING_STRATEGIES.__dict__.values()

# adding extra attributes to NapistuGraphs

VERTEX_SBML_DFS_SUMMARIES = SimpleNamespace(
    SOURCES="sources",
    ONTOLOGIES="ontologies",
)

VALID_VERTEX_SBML_DFS_SUMMARIES = list(VERTEX_SBML_DFS_SUMMARIES.__dict__.values())

# edge reversal

NAPISTU_GRAPH_EDGE_DIRECTIONS = SimpleNamespace(
    FORWARD="forward", REVERSE="reverse", UNDIRECTED="undirected"
)

EDGE_REVERSAL_ATTRIBUTE_MAPPING = {
    NAPISTU_GRAPH_EDGES.FROM: NAPISTU_GRAPH_EDGES.TO,
    NAPISTU_GRAPH_EDGES.TO: NAPISTU_GRAPH_EDGES.FROM,
    NAPISTU_GRAPH_EDGES.SC_PARENTS: NAPISTU_GRAPH_EDGES.SC_CHILDREN,
    NAPISTU_GRAPH_EDGES.SC_CHILDREN: NAPISTU_GRAPH_EDGES.SC_PARENTS,
    NAPISTU_GRAPH_EDGES.WEIGHT: NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
    NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM: NAPISTU_GRAPH_EDGES.WEIGHT,
    NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM: NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM,
    NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM: NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM,
    NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM: NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM,
    NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM: NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM,
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
    BYSTANDER="bystander",
    AMBIGUOUS_ACTIVATION="ambiguous activation",
    AMBIGUOUS_INHIBITION="ambiguous inhibition",
)

VALID_LINK_POLARITIES = [
    NET_POLARITY.ACTIVATION,
    NET_POLARITY.INHIBITION,
    NET_POLARITY.AMBIGUOUS,
    NET_POLARITY.BYSTANDER,
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

ADDING_ENTITY_DATA_DEFS = SimpleNamespace(
    FRESH="fresh",
    EXTEND="extend",
)

VALID_ADDING_ENTITY_DATA_DEFS = list(ADDING_ENTITY_DATA_DEFS.__dict__.values())

WEIGHTING_SPEC = SimpleNamespace(
    TABLE="table",
    VARIABLE="variable",
    TRANSFORMATION="trans",
)

WEIGHT_TRANSFORMATIONS = SimpleNamespace(
    IDENTITY="identity",
    STRING="string",
    STRING_INV="string_inv",
)

DEFAULT_WT_TRANS = WEIGHT_TRANSFORMATIONS.IDENTITY

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

UNIVERSE_GATES = SimpleNamespace(
    AND="and",
    OR="or",
)

VALID_UNIVERSE_GATES = list(UNIVERSE_GATES.__dict__.values())

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
    PATH_WEIGHT_UPSTREAM="path_weight_upstream",
    PATH_WEIGHT="path_weight",
    # other attributes associated with paths/distances
    FINAL_FROM="final_from",
    FINAL_TO="final_to",
)
