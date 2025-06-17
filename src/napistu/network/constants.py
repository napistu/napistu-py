"""Module to contain all constants used for representing and working with networks"""

from __future__ import annotations

from types import SimpleNamespace

from napistu.constants import SBML_DFS
from napistu.constants import SBOTERM_NAMES

# Graph types
NAPISTU_GRAPH_TYPES = SimpleNamespace(
    BIPARTITE="bipartite", REGULATORY="regulatory", SURROGATE="surrogate"
)

VALID_NAPISTU_GRAPH_TYPES = [
    NAPISTU_GRAPH_TYPES.BIPARTITE,
    NAPISTU_GRAPH_TYPES.REGULATORY,
    NAPISTU_GRAPH_TYPES.SURROGATE,
]

NAPISTU_GRAPH_NODES = SimpleNamespace(NAME="name")

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
    UPSTREAM_WEIGHTS="upstream_weights",
    WEIGHTS="weights",
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
    NAPISTU_GRAPH_EDGES.WEIGHTS: NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHTS,
    NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHTS: NAPISTU_GRAPH_EDGES.WEIGHTS,
    # Note: stoichiometry requires special handling (* -1)
}

# Direction enum values
EDGE_DIRECTION_MAPPING = {
    NAPISTU_GRAPH_EDGE_DIRECTIONS.FORWARD: NAPISTU_GRAPH_EDGE_DIRECTIONS.REVERSE,
    NAPISTU_GRAPH_EDGE_DIRECTIONS.REVERSE: NAPISTU_GRAPH_EDGE_DIRECTIONS.FORWARD,
    NAPISTU_GRAPH_EDGE_DIRECTIONS.UNDIRECTED: NAPISTU_GRAPH_EDGE_DIRECTIONS.UNDIRECTED,  # unchanged
}

# the regulatory graph defines a hierarchy of upstream and downstream
# entities in a reaction
# modifier/stimulator/inhibitor -> catalyst -> reactant -> reaction -> product

REGULATORY_GRAPH_HIERARCHY = [
    [SBOTERM_NAMES.MODIFIER, SBOTERM_NAMES.STIMULATOR, SBOTERM_NAMES.INHIBITOR],
    [SBOTERM_NAMES.CATALYST],
    [SBOTERM_NAMES.REACTANT],
    [NAPISTU_GRAPH_NODE_TYPES.REACTION],
    # normally we don't expect interactors to be defined because they are handled by
    # net_create._format_interactors_for_regulatory_graph() but include them here
    # until Issue #102 is solved
    [SBOTERM_NAMES.INTERACTOR],
    [SBOTERM_NAMES.PRODUCT],
]

# an alternative layout to regulatory where enyzmes are downstream of substrates.
# this doesn't make much sense from a regulatory perspective because
# enzymes modify substrates not the other way around. but, its what one might
# expect if catalysts are a surrogate for reactions as is the case for metabolic
# network layouts

SURROGATE_GRAPH_HIERARCHY = [
    [SBOTERM_NAMES.MODIFIER, SBOTERM_NAMES.STIMULATOR, SBOTERM_NAMES.INHIBITOR],
    [SBOTERM_NAMES.REACTANT],
    [SBOTERM_NAMES.CATALYST],
    [NAPISTU_GRAPH_NODE_TYPES.REACTION],
    # normally we don't expect interactors to be defined because they are handled by
    # net_create._format_interactors_for_regulatory_graph() but include them here
    # until Issue #102 is solved
    [SBOTERM_NAMES.INTERACTOR],
    [SBOTERM_NAMES.PRODUCT],
]

NEIGHBORHOOD_NETWORK_TYPES = SimpleNamespace(
    DOWNSTREAM="downstream", HOURGLASS="hourglass", UPSTREAM="upstream"
)

VALID_NEIGHBORHOOD_NETWORK_TYPES = [
    NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
    NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM,
]
