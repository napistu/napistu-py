"""Module to contain all constants used for representing and working with networks"""

from __future__ import annotations

from types import SimpleNamespace

from napistu.constants import SBML_DFS
from napistu.constants import SBOTERM_NAMES

CPR_GRAPH_NODES = SimpleNamespace(NAME="name")

CPR_GRAPH_EDGES = SimpleNamespace(
    DIRECTED="directed",
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

# variables which should be in cpr graph's edges
CPR_GRAPH_REQUIRED_EDGE_VARS = {
    CPR_GRAPH_EDGES.FROM,
    CPR_GRAPH_EDGES.TO,
    CPR_GRAPH_EDGES.SBO_TERM,
    CPR_GRAPH_EDGES.STOICHIOMETRY,
    CPR_GRAPH_EDGES.SC_PARENTS,
    CPR_GRAPH_EDGES.SC_CHILDREN,
}

# nomenclature for individual fields

CPR_GRAPH_NODE_TYPES = SimpleNamespace(REACTION="reaction", SPECIES="species")

VALID_CPR_GRAPH_NODE_TYPES = [
    CPR_GRAPH_NODE_TYPES.REACTION,
    CPR_GRAPH_NODE_TYPES.SPECIES,
]

CPR_GRAPH_EDGE_DIRECTIONS = SimpleNamespace(
    FORWARD="forward", REVERSE="reverse", UNDIRECTED="undirected"
)

# network-level nomenclature

CPR_GRAPH_TYPES = SimpleNamespace(
    BIPARTITE="bipartite", REGULATORY="regulatory", SURROGATE="surrogate"
)

VALID_CPR_GRAPH_TYPES = [
    CPR_GRAPH_TYPES.BIPARTITE,
    CPR_GRAPH_TYPES.REGULATORY,
    CPR_GRAPH_TYPES.SURROGATE,
]

CPR_WEIGHTING_STRATEGIES = SimpleNamespace(
    CALIBRATED="calibrated", MIXED="mixed", TOPOLOGY="topology", UNWEIGHTED="unweighted"
)

VALID_WEIGHTING_STRATEGIES = [
    CPR_WEIGHTING_STRATEGIES.UNWEIGHTED,
    CPR_WEIGHTING_STRATEGIES.TOPOLOGY,
    CPR_WEIGHTING_STRATEGIES.MIXED,
    CPR_WEIGHTING_STRATEGIES.CALIBRATED,
]

# the regulatory graph defines a hierarchy of upstream and downstream
# entities in a reaction
# modifier/stimulator/inhibitor -> catalyst -> reactant -> reaction -> product

REGULATORY_GRAPH_HIERARCHY = [
    [SBOTERM_NAMES.MODIFIER, SBOTERM_NAMES.STIMULATOR, SBOTERM_NAMES.INHIBITOR],
    [SBOTERM_NAMES.CATALYST],
    [SBOTERM_NAMES.REACTANT],
    [CPR_GRAPH_NODE_TYPES.REACTION],
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
    [CPR_GRAPH_NODE_TYPES.REACTION],
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
