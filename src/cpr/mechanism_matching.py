from __future__ import annotations

import logging

import igraph as ig
import pandas as pd
from cpr import sbml_dfs_core
from cpr import utils
from cpr.constants import SBML_DFS
from cpr.constants import CPR_EDGELIST
from cpr.constants import CPR_EDGELIST_REQ_VARS
from cpr.constants import IDENTIFIERS
from cpr.constants import IDENTIFIER_EDGELIST_REQ_VARS
from cpr.constants import SPECIES_IDENTIFIERS_REQUIRED_VARS
from cpr.network.constants import CPR_GRAPH_EDGES
from cpr.network import paths

logger = logging.getLogger(__name__)


def features_to_pathway_species(
    feature_identifiers: pd.DataFrame,
    species_identifiers: pd.DataFrame,
    ontologies: set,
    feature_id_var: str,
) -> pd.DataFrame:
    """
    Features to Pathway Species

    Match a table of molecular species to their corresponding species in a pathway representation.

    Parameters:
    feature_identifiers: pd.DataFrame
        pd.Dataframe containing a "feature_id_var" variable used to match entries
    species_identifiers: pd.DataFrame
        A table of molecular species identifiers produced from sbml_dfs.get_identifiers("species")
        generally using sbml_dfs_core.export_sbml_dfs()
    ontologies: set
        A set of ontologies used to match features to pathway species
    feature_id_var: str
        Variable in "feature_identifiers" containing identifiers

    Returns:
    pathway_species: pd.DataFrame
        species_identifiers joined to feature_identifiers based on shared identifiers
    """

    # map features to molecular features in the pathway
    if feature_id_var not in feature_identifiers.columns.to_list():
        raise ValueError(
            f"{feature_id_var} must be a variable in 'feature_identifiers', "
            f"possible variables are {', '.join(feature_identifiers.columns.tolist())}"
        )

    # check identifiers table
    _check_species_identifiers_table(species_identifiers)

    available_ontologies = set(species_identifiers[IDENTIFIERS.ONTOLOGY].tolist())
    unavailable_ontologies = ontologies.difference(available_ontologies)

    # no ontologies present
    if len(unavailable_ontologies) == len(ontologies):
        raise ValueError(
            f"None of the requested ontologies ({', '.join(ontologies)}) "
            "were used to annotate pathway species. Available ontologies are: "
            f"{', '.join(available_ontologies)}"
        )

    # 1+ desired ontologies are not present
    if len(unavailable_ontologies) > 0:
        raise ValueError(
            f"Some of the requested ontologies ({', '.join(unavailable_ontologies)}) "
            "were NOT used to annotate pathway species. Available ontologies are: "
            f"{', '.join(available_ontologies)}"
        )

    relevant_identifiers = species_identifiers[
        species_identifiers[IDENTIFIERS.ONTOLOGY].isin(ontologies)
    ]

    # map features to pathway species
    pathway_species = feature_identifiers.merge(
        relevant_identifiers, left_on=feature_id_var, right_on=IDENTIFIERS.IDENTIFIER
    )

    if pathway_species.shape[0] == 0:
        logger.warning(
            "None of the provided species identifiers matched entries of the pathway; returning None"
        )
        None

    # report the fraction of unmapped species

    return pathway_species


def edgelist_to_pathway_species(
    formatted_edgelist: pd.DataFrame, species_identifiers: pd.DataFrame, ontologies: set
) -> pd.DataFrame:
    """
    Edgelist to Pathway Species

    Match an edgelist of molecular species pairs to their corresponding species in a pathway representation.

    Parameters:
    formatted_edgelist: pd.DataFrame
        pd.Dataframe containing a "identifier_upstream" and "identifier_downstream" variables used to to match entries
    species_identifiers: pd.DataFrame
        A table of molecular species identifiers produced from sbml_dfs.get_identifiers("species") generally using
        sbml_dfs_core.export_sbml_dfs()
    ontologies: set
        A set of ontologies used to match features to pathway species

    Returns:
    edges_on_pathway: pd.DataFrame
        formatted_edgelist with upstream features mapped
        to "s_id_upstream" and downstream species mapped
        to "s_id_downstream"
    """

    required_vars_distinct_features = {
        CPR_EDGELIST.IDENTIFIER_UPSTREAM,
        CPR_EDGELIST.IDENTIFIER_DOWNSTREAM,
    }
    missing_required_vars_distinct_features = (
        required_vars_distinct_features.difference(
            set(formatted_edgelist.columns.tolist())
        )
    )

    if len(missing_required_vars_distinct_features) > 0:
        raise ValueError(
            f"{len(missing_required_vars_distinct_features)} required variables were "
            "missing from 'formatted_edgelist': "
            f"{', '.join(missing_required_vars_distinct_features)}"
        )

    # define all distinct identifiers in edgelist
    distinct_identifiers = (
        pd.concat(
            [
                formatted_edgelist[CPR_EDGELIST.IDENTIFIER_UPSTREAM],
                formatted_edgelist[CPR_EDGELIST.IDENTIFIER_DOWNSTREAM],
            ]
        )
        .drop_duplicates()
        .reset_index(drop=True)
        .to_frame()
        .rename({0: "feature_id"}, axis=1)
    )

    # merge edgelist identifiers with pathway identifiers to map s_ids to identifiers
    features_on_pathway = features_to_pathway_species(
        feature_identifiers=distinct_identifiers,
        species_identifiers=species_identifiers,
        ontologies=ontologies,
        feature_id_var="feature_id",
    )

    # add s_ids of both upstream and downstream edges to pathway
    edges_on_pathway = formatted_edgelist.merge(
        features_on_pathway[[SBML_DFS.S_ID, IDENTIFIERS.IDENTIFIER]].rename(
            {
                SBML_DFS.S_ID: CPR_EDGELIST.S_ID_UPSTREAM,
                IDENTIFIERS.IDENTIFIER: CPR_EDGELIST.IDENTIFIER_UPSTREAM,
            },
            axis=1,
        )
    ).merge(
        features_on_pathway[[SBML_DFS.S_ID, IDENTIFIERS.IDENTIFIER]].rename(
            {
                SBML_DFS.S_ID: CPR_EDGELIST.S_ID_DOWNSTREAM,
                IDENTIFIERS.IDENTIFIER: CPR_EDGELIST.IDENTIFIER_DOWNSTREAM,
            },
            axis=1,
        )
    )

    return edges_on_pathway


def edgelist_to_scids(
    formatted_edgelist: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_identifiers: pd.DataFrame,
    ontologies: set,
):
    """

    Edgelist to Compartmentalized Species IDds

    Map an edgelist of possible mechanistic interactions onto a
    pathadex pathway

    Parameters:
    formatted_edgelist: pd.DataFrame
        pd.Dataframe containing a "identifier_upstream" and
        "identifier_downstream" variables used to to match entries
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic model
    species_identifiers: pd.DataFrame
        A table of molecular species identifiers produced from
        sbml_dfs.get_identifiers("species") generally using sbml_dfs_core.export_sbml_dfs()
    ontologies: set
        A set of ontologies used to match features to pathway species

    Returns:
    edgelist_w_scids: pd.DataFrame
        formatted_edgelist with upstream features mapped to "sc_id_upstream" and
        downstream species mapped to "sc_id_downstream"
    """

    _check_species_identifiers_table(species_identifiers)

    # map edges onto pathway entities based on shared identifiers
    edges_on_pathway = edgelist_to_pathway_species(
        formatted_edgelist=formatted_edgelist,
        species_identifiers=species_identifiers,
        ontologies=ontologies,
    )

    # expand from s_ids to sc_ids
    s_id_pairs = edges_on_pathway[
        [CPR_EDGELIST.S_ID_UPSTREAM, CPR_EDGELIST.S_ID_DOWNSTREAM]
    ].drop_duplicates()
    sc_id_pairs = s_id_pairs.merge(
        sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID]]
        .reset_index()
        .rename(
            {
                SBML_DFS.S_ID: CPR_EDGELIST.S_ID_UPSTREAM,
                SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_UPSTREAM,
            },
            axis=1,
        )
    ).merge(
        sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID]]
        .reset_index()
        .rename(
            {
                SBML_DFS.S_ID: CPR_EDGELIST.S_ID_DOWNSTREAM,
                SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_DOWNSTREAM,
            },
            axis=1,
        )
    )

    # map sc_ids back to edges_on_pathway
    # join lookup table of s_id_upstream, s_id_downstream -> sc_ids
    edgelist_w_scids = edges_on_pathway.merge(sc_id_pairs)

    logger_msg = (
        f"{edgelist_w_scids.shape[0]} interactions mapped "
        "onto pairs of compartmentalized species in the mechanistic model"
    )
    if edgelist_w_scids.shape[0] == 0:
        logger.warning(logger_msg)
    else:
        logger.info(logger_msg)

    return edgelist_w_scids


def filter_to_direct_mechanistic_interactions(
    formatted_edgelist: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_identifiers: pd.DataFrame,
    ontologies: set,
) -> pd.DataFrame:
    """
    Filter to Direct Mechanistic Interactions

    Filter an edgelist to direct mechanistic interactions

    Parameters:
    formatted_edgelist: pd.DataFrame
        pd.Dataframe containing a "identifier_upstream" and "identifier_downstream" variables used to to match entries
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic model
    species_identifiers: pd.DataFrame
        A table of molecular species identifiers
        produced from sbml_dfs.get_identifiers("species") generally
        using sbml_dfs_core.export_sbml_dfs()
    ontologies: set
        A set of ontologies used to match features to pathway species

    Returns:
    edgelist_w_direct_mechanistic_interactions: pd.DataFrame
        formatted_edgelist filtered to mechanistic reactions present in the pathway representation
    """

    edgelist_w_scids = _edgelist_to_scids_if_needed(
        formatted_edgelist, sbml_dfs, species_identifiers, ontologies
    )

    # reduce to distinct sc_id pairs
    sc_id_pairs = edgelist_w_scids[CPR_EDGELIST_REQ_VARS].drop_duplicates()

    # define all existing direct regulatory interactions
    pathway_interactions = pd.concat(
        [
            # pair 0 -> <0 # modifiers affect substrates
            sbml_dfs.reaction_species[
                sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] == 0
            ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]]
            .rename({SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_UPSTREAM}, axis=1)
            .merge(
                sbml_dfs.reaction_species[
                    sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] < 0
                ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]].rename(
                    {SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_DOWNSTREAM}, axis=1
                )
            ),
            # pair <0 -> >0 # substrates affect products
            sbml_dfs.reaction_species[
                sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] < 0
            ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]]
            .rename({SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_UPSTREAM}, axis=1)
            .merge(
                sbml_dfs.reaction_species[
                    sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] > 0
                ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]].rename(
                    {SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_DOWNSTREAM}, axis=1
                )
            ),
            # pair 0 -> >0 # modifiers affect products
            sbml_dfs.reaction_species[
                sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] == 0
            ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]]
            .rename({SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_UPSTREAM}, axis=1)
            .merge(
                sbml_dfs.reaction_species[
                    sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] > 0
                ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]].rename(
                    {SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_DOWNSTREAM}, axis=1
                )
            ),
        ]
    ).reset_index(drop=True)

    # filter pathway interactions based on matches to sc_id_pairs
    direct_edge_interactions = (
        sc_id_pairs.merge(pathway_interactions)
        .merge(
            sbml_dfs.species[SBML_DFS.S_NAME]
            .to_frame()
            .rename({SBML_DFS.S_NAME: CPR_EDGELIST.S_NAME_UPSTREAM}, axis=1),
            left_on=CPR_EDGELIST.S_ID_UPSTREAM,
            right_index=True,
            # add species metadata for matches
        )
        .merge(
            sbml_dfs.species[SBML_DFS.S_NAME]
            .to_frame()
            .rename({SBML_DFS.S_NAME: CPR_EDGELIST.S_NAME_DOWNSTREAM}, axis=1),
            left_on=CPR_EDGELIST.S_ID_DOWNSTREAM,
            right_index=True,
            # add metadata for reactions where interaction occurs
        )
        .merge(
            sbml_dfs.reactions[SBML_DFS.R_NAME].to_frame(),
            left_on=SBML_DFS.R_ID,
            right_index=True,
        )
    )

    edgelist_w_direct_mechanistic_interactions = edgelist_w_scids.merge(
        direct_edge_interactions[
            [
                CPR_EDGELIST.SC_ID_UPSTREAM,
                CPR_EDGELIST.SC_ID_DOWNSTREAM,
                SBML_DFS.R_ID,
                CPR_EDGELIST.S_NAME_UPSTREAM,
                CPR_EDGELIST.S_NAME_DOWNSTREAM,
                SBML_DFS.R_NAME,
            ]
        ]
    )

    return edgelist_w_direct_mechanistic_interactions


def filter_to_indirect_mechanistic_interactions(
    formatted_edgelist: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_identifiers: pd.DataFrame,
    cpr_graph: ig.Graph,
    ontologies: set,
    precomputed_distances=None,
    max_path_length=10,
):
    """
    Filter to Indirect Mechanistic Interactions

    Filter an edgelist to indirect mechanistic interactions.
    Indirect relationships are identified by searching a
    network for paths from an upstream species to a downstream species

    Parameters:
    formatted_edgelist: pd.DataFrame
        pd.Dataframe containing a "identifier_upstream" and
        "identifier_downstream" variables used to to match entries
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic model
    species_identifiers: pandas.DataFrame
        A table of molecular species identifiers produced from
        sbml_dfs.get_identifiers("species") generally using sbml_dfs_core.export_sbml_dfs()
    cpr_graph: igraph.Graph
        A network representation of the sbml_dfs model
    ontologies: set
        A set of ontologies used to match features to pathway species
    precomputed_distances: None or a pd.DataFrame containing path lengths and weights
        between pairs of cspecies.
    max_path_length: int
        Maximum number of steps to consider.

    Returns:
    edgelist_w_indirect_mechanistic_interactions: pd.DataFrame
        formatted_edgelist filtered to mechanistic reactions which can be described
        by an indirect mechanism. The mechanism is described by a path weight, length,
        and a vpath and epath list of vertices and edges which were traversed to create the path.
    """

    edgelist_w_scids = _edgelist_to_scids_if_needed(
        formatted_edgelist, sbml_dfs, species_identifiers, ontologies
    )

    if precomputed_distances is not None:
        # rename to match conventions in precomputed_distances
        # filter by these precomputed distances and then restore naming
        edgelist_w_scids = paths._filter_paths_by_precomputed_distances(
            edgelist_w_scids.rename(
                {
                    CPR_EDGELIST.SC_ID_UPSTREAM: CPR_EDGELIST.SC_ID_ORIGIN,
                    CPR_EDGELIST.SC_ID_DOWNSTREAM: CPR_EDGELIST.SC_ID_DEST,
                },
                axis=1,
            ),
            precomputed_distances,
        ).rename(
            {
                CPR_EDGELIST.SC_ID_ORIGIN: CPR_EDGELIST.SC_ID_UPSTREAM,
                CPR_EDGELIST.SC_ID_DEST: CPR_EDGELIST.SC_ID_DOWNSTREAM,
            },
            axis=1,
        )

    # find paths from 1 upstream to all desired downstream sc_ids
    # (this is the convention with igraph)
    indexed_origin_vertices = edgelist_w_scids.set_index(CPR_EDGELIST.SC_ID_UPSTREAM)

    # loop through upstream cspecies and find paths to all downstream species
    global_dict = dict()
    for an_origin_index in indexed_origin_vertices.index.unique():  # type: ignore
        origin_targets = indexed_origin_vertices.loc[
            an_origin_index
        ]  # type: pd.DataFrame

        # if indexing only a single entry pd.DataFrame becomes a pd.Series
        # convert back to DataFrame for consistency
        origin_targets = utils.ensure_pd_df(origin_targets)

        # log entry for debugging
        logger.debug(
            f"finding paths from {an_origin_index} to "
            f"{origin_targets.shape[0]} target vertices"
        )

        # find all paths from indexed_origin to desired destination
        shortest_paths = paths.find_shortest_reaction_paths(
            cpr_graph,
            sbml_dfs,
            origin=an_origin_index,
            # find all unique destinations (as a list for compatibility with igraph dest)
            dest=origin_targets[CPR_EDGELIST.SC_ID_DOWNSTREAM].unique().tolist(),
            weight_var=CPR_GRAPH_EDGES.WEIGHTS,
        )

        if shortest_paths is None:
            continue

        vertices, edges = shortest_paths
        indexed_edges = edges.set_index("path")
        indexed_vertices = vertices.set_index("path")

        paths_list = list()
        for ind in indexed_edges.index.unique():
            one_path = indexed_edges.loc[ind]

            # make sure that we are working with a DF
            if type(one_path) is pd.Series:
                one_path = one_path.to_frame().T

            if one_path.shape[0] > max_path_length:
                continue

            # find the destination node
            # this is annoying because if the graph is undirected
            # its not clear if the from or to edge is the actual destination
            # when taking advantage of the fact that igraph lets you
            # look up multiple destinations at once this information is lost
            ancestor_species = {an_origin_index}
            if one_path.shape[0] > 1:
                penultimate_edge = one_path.iloc[one_path.shape[0] - 2]
                ancestor_species = ancestor_species.union(
                    {
                        penultimate_edge[CPR_GRAPH_EDGES.FROM],
                        penultimate_edge[CPR_GRAPH_EDGES.TO],
                    }
                )

            terminal_edge = one_path.iloc[one_path.shape[0] - 1]
            ending_cspecies = {terminal_edge[CPR_GRAPH_EDGES.FROM], terminal_edge[CPR_GRAPH_EDGES.TO]}.difference(ancestor_species)  # type: ignore

            if len(ending_cspecies) != 1:
                raise ValueError(
                    "The terminal edge could not be determined when summarizing paths"
                )
            ending_cspecies = ending_cspecies.pop()

            path_series = pd.Series(
                {
                    CPR_GRAPH_EDGES.FROM: an_origin_index,
                    CPR_GRAPH_EDGES.TO: ending_cspecies,
                    "weight": sum(one_path[CPR_GRAPH_EDGES.WEIGHTS]),
                    "path_length": one_path.shape[0],
                    "vpath": indexed_vertices.loc[ind],
                    "epath": one_path,
                }  # type: ignore
            )  # type: pd.Series

            paths_list.append(path_series)

        if len(paths_list) > 0:
            origin_paths = pd.DataFrame(paths_list)
            global_dict[an_origin_index] = origin_paths

    if len(global_dict.keys()) == 0:
        logger.warning(
            "None of the provide molecular pairs could be mechanistically linked with a network path"
        )
        return None

    all_shortest_paths = pd.concat(global_dict.values())

    indirect_shortest_paths = edgelist_w_scids.merge(
        all_shortest_paths,
        left_on=[CPR_EDGELIST.SC_ID_UPSTREAM, CPR_EDGELIST.SC_ID_DOWNSTREAM],
        right_on=[CPR_GRAPH_EDGES.FROM, CPR_GRAPH_EDGES.TO],
    )

    return indirect_shortest_paths


def _edgelist_to_scids_if_needed(
    edgelist: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_identifiers: pd.DataFrame,
    ontologies: set,
) -> pd.DataFrame:
    """Map a set of edgelist species to cspecies or skip if cspecies were provided."""

    if utils.match_pd_vars(edgelist, CPR_EDGELIST_REQ_VARS).are_present:
        logger.info(
            f"An edgelist with {', '.join(CPR_EDGELIST_REQ_VARS)} was provided; identifier matching will be skipped"
        )
        return edgelist
    else:
        utils.match_pd_vars(edgelist, IDENTIFIER_EDGELIST_REQ_VARS).assert_present()

        _check_species_identifiers_table(species_identifiers)

        edgelist_w_scids = edgelist_to_scids(
            edgelist,
            sbml_dfs=sbml_dfs,
            species_identifiers=species_identifiers,
            ontologies=ontologies,
        )

        return edgelist_w_scids


def _check_species_identifiers_table(
    species_identifiers: pd.DataFrame,
    required_vars: set = SPECIES_IDENTIFIERS_REQUIRED_VARS,
):
    missing_required_vars = required_vars.difference(
        set(species_identifiers.columns.tolist())
    )
    if len(missing_required_vars) > 0:
        raise ValueError(
            f"{len(missing_required_vars)} required variables "
            "were missing from the species_identifiers table: "
            f"{', '.join(missing_required_vars)}"
        )

    return None
