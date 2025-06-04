"""Module containing functions to interoperate with rcpr's netcontextr functions"""

from __future__ import annotations

import logging
from typing import Any
from typing import Iterable

from napistu import sbml_dfs_core
from napistu.rpy2 import has_rpy2

from napistu.rpy2.constants import FIELD_REACTIONS
from napistu.rpy2.constants import COL_REACTION_ID

if has_rpy2:
    from rpy2.robjects import ListVector
    import rpy2.robjects as robjs

logger = logging.getLogger(__name__)


def trim_network_by_gene_attribute(
    rcpr,
    rcpr_graph: ListVector,
    field_name: str,
    field_value: Any = None,
    **kwargs,
) -> ListVector:
    """Trims the network by a gene attribute

    See the R function `rcpr::trim_network_by_gene_attribute` for
    more details.

    Args:
        rcpr (): The rpy2 rcpr object
        rcpr_graph (ListVector): The graph to trim
        field_name (str): The name of the column in the gene data to trim by
        field_value (Any): One or more values to trim by

    Returns:
        ListVector: The trimmed graph
    """
    if field_value is None:
        field_value = robjs.r("NaN")
    rcpr_graph_trimmed = rcpr.trim_network_by_gene_attribute(
        rcpr_graph, field_name=field_name, field_value=field_value, **kwargs
    )
    return rcpr_graph_trimmed


def apply_context_to_sbml_dfs(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    rcpr_graph: ListVector,
    inplace=True,
    remove_species=False,
) -> sbml_dfs_core.SBML_dfs:
    """Applies the context to the SBML dfs

    This is currently an in-place modification of
    the sbml_dfs object.

    Args:
        sbml_dfs (SbmlDfs): The SBML dfs to apply the context to
        rcpr_graph (ListVector): The graph to apply the context from
        inplace (bool, optional): Whether to modify the sbml_dfs in-place
            when applying the context. Defaults to True. "False" not yet implemented.
        remove_species (bool, optional): Whether to remove
            (compartmentalized) species that are no longer in the reactions.
            Defaults to False.

    Returns:
        SbmlDfs: The SBML dfs with the context applied
    """
    if not inplace:
        raise NotImplementedError("Only inplace is currently supported")

    # r_ids after trimming
    r_ids_new = set(rcpr_graph.rx("interactions")[0].rx("r_id")[0])

    # find original r_ids
    r_ids_old = set(sbml_dfs.reactions.index.tolist())

    # find the r_ids that are in the original but not in the new
    r_ids_to_remove = r_ids_old - r_ids_new

    # assert that no new r_ids were added
    if len(diff_ids := r_ids_new - r_ids_old) != 0:
        raise ValueError(
            f"New reactions present in rcpr, not present in smbl_dfs: {', '.join(diff_ids)}"
        )

    sbml_dfs.remove_reactions(r_ids_to_remove, remove_species=remove_species)

    return sbml_dfs


def trim_reactions_by_gene_attribute(
    rcpr,
    rcpr_reactions: ListVector,
    field_name: str,
    field_value: Any = None,
    **kwargs,
) -> ListVector:
    """Trims rcpr reactions by a gene attribute

    See the R function `rcpr::trim_reactions_by_gene_attribute` for
    more details.

    Args:
        rcpr (): The rpy2 rcpr object
        rcpr_reactions (ListVector): The graph to trim
        field_name (str): The name of the column in the gene data to trim by
        field_value (Any): One or more values to trim by

    Returns:
        ListVector: The trimmed graph
    """
    if field_value is None:
        field_value = robjs.r("NaN")
    rcpr_reactions_trimmed = rcpr.trim_reactions_by_gene_attribute(
        rcpr_reactions, field_name=field_name, field_value=field_value, **kwargs
    )
    return rcpr_reactions_trimmed


def apply_reactions_context_to_sbml_dfs(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    rcpr_reactions: ListVector,
    considered_reactions: Iterable[str] | None = None,
    inplace=True,
    remove_species=False,
) -> sbml_dfs_core.SBML_dfs:
    """Applies the context to the SBML dfs

    This is currently an in-place modification of
    the sbml_dfs object.

    Args:
        sbml_dfs (sbml_dfs_core.SBML_dfs): The SBML dfs to apply the context to
        rcpr_reactions (ListVector): The contextualized
        considered_reactions (Iterable[str], optional): The reactions that were
            considered for contextualisation. If None, all reactions that are
            in the sbml_dfs are considered and filtered out if they are not part of
            the rcpr_reactions. If provided, only reactions considered and not part
            of the rcpr_reactions are removed. Defaults to None.
        inplace (bool, optional): Whether to apply the context inplace.
            Only True currently implemented.
        remove_species (bool, optional): Whether to remove
            (compartmentalized) species that are no longer in the reactions.
            Defaults to False.

    Returns:
        SbmlDfs: The SBML dfs with the context applied
    """
    if not inplace:
        raise NotImplementedError("Only inplace is currently supported")

    # r_ids after trimming
    r_ids_new = _get_rids_from_rcpr_reactions(rcpr_reactions)

    # find original r_ids
    if considered_reactions is None:
        r_ids_old = set(sbml_dfs.reactions.index.tolist())
    else:
        r_ids_old = set(considered_reactions)

    # find the r_ids that are in the original but not in the new
    r_ids_to_remove = r_ids_old - r_ids_new

    # assert that no new r_ids were added
    if len(diff_ids := r_ids_new - r_ids_old) != 0:
        raise ValueError(
            "New reactions present in rcpr, not present in the considered "
            f"reactions: {', '.join(diff_ids)}"
        )

    sbml_dfs.remove_reactions(r_ids_to_remove, remove_species=remove_species)

    return sbml_dfs


def _get_rids_from_rcpr_reactions(rcpr_reactions: ListVector) -> set[str]:
    """Gets the r_ids from the rcpr reactions"""
    return set(rcpr_reactions.rx(FIELD_REACTIONS)[0].rx(COL_REACTION_ID)[0])
