"""
Module with helper functions to deal with edgelists

Edgelists are assumed to be DataFrames whose first two columns represent an Edge relation, eg From, To
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def remove_reciprocal_interactions(
    edgelist: pd.DataFrame, extra_defining_vars: list = list()
) -> pd.DataFrame:
    """Remove reciprocal edges from an edgelist (i.e., if B-A always exists for every A-B then remove B-A)

    Args:
        edgelist (pd.DataFrame): edgelist (pd.DataFrame): edgelist where the first two
            columns are assumed to be the edge vertices
        extra_defining_vars (list): list (which can be empty) of variables which define
            a unique interaction beyond the vertices

    Returns:
        indegenerate_edgelist (pd.DataFrame): edgelist with B-A edges removed and A-B retained

    """

    edgelist_vars = edgelist.columns.tolist()[0:2]
    logger.info(
        "Removing reciprocal interactions treating "
        f"{edgelist_vars[0]} and {edgelist_vars[1]} as vertices"
    )

    reciprocal_interaction_fraction = count_fraction_of_reciprocal_interactions(
        edgelist, extra_defining_vars
    )
    if reciprocal_interaction_fraction != 1:
        raise ValueError(
            f"Only {reciprocal_interaction_fraction} of edges are present as reciprocal edges;"
            " this method of removing reciprocal edges will be unreliable"
        )

    indegenerate_edgelist = edgelist.loc[
        edgelist[edgelist_vars[0]] < edgelist[edgelist_vars[1]]
    ]

    return indegenerate_edgelist


def count_fraction_of_reciprocal_interactions(
    edgelist: pd.DataFrame, extra_defining_vars: list = list()
) -> float:
    """Count the fraction of A-B edges which also show up as B-A edges

    Args:
        edgelist (pd.DataFrame): edgelist (pd.DataFrame): edgelist where the first two
            columns are assumed to be the edge vertices
        extra_defining_vars (list): list (which can be empty) of variables which define
            a unique interaction beyond the vertices

    Returns:
        fraction (float): fraction of A-B edges which are also included as B-A edges

    """

    # first two variables are assumed to be vertices of edgelist
    edgelist_vars = edgelist.columns.tolist()[0:2]
    logger.info(
        "Counting the fraction of reciprocal interactions treating "
        f"{edgelist_vars[0]} and {edgelist_vars[1]} as vertices"
    )

    # extra defining variables must exist
    missing_extra_defining_vars = set(extra_defining_vars).difference(
        set(edgelist.columns)
    )
    if len(missing_extra_defining_vars) > 0:
        raise ValueError(
            f"{', '.join(missing_extra_defining_vars)} are \"extra_defining_vars\" "
            "but were missing from the edgelist"
        )

    extended_edgelist_vars = [*edgelist_vars, *extra_defining_vars]
    logger.info(
        f"{', '.join(extra_defining_vars)} will be used as \"extra_defining_vars\" "
        "which much match across reciprocal edges for the edge to be identical"
    )

    possible_reciprocal_interactions = (
        edgelist[extended_edgelist_vars]
        .rename(
            {edgelist_vars[0]: edgelist_vars[1], edgelist_vars[1]: edgelist_vars[0]},
            axis=1,
        )
        .assign(reciprocal_exists=True)
    )

    reciprocal_interaction_test = edgelist[extended_edgelist_vars].merge(
        possible_reciprocal_interactions
    )

    return reciprocal_interaction_test.shape[0] / edgelist.shape[0]
