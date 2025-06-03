from __future__ import annotations

import collections
import os
from itertools import chain
from typing import Any

import igraph as ig
import pandas as pd
from napistu import utils
from napistu.ingestion.constants import OBO_GO_BASIC_LOCAL_TMP
from napistu.ingestion.constants import OBO_GO_BASIC_URL


def create_go_parents_df(go_basic_obo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the GO Parents Table

    Reformat a table with GO attributes into a table with child-parent relationships

    Args:
        go_basic_obo_df (pd.DataFrame): Table generated from parsing go-basic.obo with
            obo.format_obo_dict_as_df

    Returns:
        go_parents_df (pd.DataFrame): a table with:
          - parent_id: GO ID of parent (from an is-a entry)
          - parent_name: common name of parent (from an is-a entry)
          - child_id: GO ID from the index

    """
    # filter to CC ontology and look at a series
    # where the index is GO IDs and values is a list of parent "is-a" relations
    cc_parents = go_basic_obo_df.query("namespace == 'cellular_component'")["is_a"]

    # this is currently at 4496 rows - this is expected to slowly increase
    if cc_parents.shape[0] < 4496:
        raise ValueError(
            f"Expected at least 4496 rows in cc_parents, got {cc_parents.shape[0]}"
        )
    if cc_parents.shape[0] >= 5000:
        raise ValueError(
            f"Expected fewer than 5000 rows in cc_parents, got {cc_parents.shape[0]}"
        )

    # convert from a list of strings to a list of dicts then expand so each
    # dict is its own row
    parent_entries = cc_parents.map(_isa_str_list_to_dict_list).explode()
    # drop orphans which will be NaN's after the explosion
    parent_entries = parent_entries[~parent_entries.isnull()]

    # convert to a DF which just has string variables
    go_parents_df = pd.DataFrame(parent_entries.tolist())
    go_parents_df["child_id"] = parent_entries.index

    # currently at 4688 rows - this may increase or decrease but will do so slowly
    if go_parents_df.shape[0] <= 4600:
        raise ValueError(
            f"Expected more than 4600 rows in go_parents_df, got {go_parents_df.shape[0]}"
        )
    if go_parents_df.shape[0] >= 5000:
        raise ValueError(
            f"Expected fewer than 5000 rows in go_parents_df, got {go_parents_df.shape[0]}"
        )

    return go_parents_df


def read_obo_as_dict(local_obo_path: str) -> dict:
    """
    Read OBO as Dictionary

    The Open Biological and Biomedical Ontologies (OBO) format is a standard format
    for representing ontologies. Many parsers exist for obo but since we are not
    relying extensively on it and we are trying to minimize dependencies here we provide a
    few functions for parsing standard obo formats.

    Args:
        local_obo_path (str): path to a local obo file.

    Returns
        term_dict (dict): dictionary where keys are ids and values are tuples
            containing (attribute, value) pairs
    """
    # create a dict where keys are term IDs and values are lists of tuples
    term_dict = dict()  # type: dict[str, Any]
    term_is_next = False
    active_term = None

    with open(local_obo_path) as file:
        for line in file:
            line_strip = line.rstrip()

            # reset the active term using the break between term definitions
            if line_strip == "":
                active_term = None

            line_as_tuple = _format_entry_tuple(line_strip)

            # catch new term definitions
            if term_is_next:
                attrib, value = line_as_tuple
                if attrib != "id":
                    raise ValueError(
                        f'{line_strip} was expected to be an "id" but it was not recongized as one'
                    )

                active_term = value
                term_dict[active_term] = list()
                term_is_next = False
                continue

            if line_strip == "[Term]":
                term_is_next = True
                continue
            else:
                term_is_next = False

            if active_term is not None:
                term_dict[active_term].append(line_as_tuple)

    return term_dict


def format_obo_dict_as_df(obo_term_dict: dict) -> pd.DataFrame:
    """
    Format an OBO Dict as a DataFrame

    Reorganize a dictionary of tuples into a DataFrame

    Args:
        term_dict (dict): dictionary where keys are ids and values are tuples
            containing (attribute, value) pairs

    Returns
        obo_df (pd.DataFrame): A pd.DataFrame with one row per identifier and one columns for unique attribute
    """
    # find attributes which can occur multiple times. These will be represented as lists within the
    # pandas DataFrame. The remaining attributes will just be strings.
    dups = [_find_obo_attrib_dups(obo_term_dict[k]) for k in obo_term_dict.keys()]
    degenerate_attribs = set(chain(*dups))

    # reorganize term as list to setup creation of pd.DataFrame
    term_dicts = list()
    for k, v in obo_term_dict.items():
        term_dict = _reformat_obo_entry_as_dict(v, degenerate_attribs)
        term_dict["id"] = k
        term_dicts.append(term_dict)

    obo_df = pd.DataFrame(term_dicts).set_index("id")

    return obo_df


def _reformat_obo_entry_as_dict(one_term, degenerate_attribs) -> dict:
    term_dict = dict()
    for attrib in degenerate_attribs:
        term_dict[attrib] = list()

    for attrib, value in one_term:
        if attrib in degenerate_attribs:
            term_dict[attrib].append(value)
        else:
            term_dict[attrib] = value

    return term_dict


def create_parent_child_graph(go_parents_df: pd.DataFrame) -> ig.Graph:
    """
    Create Parent:Child Graph

    Format the Simple GO CC Ontology as a Directed Acyclic Graph (DAG).

    Args:
        go_parents_df (pd.DataFrame): a table with:
          - parent_id: GO ID of parent (from an is-a entry)
          - parent_name: common name of parent (from an is-a entry)
          - child_id: GO ID from the index

    Returns:
        parent_child_graph (ig.Graph): a DAG formed from parent-child relationships.

    """
    valid_go_ids = {
        *go_parents_df["parent_id"].tolist(),
        *go_parents_df["child_id"].tolist(),
    }
    valid_go_ids_df = pd.DataFrame(valid_go_ids)
    valid_go_ids_df.columns = ["go_id"]  # type: ignore

    # format edgelist as an igraph network
    parent_child_graph = ig.Graph.DictList(
        vertices=valid_go_ids_df.to_dict("records"),
        edges=go_parents_df[["child_id", "parent_id"]].to_dict("records"),
        directed=True,
        vertex_name_attr="go_id",
        edge_foreign_keys=("child_id", "parent_id"),
    )

    # is it a fully connected DAG as expected?
    if not parent_child_graph.is_dag():
        raise ValueError("parent_child_graph is not a DAG as expected")
    if not parent_child_graph.is_connected("weak"):
        raise ValueError("parent_child_graph is not weakly connected as expected")

    return parent_child_graph


def create_go_ancestors_df(parent_child_graph: ig.Graph) -> pd.DataFrame:
    """
    Create GO Ancestors DataFrame

    Args:
        parent_child_graph (ig.Graph): a DAG formed from parent-child relationships.

    Returns:
        go_ancestors_df (pd.DataFrame): a table with:
          - go_id: GO ID of a CC GO term of interest
          - ancestor_id: An ancestor (parent, parent of parent, ...)'s GO CC ID
    """
    # find the ancestors of each vertex
    ancestor_dict = [
        {
            "go_id": v["go_id"],
            "ancestor_id": parent_child_graph.vs(
                parent_child_graph.subcomponent(v, mode=ig.OUT)
            ).get_attribute_values("go_id"),
        }
        for v in parent_child_graph.vs
    ]

    go_ancestors_df = pd.DataFrame(ancestor_dict).explode("ancestor_id")
    # drop self edges
    go_ancestors_df = go_ancestors_df[
        go_ancestors_df["go_id"] != go_ancestors_df["ancestor_id"]
    ]

    return go_ancestors_df


def _download_go_basic_obo(local_obo_path: str = OBO_GO_BASIC_LOCAL_TMP) -> None:
    """Download an OBO file containing GO categories and their relations (but not the genes in each category)."""

    utils.download_wget(OBO_GO_BASIC_URL, local_obo_path)

    if not os.path.isfile(local_obo_path):
        raise FileNotFoundError(
            f"{local_obo_path} was not found after trying to download from {OBO_GO_BASIC_URL}"
        )


def _isa_str_list_to_dict_list(isa_list: list) -> list[dict[str, Any]]:
    """Split parent-child relationships from individual strings to dictionaries where parent and child are separated."""

    split_vals = [tuple(val.split(" ! ")) for val in isa_list]

    isa_dict_list = list()
    for split_val in split_vals:
        if len(split_val) != 2:
            raise ValueError(
                f"Expected tuple of length 2, got {len(split_val)}: {split_val}"
            )
        isa_dict_list.append({"parent_id": split_val[0], "parent_name": split_val[1]})

    return isa_dict_list


def _format_entry_tuple(line_str: str) -> tuple | None:
    """Split and return a colon-separated tuple."""

    entry = line_str.split(": ", maxsplit=1)
    if len(entry) == 2:
        return tuple(entry)
    return None


def _find_obo_attrib_dups(one_term) -> list:
    """Identify attributes which are present multiple times."""

    attrib_count = collections.Counter([v[0] for v in one_term])
    duplicated_attributes = [item for item, count in attrib_count.items() if count > 1]

    return duplicated_attributes
