"""Module containing functions to interoperate with rcpr's netcontextr functions"""

from __future__ import annotations

import logging
import os
from tempfile import NamedTemporaryFile
from typing import Any
from typing import Callable
from typing import Iterable

import pandas as pd
from napistu import sbml_dfs_core
from napistu import utils
from napistu.rpy2 import has_rpy2
from napistu.rpy2 import warn_if_no_rpy2

from napistu.rpy2.constants import COL_GENE
from napistu.rpy2.constants import COL_PROTEIN_1
from napistu.rpy2.constants import COL_PROTEIN_2
from napistu.rpy2.constants import FIELD_INTERACTIONS
from napistu.rpy2.constants import FIELD_GENES
from napistu.rpy2.constants import FIELD_REACTIONS
from napistu.rpy2.constants import COL_ROLE
from napistu.rpy2.constants import COL_REACTION_ID
from napistu.rpy2.constants import COL_STOICHIOMETRY
from napistu.rpy2.constants import NETCONTEXTR_ONTOLOGY
from napistu.rpy2.constants import NETCONTEXTR_SBO_MAP

if has_rpy2:
    from napistu.rpy2.callr import pandas_to_r_dataframe
    from rpy2.robjects import ListVector
    import rpy2.robjects as robjs

logger = logging.getLogger(__name__)


@warn_if_no_rpy2
def _none2null(none_obj):
    return robjs.r("NULL")


@warn_if_no_rpy2
def sbml_dfs_to_rcpr_string_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    reaction_data: str = "string",
    identifier_ontology: str = "ensembl_gene",
    rescale_data: Callable[[pd.DataFrame], pd.DataFrame] | None = lambda x: x / 1000,
) -> ListVector:
    """Converts an sbml_dfs to a rcpr string graph

    This utility converts the sbml_dfs to the format returned
    by `rcpr::createStringGraph`.

    Args:
        sbml_dfs (SBML_dfs): the sbml_dfs from string.
            It is assumed that this sbml_dfs has only reactions with exactly
            two reactands and a 1:1 mapping between s_id and sc_id.
        reaction_data (str, optional): The reaction data that contains
            the string scores. Defaults to 'string'.
        identifier_ontology (str, optional): The ontology to use for the
            protein identifiers. Defaults to `ensembl_gene` (default in rcpr)
        rescale_data (Callable[pd.DataFrame], optional): A function to rescale
            the data. Defaults to lambda x: x/1000 (default in rcpr)

    Returns:
        This is a list of dataframes almost the same as `rcpr::createStringGraph`:
        - `genes`: a dataframe with column `gene`
            and the extra column `s_id`, `sc_id`
        - `interactions`: a dataframe with columns `protein1`, `protein2` and
            the scores from string
            and the extra column `r_id`

        The extra columns `s_id` and `r_id` are used to map the genes and reactions
        to the sbml_dfs. This is useful for mapping back rcpr results to the
        sbml_dfs.
    """

    dat_gene = (
        sbml_dfs.species["s_Identifiers"]
        # Get the identifiers for the given ontology
        .map(lambda ids: ids.hoist(identifier_ontology))
        .rename(COL_GENE)
        .to_frame()
        # Merge with compartmentalized species to get the sc_id
        .merge(
            sbml_dfs.compartmentalized_species[["s_id"]].reset_index(drop=False),
            left_index=True,
            right_on="s_id",
        )[[COL_GENE, "sc_id", "s_id"]]
    )

    # Perform validations
    if not dat_gene["s_id"].is_unique:
        raise ValueError("dat_gene['s_id'] must be unique")
    if not dat_gene["sc_id"].is_unique:
        raise ValueError("dat_gene['sc_id'] must be unique")
    if not dat_gene[COL_GENE].is_unique:
        raise ValueError("dat_gene[COL_GENE] must be unique")
    if dat_gene[COL_GENE].hasnans:
        raise ValueError("dat_gene[COL_GENE] must not have NaNs")

    # Reshape into the correct format
    dat_reactions = dat_gene[["sc_id", COL_GENE]].merge(
        sbml_dfs.reaction_species[["r_id", "sc_id"]], on="sc_id"
    )[[COL_GENE, "r_id"]]
    # assert that this has the correct shape, ie 2x the shape of the number
    # of reactions
    if dat_reactions.shape[0] != 2 * sbml_dfs.reactions.shape[0]:
        raise ValueError("There should be exactly 2 reactants per reaction")

    # This is the fastest way I found to reshape this into the
    # Edgelist format
    dat_reactions["flag"] = dat_reactions["r_id"].duplicated()
    dat_interactions = dat_reactions.pivot(
        index="r_id", columns="flag", values=COL_GENE
    )
    dat_interactions.columns = pd.Index([COL_PROTEIN_1, COL_PROTEIN_2], dtype=object)
    if rescale_data is not None:
        reaction_df = rescale_data(sbml_dfs.reactions_data[reaction_data])
    else:
        reaction_df = sbml_dfs.reactions_data[reaction_data]

    dat_interactions = dat_interactions.join(reaction_df).reset_index(drop=False)

    genes = pandas_to_r_dataframe(dat_gene)
    interactions = pandas_to_r_dataframe(dat_interactions)

    out = ListVector({FIELD_GENES: genes, FIELD_INTERACTIONS: interactions})
    return out


@warn_if_no_rpy2
def load_and_clean_hpa_data(
    rcpr,
    uri_hpa: str,
):
    """Load and cleans HPA data using rcpr

    Args:
        rcpr (): The rpy2 rcpr object
        uri_hpa (str): The uri of the HPA data

    Returns:
        rpy2 object: The cleaned HPA data
    """

    with NamedTemporaryFile() as f:
        # R cannot work with gcs uris
        # thus download the file to a temporary
        # location incase it is a gcs uri
        if os.path.exists(uri_hpa):
            # if the file is already a local
            # file, just use it
            path_hpa = uri_hpa
        else:
            path_hpa = f.name
            utils.copy_uri(uri_hpa, path_hpa)

        hpa_localization_data = rcpr.load_and_clean_hpa_data(path_hpa)
    return hpa_localization_data


@warn_if_no_rpy2
def load_and_clean_gtex_data(rcpr_rpy2, uri_gtex: str, by_tissue_zfpkm: bool = False):
    """Load and cleans GTEx data using rcpr

    Args:
        rcpr_rpy2 (): The rpy2 rcpr object
        uri_gtex (str): The uri of the GTEx data
        by_tissue_zfpkm (bool, optional): Whether to return the data normalized
          by tissue using zfpkm. Defaults to False.
    Returns:
        rpy2 object: The cleaned GTEx data
    """
    with NamedTemporaryFile() as f:
        # R cannot work with gcs uris
        # thus download the file to a temporary
        # location incase it is a gcs uri
        if os.path.exists(uri_gtex):
            # if the file is already a local
            # file, just use it
            path_gtex = uri_gtex
        else:
            path_gtex = f.name
            utils.copy_uri(uri_gtex, path_gtex)

        gtex_tissue_data = rcpr_rpy2.load_and_clean_gtex_data(path_gtex)

    if by_tissue_zfpkm:
        gtex_tissue_data = rcpr_rpy2.gene_expression_by_tissue(gtex_tissue_data)
    return gtex_tissue_data


def annotate_genes(
    rcpr, rcpr_graph: ListVector, data, field_name: str, **kwargs
) -> ListVector:
    """Annotates the genes in the graph with the given gene data

    See the rcpr documentation about the exact format
    required.

    Args:
        rcpr (): The rpy2 rcpr object
        rcpr_graph (ListVector): The graph to annotate
        data (complicated): "
        field_name (str): The name of the column in the gene data to annotate with

    Returns:
        ListVector: The annotated graph
    """
    # Annotate the genes
    rcpr_graph_annot = rcpr.annotate_genes(rcpr_graph, data, field_name, **kwargs)
    return rcpr_graph_annot


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


def sbml_dfs_to_rcpr_reactions(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    identifier_ontology: str = NETCONTEXTR_ONTOLOGY,
) -> ListVector:
    """Converts an sbml_dfs to a rcpr reaction graph

    This utility converts the sbml_dfs to the format validated by
    by `rcpr::validate_netcontextr_reactions`.

    It converts the smbl_dfs into a reaction graph by:
    - Building the `reactions` dataframe:
        - Using the `species` identifiers to map `reaction_species`
            to `genes` using the `identifier_ontology`.
            Note that one species may be split into multiple `genes`
            and multiple species may be combined into a single `gene`.
        - Converting `sbo_terms` to roles.
        - renaming `r_id` to `reaction_id`
    - Building `genes` dataframe by taking all unique `genes` from the `reactions`

    Args:
        sbml_dfs (SBML_dfs): an sbml_dfs.
        identifier_ontology (str, optional): The ontology to use for the
             identifiers. Defaults to `ensembl_gene` (default in rcpr)

    Returns:
        This is a list of dataframes that validate with validate_netcontextr_reactions:
        - `genes`: a dataframe with column `gene`
        - `reactions`: a dataframe with columns "gene", "reaction_id", "role", "rsc_id"
            representing the reaction data split up into individual reactions.
    """

    # Get the reactions
    dat_reactions = _get_reactions(sbml_dfs, identifier_ontology)
    # Get the genes
    dat_gene = dat_reactions[[COL_GENE]].drop_duplicates()
    # Note that no 1:1 mapping between genes and species can be made
    # as multiple species could have the same gene annotation
    # and also even one species could have multiple gene identifiers
    genes = pandas_to_r_dataframe(dat_gene)
    reactions = pandas_to_r_dataframe(dat_reactions)

    out = ListVector({FIELD_GENES: genes, FIELD_REACTIONS: reactions})
    return out


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


def _get_reactions(
    sbml_dfs: sbml_dfs_core.SBML_dfs, identifier_ontology: str = NETCONTEXTR_ONTOLOGY
) -> pd.DataFrame:
    """Gets the reactions from the sbml_dfs"""
    dat_reaction = (
        sbml_dfs.species["s_Identifiers"]
        # Get the identifiers for the given ontology
        .map(lambda ids: ids.hoist(identifier_ontology, squeeze=False))
        .map(lambda x: x if len(x) > 0 else None)
        .dropna()
        .rename(COL_GENE)
        .to_frame()
        .explode(COL_GENE)
        # Merge with compartmentalized species to get the sc_id
        .merge(
            sbml_dfs.compartmentalized_species[["s_id"]].reset_index(drop=False),
            left_index=True,
            right_on="s_id",
        )[[COL_GENE, "sc_id"]]
        .merge(
            sbml_dfs.reaction_species[
                ["r_id", "sc_id", "sbo_term", "stoichiometry"]
            ].reset_index(drop=False),
            on="sc_id",
        )
        .assign(**{COL_ROLE: lambda x: x["sbo_term"].map(NETCONTEXTR_SBO_MAP)})
        .rename({"r_id": COL_REACTION_ID, "stoichiometry": COL_STOICHIOMETRY}, axis=1)
    )
    fil = dat_reaction[COL_ROLE].isna()
    if fil.sum() > 0:
        missing_sbo_terms = dat_reaction.loc[fil, "sbo_term"].unique()
        logger.warning(
            f"Found {fil.sum()} reactions had an sbo term that was not"
            "mappable to a rcpr role. These are ignored. "
            f"The sbo terms are: {', '.join(missing_sbo_terms)}"
        )

        dat_reaction = dat_reaction.loc[~fil, :]
    return dat_reaction[
        [COL_ROLE, COL_GENE, COL_REACTION_ID, COL_STOICHIOMETRY, "rsc_id"]
    ]
