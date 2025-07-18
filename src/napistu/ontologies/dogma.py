from __future__ import annotations

import logging
import pandas as pd

from napistu import sbml_dfs_core
from napistu import sbml_dfs_utils
from napistu import source
from napistu import identifiers
from napistu import utils
from napistu.ontologies.genodexito import Genodexito
from napistu.constants import (
    BQB,
    SBOTERM_NAMES,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    ONTOLOGIES,
    SBML_DFS,
)
from napistu.ontologies.constants import (
    INTERCONVERTIBLE_GENIC_ONTOLOGIES,
    GENE_ONTOLOGIES,  # noqa: F401
    GENODEXITO_DEFS,
    NAME_ONTOLOGIES,
    PROTEIN_ONTOLOGIES,
)

logger = logging.getLogger(__name__)


def create_dogmatic_sbml_dfs(
    species: str,
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
    r_paths: str | None = None,
) -> sbml_dfs_core.SBML_dfs:
    """
    Create Dogmatic SMBL_DFs

    Create an SBML_dfs model which is pretty much just proteins and no
    reactions, as well as annotations linking proteins to genes, and
    creating nice labels for genes/proteins.

    Args:
        species (str):
            An organismal species (e.g., Homo sapiens)
        r_paths (str or None)
            Optional, p]ath to an R packages directory

    Returns:
        dogmatic_sbml_dfs (sbml.SBML_dfs)
            A pathway model which (pretty much) just contains proteins and
            diverse identifiers
    """

    dogmatic_mappings = _connect_dogmatic_mappings(
        species, preferred_method, allow_fallback, r_paths
    )

    logger.info("Creating inputs for sbml_dfs_from_edgelist()")

    # format entries for sbml_dfs_from_edgelist()
    species_df = dogmatic_mappings["cluster_consensus_identifiers_df"].join(
        dogmatic_mappings["s_name_series"]
    )

    # stub required but invariant variables
    compartments_df = sbml_dfs_utils.stub_compartments()
    interaction_source = source.Source(init=True)

    # interactions table. This is required to create the sbml_dfs but we'll drop the info later
    interaction_edgelist = species_df.rename(
        columns={
            "s_name": "upstream_name",
            SBML_DFS.S_IDENTIFIERS: SBML_DFS.R_IDENTIFIERS,
        }
    )
    interaction_edgelist["downstream_name"] = interaction_edgelist["upstream_name"]
    interaction_edgelist["upstream_compartment"] = "cellular_component"
    interaction_edgelist["downstream_compartment"] = "cellular_component"
    interaction_edgelist["r_name"] = interaction_edgelist["upstream_name"]
    interaction_edgelist["sbo_term"] = MINI_SBO_FROM_NAME["reactant"]
    interaction_edgelist["r_isreversible"] = False

    dogmatic_sbml_dfs = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist=interaction_edgelist,
        species_df=species_df,
        compartments_df=compartments_df,
        interaction_source=interaction_source,
        upstream_stoichiometry=-1,
        downstream_stoichiometry=1,
        downstream_sbo_name=SBOTERM_NAMES.PRODUCT,
    )

    # remove all reactions except 1 (so it still passes sbml_dfs.validate())
    # this self reaction will be removed when creating the graph
    dogmatic_sbml_dfs.remove_reactions(dogmatic_sbml_dfs.reactions.index.tolist()[1::])

    return dogmatic_sbml_dfs


def _connect_dogmatic_mappings(
    species: str,
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
    r_paths: str | None = None,
) -> dict:
    """
    Connect Dogmatic Mappings

    Merge all ontologies into greedy clusters based on shared associations to entrez ids

    Args:
        species (str):
            An organismal species (e.g., Homo sapiens)
        r_paths (str or None)
            Optional, p]ath to an R packages directory

    Returns:
        dict with:
        - s_name_series: a series where the index is distinct molecular species and the values are names.
        - cluster_consensus_identifiers_df: a pd.DataFrame where the index is distinct molecular species
        and values are identifiers objects.
    """

    genodexito = Genodexito(
        species=species,
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
        r_paths=r_paths,
    )

    genodexito.create_mapping_tables(mappings=INTERCONVERTIBLE_GENIC_ONTOLOGIES)

    genodexito.stack_mappings(ontologies=set(PROTEIN_ONTOLOGIES))
    protein_mappings = genodexito.stacked_mappings

    # apply greedy graph-based clustering to connect proteins with a common mapping to entrez
    edgelist_df = utils.format_identifiers_as_edgelist(
        protein_mappings, [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]
    )
    connected_indices = utils.find_weakly_connected_subgraphs(
        edgelist_df[["ind", "id"]]
    )

    # add clusters to proteins. Each cluster will be a distinct molecular species
    protein_mappings_w_clusters = protein_mappings.reset_index().merge(
        connected_indices
    )

    # combine entrez + cluster so we can pass cluster to non-protein attributes
    entrez_clusters = protein_mappings_w_clusters[
        [ONTOLOGIES.NCBI_ENTREZ_GENE, "cluster"]
    ].drop_duplicates()
    # check for the other ontologies aside from proteins and entrez (since that's in the index)
    other_ontologies = INTERCONVERTIBLE_GENIC_ONTOLOGIES.difference(
        set(PROTEIN_ONTOLOGIES)
    )
    other_ontologies.discard(ONTOLOGIES.NCBI_ENTREZ_GENE)

    genodexito.stack_mappings(ontologies=other_ontologies)
    other_mappings = genodexito.stacked_mappings

    other_mappings_w_clusters = entrez_clusters.merge(
        other_mappings, left_on=ONTOLOGIES.NCBI_ENTREZ_GENE, right_index=True
    )

    possible_names = pd.concat(
        [
            protein_mappings_w_clusters.query("ontology in @NAME_ONTOLOGIES.keys()"),
            other_mappings_w_clusters.query("ontology in @NAME_ONTOLOGIES.keys()"),
        ]
    )[["cluster", IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]]

    possible_names.loc[:, "ontology_preference"] = possible_names[
        IDENTIFIERS.ONTOLOGY
    ].map(NAME_ONTOLOGIES)

    # remove possible names which are present in multiple clusters.
    # all clusters will need unique names to use sbml_dfs_from_edgelist()
    id_counts = (
        possible_names[["cluster", IDENTIFIERS.IDENTIFIER]]
        .drop_duplicates()
        .value_counts(IDENTIFIERS.IDENTIFIER)
    )
    possible_names = possible_names[
        ~possible_names[IDENTIFIERS.IDENTIFIER].isin(
            id_counts[id_counts > 1].index.tolist()
        )
    ]

    s_name_series = (
        utils._add_nameness_score(possible_names, IDENTIFIERS.IDENTIFIER)
        .sort_values(["ontology_preference", "nameness_score"])
        .groupby("cluster")
        .first()
        .rename(columns={IDENTIFIERS.IDENTIFIER: SBML_DFS.S_NAME})[SBML_DFS.S_NAME]
    )

    protein_ids = protein_mappings_w_clusters.assign(bqb=BQB.IS)[
        ["cluster", IDENTIFIERS.IDENTIFIER, IDENTIFIERS.ONTOLOGY, IDENTIFIERS.BQB]
    ]
    gene_ids = other_mappings_w_clusters.query("ontology in @GENE_ONTOLOGIES").assign(
        bqb=BQB.IS_ENCODED_BY
    )[["cluster", IDENTIFIERS.IDENTIFIER, IDENTIFIERS.ONTOLOGY, IDENTIFIERS.BQB]]
    entrez_ids = entrez_clusters.assign(
        ontology=ONTOLOGIES.NCBI_ENTREZ_GENE, bqb=BQB.IS_ENCODED_BY
    ).rename(columns={ONTOLOGIES.NCBI_ENTREZ_GENE: IDENTIFIERS.IDENTIFIER})[
        ["cluster", IDENTIFIERS.IDENTIFIER, IDENTIFIERS.ONTOLOGY, IDENTIFIERS.BQB]
    ]

    # combine all ids to setup a single cluster-level Identifiers
    all_ids = pd.concat([protein_ids, gene_ids, entrez_ids])
    all_ids.loc[:, IDENTIFIERS.URL] = [
        identifiers.create_uri_url(x, y)
        for x, y in zip(all_ids[IDENTIFIERS.ONTOLOGY], all_ids[IDENTIFIERS.IDENTIFIER])
    ]

    # create one Identifiers object for each new species
    cluster_consensus_identifiers = {
        k: identifiers.Identifiers(
            list(
                v[
                    [
                        IDENTIFIERS.ONTOLOGY,
                        IDENTIFIERS.IDENTIFIER,
                        IDENTIFIERS.URL,
                        IDENTIFIERS.BQB,
                    ]
                ]
                .reset_index(drop=True)
                .T.to_dict()
                .values()
            )
        )
        for k, v in all_ids.groupby("cluster")
    }

    cluster_consensus_identifiers_df = pd.DataFrame(
        cluster_consensus_identifiers, index=[SBML_DFS.S_IDENTIFIERS]
    ).T
    cluster_consensus_identifiers_df.index.name = "cluster"

    out_dict = {
        "s_name_series": s_name_series,
        "cluster_consensus_identifiers_df": cluster_consensus_identifiers_df,
    }

    return out_dict
