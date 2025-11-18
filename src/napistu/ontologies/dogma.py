from __future__ import annotations

import datetime
import logging

import pandas as pd

from napistu import identifiers, sbml_dfs_core, sbml_dfs_utils, utils
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    SBML_DFS,
)
from napistu.ingestion.constants import (
    DATA_SOURCE_DESCRIPTIONS,
    DATA_SOURCES,
    INTERACTION_EDGELIST_DEFS,
)
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator
from napistu.ontologies.constants import (
    GENE_ONTOLOGIES,  # noqa: F401
    GENODEXITO_DEFS,
    INTERCONVERTIBLE_GENIC_ONTOLOGIES,
    NAME_ONTOLOGIES,
    PROTEIN_ONTOLOGIES,
)
from napistu.ontologies.genodexito import Genodexito
from napistu.source import Source

logger = logging.getLogger(__name__)


def create_dogmatic_sbml_dfs(
    organismal_species: str | OrganismalSpeciesValidator,
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
    r_paths: str | None = None,
) -> sbml_dfs_core.SBML_dfs:
    """
    Create Dogmatic SMBL_DFs

    Create an SBML_dfs model which is pretty much just proteins and no
    reactions, as well as annotations linking proteins to genes, and
    creating nice labels for genes/proteins.

    Parameters
    ----------
    organismal_species : str | OrganismalSpeciesValidator
        An organismal species (e.g., Homo sapiens)
    r_paths : str or None
        Optional, p]ath to an R packages directory

    Returns:
        dogmatic_sbml_dfs (sbml.SBML_dfs)
            A pathway model which (pretty much) just contains proteins and
            diverse identifiers
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)

    dogmatic_mappings = _connect_dogmatic_mappings(
        organismal_species.latin_name, preferred_method, allow_fallback, r_paths
    )

    logger.info("Creating inputs for SBML_dfs.from_edgelist()")

    # format entries for SBML_dfs.from_edgelist()
    species_df = dogmatic_mappings["cluster_consensus_identifiers_df"].join(
        dogmatic_mappings["s_name_series"]
    )

    # stub required but invariant variables
    compartments_df = sbml_dfs_utils.stub_compartments()

    model_source = Source.single_entry(
        model=DATA_SOURCES.DOGMA,
        pathway_id=DATA_SOURCES.DOGMA,
        data_source=DATA_SOURCES.DOGMA,
        organismal_species=organismal_species.latin_name,
        name=DATA_SOURCE_DESCRIPTIONS[DATA_SOURCES.DOGMA],
        date=datetime.date.today().strftime("%Y%m%d"),
    )

    # interactions table. This is required to create the sbml_dfs but we'll drop the info later
    interaction_edgelist = species_df.rename(
        columns={
            SBML_DFS.S_NAME: INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            SBML_DFS.S_IDENTIFIERS: SBML_DFS.R_IDENTIFIERS,
        }
    )
    interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM] = (
        interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM]
    )
    interaction_edgelist[SBML_DFS.R_NAME] = interaction_edgelist[
        INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM
    ]

    dogmatic_sbml_dfs = sbml_dfs_core.SBML_dfs.from_edgelist(
        interaction_edgelist=interaction_edgelist,
        species_df=species_df,
        compartments_df=compartments_df,
        model_source=model_source,
    )

    # remove all reactions except 1 (so it still passes sbml_dfs.validate())
    # this self reaction will be removed when creating the graph
    dogmatic_sbml_dfs.remove_entities(
        SBML_DFS.REACTIONS,
        dogmatic_sbml_dfs.reactions.index.tolist()[1::],
        remove_references=False,
    )

    return dogmatic_sbml_dfs


def _connect_dogmatic_mappings(
    organismal_species: str,
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
    r_paths: str | None = None,
) -> dict:
    """
    Connect Dogmatic Mappings

    Merge all ontologies into greedy clusters based on shared associations to entrez ids

    Parameters
    ----------
    organismal_species : str
        An organismal species (e.g., Homo sapiens)
    r_paths : str or None
        Optional, p]ath to an R packages directory

    Returns:
        dict with:
        - s_name_series: a series where the index is distinct molecular species and the values are names.
        - cluster_consensus_identifiers_df: a pd.DataFrame where the index is distinct molecular species
        and values are identifiers objects.
    """

    genodexito = Genodexito(
        organismal_species=organismal_species,
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
    # all clusters will need unique names to use SBML_dfs.from_edgelist()
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
