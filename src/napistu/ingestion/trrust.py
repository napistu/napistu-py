from __future__ import annotations

import datetime
import os
import warnings
from itertools import chain
from typing import Union

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs
import pandas as pd

from napistu import identifiers, utils
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    SBML_DFS,
    SBOTERM_NAMES,
)
from napistu.ingestion.constants import (
    DATA_SOURCE_DESCRIPTIONS,
    DATA_SOURCES,
    INTERACTION_EDGELIST_DEFS,
    LATIN_SPECIES_NAMES,
    TRRUST_COMPARTMENT_NUCLEOPLASM,
    TRRUST_COMPARTMENT_NUCLEOPLASM_GO_ID,
    TRRUST_SIGNS,
    TRRUST_SYMBOL,
    TRRUST_UNIPROT,
    TRRUST_UNIPROT_ID,
    TTRUST_URL_RAW_DATA_HUMAN,
)
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator
from napistu.network.constants import NAPISTU_GRAPH_EDGES
from napistu.ontologies.constants import GENODEXITO_DEFS
from napistu.ontologies.genodexito import Genodexito
from napistu.sbml_dfs_core import SBML_dfs
from napistu.source import Source


def download_trrust(target_uri: str) -> None:
    """Downloads trrust to the target uri

    Parameters
    ----------
    target_uri : str
        target url

    Returns
    -------
    None
    """
    utils.download_wget(TTRUST_URL_RAW_DATA_HUMAN, target_uri)

    return None


def convert_trrust_to_sbml_dfs(
    trrust_uri: str,
    organismal_species: Union[
        str, OrganismalSpeciesValidator
    ] = LATIN_SPECIES_NAMES.HOMO_SAPIENS,
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
) -> SBML_dfs:
    """Ingests trrust to sbml dfs

    Parameters
    ----------
    trrust_uri : str
        trrust uri
    organismal_species : str | OrganismalSpeciesValidator
        organismal species
    preferred_method : str
        preferred method
    allow_fallback : bool
        allow fallback

    Returns
    -------
    SBML_dfs
        sbml dfs
    """

    # Read trrust raw data
    trrust_edgelist = _read_trrust(trrust_uri)

    # validate species; not really needed since only the human URI is tracked but
    # mouse could be added and this creates a relatively consistent interface
    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)
    organismal_species.assert_supported(
        supported_species=[LATIN_SPECIES_NAMES.HOMO_SAPIENS]
    )

    # Get uniprot to symbol mapping
    uniprot_2_symbol = _get_uniprot_2_symbol_mapping(
        organismal_species, preferred_method, allow_fallback
    )

    # Start building new sbml dfs
    # define model-level metadata
    model_source = Source.single_entry(
        model=DATA_SOURCES.TRRUST,
        pathway_id=DATA_SOURCES.TRRUST,
        data_source=DATA_SOURCES.TRRUST,
        organismal_species=organismal_species.latin_name,
        name=DATA_SOURCE_DESCRIPTIONS[DATA_SOURCES.TRRUST],
        date=datetime.date.today().strftime("%Y%m%d"),
    )

    # Summarize edges

    edge_summaries_df = (
        trrust_edgelist.groupby(["from", "to"], as_index=True)
        .apply(_summarize_trrust_pairs)
        .reset_index(drop=False)
    )

    # define distinct species
    species_df = (
        pd.DataFrame(
            {
                SBML_DFS.S_NAME: list(
                    {*edge_summaries_df["from"], *edge_summaries_df["to"]}
                )
            }
        )
        .merge(
            uniprot_2_symbol.rename({TRRUST_SYMBOL: SBML_DFS.S_NAME}, axis=1),
            how="left",
        )
        .set_index(SBML_DFS.S_NAME)
    )

    # create Identifiers objects for all species with uniprot IDs
    species_w_ids = species_df[~species_df[TRRUST_UNIPROT_ID].isnull()].sort_index()

    species_w_ids[IDENTIFIERS.URL] = [
        identifiers.create_uri_url(ontology=TRRUST_UNIPROT, identifier=x)
        for x in species_w_ids[TRRUST_UNIPROT_ID]
    ]

    # create a series where each row is a gene with 1+ uniprot ids and the value is an
    # identifiers objects with all uniprot ids
    species_w_ids_series = pd.Series(
        [
            identifiers.Identifiers(
                [
                    identifiers.format_uri(uri=x, bqb=BQB.IS)
                    for x in species_w_ids.loc[[ind]][IDENTIFIERS.URL].tolist()
                ]
            )
            for ind in species_w_ids.index.unique()
        ],
        index=species_w_ids.index.unique(),
    ).rename(SBML_DFS.S_IDENTIFIERS)

    # just retain s_name and s_Identifiers
    # this just needs a source object which will be added later
    species_df = (
        species_df.reset_index()
        .drop(TRRUST_UNIPROT_ID, axis=1)
        .drop_duplicates()
        .merge(
            species_w_ids_series,
            how="left",
            left_on=SBML_DFS.S_NAME,
            right_index=True,
        )
        .reset_index(drop=True)
    )
    # stub genes with missing IDs
    species_df[SBML_DFS.S_IDENTIFIERS] = species_df[SBML_DFS.S_IDENTIFIERS].fillna(  # type: ignore
        value=identifiers.Identifiers([])
    )

    # define distinct compartments
    compartments_df = pd.DataFrame(
        {
            SBML_DFS.C_NAME: TRRUST_COMPARTMENT_NUCLEOPLASM,
            SBML_DFS.C_IDENTIFIERS: identifiers.Identifiers(
                [
                    identifiers.format_uri(
                        uri=identifiers.create_uri_url(
                            ontology=ONTOLOGIES.GO,
                            identifier=TRRUST_COMPARTMENT_NUCLEOPLASM_GO_ID,
                        ),
                        bqb=BQB.IS,
                    )
                ]
            ),
        },
        index=[0],
    )

    gene_gene_identifier_edgelist = edge_summaries_df.rename(
        {
            NAPISTU_GRAPH_EDGES.FROM: INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            NAPISTU_GRAPH_EDGES.TO: INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
        },
        axis=1,
    ).assign(
        upstream_compartment=TRRUST_COMPARTMENT_NUCLEOPLASM,
        downstream_compartment=TRRUST_COMPARTMENT_NUCLEOPLASM,
    )
    gene_gene_identifier_edgelist[SBML_DFS.R_NAME] = [
        f"{x} {y} of {z}"
        for x, y, z in zip(
            gene_gene_identifier_edgelist[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM],
            gene_gene_identifier_edgelist["sign"],
            gene_gene_identifier_edgelist[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM],
        )
    ]

    # convert relationships to SBO terms
    interaction_edgelist = gene_gene_identifier_edgelist.rename(
        {"sign": INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM}, axis=1
    )

    # format pubmed identifiers of interactions
    interaction_edgelist[SBML_DFS.R_IDENTIFIERS] = [
        _format_pubmed_for_interactions(x) for x in interaction_edgelist["reference"]
    ]

    # directionality: by default, set r_isreversible to False for TRRUST data
    interaction_edgelist[SBML_DFS.R_ISREVERSIBLE] = False

    # reduce to essential variables
    interaction_edgelist = interaction_edgelist[
        [
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM,
            SBML_DFS.R_NAME,
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM,
            SBML_DFS.R_IDENTIFIERS,
            SBML_DFS.R_ISREVERSIBLE,
        ]
    ]

    # Build sbml dfs
    sbml_dfs = SBML_dfs.from_edgelist(
        interaction_edgelist=interaction_edgelist,
        species_df=species_df,
        compartments_df=compartments_df,
        model_source=model_source,
    )
    sbml_dfs.validate()
    return sbml_dfs


# utility functions


def _format_pubmed_for_interactions(pubmed_set):
    """Format a set of pubmed ids as an Identifiers object."""

    ids = list()
    for p in pubmed_set:
        # some pubmed IDs are bogus
        url = identifiers.create_uri_url(
            ontology=ONTOLOGIES.PUBMED, identifier=p, strict=False
        )
        if url is not None:
            valid_url = identifiers.format_uri(uri=url, bqb=BQB.IS_DESCRIBED_BY)

            ids.append(valid_url)

    return identifiers.Identifiers(ids)


def _get_uniprot_2_symbol_mapping(
    organismal_species: Union[
        str, OrganismalSpeciesValidator
    ] = LATIN_SPECIES_NAMES.HOMO_SAPIENS,
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """Create a mapping from Uniprot IDs to gene symbols using Genodexito.

    Parameters
    ----------
    species : str, optional
        The species to create mappings for, by default LATIN_SPECIES_NAMES.HOMO_SAPIENS
    preferred_method : str, optional
        Preferred method for identifier mapping, by default GENODEXITO_DEFS.BIOCONDUCTOR
    allow_fallback : bool, optional
        Whether to allow fallback to alternative mapping methods, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for gene symbol and UniProt ID mappings
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)

    genodexito = Genodexito(
        organismal_species=organismal_species.latin_name,
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    )

    # Create mapping tables for the required ontologies
    genodexito.create_mapping_tables(
        mappings={ONTOLOGIES.SYMBOL, ONTOLOGIES.UNIPROT, ONTOLOGIES.NCBI_ENTREZ_GENE}
    )

    # Merge mappings to create a combined table
    genodexito.merge_mappings(
        [ONTOLOGIES.SYMBOL, ONTOLOGIES.UNIPROT, ONTOLOGIES.NCBI_ENTREZ_GENE]
    )

    # Rename columns to match the original function's output format
    uniprot_2_symbol = (
        genodexito.merged_mappings.drop(ONTOLOGIES.NCBI_ENTREZ_GENE, axis=1)
        .dropna()
        .rename(
            columns={
                ONTOLOGIES.SYMBOL: TRRUST_SYMBOL,
                ONTOLOGIES.UNIPROT: TRRUST_UNIPROT_ID,
            }
        )
    )

    return uniprot_2_symbol


def _read_trrust(trrust_uri: str) -> pd.DataFrame:
    """Read trrust csv

    Parameters
    ----------
    trrust_uri : str
        uri to the trrust csv

    Returns
    -------
    pd.DataFrame
        Data Frame
    """

    base_path = os.path.dirname(trrust_uri)
    file_name = os.path.basename(trrust_uri)
    with open_fs(base_path) as base_fs:
        with base_fs.open(file_name) as f:
            trrust_edgelist = pd.read_csv(
                f, sep="\t", names=["from", "to", "sign", "reference"]
            ).drop_duplicates()
    return trrust_edgelist


def _summarize_trrust_pairs(pair_data: pd.DataFrame) -> pd.Series:
    """Summarize a TF->target relationship based on the sign and source of the interaction."""

    signs = set(pair_data["sign"].tolist())
    if (TRRUST_SIGNS.ACTIVATION in signs) and (TRRUST_SIGNS.REPRESSION in signs):
        sign = SBOTERM_NAMES.MODIFIER
    elif TRRUST_SIGNS.ACTIVATION in signs:
        sign = SBOTERM_NAMES.STIMULATOR
    elif TRRUST_SIGNS.REPRESSION in signs:
        sign = SBOTERM_NAMES.INHIBITOR
    else:
        sign = SBOTERM_NAMES.MODIFIER

    refs = set(chain(*[x.split(";") for x in pair_data["reference"]]))
    return pd.Series({"sign": sign, "reference": refs})
