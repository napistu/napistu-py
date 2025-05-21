from __future__ import annotations

import logging

import pandas as pd
from napistu import constants
from napistu import identifiers
from napistu import sbml_dfs_core
from napistu import source
from napistu import utils
from napistu.rpy2 import callr
from napistu.rpy2 import report_r_exceptions
from napistu.rpy2 import warn_if_no_rpy2

from napistu.constants import SBML_DFS
from napistu.constants import BQB
from napistu.constants import IDENTIFIERS
from napistu.constants import ONTOLOGIES
from napistu.constants import ONTOLOGY_ALIASES
from napistu.rpy2.constants import BIOC_VALID_EXPANDED_SPECIES_ONTOLOGIES
from napistu.rpy2.constants import BIOC_DOGMATIC_MAPPING_ONTOLOGIES
from napistu.rpy2.constants import BIOC_PROTEIN_ONTOLOGIES
from napistu.rpy2.constants import BIOC_NAME_ONTOLOGIES
from napistu.rpy2.constants import BIOC_GENE_ONTOLOGIES  # noqa
from napistu.rpy2.constants import BIOC_NOMENCLATURE

logger = logging.getLogger(__name__)


@warn_if_no_rpy2
@report_r_exceptions
def expand_identifiers(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    id_type: str,
    species: str,
    expanded_ontologies: list[str],
    r_paths: str | None = None,
) -> pd.Series:
    """
    Expand Identifiers

    Update a table's identifiers to include additional related ontologies

    Ontologies are pulled from the bioconductor "org" packages. This is effective, but inelegant.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        A relational pathway model built around reactions interconverting compartmentalized species.
    id_type: str
        Identifiers to expand: species, compartments, or reactions
    species: str
        Species name
    expanded_ontologies: list
        Ontologies to add or complete
    r_paths: str
        Path to an R packages directory

    Returns
    -------
    a pd.Series with identifiers as the index and updated Identifiers objects as values
    """

    if not isinstance(sbml_dfs, sbml_dfs_core.SBML_dfs):
        raise TypeError("sbml_dfs is not an sbml_dfs_core.SBML_dfs object")

    # pull out all identifiers as a pd.DataFrame
    all_entity_identifiers = sbml_dfs.get_identifiers(id_type)
    if not isinstance(all_entity_identifiers, pd.DataFrame):
        raise TypeError("all_entity_identifiers must be a pandas DataFrame")

    if id_type == "species":
        all_entity_identifiers = _check_species_identifiers_entrez_gene_ontology(
            all_entity_identifiers
        )

        valid_expanded_ontologies = BIOC_VALID_EXPANDED_SPECIES_ONTOLOGIES
    elif id_type in ["reactions", "compartments"]:
        raise NotImplementedError(
            f"No converters implemented to expand {id_type} annotations"
        )
    else:
        raise ValueError(f"{id_type} is an invalid id_type")

    invalid_expanded_ontologies = set(expanded_ontologies).difference(
        valid_expanded_ontologies
    )

    if len(invalid_expanded_ontologies) != 0:
        raise NotImplementedError(
            f"No converters implemented to expand {id_type} annotations to {', '.join(invalid_expanded_ontologies)}"
        )

    # find entries in valid_expanded_ontologies which are already present
    # these are the entries that will be used to expand to other ontologies
    # or fill in ontologies with incomplete annotations
    starting_ontologies = valid_expanded_ontologies.intersection(
        set(all_entity_identifiers["ontology"])
    )

    if len(starting_ontologies) == 0:
        raise ValueError(f"No ontologies with {id_type} converters are present")

    required_conversion_ontologies = set(starting_ontologies).union(
        set(expanded_ontologies)
    )

    # pull down entrez ids + mapping to other ontologies
    mapping_ontologies = required_conversion_ontologies.intersection(
        BIOC_VALID_EXPANDED_SPECIES_ONTOLOGIES
    )

    mappings_dict = create_bioconductor_mapping_tables(
        mappings=mapping_ontologies, species=species, r_paths=r_paths
    )

    # start with entrez IDs (since all other ontologies are mapped to them in the
    # bioconductor "org" packages)

    # get these values by just looking up the mappings between entrez genes and genomic loci
    running_ids = merge_bioconductor_mappings(mappings_dict, mapping_ontologies)

    # map from existing ontologies to expanded ontologies
    ontology_mappings = list()
    # starting w/
    for start in starting_ontologies:
        # ending w/
        for end in expanded_ontologies:
            if start == end:
                continue
            lookup = (
                running_ids[[start, end]]
                .rename(columns={start: IDENTIFIERS.IDENTIFIER, end: "new_identifier"})
                .assign(ontology=start)
                .assign(new_ontology=end)
            )
            ontology_mappings.append(lookup)

    ontology_mappings_df = pd.concat(ontology_mappings).dropna()

    # old identifiers joined with new identifiers

    # first, define the names of keys and ids
    table_pk_var = sbml_dfs.schema[id_type]["pk"]
    table_id_var = sbml_dfs.schema[id_type]["id"]

    # retain bqb terms to define how an identifier is related to sid
    # this relation will be preserved for the new ids

    merged_identifiers = all_entity_identifiers[
        [
            table_pk_var,
            IDENTIFIERS.ONTOLOGY,
            IDENTIFIERS.IDENTIFIER,
            IDENTIFIERS.BQB,
        ]
    ].merge(ontology_mappings_df)

    # new, possibly redundant identifiers
    new_identifiers = merged_identifiers[
        [table_pk_var, "new_ontology", "new_identifier", IDENTIFIERS.BQB]
    ].rename(
        columns={
            "new_ontology": IDENTIFIERS.ONTOLOGY,
            "new_identifier": IDENTIFIERS.IDENTIFIER,
        }
    )

    expanded_identifiers_df = (
        pd.concat(
            [
                all_entity_identifiers[
                    [
                        table_pk_var,
                        IDENTIFIERS.ONTOLOGY,
                        IDENTIFIERS.IDENTIFIER,
                        IDENTIFIERS.URL,
                        IDENTIFIERS.BQB,
                    ]
                ],
                new_identifiers,
                # ignore new identifier if it already exists
            ]
        )
        # remove duplicated identifiers
        .groupby([table_pk_var, IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER])
        .first()
        .reset_index()
        .set_index(table_pk_var)
    )

    # create a dictionary of new Identifiers objects
    expanded_identifiers_dict = {
        i: _expand_identifiers_new_entries(i, expanded_identifiers_df)
        for i in expanded_identifiers_df.index.unique()
    }

    output = pd.Series(expanded_identifiers_dict).rename(table_id_var)
    output.index.name = table_pk_var

    return output


@warn_if_no_rpy2
@report_r_exceptions
def create_bioconductor_mapping_tables(
    mappings: set[str], species: str, r_paths: str | None = None
) -> dict[str, pd.DataFrame]:
    """
    Create Bioconductor Mapping Tables

    Creating a dictionary of mappings between entrez and other ontologies.

    Args:
        mappings (set):
            A set of ontologies to work with. The valid ontologies are:
            "ensembl_gene", "ensembl_transcript", and "uniprot".
        species (str):
            The organismal species that we are working with (e.g., Homo sapiens).
        r_paths (str, optional):
            Optional path to a library of R packages.

    Returns:
        mappings_dict (dict):
            A table of entrez ids, and tables mapping from each ontology in "mappings" to entrez.

    """

    if not isinstance(mappings, set):
        raise TypeError(f"mappings must be a set, but got {type(mappings).__name__}")
    if not isinstance(species, str):
        raise TypeError(f"species must be a str, but got {type(species).__name__}")

    logger.info(
        f"Creating mapping tables from entrez genes to/from {', '.join(mappings)}"
    )

    invalid_mappings = set(mappings).difference(BIOC_VALID_EXPANDED_SPECIES_ONTOLOGIES)

    if len(invalid_mappings) > 0:
        raise ValueError(
            f"{len(invalid_mappings)} mappings could not be created: {', '.join(invalid_mappings)}.\n"
            f"The valid mappings are {', '.join(BIOC_VALID_EXPANDED_SPECIES_ONTOLOGIES)}"
        )

    mappings_dict = dict()

    # all mappings are with respect to entrez. so we will always want to obtain entrez ids
    mappings_dict[ONTOLOGIES.NCBI_ENTREZ_GENE] = (
        callr.r_dataframe_to_pandas(
            callr.bioconductor_org_r_function(
                BIOC_NOMENCLATURE.CHR_TBL, species, r_paths=None
            )
        )
        .drop(BIOC_NOMENCLATURE.CHROMOSOME, axis=1)
        .rename(
            columns={BIOC_NOMENCLATURE.NCBI_ENTREZ_GENE: ONTOLOGIES.NCBI_ENTREZ_GENE}
        )
        .set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)
    )

    if ONTOLOGIES.ENSEMBL_GENE in mappings:
        # "entrez <> ensembl genes"
        mappings_dict[ONTOLOGIES.ENSEMBL_GENE] = (
            callr.r_dataframe_to_pandas(
                callr.bioconductor_org_r_function(
                    BIOC_NOMENCLATURE.ENSG_TBL, species, r_paths=r_paths
                )
            )
            .rename(
                columns={
                    BIOC_NOMENCLATURE.NCBI_ENTREZ_GENE: ONTOLOGIES.NCBI_ENTREZ_GENE,
                    BIOC_NOMENCLATURE.ENSEMBL_GENE: ONTOLOGIES.ENSEMBL_GENE,
                }
            )
            .set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)
        )

    if ONTOLOGIES.ENSEMBL_TRANSCRIPT in mappings:
        # "entrez <> ensembl transcripts"
        mappings_dict[ONTOLOGIES.ENSEMBL_TRANSCRIPT] = (
            callr.r_dataframe_to_pandas(
                callr.bioconductor_org_r_function(
                    BIOC_NOMENCLATURE.ENST_TBL, species, r_paths=r_paths
                )
            )
            .rename(
                columns={
                    BIOC_NOMENCLATURE.NCBI_ENTREZ_GENE: ONTOLOGIES.NCBI_ENTREZ_GENE,
                    BIOC_NOMENCLATURE.ENSEMBL_TRANSCRIPT: ONTOLOGIES.ENSEMBL_TRANSCRIPT,
                }
            )
            .set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)
        )

    if ONTOLOGIES.ENSEMBL_PROTEIN in mappings:
        # "entrez <> ensembl proteins"
        mappings_dict[ONTOLOGIES.ENSEMBL_PROTEIN] = (
            callr.r_dataframe_to_pandas(
                callr.bioconductor_org_r_function(
                    BIOC_NOMENCLATURE.ENSP_TBL, species, r_paths=r_paths
                )
            )
            .rename(
                columns={
                    BIOC_NOMENCLATURE.NCBI_ENTREZ_GENE: ONTOLOGIES.NCBI_ENTREZ_GENE,
                    BIOC_NOMENCLATURE.ENSEMBL_PROTEIN: ONTOLOGIES.ENSEMBL_PROTEIN,
                }
            )
            .set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)
        )

    if ONTOLOGIES.UNIPROT in mappings:
        # "entrez <> uniprot"
        mappings_dict[ONTOLOGIES.UNIPROT] = (
            callr.r_dataframe_to_pandas(
                callr.bioconductor_org_r_function(
                    BIOC_NOMENCLATURE.UNIPROT_TBL, species, r_paths=r_paths
                )
            )
            .rename(
                columns={
                    BIOC_NOMENCLATURE.NCBI_ENTREZ_GENE: ONTOLOGIES.NCBI_ENTREZ_GENE,
                    BIOC_NOMENCLATURE.UNIPROT: ONTOLOGIES.UNIPROT,
                }
            )
            .set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)
        )

    if ONTOLOGIES.GENE_NAME in mappings:
        # "entrez <> gene name"
        mappings_dict[ONTOLOGIES.GENE_NAME] = (
            callr.r_dataframe_to_pandas(
                callr.bioconductor_org_r_function(
                    BIOC_NOMENCLATURE.NAME_TBL, species, r_paths=r_paths
                )
            )
            .rename(
                columns={
                    BIOC_NOMENCLATURE.NCBI_ENTREZ_GENE: ONTOLOGIES.NCBI_ENTREZ_GENE,
                    BIOC_NOMENCLATURE.GENE_NAME: ONTOLOGIES.GENE_NAME,
                }
            )
            .set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)
        )

    if ONTOLOGIES.SYMBOL in mappings:
        # "entrez <> gene symbol"
        mappings_dict[ONTOLOGIES.SYMBOL] = (
            callr.r_dataframe_to_pandas(
                callr.bioconductor_org_r_function(
                    BIOC_NOMENCLATURE.SYMBOL_TBL, species, r_paths=r_paths
                )
            )
            .rename(
                columns={
                    BIOC_NOMENCLATURE.NCBI_ENTREZ_GENE: ONTOLOGIES.NCBI_ENTREZ_GENE,
                    BIOC_NOMENCLATURE.SYMBOL: ONTOLOGIES.SYMBOL,
                }
            )
            .set_index(ONTOLOGIES.NCBI_ENTREZ_GENE)
        )

    return mappings_dict


def merge_bioconductor_mappings(
    mappings_dict: dict, mapping_ontologies: set[str]
) -> pd.DataFrame:
    """Combine multiple ontologies by recursively joining on Entrez Gene"""

    running_ids = mappings_dict[ONTOLOGIES.NCBI_ENTREZ_GENE]

    for mapping in mapping_ontologies:
        logger.debug(f"adding entries for {mapping} to running_ids")
        mapping_df = mappings_dict[mapping]

        running_ids = running_ids.join(mapping_df)

    running_ids = running_ids.reset_index()

    return running_ids


def stack_bioconductor_mappings(
    mappings_dict: dict[str, pd.DataFrame], mapping_ontologies: set[str]
) -> pd.DataFrame:
    """
    Stack Bioconductor Mappings

    Convert a dict of mappings between entrez identifiers and other identifiers to a single table.

    Args:
        mappings_dict (dict):
            A dictionary containing mappings between entrez and other ontologies.
        mapping_ontologies (set):
            A set of mappings to combine.

    Returns:
        mappings_df (pd.DataFrame):
            A table containing entrez_gene_id, ontology, and identifier.
    """

    mappings_list = list()
    for ont in mapping_ontologies:
        one_mapping_df = (
            mappings_dict[ont].assign(ontology=ont).rename({ont: "identifier"}, axis=1)
        )

        mappings_list.append(one_mapping_df)

    return pd.concat(mappings_list)


def _check_species_identifiers_entrez_gene_ontology(
    entity_identifiers_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check whether species ontologies contain ncbigene or ncbi_gene
    If so, replaced them to ncbi_entrez_gene.
    Return: entity_identifiers_df with proper gene ontology types.
    """

    intersect_gene_onto = set(entity_identifiers_df["ontology"]).intersection(
        ONTOLOGY_ALIASES.NCBI_ENTREZ_GENE
    )

    # if entity_identifiers_df contains members of ENTREZ_ONTOLOGY_ALIASES,
    # replace to ncbi_entrez_gene
    if intersect_gene_onto:
        logger.info(
            f" Replace unmatching ontology {', '.join(intersect_gene_onto)} to {ONTOLOGIES.NCBI_ENTREZ_GENE}."
        )

        filtered_onto_df = entity_identifiers_df[
            entity_identifiers_df["ontology"].isin(list(intersect_gene_onto))
        ]

        entity_identifiers_df.loc[filtered_onto_df.index, "ontology"] = (
            ONTOLOGIES.NCBI_ENTREZ_GENE
        )

    return entity_identifiers_df


def update_expanded_identifiers(
    model: sbml_dfs_core.SBML_dfs, id_type: str, expanded_ids: pd.Series
) -> sbml_dfs_core.SBML_dfs:
    """Update the expanded identifiers for a model.

    Args:
        model (sbml_dfs_core.SBML_dfs): _description_
        id_type (str): _description_
        expanded_ids (str): _description_
    """
    ids = getattr(model, id_type)

    # make sure expanded_ids and original model.species have same number of s_ids
    # if a s_id only in model.species, adding it to expanded_ids.
    if ids.shape[0] != expanded_ids.shape[0]:
        matched_expanded_ids = expanded_ids.combine_first(ids[SBML_DFS.S_IDENTIFIERS])
        logger.debug(
            f"{ids.shape[0] - expanded_ids.shape[0]} "
            "ids are not included in expanded ids"
        )
    else:
        matched_expanded_ids = expanded_ids

    updated_ids = ids.drop(SBML_DFS.S_IDENTIFIERS, axis=1).join(
        pd.DataFrame(matched_expanded_ids)
    )

    setattr(model, id_type, updated_ids)

    return model


def create_dogmatic_sbml_dfs(
    species: str, r_paths: str | None = None
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

    dogmatic_mappings = connect_dogmatic_mappings(species)

    logger.info("Creating inputs for sbml_dfs_from_edgelist()")

    # format entries for sbml_dfs_from_edgelist()
    species_df = dogmatic_mappings["cluster_consensus_identifiers_df"].join(
        dogmatic_mappings["s_name_series"]
    )

    # stub required but invariant variables
    compartments_df = sbml_dfs_core._stub_compartments()
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
    interaction_edgelist["sbo_term"] = constants.MINI_SBO_FROM_NAME["reactant"]
    interaction_edgelist["r_isreversible"] = False

    dogmatic_sbml_dfs = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist=interaction_edgelist,
        species_df=species_df,
        compartments_df=compartments_df,
        interaction_source=interaction_source,
        upstream_stoichiometry=-1,
        downstream_stoichiometry=1,
        downstream_sbo_name="product",
    )

    # remove all reactions except 1 (so it still passes sbml_dfs.validate())
    # this self reaction will be removed when creating the graph
    dogmatic_sbml_dfs.remove_reactions(dogmatic_sbml_dfs.reactions.index.tolist()[1::])

    return dogmatic_sbml_dfs


def connect_dogmatic_mappings(species: str, r_paths: str | None = None) -> dict:
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

    mappings_dict = create_bioconductor_mapping_tables(
        mappings=BIOC_DOGMATIC_MAPPING_ONTOLOGIES,
        species=species,
        r_paths=r_paths,
    )

    protein_mappings = stack_bioconductor_mappings(
        mappings_dict, set(BIOC_PROTEIN_ONTOLOGIES)
    )

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
    other_ontologies = BIOC_DOGMATIC_MAPPING_ONTOLOGIES.difference(
        set(BIOC_PROTEIN_ONTOLOGIES)
    )
    other_mappings = stack_bioconductor_mappings(mappings_dict, other_ontologies)
    other_mappings_w_clusters = entrez_clusters.merge(
        other_mappings, left_on=ONTOLOGIES.NCBI_ENTREZ_GENE, right_index=True
    )

    possible_names = pd.concat(
        [
            protein_mappings_w_clusters.query(
                "ontology in @BIOC_NAME_ONTOLOGIES.keys()"
            ),
            other_mappings_w_clusters.query("ontology in @BIOC_NAME_ONTOLOGIES.keys()"),
        ]
    )[["cluster", IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]]

    possible_names.loc[:, "ontology_preference"] = possible_names[
        IDENTIFIERS.ONTOLOGY
    ].map(BIOC_NAME_ONTOLOGIES)

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
    gene_ids = other_mappings_w_clusters.query(
        "ontology in @BIOC_GENE_ONTOLOGIES"
    ).assign(bqb=BQB.IS_ENCODED_BY)[
        ["cluster", IDENTIFIERS.IDENTIFIER, IDENTIFIERS.ONTOLOGY, IDENTIFIERS.BQB]
    ]
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


@warn_if_no_rpy2
def _expand_identifiers_new_entries(
    sysid: str, expanded_identifiers_df: pd.DataFrame
) -> identifiers.Identifiers:
    """Expand Identifiers to include Bioconductor annotations"""
    entry = expanded_identifiers_df.loc[sysid]

    if type(entry) is pd.Series:
        sysis_id_list = [entry.to_dict()]
    else:
        # multiple annotations
        sysis_id_list = list(entry.reset_index(drop=True).T.to_dict().values())

    return identifiers.Identifiers(sysis_id_list)
