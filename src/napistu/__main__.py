"""The CLI for Napistu"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Sequence

import click
import click_logging
import napistu
import igraph as ig
import pandas as pd
from napistu import consensus as napistu_consensus
from napistu import indices
from napistu import sbml_dfs_core
from napistu import utils
from napistu.context import filtering
from napistu.matching import mount
from napistu.ingestion import bigg
from napistu.ingestion import gtex
from napistu.ingestion import hpa
from napistu.ingestion import reactome
from napistu.ingestion import sbml
from napistu.ingestion import string
from napistu.ingestion import trrust
from napistu.modify import curation
from napistu.modify import gaps
from napistu.modify import pathwayannot
from napistu.modify import uncompartmentalize
from napistu.network import net_create
from napistu.network.ig_utils import get_graph_summary
from napistu.network.ng_utils import read_graph_attrs_spec
from napistu.network import precompute
from napistu.ontologies.genodexito import Genodexito
from napistu.ontologies import dogma
from napistu.constants import ONTOLOGIES
from napistu.constants import RESOLVE_MATCHES_AGGREGATORS
from napistu.ingestion.constants import PROTEINATLAS_SUBCELL_LOC_URL
from napistu.ingestion.constants import GTEX_RNASEQ_EXPRESSION_URL
from napistu.network.constants import NAPISTU_GRAPH_EDGES
from fs import open_fs

logger = logging.getLogger(napistu.__name__)
click_logging.basic_config(logger)

ALL = "all"


@click.group()
def cli():
    """The Calico Pathway Resources CLI"""
    pass


@click.group()
def ingestion():
    """Command line tools to retrieve raw data."""
    pass


@ingestion.command(name="reactome")
@click.argument("base_folder", type=str)
@click.option(
    "--overwrite", "-o", is_flag=True, default=False, help="Overwrite existing files?"
)
@click_logging.simple_verbosity_option(logger)
def ingest_reactome(base_folder: str, overwrite=True):
    logger.info("Start downloading Reactome to %s", base_folder)
    reactome.reactome_sbml_download(f"{base_folder}/sbml", overwrite=overwrite)


@ingestion.command(name="bigg")
@click.argument("base_folder", type=str)
@click.option(
    "--overwrite", "-o", is_flag=True, default=False, help="Overwrite existing files?"
)
@click_logging.simple_verbosity_option(logger)
def ingest_bigg(base_folder: str, overwrite: bool):
    logger.info("Start downloading Bigg to %s", base_folder)
    bigg.bigg_sbml_download(base_folder, overwrite)


@ingestion.command(name="trrust")
@click.argument("target_uri", type=str)
@click_logging.simple_verbosity_option(logger)
def ingest_ttrust(target_uri: str):
    logger.info("Start downloading TRRUST to %s", target_uri)
    trrust.download_trrust(target_uri)


@ingestion.command(name="proteinatlas-subcell")
@click.argument("target_uri", type=str)
@click.option(
    "--url",
    type=str,
    default=PROTEINATLAS_SUBCELL_LOC_URL,
    help="URL to download the zipped protein atlas subcellular localization tsv from.",
)
@click_logging.simple_verbosity_option(logger)
def ingest_proteinatlas_subcell(target_uri: str, url: str):
    hpa.download_hpa_data(target_uri, url)


@ingestion.command(name="gtex-rnaseq-expression")
@click.argument("target_uri", type=str)
@click.option(
    "--url",
    type=str,
    default=GTEX_RNASEQ_EXPRESSION_URL,
    help="URL to download the gtex file from.",
)
@click_logging.simple_verbosity_option(logger)
def ingest_gtex_rnaseq(target_uri: str, url: str):
    gtex.download_gtex_rnaseq(target_uri, url)


@ingestion.command(name="string-db")
@click.argument("target_uri", type=str)
@click.option(
    "--species",
    type=str,
    default="Homo sapiens",
    help="Species name (e.g., Homo sapiens).",
)
@click_logging.simple_verbosity_option(logger)
def ingest_string_db(target_uri: str, species: str):
    string.download_string(target_uri, species)


@ingestion.command(name="string-aliases")
@click.argument("target_uri", type=str)
@click.option(
    "--species",
    type=str,
    default="Homo sapiens",
    help="Species name (e.g., Homo sapiens).",
)
@click_logging.simple_verbosity_option(logger)
def ingest_string_aliases(target_uri: str, species: str):
    string.download_string_aliases(target_uri, species)


@click.group()
def integrate():
    """Command line tools to integrate raw models into a single SBML_dfs model"""
    pass


@integrate.command(name="reactome")
@click.argument("pw_index_uri", type=str)
@click.argument("output_model_uri", type=str)
@click.option("--species", "-s", multiple=True, default=(ALL,))
@click.option(
    "--overwrite", "-o", is_flag=True, default=False, help="Overwrite existing files?"
)
@click.option(
    "--permissive",
    "-p",
    is_flag=True,
    default=False,
    help="Can parsing failures in submodels throw warnings instead of exceptions?",
)
@click_logging.simple_verbosity_option(logger)
def integrate_reactome(
    pw_index_uri: str,
    output_model_uri: str,
    species: Sequence[str] | None,
    overwrite=False,
    permissive=False,
):
    """Integrates reactome models based on a pw_index"""
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")
    if species is not None and len(species) == 1 and species[0] == ALL:
        species = None

    strict = not permissive
    logger.debug(f"permissive = {permissive}; strict = {strict}")

    consensus_model = reactome.construct_reactome_consensus(
        pw_index_uri, species=species, strict=strict
    )
    utils.save_pickle(output_model_uri, consensus_model)


@integrate.command(name="bigg")
@click.argument("pw_index_uri", type=str)
@click.argument("output_model_uri", type=str)
@click.option("--species", "-s", multiple=True, default=(ALL,))
@click.option(
    "--overwrite", "-o", is_flag=True, default=False, help="Overwrite existing files?"
)
@click_logging.simple_verbosity_option(logger)
def integrate_bigg(
    pw_index_uri: str,
    output_model_uri: str,
    species: Sequence[str] | None,
    overwrite=False,
):
    """Integrates bigg models based on a pw_index"""
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")
    if species is not None and len(species) == 1 and species[0] == ALL:
        species = None
    consensus_model = bigg.construct_bigg_consensus(pw_index_uri, species=species)
    utils.save_pickle(output_model_uri, consensus_model)


@integrate.command(name="trrust")
@click.argument("trrust_csv_uri", type=str)
@click.argument("output_model_uri", type=str)
@click.option(
    "--overwrite", "-o", is_flag=True, default=False, help="Overwrite existing files?"
)
@click_logging.simple_verbosity_option(logger)
def integrate_trrust(
    trrust_csv_uri: str,
    output_model_uri: str,
    overwrite=False,
):
    """Converts TRRUST csv to SBML_dfs model"""
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")
    logger.info("Start converting TRRUST csv to SBML_dfs")
    sbmldfs_model = trrust.convert_trrust_to_sbml_dfs(trrust_csv_uri)
    logger.info("Save SBML_dfs model to %s", output_model_uri)
    utils.save_pickle(output_model_uri, sbmldfs_model)


@integrate.command(name="string-db")
@click.argument("string_db_uri", type=str)
@click.argument("string_aliases_uri", type=str)
@click.argument("output_model_uri", type=str)
@click.option(
    "--overwrite", "-o", is_flag=True, default=False, help="Overwrite existing files?"
)
@click_logging.simple_verbosity_option(logger)
def integrate_string_db(
    string_db_uri: str, string_aliases_uri: str, output_model_uri: str, overwrite=False
):
    """Converts string-db to the sbml_dfs format"""
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")
    logger.info("Start converting string-db to SBML_dfs")
    sbmldfs_model = string.convert_string_to_sbml_dfs(string_db_uri, string_aliases_uri)
    logger.info("Save SBML_dfs model to %s", output_model_uri)
    utils.save_pickle(output_model_uri, sbmldfs_model)


@click.group()
def consensus():
    """Command line tools to create a consensus model from SBML_dfs"""
    pass


@consensus.command(name="create")
@click.argument("sbml_dfs_uris", type=str, nargs=-1)
@click.argument("output_model_uri", type=str, nargs=1)
@click.option(
    "--nondogmatic",
    "-n",
    is_flag=True,
    default=False,
    help="Run in non-dogmatic mode (trying to merge genes and proteins)?",
)
@click_logging.simple_verbosity_option(logger)
def create_consensus(
    sbml_dfs_uris: Sequence[str], output_model_uri: str, nondogmatic: bool
):
    """Create a consensus model from a list of SBML_dfs"""

    dogmatic = not nondogmatic
    logger.debug(f"nondogmatic = {nondogmatic}; dogmatic = {dogmatic}")
    logger.info(
        f"Creating a consensus from {len(sbml_dfs_uris)} sbml_dfs where dogmatic = {dogmatic}"
    )

    sbml_dfs_dict = {uri: utils.load_pickle(uri) for uri in sbml_dfs_uris}
    pw_index_df = pd.DataFrame(
        {
            "file": sbml_dfs_uris,
            "pathway_id": sbml_dfs_dict.keys(),
            "source": sbml_dfs_dict.keys(),
            "name": sbml_dfs_dict.keys(),
            # TODO: Discuss with Sean how to deal with date in pw_index
            "date": "1900-01-01",
        }
    )
    pw_index_df["species"] = "unknown"
    pw_index = indices.PWIndex(pw_index=pw_index_df, validate_paths=False)
    consensus_model = napistu_consensus.construct_consensus_model(
        sbml_dfs_dict, pw_index, dogmatic
    )
    utils.save_pickle(output_model_uri, consensus_model)


@click.group()
def refine():
    """Command line tools to refine a consensus model"""
    pass


@refine.command(name="add_reactome_entity_sets")
@click.argument("model_uri", type=str)
@click.argument("entity_set_csv", type=str)
@click.argument("output_model_uri", type=str)
def add_reactome_entity_sets(
    model_uri: str, entity_set_csv: str, output_model_uri: str
):
    """Add reactome entity sets to a consensus model

    The entity set csv is classically exported from the neo4j reactome
    database.
    """
    model = utils.load_pickle(model_uri)
    model = pathwayannot.add_entity_sets(model, entity_set_csv)
    utils.save_pickle(output_model_uri, model)


@refine.command(name="add_reactome_identifiers")
@click.argument("model_uri", type=str)
@click.argument("crossref_csv", type=str)
@click.argument("output_model_uri", type=str)
def add_reactome_identifiers(model_uri: str, crossref_csv: str, output_model_uri: str):
    """Add reactome identifiers to a consensus model

    The crossref csv is classically exported from the neo4j reactome
    database.
    """
    model = utils.load_pickle(model_uri)
    model = pathwayannot.add_reactome_identifiers(model, crossref_csv)
    utils.save_pickle(output_model_uri, model)


@refine.command(name="infer_uncompartmentalized_species_location")
@click.argument("model_uri", type=str)
@click.argument("output_model_uri", type=str)
def infer_uncompartmentalized_species_location(model_uri: str, output_model_uri: str):
    """
    Infer Uncompartmentalized Species Location

    If the compartment of a subset of compartmentalized species was
    not specified, infer an appropriate compartment from other members of reactions they particpate in
    """
    sbml_dfs = utils.load_pickle(model_uri)
    sbml_dfs.infer_uncompartmentalized_species_location()
    utils.save_pickle(output_model_uri, sbml_dfs)


@refine.command(name="name_compartmentalized_species")
@click.argument("model_uri", type=str)
@click.argument("output_model_uri", type=str)
def name_compartmentalized_species(model_uri: str, output_model_uri: str):
    """
    Name Compartmentalized Species

    Rename compartmentalized species if they have the same name as their species
    """
    sbml_dfs = utils.load_pickle(model_uri)
    sbml_dfs.name_compartmentalized_species()
    utils.save_pickle(output_model_uri, sbml_dfs)


@refine.command(name="merge_model_compartments")
@click.argument("model_uri", type=str)
@click.argument("output_model_uri", type=str)
def merge_model_compartments(model_uri: str, output_model_uri: str):
    """Take a compartmentalized mechanistic model and merge all of the compartments."""
    model = utils.load_pickle(model_uri)
    model = uncompartmentalize.uncompartmentalize_sbml_dfs(model)
    utils.save_pickle(output_model_uri, model)


@refine.command(name="drop_cofactors")
@click.argument("model_uri", type=str)
@click.argument("output_model_uri", type=str)
def drop_cofactors(model_uri: str, output_model_uri: str):
    """Remove reaction species acting as cofactors"""
    model = utils.load_pickle(model_uri)
    model = pathwayannot.drop_cofactors(model)
    utils.save_pickle(output_model_uri, model)


@refine.command(name="add_transportation_reactions")
@click.argument("model_uri", type=str)
@click.argument("output_model_uri", type=str)
@click.option(
    "--exchange-compartment",
    "-e",
    default="cytosol",
    help="Exchange compartment for new transport reactions.",
)
@click_logging.simple_verbosity_option(logger)
def add_transportation_reaction(
    model_uri, output_model_uri, exchange_compartment="cytosol"
):
    """Add transportation reactions to a consensus model"""

    model = utils.load_pickle(model_uri)
    model = gaps.add_transportation_reactions(
        model, exchange_compartment=exchange_compartment
    )
    utils.save_pickle(output_model_uri, model)


@refine.command(name="apply_manual_curations")
@click.argument("model_uri", type=str)
@click.argument("curation_dir", type=str)
@click.argument("output_model_uri", type=str)
def apply_manual_curations(model_uri: str, curation_dir: str, output_model_uri: str):
    """Apply manual curations to a consensus model

    The curation dir is a directory containing the manual curations
    Check napistu.modify.curation.curate_sbml_dfs for more information.
    """
    model = utils.load_pickle(model_uri)
    model = curation.curate_sbml_dfs(curation_dir=curation_dir, sbml_dfs=model)
    utils.save_pickle(output_model_uri, model)


@refine.command(name="expand_identifiers")
@click.argument("sbml_dfs_uri", type=str)
@click.argument("output_model_uri", type=str)
@click.option("--species", "-s", default="Homo sapiens", type=str)
@click.option(
    "--ontologies", "-o", multiple=True, type=str, help="Ontologies to add or complete"
)
@click.option(
    "--preferred_method",
    "-p",
    default="bioconductor",
    type=str,
    help="Preferred method to use for identifier expansion",
)
@click.option(
    "--allow_fallback",
    "-a",
    default=True,
    type=bool,
    help="Allow fallback to other methods if preferred method fails",
)
def expand_identifiers(
    sbml_dfs_uri: str,
    output_model_uri: str,
    species: str,
    ontologies: set[str],
    preferred_method: str,
    allow_fallback: bool,
):
    """Expand identifiers of a model

    Args:
        sbml_dfs_uri (str): uri of model in sbml dfs format
        output_model_uri (str): output uri of model in sbml dfs format
        species (str): Species to use
        ontologies (set[str]): ontologies to add or update

    Example call:
    > napistu refine expand_identifiers gs://<uri> ./test.pickle -o ensembl_gene
    """
    sbml_dfs: sbml.SBML_dfs = utils.load_pickle(sbml_dfs_uri)  # type: ignore
    if len(ontologies) == 0:
        raise ValueError("No ontologies to expand specified.")

    Genodexito(
        species=species,
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    ).expand_sbml_dfs_ids(sbml_dfs, ontologies=ontologies)

    utils.save_pickle(output_model_uri, sbml_dfs)


@integrate.command(name="dogmatic_scaffold")
@click.argument("output_model_uri", type=str)
@click.option("--species", "-s", default="Homo sapiens", type=str)
@click.option(
    "--preferred_method",
    "-p",
    default="bioconductor",
    type=str,
    help="Preferred method to use for identifier expansion",
)
@click.option(
    "--allow_fallback",
    "-a",
    default=True,
    type=bool,
    help="Allow fallback to other methods if preferred method fails",
)
def dogmatic_scaffold(
    output_model_uri: str,
    species: str,
    preferred_method: str,
    allow_fallback: bool,
):
    """Dogmatic Scaffold

    Args:
        output_model_uri (str): output uri of model in sbml dfs format
        species (str): Species to use

    Example call:
    > cpr integrate dogmatic_scaffold ./test.pickle
    """

    dogmatic_sbml_dfs = dogma.create_dogmatic_sbml_dfs(
        species=species,
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    )

    utils.save_pickle(output_model_uri, dogmatic_sbml_dfs)


@refine.command(name="filter_gtex_tissue")
@click.argument("sbml_dfs_uri", type=str)
@click.argument("gtex_file_uri", type=str)
@click.argument("output_model_uri", type=str)
@click.argument("tissue", type=str)
@click_logging.simple_verbosity_option(logger)
def filter_gtex_tissue(
    sbml_dfs_uri: str,
    gtex_file_uri: str,
    output_model_uri: str,
    tissue: str,
    filter_non_genic_reactions: bool,
):
    """Filter model by the gtex tissue expression

    This uses zfpkm values derived from gtex to filter the model.
    """

    logger.info("Load sbml_dfs model")
    sbml_dfs: sbml.SBML_dfs = utils.load_pickle(sbml_dfs_uri)  # type: ignore
    logger.info("Load and clean gtex tissue expression")
    dat_gtex = gtex.load_and_clean_gtex_data(gtex_file_uri)
    logger.info("Annotate genes with gtex tissue expression")
    mount.bind_wide_results(
        sbml_dfs=sbml_dfs,
        results_df=dat_gtex.reset_index(drop=False),
        results_name="gtex",
        ontologies={ONTOLOGIES.ENSEMBL_GENE},
        numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MAX,
    )
    logger.info("Trim network by gene attribute")
    filtering.filter_species_by_attribute(
        sbml_dfs,
        "gtex",
        attribute_name=tissue,
        # remove entries which are NOT in the liver
        attribute_value=0,
        inplace=True,
    )
    # remove the gtex species data from the sbml_dfs
    sbml_dfs.remove_species_data("gtex")

    logger.info("Save sbml_dfs to %s", output_model_uri)
    utils.save_pickle(output_model_uri, sbml_dfs)


@refine.command(name="filter_hpa_compartments")
@click.argument("sbml_dfs_uri", type=str)
@click.argument("hpa_file_uri", type=str)
@click.argument("output_model_uri", type=str)
@click_logging.simple_verbosity_option(logger)
def filter_hpa_gene_compartments(
    sbml_dfs_uri: str, hpa_file_uri: str, output_model_uri: str
):
    """Filter an interaction network using the human protein atlas

    This uses loads the human proteome atlas and removes reactions (including interactions)
    containing genes which are not colocalized.

    Only interactions between genes in the same compartment are kept.
    """

    logger.info("Load sbml_dfs model")
    sbml_dfs: sbml.SBML_dfs = utils.load_pickle(sbml_dfs_uri)  # type: ignore
    logger.info("Load and clean hpa data")
    dat_hpa = hpa.load_and_clean_hpa_data(hpa_file_uri)
    logger.info("Annotate genes with HPA compartments")
    mount.bind_wide_results(
        sbml_dfs=sbml_dfs,
        results_df=dat_hpa.reset_index(drop=False),
        results_name="hpa",
        ontologies={ONTOLOGIES.ENSEMBL_GENE},
        numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MAX,
    )
    logger.info(
        "Trim network removing reactions with species in different compartments"
    )
    filtering.filter_reactions_with_disconnected_cspecies(
        sbml_dfs, "hpa", inplace=False
    )
    sbml_dfs.remove_species_data("hpa")

    logger.info("Save sbml_dfs to %s", output_model_uri)
    utils.save_pickle(output_model_uri, sbml_dfs)


@click.group()
def exporter():
    """Command line tools to export a consensus model
    to various formats
    """
    pass


@exporter.command(name="export_igraph")
@click.argument("model_uri", type=str)
@click.argument("output_uri", type=str)
@click.option(
    "--graph_attrs_spec_uri",
    "-a",
    default=None,
    help="File specifying reaction and/or species attributes to add to the graph",
)
@click.option(
    "--format", "-f", default="pickle", help="Output format: gml, edgelist, pickle"
)
@click.option(
    "--wiring_approach",
    "-g",
    type=str,
    default="bipartite",
    help="bipartite or regulatory",
)
@click.option(
    "--weighting_strategy",
    "-w",
    type=str,
    default="unweighted",
    help="Approach to adding weights to the network",
)
@click.option(
    "--directed", "-d", type=bool, default=True, help="Directed or undirected graph?"
)
@click.option(
    "--reverse",
    "-r",
    type=bool,
    default=False,
    help="Reverse edges so they flow from effects to causes?",
)
def export_igraph(
    model_uri: str,
    output_uri: str,
    graph_attrs_spec_uri: str | None,
    format: str,
    wiring_approach: str,
    weighting_strategy: str,
    directed: bool,
    reverse: bool,
):
    """Export the consensus model as an igraph object"""
    model = utils.load_pickle(model_uri)

    if graph_attrs_spec_uri is None:
        graph_attrs_spec = None
    else:
        graph_attrs_spec = read_graph_attrs_spec(graph_attrs_spec_uri)

    napistu_graph = net_create.process_napistu_graph(
        model,
        reaction_graph_attrs=graph_attrs_spec,
        directed=directed,
        edge_reversed=reverse,
        wiring_approach=wiring_approach,
        weighting_strategy=weighting_strategy,
        verbose=True,
    )

    base, path = os.path.split(output_uri)
    with open_fs(base, create=True, writeable=True) as fs:
        with fs.openbin(path, "wb") as f:
            if format == "gml":
                napistu_graph.write_gml(f)
            elif format == "edgelist":
                napistu_graph.write_edgelist(f)
            elif format == "pickle":
                pickle.dump(napistu_graph, f)
            else:
                raise ValueError("Unknown format: %s" % format)


@exporter.command(name="export_precomputed_distances")
@click.argument("graph_uri", type=str)
@click.argument("output_uri", type=str)
@click.option(
    "--format",
    "-f",
    type=str,
    default="pickle",
    help="Input igraph format: gml, edgelist, pickle",
)
@click.option(
    "--max_steps",
    "-s",
    type=int,
    default=-1,
    help="The max number of steps between pairs of species to save a distance",
)
@click.option(
    "--max_score_q",
    "-q",
    type=float,
    default=1,
    help='Retain up to the "max_score_q" quantiles of all scores (small scores are better)',
)
@click.option(
    "--partition_size",
    "-p",
    type=int,
    default=5000,
    help="The number of species to process together when computing distances",
)
@click.option(
    "--weight_vars",
    "-w",
    type=str,
    default=[NAPISTU_GRAPH_EDGES.WEIGHT, NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT],
    help="One or more variables defining edge weights to use when calculating weighted shortest paths.",
)
def export_precomputed_distances(
    graph_uri: str,
    output_uri: str,
    format: str,
    max_steps: int,
    max_score_q: float,
    partition_size: int,
    weight_vars: str,
):
    """Export precomputed distances for the igraph object"""

    base, path = os.path.split(graph_uri)
    with open_fs(base) as fs:
        with fs.openbin(path) as f:
            if format == "gml":
                napistu_graph = ig.Graph.Read_GML(f)
            elif format == "edgelist":
                napistu_graph = ig.Graph.Read_Edgelist(f)
            elif format == "pickle":
                napistu_graph = ig.Graph.Read_Pickle(f)
            else:
                raise ValueError("Unknown format: %s" % format)

    # convert weight vars from a str to list
    weight_vars_list = utils.click_str_to_list(weight_vars)

    precomputed_distances = precompute.precompute_distances(
        napistu_graph,
        max_steps=max_steps,
        max_score_q=max_score_q,
        partition_size=partition_size,
        weight_vars=weight_vars_list,
    )

    utils.save_parquet(precomputed_distances, output_uri)


@exporter.command(name="export_smbl_dfs_tables")
@click.argument("model_uri", type=str)
@click.argument("output_uri", type=str)
@click.option(
    "--overwrite", "-o", is_flag=True, default=False, help="Overwrite existing files?"
)
@click.option(
    "--model-prefix", "-m", type=str, default="", help="Model prefix for files?"
)
@click.option(
    "--nondogmatic",
    "-n",
    is_flag=True,
    default=False,
    help="Run in non-dogmatic mode (trying to merge genes and proteins)?",
)
@click_logging.simple_verbosity_option(logger)
def export_sbml_dfs_tables(
    model_uri: str,
    output_uri: str,
    overwrite=False,
    model_prefix="",
    nondogmatic: bool = True,
):
    """Export the consensus model as a collection of table"""

    dogmatic = not nondogmatic
    logger.debug(f"nondogmatic = {nondogmatic}; dogmatic = {dogmatic}")
    logger.info(f"Exporting tables with dogmatic = {dogmatic}")

    sbml_dfs = utils.load_pickle(model_uri)
    sbml_dfs.export_sbml_dfs(
        model_prefix, output_uri, overwrite=overwrite, dogmatic=dogmatic
    )


@click.group()
def importer():
    """Tools to import sbml_dfs directly form other sources"""
    pass


@importer.command(name="sbml_dfs")
@click.argument("input_uri", type=str)
@click.argument("output_uri", type=str)
@click_logging.simple_verbosity_option(logger)
def import_sbml_dfs_from_sbml_dfs_uri(input_uri, output_uri):
    """Import sbml_dfs from an uri, eg another GCS bucket"""
    logger.info("Load sbml_dfs from %s", input_uri)
    # We could also just copy the file, but I think validating
    # the filetype is a good idea to prevent downstream errors.
    sbml_dfs = utils.load_pickle(input_uri)
    if not (isinstance(sbml_dfs, sbml.SBML_dfs)):
        raise ValueError(
            f"Pickled input is not an SBML_dfs object but {type(sbml_dfs)}: {input_uri}"
        )
    logger.info("Save file to %s", output_uri)
    utils.save_pickle(output_uri, sbml_dfs)


@importer.command(name="sbml")
@click.argument("input_uri", type=str)
@click.argument("output_uri", type=str)
@click_logging.simple_verbosity_option(logger)
def import_sbml_dfs_from_sbml(input_uri, output_uri):
    """Import sbml_dfs from a sbml file"""
    logger.info("Load sbml from %s", input_uri)
    # We could also just copy the file, but I think validating
    # the filetype is a good idea to prevent downstream errors.
    sbml_file = sbml.SBML(input_uri)
    logger.info("Convert file to sbml_dfs")
    sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_file)
    logger.info("Save file to %s", output_uri)
    utils.save_pickle(output_uri, sbml_dfs)


@click.group()
def contextualizer():
    """Command line tools to contextualize a pathway model"""
    pass


@click.group()
def helpers():
    """Various helper functions"""
    pass


@helpers.command(name="copy_uri")
@click.argument("input_uri", type=str)
@click.argument("output_uri", type=str)
@click.option("--is-file", type=bool, default=True, help="Is the input a file?")
@click_logging.simple_verbosity_option(logger)
def copy_uri(input_uri, output_uri, is_file=True):
    """Copy a uri representing a file or folder from one location to another"""
    logger.info("Copy uri from %s to %s", input_uri, output_uri)
    utils.copy_uri(input_uri, output_uri, is_file=is_file)


@helpers.command(name="validate_sbml_dfs")
@click.argument("input_uri", type=str)
@click_logging.simple_verbosity_option(logger)
def validate_sbml_dfs(input_uri):
    """Validate a sbml_dfs object"""
    sbml_dfs = utils.load_pickle(input_uri)
    sbml_dfs.validate()

    logger.info(f"Successfully validated: {input_uri}")


@click.group()
def stats():
    """Various functions to calculate network statistics

    The statistics are saved as json files
    """
    pass


@stats.command(name="sbml_dfs_network")
@click.argument("input_uri", type=str)
@click.argument("output_uri", type=str)
def calculate_sbml_dfs_stats(input_uri, output_uri):
    """Calculate statistics for a sbml_dfs object"""
    model: sbml_dfs_core.SBML_dfs = utils.load_pickle(input_uri)  # type: ignore
    stats = model.get_network_summary()
    utils.save_json(output_uri, stats)


@stats.command(name="igraph_network")
@click.argument("input_uri", type=str)
@click.argument("output_uri", type=str)
def calculate_igraph_stats(input_uri, output_uri):
    """Calculate statistics for an igraph object"""
    graph: ig.Graph = utils.load_pickle(input_uri)  # type: ignore
    stats = get_graph_summary(graph)
    utils.save_json(output_uri, stats)


cli.add_command(ingestion)
cli.add_command(integrate)
cli.add_command(consensus)
cli.add_command(refine)
cli.add_command(exporter)
cli.add_command(importer)
cli.add_command(helpers)
cli.add_command(stats)

if __name__ == "__main__":
    cli()
