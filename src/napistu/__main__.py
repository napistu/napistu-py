"""The CLI for Napistu"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Sequence

import click
import igraph as ig
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs

from napistu import consensus as napistu_consensus
from napistu import utils
from napistu._cli import (
    click_str_to_list,
    genodexito_options,
    nondogmatic_option,
    organismal_species_argument,
    overwrite_option,
    sbml_dfs_input,
    sbml_dfs_io,
    sbml_dfs_output,
    setup_logging,
    target_uri_output,
    verbosity_option,
)
from napistu.constants import ONTOLOGIES, RESOLVE_MATCHES_AGGREGATORS
from napistu.context import filtering
from napistu.ingestion import (
    bigg,
    gtex,
    hpa,
    intact,
    omnipath,
    reactome,
    reactome_fi,
    string,
    trrust,
)
from napistu.ingestion.constants import (
    GTEX_RNASEQ_EXPRESSION_URL,
    PROTEINATLAS_SUBCELL_LOC_URL,
    REACTOME_FI_URL,
)
from napistu.ingestion.sbml import SBML
from napistu.matching.mount import bind_wide_results
from napistu.modify import pathwayannot
from napistu.modify.cofactors import drop_cofactors as drop_cofactors_func
from napistu.modify.curation import curate_sbml_dfs
from napistu.modify.gaps import add_transportation_reactions
from napistu.modify.uncompartmentalize import uncompartmentalize_sbml_dfs
from napistu.network.constants import DROP_REACTIONS_WHEN, NAPISTU_GRAPH_EDGES
from napistu.network.ig_utils import get_graph_summary
from napistu.network.net_create import process_napistu_graph
from napistu.network.ng_core import NapistuGraph
from napistu.network.ng_utils import read_graph_attrs_spec
from napistu.network.ng_utils import validate_assets as validate_assets_func
from napistu.network.precompute import precompute_distances
from napistu.ontologies import dogma
from napistu.ontologies.genodexito import Genodexito
from napistu.sbml_dfs_core import SBML_dfs
from napistu.sbml_dfs_utils import display_post_consensus_checks

# Module-level logger and console - will be initialized when CLI is invoked
logger = None
console = None


@click.group()
def cli():
    """The Napistu CLI"""
    # Set up logging only when CLI is actually invoked, not at import time
    # This prevents interfering with pytest's caplog fixture during tests
    global logger, console
    if logger is None:
        logger, console = setup_logging()


@click.group()
def ingestion():
    """Command line tools to retrieve raw data."""
    pass


@ingestion.command(name="reactome")
@click.argument("base_folder", type=str)
@overwrite_option
@verbosity_option
def ingest_reactome(base_folder: str, overwrite=True):
    logger.info("Start downloading Reactome to %s", base_folder)
    reactome.reactome_sbml_download(f"{base_folder}/sbml", overwrite=overwrite)


@ingestion.command(name="bigg")
@click.argument("base_folder", type=str)
@overwrite_option
@verbosity_option
def ingest_bigg(base_folder: str, overwrite: bool):
    logger.info("Start downloading Bigg to %s", base_folder)
    bigg.bigg_sbml_download(base_folder, overwrite)


@ingestion.command(name="trrust")
@target_uri_output
@verbosity_option
def ingest_ttrust(target_uri: str):
    logger.info("Start downloading TRRUST to %s", target_uri)
    trrust.download_trrust(target_uri)


@ingestion.command(name="proteinatlas_subcell")
@target_uri_output
@click.option(
    "--url",
    type=str,
    default=PROTEINATLAS_SUBCELL_LOC_URL,
    help="URL to download the zipped protein atlas subcellular localization tsv from.",
)
@verbosity_option
def ingest_proteinatlas_subcell(target_uri: str, url: str):
    hpa.download_hpa_data(target_uri, url)


@ingestion.command(name="gtex-rnaseq-expression")
@target_uri_output
@click.option(
    "--url",
    type=str,
    default=GTEX_RNASEQ_EXPRESSION_URL,
    help="URL to download the gtex file from.",
)
@verbosity_option
def ingest_gtex_rnaseq(target_uri: str, url: str):
    gtex.download_gtex_rnaseq(target_uri, url)


@ingestion.command(name="string_db")
@organismal_species_argument
@target_uri_output
@verbosity_option
def ingest_string_db(organismal_species: str, target_uri: str):
    string.download_string(target_uri, organismal_species)


@ingestion.command(name="string_aliases")
@organismal_species_argument
@target_uri_output
@verbosity_option
def ingest_string_aliases(organismal_species: str, target_uri: str):
    string.download_string_aliases(target_uri, organismal_species)


@ingestion.command(name="reactome_fi")
@target_uri_output
@click.option(
    "--url",
    type=str,
    default=REACTOME_FI_URL,
    help="URL to download the Reactome FI data from. If not provided, uses default URL.",
)
@overwrite_option
@verbosity_option
def ingest_reactome_fi(target_uri: str, url: str, overwrite: bool):
    """Download Reactome Functional Interactions (FI) dataset as a TSV file."""
    if overwrite is False and utils.path_exists(target_uri):
        raise FileExistsError(f"'{target_uri}' exists but overwrite set False.")

    logger.info("Start downloading Reactome FI from %s to %s", url, target_uri)
    reactome_fi.download_reactome_fi(target_uri, url=url)


@ingestion.command(name="intact")
@click.argument("output_dir_path", type=str)
@organismal_species_argument
@overwrite_option
@verbosity_option
def ingest_intact(output_dir_path: str, organismal_species: str, overwrite: bool):
    """Download IntAct PSI-MI XML files for a specific species.

    OUTPUT_DIR_PATH: Local directory to create and unzip files into
    ORGANISMAL_SPECIES: The species name (e.g., "Homo sapiens") to work with
    """
    if overwrite is False and utils.path_exists(output_dir_path):
        raise FileExistsError(f"'{output_dir_path}' exists but overwrite set False.")

    logger.info(
        "Start downloading IntAct PSI-MI XML files for %s to %s",
        organismal_species,
        output_dir_path,
    )
    intact.download_intact_xmls(
        output_dir_path, organismal_species, overwrite=overwrite
    )


@click.group()
def integrate():
    """Command line tools to integrate raw models into a single SBML_dfs model"""
    pass


@integrate.command(name="reactome")
@click.argument("pw_index_uri", type=str)
@organismal_species_argument
@sbml_dfs_output
@overwrite_option
@click.option(
    "--permissive",
    "-p",
    is_flag=True,
    default=False,
    help="Can parsing failures in submodels throw warnings instead of exceptions?",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Include detailed logs?",
)
@verbosity_option
def integrate_reactome(
    pw_index_uri: str,
    organismal_species: str,
    output_model_uri: str,
    overwrite=False,
    permissive=False,
    verbose=False,
):
    """Integrates reactome models based on a pw_index"""
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")

    strict = not permissive
    logger.debug(f"permissive = {permissive}; strict = {strict}")

    consensus_model = reactome.construct_reactome_consensus(
        pw_index_uri,
        organismal_species=organismal_species,
        strict=strict,
        verbose=verbose,
    )
    consensus_model.to_pickle(output_model_uri)


@integrate.command(name="bigg")
@click.argument("pw_index_uri", type=str)
@organismal_species_argument
@sbml_dfs_output
@overwrite_option
@verbosity_option
def integrate_bigg(
    pw_index_uri: str,
    organismal_species: str,
    output_model_uri: str,
    overwrite=False,
):
    """Integrates bigg models based on a pw_index"""
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")

    consensus_model = bigg.construct_bigg_consensus(pw_index_uri, organismal_species)
    consensus_model.to_pickle(output_model_uri)


@integrate.command(name="trrust")
@click.argument("trrust_csv_uri", type=str)
@sbml_dfs_output
@overwrite_option
@verbosity_option
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
    sbmldfs_model.to_pickle(output_model_uri)


@integrate.command(name="reactome_fi")
@click.argument("reactome_fi_uri", type=str)
@sbml_dfs_output
@genodexito_options
@overwrite_option
@verbosity_option
def integrate_reactome_fi(
    reactome_fi_uri: str,
    output_model_uri: str,
    preferred_method: str = "bioconductor",
    allow_fallback: bool = True,
    overwrite: bool = False,
):
    """Converts Reactome FI TSV to SBML_dfs model"""
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")

    logger.info("Start converting Reactome FI TSV to an SBML_dfs model")
    sbml_dfs = reactome_fi.convert_reactome_fi_to_sbml_dfs(
        pd.read_csv(reactome_fi_uri, sep="\t"),
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    )

    logger.info("Saving the SBML_dfs model to %s", output_model_uri)
    sbml_dfs.to_pickle(output_model_uri)


@integrate.command(name="string_db")
@click.argument("string_db_uri", type=str)
@click.argument("string_aliases_uri", type=str)
@organismal_species_argument
@sbml_dfs_output
@overwrite_option
@verbosity_option
def integrate_string_db(
    string_db_uri: str,
    string_aliases_uri: str,
    organismal_species: str,
    output_model_uri: str,
    overwrite=False,
):
    """Converts STRING database to the sbml_dfs format"""
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")
    logger.info("Start converting STRING database to an SBML_dfs model")
    sbml_dfs = string.convert_string_to_sbml_dfs(
        string_db_uri, string_aliases_uri, organismal_species
    )
    logger.info("Saving the SBML_dfs model to %s", output_model_uri)
    sbml_dfs.to_pickle(output_model_uri)


@integrate.command(name="intact")
@click.argument("intact_xml_dir", type=str)
@organismal_species_argument
@sbml_dfs_output
@overwrite_option
@verbosity_option
def integrate_intact(
    intact_xml_dir: str,
    organismal_species: str,
    output_model_uri: str,
    overwrite: bool = False,
):
    """Converts IntAct PSI-MI XML files to SBML_dfs model.

    INTACT_XML_DIR: Directory containing the IntAct PSI-MI XML files
    ORGANISMAL_SPECIES: The species name (e.g., "Homo sapiens") to work with
    OUTPUT_MODEL_URI: Output URI for the SBML_dfs model
    OVERWRITE: Overwrite existing files?
    """
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")

    logger.info("Start converting IntAct PSI-MI XML files to an SBML_dfs model")

    # Import PSI-MI formatting functions
    from napistu.ingestion import psi_mi

    # Format the PSI-MI XML files
    formatted_psi_mis = psi_mi.format_psi_mis(intact_xml_dir)

    # Aggregate PSI-MI interactions
    intact_summaries = psi_mi.aggregate_psi_mis(formatted_psi_mis)

    # Convert to SBML_dfs
    sbml_dfs = intact.intact_to_sbml_dfs(intact_summaries, organismal_species)

    # Validate the model
    logger.info("Validating SBML_dfs model")
    sbml_dfs.validate()

    logger.info("Save SBML_dfs model to %s", output_model_uri)
    sbml_dfs.to_pickle(output_model_uri)


@integrate.command(name="omnipath")
@organismal_species_argument
@sbml_dfs_output
@genodexito_options
@overwrite_option
@verbosity_option
def integrate_omnipath(
    organismal_species: str,
    output_model_uri: str,
    preferred_method: str = "bioconductor",
    allow_fallback: bool = True,
    overwrite: bool = False,
):
    """Downloads OmniPath data and formats interactions as an SBML_dfs model.

    ORGANISMAL_SPECIES: The species name (e.g., "Homo sapiens") to work with
    OUTPUT_MODEL_URI: Output URI for the SBML_dfs model
    PREFERRED_METHOD: Preferred Genodexito method for identifier mapping (default: bioconductor).
    ALLOW_FALLBACK: Allow fallback to other Genodexito methods if preferred method fails (default: True).
    OVERWRITE: Overwrite existing files?
    """
    if overwrite is False and utils.path_exists(output_model_uri):
        raise FileExistsError("'output_model_uri' exists but overwrite set False.")

    logger.info(
        "Start downloading OmniPath data and formatting interactions as SBML_dfs"
    )

    # Format OmniPath as SBML_dfs
    sbml_dfs = omnipath.format_omnipath_as_sbml_dfs(
        organismal_species=organismal_species,
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    )

    # Validate the model
    logger.info("Validating SBML_dfs model")
    sbml_dfs.validate()

    logger.info("Save SBML_dfs model to %s", output_model_uri)
    sbml_dfs.to_pickle(output_model_uri)


@click.group()
def consensus():
    """Command line tools to create a consensus model from SBML_dfs"""
    pass


@consensus.command(name="create")
@click.argument("sbml_dfs_uris", type=str, nargs=-1)
@sbml_dfs_output
@nondogmatic_option
@click.option(
    "--ignore-mergeability",
    is_flag=True,
    default=False,
    help="Ignore issues which will prevent merging across models",
)
@verbosity_option
def create_consensus(
    sbml_dfs_uris: Sequence[str],
    output_model_uri: str,
    nondogmatic: bool,
    ignore_mergeability: bool,
):
    """Create a consensus model from a list of SBML_dfs"""

    dogmatic = not nondogmatic
    check_mergeability = not ignore_mergeability
    logger.debug(f"dogmatic = {dogmatic}; check_mergeability = {check_mergeability}")
    logger.info(
        f"Creating a consensus from {len(sbml_dfs_uris)} sbml_dfs where dogmatic = {dogmatic}"
    )

    # create a list of sbml_dfs objects
    sbml_dfs_list = [SBML_dfs.from_pickle(uri) for uri in sbml_dfs_uris]

    # reorganize as a list and table containing model-level metadata from the individual SBML_dfs
    sbml_dfs_dict, pw_index = napistu_consensus.prepare_consensus_model(sbml_dfs_list)

    consensus_model = napistu_consensus.construct_consensus_model(
        sbml_dfs_dict,
        pw_index,
        dogmatic=dogmatic,
        check_mergeability=check_mergeability,
    )

    consensus_model.to_pickle(output_model_uri)


@consensus.command(name="check")
@sbml_dfs_input
@verbosity_option
def check_consensus(sbml_dfs_uri: str):
    """Check a consensus model for potential issues"""

    logger.info(f"Checking consensus model: {sbml_dfs_uri}")

    # Load the consensus model
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)

    # Run post-consensus checks
    results = sbml_dfs.post_consensus_checks()
    display_post_consensus_checks(results)

    logger.info("Consensus model check completed")


@click.group()
def refine():
    """Command line tools to refine a consensus model"""
    pass


@refine.command(name="add_reactome_entity_sets")
@sbml_dfs_input
@click.argument("entity_set_csv", type=str)
@sbml_dfs_output
def add_reactome_entity_sets(
    sbml_dfs_uri: str, entity_set_csv: str, output_model_uri: str
):
    """Add reactome entity sets to a consensus model

    The entity set csv is classically exported from the neo4j reactome
    database.
    """
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    annotated_sbml_dfs = pathwayannot.add_entity_sets(sbml_dfs, entity_set_csv)
    annotated_sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="add_reactome_identifiers")
@sbml_dfs_input
@click.argument("crossref_csv", type=str)
@sbml_dfs_output
def add_reactome_identifiers(
    sbml_dfs_uri: str, crossref_csv: str, output_model_uri: str
):
    """Add reactome identifiers to a consensus model

    The crossref csv is classically exported from the neo4j reactome
    database.
    """
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    annotated_sbml_dfs = pathwayannot.add_reactome_identifiers(sbml_dfs, crossref_csv)
    annotated_sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="infer_uncompartmentalized_species_location")
@sbml_dfs_io
def infer_uncompartmentalized_species_location(
    sbml_dfs_uri: str, output_model_uri: str
):
    """
    Infer Uncompartmentalized Species Location

    If the compartment of a subset of compartmentalized species was
    not specified, infer an appropriate compartment from other members of reactions they particpate in
    """
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    sbml_dfs.infer_uncompartmentalized_species_location()
    sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="name_compartmentalized_species")
@sbml_dfs_io
def name_compartmentalized_species(sbml_dfs_uri: str, output_model_uri: str):
    """
    Name Compartmentalized Species

    Rename compartmentalized species if they have the same name as their species
    """
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    sbml_dfs.name_compartmentalized_species()
    sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="merge_model_compartments")
@sbml_dfs_io
def merge_model_compartments(sbml_dfs_uri: str, output_model_uri: str):
    """Take a compartmentalized mechanistic model and merge all of the compartments."""
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    uncompartmentalize_sbml_dfs(sbml_dfs, inplace=True)
    sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="drop_cofactors")
@sbml_dfs_io
def drop_cofactors(sbml_dfs_uri: str, output_model_uri: str):
    """Remove reaction species acting as cofactors"""
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    cofactor_filtered_sbml_dfs = drop_cofactors_func(sbml_dfs)
    cofactor_filtered_sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="add_transportation_reactions")
@sbml_dfs_io
@click.option(
    "--exchange-compartment",
    "-e",
    default="cytosol",
    help="Exchange compartment for new transport reactions.",
)
@verbosity_option
def add_transportation_reaction(
    sbml_dfs_uri, output_model_uri, exchange_compartment="cytosol"
):
    """Add transportation reactions to a consensus model"""

    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    annotated_sbml_dfs = add_transportation_reactions(
        sbml_dfs, exchange_compartment=exchange_compartment
    )
    annotated_sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="apply_manual_curations")
@sbml_dfs_input
@click.argument("curation_dir", type=str)
@sbml_dfs_output
def apply_manual_curations(sbml_dfs_uri: str, curation_dir: str, output_model_uri: str):
    """Apply manual curations to a consensus model

    The curation dir is a directory containing the manual curations
    Check napistu.modify.curation.curate_sbml_dfs for more information.
    """
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    annotated_sbml_dfs = curate_sbml_dfs(curation_dir=curation_dir, sbml_dfs=sbml_dfs)
    annotated_sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="expand_identifiers")
@sbml_dfs_input
@organismal_species_argument
@sbml_dfs_output
@click.option(
    "--ontologies", "-o", multiple=True, type=str, help="Ontologies to add or complete"
)
@genodexito_options
def expand_identifiers(
    sbml_dfs_uri: str,
    organismal_species: str,
    output_model_uri: str,
    ontologies: set[str],
    preferred_method: str,
    allow_fallback: bool,
):
    """Expand identifiers of a model

    Parameters
    ----------
    sbml_dfs_uri : str
        sbml_dfs_uri (str): uri of model in sbml dfs format
    organismal_species : str
        Species to use
    output_model_uri : str
        output uri of model in sbml dfs format
    ontologies : set[str]
        ontologies to add or update

    Example call:
    > napistu refine expand_identifiers gs://<uri> ./test.pickle -o ensembl_gene
    """
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    if len(ontologies) == 0:
        raise ValueError("No ontologies to expand specified.")

    Genodexito(
        organismal_species=organismal_species,
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    ).expand_sbml_dfs_ids(sbml_dfs, ontologies=ontologies)

    sbml_dfs.to_pickle(output_model_uri)


@integrate.command(name="dogmatic_scaffold")
@organismal_species_argument
@sbml_dfs_output
@genodexito_options
@verbosity_option
def dogmatic_scaffold(
    organismal_species: str,
    output_model_uri: str,
    preferred_method: str,
    allow_fallback: bool,
):
    """Dogmatic Scaffold

    Parameters
    ----------
    organismal_species : str
        Species to use
    output_model_uri : str
        output uri of model in sbml dfs format
    preferred_method : str
        Preferred method to use for identifier expansion
    allow_fallback : bool
        Allow fallback to other methods if preferred method fails

    Example call:
    > cpr integrate dogmatic_scaffold ./test.pickle
    """

    dogmatic_sbml_dfs = dogma.create_dogmatic_sbml_dfs(
        organismal_species=organismal_species,
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    )

    dogmatic_sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="filter_gtex_tissue")
@sbml_dfs_input
@click.argument("gtex_file_uri", type=str)
@click.argument("tissue", type=str)
@sbml_dfs_output
@verbosity_option
def filter_gtex_tissue(
    sbml_dfs_uri: str, gtex_file_uri: str, output_model_uri: str, tissue: str
):
    """Filter model by the gtex tissue expression

    This uses zfpkm values derived from gtex to filter the model.
    """

    logger.info("Load sbml_dfs model")
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    logger.info("Load and clean gtex tissue expression")
    dat_gtex = gtex.load_and_clean_gtex_data(gtex_file_uri)
    logger.info("Annotate genes with gtex tissue expression")
    bind_wide_results(
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
    sbml_dfs.to_pickle(output_model_uri)


@refine.command(name="filter_hpa_compartments")
@sbml_dfs_input
@click.argument("hpa_file_uri", type=str)
@sbml_dfs_output
@verbosity_option
def filter_hpa_gene_compartments(
    sbml_dfs_uri: str, hpa_file_uri: str, output_model_uri: str
):
    """Filter an interaction network using the human protein atlas

    This uses loads the human proteome atlas and removes reactions (including interactions)
    containing genes which are not colocalized.

    Only interactions between genes in the same compartment are kept.
    """

    logger.info("Load sbml_dfs model")
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    logger.info("Load and clean hpa data")
    dat_hpa = hpa.load_and_clean_hpa_data(hpa_file_uri)
    logger.info("Annotate genes with HPA compartments")
    bind_wide_results(
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
    sbml_dfs.to_pickle(output_model_uri)


@click.group()
def exporter():
    """Command line tools to export a consensus model
    to various formats
    """
    pass


@exporter.command(name="export_napistu_graph")
@sbml_dfs_input
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
@click.option(
    "--drop_reactions_when",
    "-x",
    type=click.Choice(
        [
            DROP_REACTIONS_WHEN.SAME_TIER,
            DROP_REACTIONS_WHEN.EDGELIST,
            DROP_REACTIONS_WHEN.ALWAYS,
        ],
        case_sensitive=False,
    ),
    default=DROP_REACTIONS_WHEN.SAME_TIER,
    help="When to drop reactions as network vertices: 'same_tier' (default), 'edgelist', or 'always'",
)
@click.option(
    "--deduplicate_edges",
    is_flag=True,
    default=False,
    help="Deduplicate edges so 0-1 edges connect each vertex.",
)
def export_napistu_graph(
    sbml_dfs_uri: str,
    output_uri: str,
    graph_attrs_spec_uri: str | None,
    format: str,
    wiring_approach: str,
    weighting_strategy: str,
    directed: bool,
    reverse: bool,
    drop_reactions_when: str,
    deduplicate_edges: bool,
):
    """Export the consensus model as an igraph object"""
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)

    if graph_attrs_spec_uri is None:
        graph_attrs_spec = None
    else:
        graph_attrs_spec = read_graph_attrs_spec(graph_attrs_spec_uri)

    napistu_graph = process_napistu_graph(
        sbml_dfs,
        directed=directed,
        wiring_approach=wiring_approach,
        weighting_strategy=weighting_strategy,
        reaction_graph_attrs=graph_attrs_spec,
        drop_reactions_when=drop_reactions_when,
        deduplicate_edges=deduplicate_edges,
        verbose=True,
    )

    if reverse:
        napistu_graph.reverse_edges()

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
    default=1000,
    help="The number of species to process together when computing distances",
)
@click.option(
    "--weight_vars",
    "-w",
    type=str,
    default=[NAPISTU_GRAPH_EDGES.WEIGHT, NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM],
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
    weight_vars_list = click_str_to_list(weight_vars)

    precomputed_distances = precompute_distances(
        napistu_graph,
        max_steps=max_steps,
        max_score_q=max_score_q,
        partition_size=partition_size,
        weight_vars=weight_vars_list,
    )

    utils.save_parquet(precomputed_distances, output_uri)


@exporter.command(name="export_smbl_dfs_tables")
@sbml_dfs_input
@click.argument("output_uri", type=str)
@click.option(
    "--model-prefix", "-m", type=str, default="", help="Model prefix for files?"
)
@nondogmatic_option
@verbosity_option
@overwrite_option
def export_sbml_dfs_tables(
    sbml_dfs_uri: str,
    output_uri: str,
    overwrite=False,
    model_prefix="",
    nondogmatic: bool = True,
):
    """Export the consensus model as a collection of table"""

    dogmatic = not nondogmatic
    logger.debug(f"nondogmatic = {nondogmatic}; dogmatic = {dogmatic}")
    logger.info(f"Exporting tables with dogmatic = {dogmatic}")

    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    sbml_dfs.export_sbml_dfs(
        model_prefix, output_uri, overwrite=overwrite, dogmatic=dogmatic
    )


@click.group()
def importer():
    """Tools to import sbml_dfs directly from other sources"""
    pass


@importer.command(name="sbml_dfs")
@click.argument("input_uri", type=str)
@sbml_dfs_io
@verbosity_option
def import_sbml_dfs_from_sbml_dfs_uri(sbml_dfs_uri, output_model_uri):
    """Import sbml_dfs from an uri, eg another GCS bucket"""
    logger.info("Load sbml_dfs from %s", sbml_dfs_uri)
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    logger.info("Save file to %s", output_model_uri)
    sbml_dfs.to_pickle(output_model_uri)


@importer.command(name="sbml")
@click.argument("input_uri", type=str)
@sbml_dfs_output
@verbosity_option
def import_sbml_dfs_from_sbml(input_uri, output_model_uri):
    """Import sbml_dfs from a sbml file"""
    logger.info("Load sbml from %s", input_uri)
    # We could also just copy the file, but I think validating
    # the filetype is a good idea to prevent downstream errors.
    sbml_file = SBML(input_uri)
    logger.info("Converting file to sbml_dfs")
    sbml_dfs = SBML_dfs(sbml_file)
    logger.info("Saving file to %s", output_model_uri)
    sbml_dfs.to_pickle(output_model_uri)


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
@verbosity_option
def copy_uri(input_uri, output_uri, is_file=True):
    """Copy a uri representing a file or folder from one location to another"""
    logger.info("Copying uri from %s to %s", input_uri, output_uri)
    utils.copy_uri(input_uri, output_uri, is_file=is_file)


@helpers.command(name="validate_sbml_dfs")
@sbml_dfs_input
@verbosity_option
def validate_sbml_dfs(sbml_dfs_uri):
    """Validate a sbml_dfs object"""
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    sbml_dfs.validate()

    logger.info(f"Successfully validated: {sbml_dfs_uri}")


@helpers.command(name="summarize_sbml_dfs")
@sbml_dfs_input
@verbosity_option
def summarize_sbml_dfs(sbml_dfs_uri):
    """Display a summary of an SBML_dfs object"""
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    sbml_dfs.show_summary()


@helpers.command(name="validate_assets")
@sbml_dfs_input
@click.option(
    "--napistu_graph_uri", "-g", type=str, help="URI to NapistuGraph pickle file"
)
@click.option(
    "--precomputed_distances_uri",
    "-d",
    type=str,
    help="URI to precomputed distances parquet file",
)
@click.option(
    "--identifiers_df_uri", "-i", type=str, help="URI to identifiers DataFrame TSV file"
)
@verbosity_option
def validate_assets(
    sbml_dfs_uri: str,
    napistu_graph_uri: str = None,
    precomputed_distances_uri: str = None,
    identifiers_df_uri: str = None,
):
    """Validate assets for consistency

    Loads an SBML_dfs object and optionally validates it against other assets:
    - NapistuGraph: Network representation of the SBML_dfs
    - Precomputed distances: Distance matrix between vertices in the graph
    - Identifiers DataFrame: Systematic identifiers for compartmentalized species

    At least one optional asset must be provided for validation to occur.
    """

    # Load the required SBML_dfs
    logger.info(f"Loading SBML_dfs from: {sbml_dfs_uri}")
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)

    # Load optional assets
    napistu_graph = None
    if napistu_graph_uri:
        logger.info(f"Loading NapistuGraph from: {napistu_graph_uri}")
        napistu_graph = NapistuGraph.from_pickle(napistu_graph_uri)

    precomputed_distances = None
    if precomputed_distances_uri:
        logger.info(f"Loading precomputed distances from: {precomputed_distances_uri}")
        precomputed_distances = pd.read_parquet(precomputed_distances_uri)

    identifiers_df = None
    if identifiers_df_uri:
        logger.info(f"Loading identifiers DataFrame from: {identifiers_df_uri}")
        identifiers_df = pd.read_csv(identifiers_df_uri, sep="\t")

    # Validate assets
    logger.info("Validating assets...")
    validate_assets_func(
        sbml_dfs=sbml_dfs,
        napistu_graph=napistu_graph,
        precomputed_distances=precomputed_distances,
        identifiers_df=identifiers_df,
    )

    logger.info("Asset validation completed successfully!")


@click.group()
def stats():
    """Various functions to calculate network statistics

    The statistics are saved as json files
    """
    pass


@stats.command(name="sbml_dfs_network")
@sbml_dfs_input
@click.argument("output_uri", type=str)
def calculate_sbml_dfs_stats(sbml_dfs_uri, output_uri):
    """Calculate statistics for a sbml_dfs object"""
    sbml_dfs = SBML_dfs.from_pickle(sbml_dfs_uri)
    stats = sbml_dfs.get_summary()
    utils.save_json(output_uri, stats)


@stats.command(name="igraph_network")
@click.argument("input_uri", type=str)
@click.argument("output_uri", type=str)
def calculate_igraph_stats(input_uri, output_uri):
    """Calculate statistics for an igraph object"""

    graph = NapistuGraph.from_pickle(input_uri)
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
