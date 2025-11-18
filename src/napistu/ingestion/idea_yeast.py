from __future__ import annotations

import datetime
import logging

import pandas as pd

from napistu import identifiers, sbml_dfs_core, sbml_dfs_utils, utils
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    SBML_DFS,
)
from napistu.ingestion.constants import (
    DATA_SOURCE_DESCRIPTIONS,
    DATA_SOURCES,
    IDEA_YEAST,
    INTERACTION_EDGELIST_DEFS,
    LATIN_SPECIES_NAMES,
)
from napistu.source import Source

logger = logging.getLogger(__name__)


def download_idea(target_uri: str) -> None:
    """Download IDEA kinetics data from a ZIP archive and save as a single file.

    Parameters
    ----------
    target_uri : str
        The URI where the IDEA kinetics data should be saved. Can be a local path
        or GCS URI. Should typically end with .tsv or .txt.

    Returns
    -------
    None

    Notes
    -----
    Downloads the IDEA kinetics dataset from the specified URL and saves
    it to the specified target URI. The data is downloaded from a ZIP archive
    and the main data file is automatically extracted.

    The IDEA (Induction Dynamics Expression Atlas) dataset provides kinetic
    information about transcriptional regulation in yeast (Saccharomyces cerevisiae).
    """

    # The ZIP archive contains a single TSV file with kinetics data
    # Based on the URL pattern, the file is likely named "idea_kinetics.tsv" or similar
    logger.info(
        "Downloading IDEA kinetics data from %s to %s",
        IDEA_YEAST.KINETICS_URL,
        target_uri,
    )

    # Use download_wget with target_filename to extract the specific file from the ZIP
    utils.download_wget(
        IDEA_YEAST.KINETICS_URL, target_uri, target_filename=IDEA_YEAST.KINETICS_FILE
    )

    return None


def convert_idea_kinetics_to_sbml_dfs(
    idea_kinetics: pd.DataFrame,
) -> sbml_dfs_core.SBML_dfs:
    """
    Convert IDEA Kinetics to SBML DFs

    Format yeast induction regulator->target relationships as a directed graph.

    Parameters
    ----------
    idea_kinetics: pd.DataFrame
        DataFrame containing IDEA Kinetics data.

    Returns
    -------
        SBML_dfs: an SBML_dfs object containing molecular species and their interactions.
        Kinetic attributes are included as reactions_data.

    """

    # separate based on whether the change is probably direct or indirect
    idea_kinetics["directness"] = [
        "direct" if t_rise < 15 else "indirect" for t_rise in idea_kinetics["t_rise"]
    ]

    # reduce cases of multiple TF-target pairs to a single entry
    distinct_edges = (
        idea_kinetics.groupby([IDEA_YEAST.SOURCE, IDEA_YEAST.TARGET], as_index=True)
        .apply(_summarize_idea_pairs)
        .reset_index()
    )

    # add some more fields are reformat
    interaction_edgelist = distinct_edges.rename(
        {
            IDEA_YEAST.SOURCE: INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            IDEA_YEAST.TARGET: INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            "role": INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM,
        },
        axis=1,
    ).assign(
        # tag reactions with the IDEA publication
        r_Identifiers=identifiers._format_Identifiers_pubmed(IDEA_YEAST.PUBMED_ID),
    )

    # create some nice interaction names before we rename the roles as SBO terms
    interaction_edgelist["r_name"] = [
        f"{u} {d} {r} of {t}"
        for u, d, r, t in zip(
            interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM],
            interaction_edgelist["directness"],
            interaction_edgelist[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM],
            interaction_edgelist[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM],
        )
    ]

    species_df = pd.DataFrame(
        {
            SBML_DFS.S_NAME: list(
                {
                    *idea_kinetics[IDEA_YEAST.SOURCE],
                    *idea_kinetics[IDEA_YEAST.TARGET],
                }
            )
        }
    )

    # create Identifiers objects for each species
    species_df[SBML_DFS.S_IDENTIFIERS] = [
        identifiers.Identifiers(
            [
                {
                    IDENTIFIERS.ONTOLOGY: "gene_name",
                    IDENTIFIERS.IDENTIFIER: x,
                    IDENTIFIERS.BQB: BQB.IS,
                }
            ]
        )
        for x in species_df[SBML_DFS.S_NAME]
    ]

    # Constant fields (for this data source)

    # setup compartments (just treat this as uncompartmentalized for now)
    compartments_df = sbml_dfs_utils.stub_compartments()

    # define model-level metadata
    model_source = Source.single_entry(
        model=DATA_SOURCES.IDEA_YEAST,
        pathway_id=DATA_SOURCES.IDEA_YEAST,
        data_source=DATA_SOURCES.IDEA_YEAST,
        organismal_species=LATIN_SPECIES_NAMES.SACCHAROMYCES_CEREVISIAE,
        name=DATA_SOURCE_DESCRIPTIONS[DATA_SOURCES.IDEA_YEAST],
        date=datetime.date.today().strftime("%Y%m%d"),
    )

    sbml_dfs = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist=interaction_edgelist,
        species_df=species_df,
        compartments_df=compartments_df,
        model_source=model_source,
        # additional attributes (directness) are added to reactions_data
        keep_reactions_data=DATA_SOURCES.IDEA_YEAST,
    )
    sbml_dfs.validate()

    return sbml_dfs


def _summarize_idea_pairs(pairs_data: pd.DataFrame) -> pd.Series:
    """Rollup multiple records of a TF->target pair into a single summary."""

    # specify how to aggregate results if there are more than one entry for a TF-target pair
    # pull most attributes from the earliest change
    # this will favor direct over indirect naturally
    earliest_change = pairs_data.sort_values("t_rise").iloc[0].to_dict()

    KEYS_SUMMARIZED = ["v_inter", "v_final", "t_rise", "t_fall", "rate", "directness"]
    kinetic_timing_dict = {k: earliest_change[k] for k in KEYS_SUMMARIZED}

    # map v_inter (log2 fold-change change following perturbation) onto SBO terms for interactions
    if (any(pairs_data["v_inter"] > 0)) and (any(pairs_data["v_inter"] < 0)):
        kinetic_timing_dict["role"] = "modifier"
    elif all(pairs_data["v_inter"] > 0):
        kinetic_timing_dict["role"] = "stimulator"
    elif all(pairs_data["v_inter"] < 0):
        kinetic_timing_dict["role"] = "inhibitor"
    else:
        ValueError("Unexpected v_inter values")

    return pd.Series(kinetic_timing_dict)
