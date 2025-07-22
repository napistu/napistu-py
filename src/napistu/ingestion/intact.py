import os
import logging

import pandas as pd

from napistu import utils
from napistu.identifiers import Identifiers
from napistu.constants import IDENTIFIERS, SBML_DFS
from napistu.ingestion.constants import (
    PSI_MI_DEFS,
    PSI_MI_INTACT_FTP_URL,
    PSI_MI_INTACT_SPECIES_TO_BASENAME,
)

logger = logging.getLogger(__name__)


def download_intact_xmls(
    output_dir_path: str,
    latin_species: str,
    overwrite: bool = False,
) -> None:
    """
    Download IntAct Species

    Download the PSM-30 XML files from IntAct for a species of interest.

    Parameters
    ----------
    output_dir_path (str):
        Local directory to create an unzip files into
    latin_species (str):
        The species name (Genus species) to work with
    overwrite (bool):
        Overwrite an existing output directory. Default: False
    """

    intact_species_basename = _get_intact_species_basename(latin_species)

    intact_species_url = os.path.join(
        PSI_MI_INTACT_FTP_URL, f"{intact_species_basename}.zip"
    )

    logger.info(f"Downloading and unzipping {intact_species_url}")

    utils.download_and_extract(
        intact_species_url,
        output_dir_path=output_dir_path,
        download_method="ftp",
        overwrite=overwrite,
    )


def create_species_df(
    raw_species_df: pd.DataFrame,
    raw_species_identifiers_df: pd.DataFrame,
    latin_species: str,
):
    """
    Create a species dataframe from the raw species dataframe and the raw species identifiers dataframe.

    Parameters
    ----------
    raw_species_df : pd.DataFrame
        The raw species dataframe.
    raw_species_identifiers_df : pd.DataFrame
        The raw species identifiers dataframe.
    latin_species : str
        The latin species name in the format "Genus species".

    Returns
    -------
    lookup_table : pd.Series
        A lookup table mapping study_id and interactor_id to the molecular species name.
    species_df : pd.DataFrame
        The molecular species dataframe.
    """

    # filter to just matchees within the same species
    intact_species_basename = _get_intact_species_basename(latin_species)

    # all partipants
    valid_species_mask = raw_species_df[PSI_MI_DEFS.INTERACTOR_LABEL].str.endswith(
        intact_species_basename
    )

    logger.info(
        f"Retaining {sum(valid_species_mask)} interactors which are from {intact_species_basename} and removing {sum(~valid_species_mask)} interactors which are not."
    )
    valid_species_df = raw_species_df.loc[valid_species_mask]

    species_df = (
        raw_species_identifiers_df.merge(
            valid_species_df[
                [
                    PSI_MI_DEFS.INTERACTOR_ID,
                    PSI_MI_DEFS.STUDY_ID,
                    PSI_MI_DEFS.INTERACTOR_LABEL,
                ]
            ],
            on=[PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTOR_ID],
            how="inner",
        )[
            [
                PSI_MI_DEFS.INTERACTOR_LABEL,
                IDENTIFIERS.ONTOLOGY,
                IDENTIFIERS.IDENTIFIER,
                IDENTIFIERS.BQB,
            ]
        ]
        .drop_duplicates()
        .groupby([PSI_MI_DEFS.INTERACTOR_LABEL])
        .apply(lambda x: Identifiers(x.to_dict(orient="records")), include_groups=False)
        .rename(SBML_DFS.S_IDENTIFIERS)
        .rename_axis(SBML_DFS.S_NAME)
        .reset_index()
    )

    lookup_table = valid_species_df.set_index(
        [PSI_MI_DEFS.STUDY_ID, PSI_MI_DEFS.INTERACTOR_ID]
    )[PSI_MI_DEFS.INTERACTOR_LABEL]

    return lookup_table, species_df


def _get_intact_species_basename(latin_species: str) -> str:

    if latin_species not in PSI_MI_INTACT_SPECIES_TO_BASENAME.keys():
        raise ValueError(
            f"The provided species {latin_species} did not match any of the species in INTACT_SPECIES_TO_BASENAME: "
            f"{', '.join(PSI_MI_INTACT_SPECIES_TO_BASENAME.keys())}"
            "If this is a species supported by IntAct please add the species to the PSI_MI_INTACT_SPECIES_TO_BASENAME dictionary."
        )

    return PSI_MI_INTACT_SPECIES_TO_BASENAME[latin_species]
