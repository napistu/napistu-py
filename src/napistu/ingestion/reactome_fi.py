import logging
import pandas as pd

from napistu.identifiers import Identifiers
from napistu import utils
from napistu.ingestion.constants import (
    REACTOME_FI,
    REACTOME_FI_RULES_FORWARD,
    REACTOME_FI_RULES_REVERSE,
    REACTOME_FI_URL,
    VALID_REACTOME_FI_DIRECTIONS,
)


logger = logging.getLogger(__name__)


def download_reactome_fi(target_uri: str, url: str = REACTOME_FI_URL) -> None:
    """
    Download the Reactome Functional Interactions (FI) dataset as a TSV file.

    Parameters
    ----------
    target_uri : str
        The URI where the Reactome FI data should be saved. Should end with .tsv
    url : str, optional
        URL to download the zipped Reactome functional interactions TSV from.
        Defaults to REACTOME_FI_URL.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If target_uri does not end with .tsv
    """

    if not target_uri.endswith(".tsv"):
        raise ValueError(f"Target URI must end with .tsv, got {target_uri}")

    file_ext = url.split(".")[-1]
    target_filename = url.split("/")[-1].split(f".{file_ext}")[0]
    logger.info("Start downloading proteinatlas %s to %s", url, target_uri)
    # target_filename is the name of the file in the zip file which will be renamed to target_uri
    utils.download_wget(url, target_uri, target_filename=target_filename)

    return None


def format_reactome_fi_edgelist(interactions: pd.DataFrame):
    """
    Format the Reactome FI interactions DataFrame as an edgelist for network analysis.

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame containing Reactome FI interactions.

    Returns
    -------
    Dictonary of:

    interaction_edgelist : pd.DataFrame
        Table containing molecular interactions with columns:
        - upstream_name : str, matches "s_name" from species_df
        - downstream_name : str, matches "s_name" from species_df
        - upstream_compartment : str, matches "c_name" from compartments_df
        - downstream_compartment : str, matches "c_name" from compartments_df
        - r_name : str, name for the interaction
        - sbo_term : str, SBO term defining interaction type
        - r_Identifiers : identifiers.Identifiers, supporting identifiers
        - r_isreversible : bool, whether reaction is reversible
    species_df : pd.DataFrame
        Table defining molecular species with columns:
        - s_name : str, name of molecular species
        - s_Identifiers : identifiers.Identifiers, species identifiers
    compartments_df : pd.DataFrame
        Table defining compartments with columns:
        - c_name : str, name of compartment
        - c_Identifiers : identifiers.Identifiers, compartment identifiers

    Notes
    -----
    This function is not yet implemented and will raise NotImplementedError.
    """

    raise NotImplementedError("TO DO - This function is incomplete")

    formatted_annotations = _parse_reactome_fi_annotations(interactions)

    # this join will expand some rows to 2 since the bidirectional relationships are captured as separate edges in Napistu
    annotated_interactions = interactions.merge(
        formatted_annotations,
        on=[REACTOME_FI.ANNOTATION, REACTOME_FI.DIRECTION],
        how="left",
    )

    # flip reverse entries so all relationships are forward or undirected
    formatted_interactions = (
        pd.concat(
            [
                annotated_interactions.query("polarity == 'forward'"),
                (
                    annotated_interactions.query("polarity == 'reverse'").rename(
                        columns={
                            REACTOME_FI.GENE1: REACTOME_FI.GENE2,
                            REACTOME_FI.GENE2: REACTOME_FI.GENE1,
                        }
                    )
                ),
            ]
        )[[REACTOME_FI.GENE1, REACTOME_FI.GENE2, "sbo_term_name", "Score"]]
        # looks like they were already unique edges
        .sort_values("Score", ascending=False)
        .groupby([REACTOME_FI.GENE1, REACTOME_FI.GENE2])
        .first()
    )

    fi_edgelist = (
        formatted_interactions.reset_index()
        .rename(
            columns={
                REACTOME_FI.GENE1: "upstream_name",
                REACTOME_FI.GENE2: "downstream_name",
            }
        )
        .assign(r_Identifiers=Identifiers([]))
    )

    return fi_edgelist


def _parse_reactome_fi_annotations(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and annotate Reactome FI interaction types and directions using regex-based rules.

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame containing Reactome FI interactions, with annotation and direction columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with annotation, direction, SBO term name, and polarity for each unique annotation/direction pair.

    Raises
    ------
    ValueError
        If an annotation/direction pair cannot be matched to a rule or if invalid directions are found.
    """

    distinct_annotations = (
        interactions[[REACTOME_FI.ANNOTATION, REACTOME_FI.DIRECTION]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    invalid_directions = distinct_annotations.loc[
        ~distinct_annotations[REACTOME_FI.DIRECTION].isin(VALID_REACTOME_FI_DIRECTIONS),
        "Direction",
    ]
    if len(invalid_directions) > 0:
        raise ValueError(f"Invalid directions: {invalid_directions}")

    annotations = list()
    for _, vals in distinct_annotations.iterrows():
        annot, direction = vals

        forward_match = utils.match_regex_dict(
            annot, REACTOME_FI_RULES_FORWARD.NAME_RULES
        )
        if not forward_match:
            if direction in REACTOME_FI_RULES_FORWARD.DIRECTION_RULES:
                forward_match = REACTOME_FI_RULES_FORWARD.DIRECTION_RULES[direction]

        reverse_match = utils.match_regex_dict(
            annot, REACTOME_FI_RULES_REVERSE.NAME_RULES
        )
        if not reverse_match:
            if direction in REACTOME_FI_RULES_REVERSE.DIRECTION_RULES:
                reverse_match = REACTOME_FI_RULES_REVERSE.DIRECTION_RULES[direction]

        if not (forward_match or reverse_match):
            raise ValueError(f"No match found for {annot} with direction {direction}")

        if forward_match:
            annotations.append(
                {
                    REACTOME_FI.ANNOTATION: annot,
                    REACTOME_FI.DIRECTION: direction,
                    "sbo_term_name": forward_match,
                    "polarity": "forward",
                }
            )

        if reverse_match:
            annotations.append(
                {
                    REACTOME_FI.ANNOTATION: annot,
                    REACTOME_FI.DIRECTION: direction,
                    "sbo_term_name": reverse_match,
                    "polarity": "reverse",
                }
            )

    return pd.DataFrame(annotations)
