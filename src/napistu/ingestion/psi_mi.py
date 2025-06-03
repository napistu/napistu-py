from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any

from napistu import utils
from napistu.ingestion.constants import PSI_MI_INTACT_DEFAULT_OUTPUT_DIR
from napistu.ingestion.constants import PSI_MI_INTACT_FTP_URL
from napistu.ingestion.constants import PSI_MI_INTACT_SPECIES_TO_BASENAME
from napistu.ingestion.constants import PSI_MI_INTACT_XML_NAMESPACE


logger = logging.getLogger(__name__)


def format_psi(
    xml_path: str, xml_namespace: str = PSI_MI_INTACT_XML_NAMESPACE
) -> list[dict[str, Any]]:
    """
    Format PSI 3.0

    Format an .xml file containing molecular interactions following the PSI 3.0 format.

    Args:
        xml_path (str): path to a .xml file
        xml_namespace (str): Namespace for the xml file

    Returns:
        entry_list (list): a list containing molecular interaction entry dicts of the format:
            - source : dict containing the database that interactions were drawn from.
            - experiment : a simple summary of the experimental design and the publication.
            - interactor_list : list containing dictionaries annotating the molecules
              (defined by their "interactor_id") involved in interactions.
            - interactions_list : list containing dictionaries annotating molecular
              interactions involving a set of "interactor_id"s.
    """

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"{xml_path} was not found")

    et = ET.parse(xml_path)

    # the root should be an entrySet if this is a PSI 3.0 file
    entry_set = et.getroot()
    if entry_set.tag != PSI_MI_INTACT_XML_NAMESPACE + "entrySet":
        raise ValueError(
            f"Expected root tag to be {PSI_MI_INTACT_XML_NAMESPACE + 'entrySet'}, got {entry_set.tag}"
        )

    entry_nodes = entry_set.findall(f"./{PSI_MI_INTACT_XML_NAMESPACE}entry")

    logger.info(f"Processing {len(entry_nodes)} entries from {xml_path}")

    formatted_entries = [_format_entry(an_entry) for an_entry in entry_nodes]

    return formatted_entries


def _download_intact_species(
    species: str,
    output_dir_path: str = PSI_MI_INTACT_DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
):
    """
    Download IntAct Species

    Download the PSM-30 XML files from IntAct for a species of interest.

    Args:
        species (str): The species name (Genus species) to work with
        output_dir_path (str): Local directory to create an unzip files into
        overwrite (bool): Overwrite an existing output directory. Default: False

    Returns:
        None

    """
    if species not in PSI_MI_INTACT_SPECIES_TO_BASENAME.keys():
        raise ValueError(
            f"The provided species {species} did not match any of the species in INTACT_SPECIES_TO_BASENAME: "
            f"{', '.join(PSI_MI_INTACT_SPECIES_TO_BASENAME.keys())}"
        )

    intact_species_url = os.path.join(
        PSI_MI_INTACT_FTP_URL, f"{PSI_MI_INTACT_SPECIES_TO_BASENAME[species]}.zip"
    )

    logger.info(f"Downloading and unzipping {intact_species_url}")

    utils.download_and_extract(
        intact_species_url,
        output_dir_path=output_dir_path,
        download_method="ftp",
        overwrite=overwrite,
    )


def _format_entry(an_entry) -> dict[str, Any]:
    """Extract a single XML entry of interactors and interactions."""

    if an_entry.tag != PSI_MI_INTACT_XML_NAMESPACE + "entry":
        raise ValueError(
            f"Expected entry tag to be {PSI_MI_INTACT_XML_NAMESPACE + 'entry'}, got {an_entry.tag}"
        )

    entry_dict = {
        "source": _format_entry_source(an_entry),
        "experiment": _format_entry_experiment(an_entry),
        "interactor_list": _format_entry_interactor_list(an_entry),
        "interactions_list": _format_entry_interactions(an_entry),
    }

    return entry_dict


def _format_entry_source(an_entry) -> dict[str, str]:
    """Format the source describing the provenance of an XML entry."""

    assert an_entry.tag == PSI_MI_INTACT_XML_NAMESPACE + "entry"

    source_names = an_entry.find(
        f".{PSI_MI_INTACT_XML_NAMESPACE}source/.{PSI_MI_INTACT_XML_NAMESPACE}names"
    )

    out = {
        "short_label": source_names.find(
            f".{PSI_MI_INTACT_XML_NAMESPACE}shortLabel"
        ).text,
        "full_name": source_names.find(f".{PSI_MI_INTACT_XML_NAMESPACE}fullName").text,
    }

    return out


def _format_entry_experiment(an_entry) -> dict[str, str]:
    """Format experiment-level information in an XML entry."""

    assert an_entry.tag == PSI_MI_INTACT_XML_NAMESPACE + "entry"

    experiment_info = an_entry.find(
        f".{PSI_MI_INTACT_XML_NAMESPACE}experimentList/.{PSI_MI_INTACT_XML_NAMESPACE}experimentDescription"
    )

    primary_ref = experiment_info.find(
        f".{PSI_MI_INTACT_XML_NAMESPACE}bibref/{PSI_MI_INTACT_XML_NAMESPACE}xref/{PSI_MI_INTACT_XML_NAMESPACE}primaryRef"
    )

    out = {
        "experiment_name": experiment_info.find(
            f".{PSI_MI_INTACT_XML_NAMESPACE}names/{PSI_MI_INTACT_XML_NAMESPACE}fullName"
        ).text,
        "interaction_method": experiment_info.find(
            f".{PSI_MI_INTACT_XML_NAMESPACE}interactionDetectionMethod/{PSI_MI_INTACT_XML_NAMESPACE}"
            f"names/{PSI_MI_INTACT_XML_NAMESPACE}fullName"
        ).text,
        "primary_ref_db": primary_ref.attrib["db"],
        "primary_ref_id": primary_ref.attrib["id"],
    }

    return out


def _format_entry_interactor_list(an_entry) -> list[dict[str, Any]]:
    """Format the molecular interactors in an XML entry."""

    assert an_entry.tag == PSI_MI_INTACT_XML_NAMESPACE + "entry"

    interactor_list = an_entry.find(f"./{PSI_MI_INTACT_XML_NAMESPACE}interactorList")

    return [_format_entry_interactor(x) for x in interactor_list]


def _format_entry_interactor(interactor) -> dict[str, Any]:
    """Format a single molecular interactor in an interaction list XML node."""

    if interactor.tag != PSI_MI_INTACT_XML_NAMESPACE + "interactor":
        raise ValueError(
            f"Expected interactor tag to be {PSI_MI_INTACT_XML_NAMESPACE + 'interactor'}, got {interactor.tag}"
        )

    # optional full name
    interactor_name_node = interactor.find(
        f"./{PSI_MI_INTACT_XML_NAMESPACE}names/{PSI_MI_INTACT_XML_NAMESPACE}fullName"
    )
    if interactor_name_node is None:
        interactor_name_value = ""  # type: ignore
    else:
        interactor_name_value = interactor_name_node.text  # type: ignore

    interactor_aliases = [
        {"alias_type": x.attrib["type"], "alias_value": x.text}
        for x in interactor.findall(
            f"./{PSI_MI_INTACT_XML_NAMESPACE}names/{PSI_MI_INTACT_XML_NAMESPACE}alias"
        )
    ]  # type: ignore

    out = {
        "interactor_id": interactor.attrib["id"],
        "interactor_label": interactor.find(
            f"./{PSI_MI_INTACT_XML_NAMESPACE}names/{PSI_MI_INTACT_XML_NAMESPACE}shortLabel"
        ).text,
        "interactor_name": interactor_name_value,
        "interactor_aliases": interactor_aliases,
        "interactor_xrefs": _format_entry_interactor_xrefs(interactor),
    }

    return out


def _format_entry_interactor_xrefs(interactor) -> list[dict[str, str]]:
    """Format the cross-references of a single interactor."""

    assert interactor.tag == PSI_MI_INTACT_XML_NAMESPACE + "interactor"

    xref_nodes = [
        *[
            interactor.find(
                f"./{PSI_MI_INTACT_XML_NAMESPACE}xref/{PSI_MI_INTACT_XML_NAMESPACE}primaryRef"
            )
        ],
        *interactor.findall(
            f"./{PSI_MI_INTACT_XML_NAMESPACE}xref/{PSI_MI_INTACT_XML_NAMESPACE}secondaryRef"
        ),
    ]

    out = [
        {"tag": x.tag, "db": x.attrib["db"], "id": x.attrib["id"]} for x in xref_nodes
    ]

    return out


def _format_entry_interactions(an_entry) -> list[dict[str, Any]]:
    """Format the molecular interaction in an XML entry."""

    assert an_entry.tag == PSI_MI_INTACT_XML_NAMESPACE + "entry"

    interaction_list = an_entry.find(f"./{PSI_MI_INTACT_XML_NAMESPACE}interactionList")

    interaction_dicts = [_format_entry_interaction(x) for x in interaction_list]

    return interaction_dicts


def _format_entry_interaction(interaction) -> dict[str, Any]:
    """Format a single interaction in an XML interaction list."""

    if interaction.tag != PSI_MI_INTACT_XML_NAMESPACE + "interaction":
        raise ValueError(
            f"Expected interaction tag to be {PSI_MI_INTACT_XML_NAMESPACE + 'interaction'}, got {interaction.tag}"
        )

    interaction_name = interaction.find(
        f"./{PSI_MI_INTACT_XML_NAMESPACE}names/{PSI_MI_INTACT_XML_NAMESPACE}shortLabel"
    ).text
    interaction_participants = interaction.findall(
        f"./{PSI_MI_INTACT_XML_NAMESPACE}participantList/{PSI_MI_INTACT_XML_NAMESPACE}participant"
    )

    # iterate through particpants and format them as a list of dicts
    interactors = [
        _format_entry_interaction_participants(x) for x in interaction_participants
    ]

    out = {"interaction_name": interaction_name, "interactors": interactors}

    return out


def _format_entry_interaction_participants(interaction_participant) -> dict[str, str]:
    """Format the participants in an XML interaction."""

    if interaction_participant.tag != PSI_MI_INTACT_XML_NAMESPACE + "participant":
        raise ValueError(
            f"Expected participant tag to be {PSI_MI_INTACT_XML_NAMESPACE + 'participant'}, got {interaction_participant.tag}"
        )

    out = {
        "interactor_id": interaction_participant.attrib["id"],
        "biological_role": interaction_participant.find(
            f"./{PSI_MI_INTACT_XML_NAMESPACE}biologicalRole/{PSI_MI_INTACT_XML_NAMESPACE}names/{PSI_MI_INTACT_XML_NAMESPACE}fullName"
        ).text,
        "experimental_role": interaction_participant.find(
            f"./{PSI_MI_INTACT_XML_NAMESPACE}experimentalRoleList/{PSI_MI_INTACT_XML_NAMESPACE}experimentalRole/"
            f"{PSI_MI_INTACT_XML_NAMESPACE}names/{PSI_MI_INTACT_XML_NAMESPACE}fullName"
        ).text,
    }

    return out
