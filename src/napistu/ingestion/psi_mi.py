from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any

from napistu import utils
from napistu.constants import IDENTIFIERS, ONTOLOGIES_LIST, BQB
from napistu.identifiers import Identifiers
from napistu.identifiers import parse_ensembl_id
from napistu.constants import SBML_DFS
import pandas as pd
from napistu.ingestion.constants import (
    PSI_MI_INTACT_FTP_URL,
    PSI_MI_INTACT_SPECIES_TO_BASENAME,
    PSI_MI_INTACT_XML_NAMESPACE,
    PSI_MI_DEFS,
    PSI_MI_REFS,
    INTACT_ONTOLOGY_CV_LOOKUP,
)

logger = logging.getLogger(__name__)


def download_intact_psi_mi_xmls(
    output_dir_path: str,
    species: str,
    overwrite: bool = False,
) -> None:
    """
    Download IntAct Species

    Download the PSM-30 XML files from IntAct for a species of interest.

    Parameters
    ----------
    species (str):
        The species name (Genus species) to work with
    output_dir_path (str):
        Local directory to create an unzip files into
    overwrite (bool):
        Overwrite an existing output directory. Default: False
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
    if entry_set.tag != xml_namespace + "entrySet":
        raise ValueError(
            f"Expected root tag to be {xml_namespace + 'entrySet'}, got {entry_set.tag}"
        )

    entry_nodes = entry_set.findall(f"./{xml_namespace}entry")

    logger.info(f"Processing {len(entry_nodes)} entries from {xml_path}")

    formatted_entries = [
        _format_entry(an_entry, xml_namespace) for an_entry in entry_nodes
    ]

    return formatted_entries


def format_psi_mis(intact_xml_dir: str) -> list[dict[str, Any]]:
    """
    Format PSI-MI XML files

    Format PSI-MI XML files into a list of dictionaries.
    """

    if not os.path.isdir(intact_xml_dir):
        raise FileNotFoundError(f"The directory {intact_xml_dir} does not exist")

    xml_files = os.listdir(intact_xml_dir)
    if len(xml_files) == 0:
        raise FileNotFoundError(f"No files found in {intact_xml_dir}")

    logger.info(f"Formatting {len(xml_files)} PSI-MI XML files")

    # only keep the first 10 files
    logger.info("Keeping only the first 10 files while implementing this feature")
    xml_files = xml_files[0:10]

    formatted_psi_mis = []
    for xml_file in xml_files:
        logger.info(f"Formatting {xml_file}")
        xml_path = os.path.join(intact_xml_dir, xml_file)
        formatted_psi_mis.append(format_psi(xml_path))

    return formatted_psi_mis


def _format_entry(an_entry, xml_namespace: str) -> dict[str, Any]:
    """Extract a single XML entry of interactors and interactions."""

    if an_entry.tag != xml_namespace + "entry":
        raise ValueError(
            f"Expected entry tag to be {xml_namespace + 'entry'}, got {an_entry.tag}"
        )

    entry_dict = {
        "source": _format_entry_source(an_entry, xml_namespace),
        "experiment": _format_entry_experiment(an_entry, xml_namespace),
        "interactor_list": _format_entry_interactor_list(an_entry, xml_namespace),
        "interactions_list": _format_entry_interactions(an_entry, xml_namespace),
    }

    return entry_dict


def _format_entry_source(an_entry, xml_namespace: str) -> dict[str, str]:
    """Format the source describing the provenance of an XML entry."""

    assert an_entry.tag == xml_namespace + "entry"

    source_names = an_entry.find(f".{xml_namespace}source/.{xml_namespace}names")

    out = {
        "short_label": _get_optional_text(source_names, f".{xml_namespace}shortLabel"),
        "full_name": _get_optional_text(source_names, f".{xml_namespace}fullName"),
    }

    return out


def _format_entry_experiment(an_entry, xml_namespace: str) -> dict[str, str]:
    """Format experiment-level information in an XML entry."""

    assert an_entry.tag == xml_namespace + "entry"

    experiment_info = an_entry.find(
        f".{xml_namespace}experimentList/.{xml_namespace}experimentDescription"
    )

    primary_ref = experiment_info.find(
        f".{xml_namespace}bibref/{xml_namespace}xref/{xml_namespace}primaryRef"
    )

    out = {
        "experiment_name": _get_optional_text(
            experiment_info, f".{xml_namespace}names/{xml_namespace}fullName"
        ),
        "interaction_method": _get_optional_text(
            experiment_info,
            f".{xml_namespace}interactionDetectionMethod/{xml_namespace}names/{xml_namespace}fullName",
        ),
        "primary_ref_db": primary_ref.attrib["db"],
        "primary_ref_id": primary_ref.attrib["id"],
    }

    return out


def _format_entry_interactor_list(an_entry, xml_namespace: str) -> list[dict[str, Any]]:
    """Format the molecular interactors in an XML entry."""

    assert an_entry.tag == xml_namespace + "entry"

    interactor_list = an_entry.find(f"./{xml_namespace}interactorList")

    return [_format_entry_interactor(x, xml_namespace) for x in interactor_list]


def _format_entry_interactor(interactor, xml_namespace: str) -> dict[str, Any]:
    """Format a single molecular interactor in an interaction list XML node."""

    if interactor.tag != xml_namespace + "interactor":
        raise ValueError(
            f"Expected interactor tag to be {xml_namespace + 'interactor'}, got {interactor.tag}"
        )

    # optional full name
    interactor_name_node = interactor.find(
        f"./{xml_namespace}names/{xml_namespace}fullName"
    )
    if interactor_name_node is None:
        interactor_name_value = ""  # type: ignore
    else:
        interactor_name_value = interactor_name_node.text  # type: ignore

    interactor_aliases = [
        {"alias_type": x.attrib["type"], "alias_value": x.text}
        for x in interactor.findall(f"./{xml_namespace}names/{xml_namespace}alias")
    ]  # type: ignore

    out = {
        "interactor_id": interactor.attrib["id"],
        "interactor_label": _get_optional_text(
            interactor, f"./{xml_namespace}names/{xml_namespace}shortLabel"
        ),
        "interactor_name": interactor_name_value,
        "interactor_aliases": interactor_aliases,
        "interactor_xrefs": _format_entry_interactor_xrefs(interactor, xml_namespace),
    }

    return out


def _format_entry_interactor_xrefs(
    interactor, xml_namespace: str
) -> list[dict[str, str]]:
    """Format the cross-references of a single interactor."""

    assert interactor.tag == xml_namespace + "interactor"

    xref_nodes = [
        *[interactor.find(f"./{xml_namespace}xref/{xml_namespace}primaryRef")],
        *interactor.findall(f"./{xml_namespace}xref/{xml_namespace}secondaryRef"),
    ]

    out = [
        {"tag": x.tag, "db": x.attrib["db"], "id": x.attrib["id"]} for x in xref_nodes
    ]

    return out


def _format_entry_interactions(an_entry, xml_namespace: str) -> list[dict[str, Any]]:
    """Format the molecular interaction in an XML entry."""

    assert an_entry.tag == xml_namespace + "entry"

    interaction_list = an_entry.find(f"./{xml_namespace}interactionList")

    interaction_dicts = [
        _format_entry_interaction(x, xml_namespace) for x in interaction_list
    ]

    return interaction_dicts


def _format_entry_interaction(interaction, xml_namespace: str) -> dict[str, Any]:
    """Format a single interaction in an XML interaction list."""

    if interaction.tag != xml_namespace + "interaction":
        raise ValueError(
            f"Expected interaction tag to be {xml_namespace + 'interaction'}, got {interaction.tag}"
        )

    interaction_participants = interaction.findall(
        f"./{xml_namespace}participantList/{xml_namespace}participant"
    )

    # iterate through particpants and format them as a list of dicts
    interactors = [
        _format_entry_interaction_participants(x, xml_namespace)
        for x in interaction_participants
    ]

    out = {
        "interaction_name": _get_optional_text(
            interaction, f"./{xml_namespace}names/{xml_namespace}shortLabel"
        ),
        "interactors": interactors,
    }

    return out


def _format_entry_interaction_participants(
    interaction_participant, xml_namespace: str
) -> dict[str, str]:
    """Format the participants in an XML interaction."""

    if interaction_participant.tag != xml_namespace + "participant":
        raise ValueError(
            f"Expected participant tag to be {xml_namespace + 'participant'}, got {interaction_participant.tag}"
        )

    out = {
        "interactor_id": interaction_participant.attrib["id"],
        "biological_role": _get_optional_text(
            interaction_participant,
            f"./{xml_namespace}biologicalRole/{xml_namespace}names/{xml_namespace}fullName",
        ),
        "experimental_role": _get_optional_text(
            interaction_participant,
            f"./{xml_namespace}experimentalRoleList/{xml_namespace}experimentalRole/{xml_namespace}names/{xml_namespace}fullName",
        ),
    }

    return out


def _sanitize_intact_identifiers(
    identifier: str,
    ontology: str,
    include_unmatched: bool = False,
    ensembl_vague: str = "ensembl",
    ontology_cv_lookup: dict = INTACT_ONTOLOGY_CV_LOOKUP,
) -> dict:

    if ontology in ontology_cv_lookup.keys():
        ontology = ontology_cv_lookup[ontology]

    if ontology == ensembl_vague:
        try:
            identifier, ontology, _ = parse_ensembl_id(identifier)
        except ValueError:
            # these are mostly obscure orthologues
            return None

    if ontology not in ONTOLOGIES_LIST:
        if not include_unmatched:
            return None
        else:
            logger.warning(f"Ontology {ontology} not in ONTOLOGIES_LIST")

    return {
        IDENTIFIERS.ONTOLOGY: ontology,
        IDENTIFIERS.IDENTIFIER: identifier,
        IDENTIFIERS.URL: None,
        IDENTIFIERS.BQB: BQB.IS,
    }


def _format_study_level_identifiers(one_study):

    study_data = one_study[PSI_MI_DEFS.EXPERIMENT]
    reaction_id = _sanitize_intact_identifiers(
        ontology=study_data[PSI_MI_REFS.PRIMARY_REF_DB],
        identifier=study_data[PSI_MI_REFS.PRIMARY_REF_ID],
    )
    return Identifiers([reaction_id if reaction_id is not None else None])


def _create_reaction_species_df(one_study):
    """
    Format the interactions in the study into a dataframe of reaction species.
    """

    reaction_species = list()
    for interaction in one_study[PSI_MI_DEFS.INTERACTIONS_LIST]:
        for participant in interaction[PSI_MI_DEFS.INTERACTORS]:
            participant[PSI_MI_DEFS.INTERACTION_NAME] = interaction[
                PSI_MI_DEFS.INTERACTION_NAME
            ]
            reaction_species.append(participant)

    return pd.DataFrame(reaction_species)


def _create_species_df(one_study):

    species = list()
    for spec in one_study[PSI_MI_DEFS.INTERACTOR_LIST]:

        ids = [
            _sanitize_intact_identifiers(x["id"], x["db"], include_unmatched=False)
            for x in spec[PSI_MI_DEFS.INTERACTOR_XREFS]
        ]
        ids = [x for x in ids if x is not None]
        ids = Identifiers(ids)

        spec_summary = {
            PSI_MI_DEFS.INTERACTOR_ID: spec[PSI_MI_DEFS.INTERACTOR_ID],
            PSI_MI_DEFS.INTERACTOR_LABEL: spec[PSI_MI_DEFS.INTERACTOR_LABEL],
            PSI_MI_DEFS.INTERACTOR_NAME: spec[PSI_MI_DEFS.INTERACTOR_NAME],
            SBML_DFS.S_IDENTIFIERS: ids,
        }
        species.append(spec_summary)

    species_df = pd.DataFrame(species)
    return species_df


def _get_optional_text(element, xpath: str, default: str = "") -> str:
    """
    Safely extract text from an optional XML element.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        The parent element to search within
    xpath : str
        The xpath expression to find the child element
    default : str, optional
        Default value to return if element is not found, by default ""

    Returns
    -------
    str
        The text content of the element, or the default value if not found
    """
    child = element.find(xpath)
    if child is None:
        return default
    return child.text or default
