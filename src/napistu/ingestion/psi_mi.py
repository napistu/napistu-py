from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any

from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    ONTOLOGIES_LIST,
)
from napistu.identifiers import Identifiers
from napistu.identifiers import parse_ensembl_id
from napistu.constants import SBML_DFS
import pandas as pd
from napistu.ingestion.constants import (
    INTACT_ONTOLOGY_CV_LOOKUP,
    INTACT_ONTOLOGY_ENSEMBL_VAGUE,
    PSI_MI_INTACT_XML_NAMESPACE,
    PSI_MI_DEFS,
    PSI_MI_RAW_ATTRS,
    PSI_MI_REFS,
)

logger = logging.getLogger(__name__)


def format_psi(
    xml_path: str,
    xml_namespace: str = PSI_MI_INTACT_XML_NAMESPACE,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    Format PSI 3.0

    Format an .xml file containing molecular interactions following the PSI 3.0 format.

    Parameters
    ----------
    xml_path (str):
        path to a .xml file
    xml_namespace (str):
        Namespace for the xml file
    verbose (bool):
        Whether to print verbose output

    Returns
    -------
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

    if verbose:
        logger.info(f"Processing {len(entry_nodes)} entries from {xml_path}")

    formatted_entries = [
        _format_entry(an_entry, xml_namespace) for an_entry in entry_nodes
    ]

    return formatted_entries


def format_psi_mis(
    intact_xml_dir: str,
    xml_namespace: str = PSI_MI_INTACT_XML_NAMESPACE,
    verbose: bool = False,
    files_to_process: int = -1,
) -> list[dict[str, Any]]:
    """
    Format PSI-MI XML files

    Format PSI-MI XML files into a list of dictionaries.

    Parameters
    ----------
    intact_xml_dir (str):
        Path to the directory containing the PSI-MI XML files
    xml_namespace (str):
        Namespace for the xml file
    verbose (bool):
        Whether to print verbose output

    Returns
    -------
    formatted_psi_mis (list): a list containing molecular interaction entry dicts of the format:
        - source : dict containing the database that interactions were drawn from.
        - experiment : a simple summary of the experimental design and the publication.
        - interactor_list : list containing dictionaries annotating the molecules
        (defined by their "interactor_id") involved in interactions.
    """

    if not os.path.isdir(intact_xml_dir):
        raise FileNotFoundError(f"The directory {intact_xml_dir} does not exist")

    xml_files = os.listdir(intact_xml_dir)
    if len(xml_files) == 0:
        raise FileNotFoundError(f"No files found in {intact_xml_dir}")

    if files_to_process > 0:
        logger.info(f"Processing only the first {files_to_process} files")
        xml_files = xml_files[:files_to_process]

    logger.info(f"Formatting {len(xml_files)} PSI-MI XML files")

    formatted_psi_mis = []
    for xml_file in xml_files:
        if verbose:
            logger.info(f"Formatting {xml_file}")
        xml_path = os.path.join(intact_xml_dir, xml_file)
        formatted_psi_mis.append(format_psi(xml_path, xml_namespace, verbose))

    return formatted_psi_mis


def _format_entry(an_entry, xml_namespace: str) -> dict[str, Any]:
    """Extract a single XML entry of interactors and interactions."""

    if an_entry.tag != xml_namespace + "entry":
        raise ValueError(
            f"Expected entry tag to be {xml_namespace + 'entry'}, got {an_entry.tag}"
        )

    entry_dict = {
        PSI_MI_DEFS.SOURCE: _format_entry_source(an_entry, xml_namespace),
        PSI_MI_DEFS.EXPERIMENT: _format_entry_experiment(an_entry, xml_namespace),
        PSI_MI_DEFS.INTERACTOR_LIST: _format_entry_interactor_list(
            an_entry, xml_namespace
        ),
        PSI_MI_DEFS.INTERACTIONS_LIST: _format_entry_interactions(
            an_entry, xml_namespace
        ),
    }

    return entry_dict


def _format_entry_source(an_entry, xml_namespace: str) -> dict[str, str]:
    """Format the source describing the provenance of an XML entry."""

    assert an_entry.tag == xml_namespace + "entry"

    source_names = an_entry.find(f".{xml_namespace}source/.{xml_namespace}names")

    out = {
        PSI_MI_DEFS.SHORT_LABEL: _get_optional_text(
            source_names, f".{xml_namespace}shortLabel"
        ),
        PSI_MI_DEFS.FULL_NAME: _get_optional_text(
            source_names, f".{xml_namespace}fullName"
        ),
    }

    return out


def _format_entry_experiment(an_entry, xml_namespace: str) -> dict[str, str]:
    """Format experiment-level information in an XML entry."""

    assert an_entry.tag == xml_namespace + "entry"

    experiment_info = an_entry.find(
        f".{xml_namespace}experimentList/.{xml_namespace}experimentDescription"
    )

    out = {
        PSI_MI_DEFS.EXPERIMENT_NAME: _get_optional_text(
            experiment_info, f".{xml_namespace}names/{xml_namespace}fullName"
        ),
        PSI_MI_DEFS.INTERACTION_METHOD: _get_optional_text(
            experiment_info,
            f".{xml_namespace}interactionDetectionMethod/{xml_namespace}names/{xml_namespace}fullName",
        ),
        PSI_MI_REFS.PRIMARY_REF_DB: _get_optional_attribute(
            experiment_info,
            f".{xml_namespace}bibref/{xml_namespace}xref/{xml_namespace}primaryRef",
            PSI_MI_RAW_ATTRS.DB,
        ),
        PSI_MI_REFS.PRIMARY_REF_ID: _get_optional_attribute(
            experiment_info,
            f".{xml_namespace}bibref/{xml_namespace}xref/{xml_namespace}primaryRef",
            PSI_MI_RAW_ATTRS.ID,
        ),
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
        {"alias_type": x.attrib.get("type", ""), "alias_value": x.text or ""}
        for x in interactor.findall(f"./{xml_namespace}names/{xml_namespace}alias")
    ]  # type: ignore

    out = {
        PSI_MI_DEFS.INTERACTOR_ID: interactor.attrib.get(PSI_MI_RAW_ATTRS.ID, ""),
        PSI_MI_DEFS.INTERACTOR_LABEL: _get_optional_text(
            interactor, f"./{xml_namespace}names/{xml_namespace}shortLabel"
        ),
        PSI_MI_DEFS.INTERACTOR_NAME: interactor_name_value,
        PSI_MI_DEFS.INTERACTOR_ALIASES: interactor_aliases,
        PSI_MI_DEFS.INTERACTOR_XREFS: _format_entry_interactor_xrefs(
            interactor, xml_namespace
        ),
    }

    return out


def _format_entry_interactor_xrefs(
    interactor, xml_namespace: str
) -> list[dict[str, str]]:
    """Format the cross-references of a single interactor."""

    assert interactor.tag == xml_namespace + PSI_MI_RAW_ATTRS.INTERACTOR

    xref_nodes = [
        *[
            interactor.find(
                f"./{xml_namespace}xref/{xml_namespace}{PSI_MI_RAW_ATTRS.PRIMARY_REF}"
            )
        ],
        *interactor.findall(
            f"./{xml_namespace}xref/{xml_namespace}{PSI_MI_RAW_ATTRS.SECONDARY_REF}"
        ),
    ]

    out = [
        {
            PSI_MI_DEFS.REF_TYPE: x.tag.endswith(PSI_MI_RAW_ATTRS.PRIMARY_REF)
            and PSI_MI_DEFS.PRIMARY
            or PSI_MI_DEFS.SECONDARY,
            PSI_MI_RAW_ATTRS.TAG: x.tag,
            PSI_MI_RAW_ATTRS.DB: x.attrib.get(PSI_MI_RAW_ATTRS.DB, ""),
            PSI_MI_RAW_ATTRS.ID: x.attrib.get(PSI_MI_RAW_ATTRS.ID, ""),
        }
        for x in xref_nodes
    ]

    return out


def _format_entry_interactions(an_entry, xml_namespace: str) -> list[dict[str, Any]]:
    """Format the molecular interaction in an XML entry."""

    assert an_entry.tag == xml_namespace + PSI_MI_RAW_ATTRS.ENTRY

    interaction_list = an_entry.find(
        f"./{xml_namespace}{PSI_MI_RAW_ATTRS.INTERACTIONS_LIST}"
    )

    interaction_dicts = [
        _format_entry_interaction(x, xml_namespace) for x in interaction_list
    ]

    return interaction_dicts


def _format_entry_interaction(interaction, xml_namespace: str) -> dict[str, Any]:
    """Format a single interaction in an XML interaction list."""

    VALID_TAGS = [
        xml_namespace + PSI_MI_RAW_ATTRS.INTERACTION,
        xml_namespace + PSI_MI_RAW_ATTRS.ABSTRACT_INTERACTION,
    ]
    if interaction.tag not in VALID_TAGS:
        logger.warning(
            f"Expected interaction tag to be {VALID_TAGS}, got {interaction.tag}"
        )

    interaction_participants = interaction.findall(
        f"./{xml_namespace}{PSI_MI_RAW_ATTRS.PARTICIPANT_LIST}/{xml_namespace}{PSI_MI_RAW_ATTRS.PARTICIPANT}"
    )

    # iterate through particpants and format them as a list of dicts
    interactors = [
        _format_entry_interaction_participants(x, xml_namespace)
        for x in interaction_participants
    ]

    out = {
        PSI_MI_DEFS.INTERACTION_NAME: _get_optional_text(
            interaction, f"./{xml_namespace}names/{xml_namespace}shortLabel"
        ),
        PSI_MI_DEFS.INTERACTION_TYPE: _get_optional_text(
            interaction,
            f"./{xml_namespace}interactionType/{xml_namespace}names/{xml_namespace}fullName",
        ),
        PSI_MI_DEFS.INTERACTORS: interactors,
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
        PSI_MI_DEFS.PARTICIPANT_ID: interaction_participant.attrib.get(
            PSI_MI_RAW_ATTRS.ID, ""
        ),
        PSI_MI_DEFS.INTERACTOR_ID: _get_optional_text(
            interaction_participant,
            f"./{xml_namespace}interactorRef",
        ),
        PSI_MI_DEFS.BIOLOGICAL_ROLE: _get_optional_text(
            interaction_participant,
            f"./{xml_namespace}biologicalRole/{xml_namespace}names/{xml_namespace}fullName",
        ),
        PSI_MI_DEFS.EXPERIMENTAL_ROLE: _get_optional_text(
            interaction_participant,
            f"./{xml_namespace}experimentalRoleList/{xml_namespace}experimentalRole/{xml_namespace}names/{xml_namespace}fullName",
        ),
    }

    return out


def _sanitize_intact_identifiers(
    identifier: str,
    ontology: str,
    include_unmatched: bool = False,
    ensembl_vague: str = INTACT_ONTOLOGY_ENSEMBL_VAGUE,
    ontology_cv_lookup: dict = INTACT_ONTOLOGY_CV_LOOKUP,
) -> dict:
    """
    Sanitize an IntAct identifier.

    Parameters
    ----------
    identifier (str):
        The identifier to sanitize
    ontology (str):
        The ontology of the identifier
    include_unmatched (bool):
        Whether to include ontologies that are not in ONTOLOGIES_LIST
    ensembl_vague (str):
        The vague ontology to use for Ensembl identifiers. This will be used to distinguish
        ENSG, ENST and ENSP identifiers.
    ontology_cv_lookup (dict):
        A dictionary mapping IntAct ontologies to entries in the Napistu ONTOLOGIES CV.

    Returns
    -------
    dict:
        A dictionary containing the sanitized identifier, ontology, and URL.
    """

    if ontology in ontology_cv_lookup.keys():
        ontology = ontology_cv_lookup[ontology]

    if ontology == ensembl_vague:
        try:
            identifier, ontology, _ = parse_ensembl_id(identifier)
        except Exception:
            # these are mostly obscure orthologues
            return {}

    if ontology not in ONTOLOGIES_LIST:
        if not include_unmatched:
            return {}
        else:
            logger.warning(f"Ontology {ontology} not in ONTOLOGIES_LIST")

    return {
        IDENTIFIERS.ONTOLOGY: ontology,
        IDENTIFIERS.IDENTIFIER: identifier,
        IDENTIFIERS.BQB: BQB.IS,
    }


def _format_study_level_data(one_study):

    study_data = one_study[PSI_MI_DEFS.EXPERIMENT]
    reaction_id = _sanitize_intact_identifiers(
        ontology=study_data[PSI_MI_REFS.PRIMARY_REF_DB],
        identifier=study_data[PSI_MI_REFS.PRIMARY_REF_ID],
    )

    out = {
        PSI_MI_DEFS.EXPERIMENT_NAME: study_data[PSI_MI_DEFS.EXPERIMENT_NAME],
        PSI_MI_DEFS.INTERACTION_METHOD: study_data[PSI_MI_DEFS.INTERACTION_METHOD],
        SBML_DFS.R_IDENTIFIERS: Identifiers(
            [reaction_id] if reaction_id is not None else []
        ),
    }

    return pd.DataFrame(out, index=[0])


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
            participant[PSI_MI_DEFS.INTERACTION_TYPE] = interaction[
                PSI_MI_DEFS.INTERACTION_TYPE
            ]
            reaction_species.append(participant)

    return pd.DataFrame(reaction_species)


def _create_species_df(one_study: dict[str, Any]) -> pd.DataFrame:

    species = list()
    species_identifiers = list()
    for spec in one_study[PSI_MI_DEFS.INTERACTOR_LIST]:

        valid_xrefs = [
            x
            for x in spec[PSI_MI_DEFS.INTERACTOR_XREFS]
            if x[PSI_MI_DEFS.REF_TYPE] == PSI_MI_DEFS.PRIMARY
            or x[PSI_MI_RAW_ATTRS.DB] == ONTOLOGIES.INTACT
        ]

        # new ids dict entries
        ids = [
            {
                **_sanitize_intact_identifiers(
                    x[PSI_MI_RAW_ATTRS.ID],
                    x[PSI_MI_RAW_ATTRS.DB],
                    include_unmatched=False,
                ),
                PSI_MI_DEFS.INTERACTOR_ID: spec[PSI_MI_DEFS.INTERACTOR_ID],
            }
            for x in valid_xrefs
        ]

        species_identifiers.append(ids)

        spec_summary = {
            PSI_MI_DEFS.INTERACTOR_ID: spec[PSI_MI_DEFS.INTERACTOR_ID],
            PSI_MI_DEFS.INTERACTOR_LABEL: spec[PSI_MI_DEFS.INTERACTOR_LABEL],
            PSI_MI_DEFS.INTERACTOR_NAME: spec[PSI_MI_DEFS.INTERACTOR_NAME],
        }
        species.append(spec_summary)

    species_df = pd.DataFrame(species)
    # nested list to shallow list
    species_identifiers = [item for sublist in species_identifiers for item in sublist]
    species_identifiers = pd.DataFrame(species_identifiers)

    return species_df, species_identifiers


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


def _get_optional_attribute(
    element, xpath: str, attribute: str, default: str = ""
) -> str:
    """
    Safely extract an attribute from an optional XML element.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        The parent element to search within
    xpath : str
        The xpath expression to find the child element
    attribute : str
        The attribute name to extract
    default : str, optional
        Default value to return if element or attribute is not found, by default ""

    Returns
    -------
    str
        The attribute value, or the default value if not found
    """
    child = element.find(xpath)
    if child is None:
        return default
    return child.attrib.get(attribute, default)
