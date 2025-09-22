from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import pandas as pd

from napistu.constants import IDENTIFIERS
from napistu.ingestion.constants import (
    PSI_MI_DEFS,
    PSI_MI_INTACT_XML_NAMESPACE,
    PSI_MI_MISSING_VALUE_STR,
    PSI_MI_RAW_ATTRS,
    PSI_MI_STUDY_TABLES,
    PSI_MI_STUDY_TABLES_LIST,
)

logger = logging.getLogger(__name__)


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
    intact_xml_dir : str
        Path to the directory containing the PSI-MI XML files
    xml_namespace : str, optional
        Namespace for the xml file, by default PSI_MI_INTACT_XML_NAMESPACE
    verbose : bool, optional
        Whether to print verbose output, by default False
    files_to_process : int, optional
        Number of files to process (-1 for all files), by default -1

    Returns
    -------
    formatted_psi_mis : list[dict[str, Any]]
        A list containing molecular interaction entry dicts of the format:
        - source : dict containing the database that interactions were drawn from.
        - experiment : a simple summary of the experimental design and the publication.
        - interactor_list : list containing dictionaries annotating the molecules
        (defined by their "interactor_id") involved in interactions.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no files
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
        formatted_psi_mis.append(format_psi_mi(xml_path, xml_namespace, verbose))

    return formatted_psi_mis


def aggregate_psi_mis(
    formatted_psi_mis: List[Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate PSI-MI molecular interactions and study metadata and format results as a dictionary of dataframes.

    Parameters
    ----------
    formatted_psi_mis : dict[str, Any]
        A dictionary of PSI-MI files, where the keys are the study IDs and the values are the PSI-MI files. As returned by `napistu.ingestion.psi_mi.format_psi_mis`.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary of dataframes, where the keys are the study IDs and the values are:
        - `reaction_species` : A dataframe of reaction species, where the columns are the reaction species and the rows are the study IDs.
        - `species` : A dataframe of species, where the columns are the species and the rows are the study IDs.
        - `species_identifiers` : A dataframe of species identifiers, where the columns are the species identifiers and the rows are the study IDs.
        - `study_level_data` : A dataframe of study level data, where the columns are the study level data and the rows are the study IDs.
    """
    # reaction sources get the pubmed id of the study
    all_studies = list()
    for file in formatted_psi_mis:

        for study in file:
            # autoincrement study id
            study_id = len(all_studies) + 100000

            species_df, species_ids = _create_study_species_df(study)

            study_tables = {
                PSI_MI_STUDY_TABLES.REACTION_SPECIES: _create_reaction_species_df(
                    study
                ).assign(study_id=study_id),
                PSI_MI_STUDY_TABLES.SPECIES: species_df.assign(study_id=study_id),
                PSI_MI_STUDY_TABLES.SPECIES_IDENTIFIERS: species_ids.assign(
                    study_id=study_id
                ),
                PSI_MI_STUDY_TABLES.STUDY_LEVEL_DATA: _format_study_level_data(
                    study
                ).assign(study_id=study_id),
            }

            all_studies.append(study_tables)

    # transpose results so a list of dicts of tables becomes a dict of tables
    all_study_tables = dict()
    for tbl in PSI_MI_STUDY_TABLES_LIST:
        all_study_tables[tbl] = pd.concat([x[tbl] for x in all_studies]).reset_index(
            drop=True
        )

    return all_study_tables


def format_psi_mi(
    xml_path: str,
    xml_namespace: str = PSI_MI_INTACT_XML_NAMESPACE,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    Format PSI 3.0

    Format an .xml file containing molecular interactions following the PSI 3.0 format.

    Parameters
    ----------
    xml_path : str
        Path to a .xml file
    xml_namespace : str, optional
        Namespace for the xml file, by default PSI_MI_INTACT_XML_NAMESPACE
    verbose : bool, optional
        Whether to print verbose output, by default False

    Returns
    -------
    entry_list : list[dict[str, Any]]
        A list containing molecular interaction entry dicts of the format:
        - source : dict containing the database that interactions were drawn from.
        - experiment : a simple summary of the experimental design and the publication.
        - interactor_list : list containing dictionaries annotating the molecules
        (defined by their "interactor_id") involved in interactions.
        - interactions_list : list containing dictionaries annotating molecular
        interactions involving a set of "interactor_id"s.

    Raises
    ------
    FileNotFoundError
        If the XML file does not exist
    ValueError
        If the XML file does not have the expected root tag
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


def _format_entry(an_entry: ET.Element, xml_namespace: str) -> Dict[str, Any]:
    """
    Extract a single XML entry of interactors and interactions.

    Parameters
    ----------
    an_entry : xml.etree.ElementTree.Element
        XML entry element to format
    xml_namespace : str
        XML namespace to use for parsing

    Returns
    -------
    dict[str, Any]
        Dictionary containing formatted entry data with keys: source, experiment, interactor_list, interactions_list

    Raises
    ------
    ValueError
        If the entry tag is not as expected
    """

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


def _format_entry_source(an_entry: ET.Element, xml_namespace: str) -> Dict[str, str]:
    """
    Format the source describing the provenance of an XML entry.

    Parameters
    ----------
    an_entry : xml.etree.ElementTree.Element
        XML entry element to format
    xml_namespace : str
        XML namespace to use for parsing

    Returns
    -------
    dict[str, str]
        Dictionary containing source information with keys: shortLabel, fullName
    """

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


def _format_entry_experiment(
    an_entry: ET.Element, xml_namespace: str
) -> Dict[str, str]:
    """
    Format experiment-level information in an XML entry.

    Parameters
    ----------
    an_entry : xml.etree.ElementTree.Element
        XML entry element to format
    xml_namespace : str
        XML namespace to use for parsing

    Returns
    -------
    dict[str, str]
        Dictionary containing experiment information with keys: experiment_name, interaction_method, ontology, identifier
    """

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
        IDENTIFIERS.ONTOLOGY: _get_optional_attribute(
            experiment_info,
            f".{xml_namespace}bibref/{xml_namespace}xref/{xml_namespace}primaryRef",
            PSI_MI_RAW_ATTRS.DB,
        ),
        IDENTIFIERS.IDENTIFIER: _get_optional_attribute(
            experiment_info,
            f".{xml_namespace}bibref/{xml_namespace}xref/{xml_namespace}primaryRef",
            PSI_MI_RAW_ATTRS.ID,
        ),
    }

    return out


def _format_entry_interactor_list(
    an_entry: ET.Element, xml_namespace: str
) -> List[Dict[str, Any]]:
    """
    Format the molecular interactors in an XML entry.

    Parameters
    ----------
    an_entry : xml.etree.ElementTree.Element
        XML entry element to format
    xml_namespace : str
        XML namespace to use for parsing

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries containing formatted interactor data
    """

    assert an_entry.tag == xml_namespace + "entry"

    interactor_list = an_entry.find(f"./{xml_namespace}interactorList")

    return [_format_entry_interactor(x, xml_namespace) for x in interactor_list]


def _format_entry_interactor(
    interactor: ET.Element, xml_namespace: str
) -> Dict[str, Any]:
    """
    Format a single molecular interactor in an interaction list XML node.

    Parameters
    ----------
    interactor : xml.etree.ElementTree.Element
        XML interactor element to format
    xml_namespace : str
        XML namespace to use for parsing

    Returns
    -------
    dict[str, Any]
        Dictionary containing formatted interactor data with keys: interactor_id, interactor_label, interactor_name, interactor_aliases, interactor_xrefs

    Raises
    ------
    ValueError
        If the interactor tag is not as expected
    """

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
    interactor: ET.Element, xml_namespace: str
) -> List[Dict[str, str]]:
    """
    Format the cross-references of a single interactor.

    Parameters
    ----------
    interactor : xml.etree.ElementTree.Element
        XML interactor element to format
    xml_namespace : str
        XML namespace to use for parsing

    Returns
    -------
    list[dict[str, str]]
        List of dictionaries containing cross-reference data with keys: ref_type, ontology, identifier
    """

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
            IDENTIFIERS.ONTOLOGY: x.attrib.get(PSI_MI_RAW_ATTRS.DB, ""),
            IDENTIFIERS.IDENTIFIER: x.attrib.get(PSI_MI_RAW_ATTRS.ID, ""),
        }
        for x in xref_nodes
    ]

    return out


def _format_entry_interactions(
    an_entry: ET.Element, xml_namespace: str
) -> List[Dict[str, Any]]:
    """
    Format the molecular interaction in an XML entry.

    Parameters
    ----------
    an_entry : xml.etree.ElementTree.Element
        XML entry element to format
    xml_namespace : str
        XML namespace to use for parsing

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries containing formatted interaction data
    """

    assert an_entry.tag == xml_namespace + PSI_MI_RAW_ATTRS.ENTRY

    interaction_list = an_entry.find(
        f"./{xml_namespace}{PSI_MI_RAW_ATTRS.INTERACTIONS_LIST}"
    )

    interaction_dicts = [
        _format_entry_interaction(x, xml_namespace) for x in interaction_list
    ]

    return interaction_dicts


def _format_entry_interaction(
    interaction: ET.Element, xml_namespace: str
) -> Dict[str, Any]:
    """
    Format a single interaction in an XML interaction list.

    Parameters
    ----------
    interaction : xml.etree.ElementTree.Element
        XML interaction element to format
    xml_namespace : str
        XML namespace to use for parsing

    Returns
    -------
    dict[str, Any]
        Dictionary containing formatted interaction data with keys: interaction_name, interaction_type, interactors
    """

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
    interaction_participant: ET.Element, xml_namespace: str
) -> Dict[str, str]:
    """
    Format the participants in an XML interaction.

    Parameters
    ----------
    interaction_participant : xml.etree.ElementTree.Element
        XML participant element to format
    xml_namespace : str
        XML namespace to use for parsing

    Returns
    -------
    dict[str, str]
        Dictionary containing formatted participant data with keys: participant_id, interactor_id, biological_role, experimental_role

    Raises
    ------
    ValueError
        If the participant tag is not as expected
    """

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


def _format_study_level_data(one_study: Dict[str, Any]) -> pd.DataFrame:
    """
    Format study-level data into a DataFrame.

    Parameters
    ----------
    one_study : dict[str, Any]
        Study data dictionary

    Returns
    -------
    pd.DataFrame
        DataFrame containing study-level data
    """
    return pd.DataFrame(one_study[PSI_MI_DEFS.EXPERIMENT], index=[0])


def _create_reaction_species_df(one_study: Dict[str, Any]) -> pd.DataFrame:
    """
    Format the interactions in the study into a dataframe of reaction species.

    Parameters
    ----------
    one_study : dict[str, Any]
        Study data dictionary

    Returns
    -------
    pd.DataFrame
        DataFrame containing reaction species data
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


def _create_study_species_df(
    one_study: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create species and species identifiers DataFrames from study data.

    Parameters
    ----------
    one_study : dict[str, Any]
        Study data dictionary

    Returns
    -------
    species_df : pd.DataFrame
        DataFrame containing species data
    species_identifiers : pd.DataFrame
        DataFrame containing species identifiers data
    """

    species = list()
    species_identifiers = list()
    for spec in one_study[PSI_MI_DEFS.INTERACTOR_LIST]:

        # Add interactor_id to each cross-reference
        spec_identifiers = [
            {**xref, PSI_MI_DEFS.INTERACTOR_ID: spec[PSI_MI_DEFS.INTERACTOR_ID]}
            for xref in spec[PSI_MI_DEFS.INTERACTOR_XREFS]
        ]
        species_identifiers.append(spec_identifiers)

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


def _get_optional_text(
    element: ET.Element, xpath: str, default: str = PSI_MI_MISSING_VALUE_STR
) -> str:
    """
    Safely extract text from an optional XML element.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        The parent element to search within
    xpath : str
        The xpath expression to find the child element
    default : str, optional
        Default value to return if element is not found, by default PSI_MI_MISSING_VALUE_STR

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
    element: ET.Element,
    xpath: str,
    attribute: str,
    default: str = PSI_MI_MISSING_VALUE_STR,
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
        Default value to return if element or attribute is not found, by default PSI_MI_MISSING_VALUE_STR

    Returns
    -------
    str
        The attribute value, or the default value if not found
    """
    child = element.find(xpath)
    if child is None:
        return default
    return child.attrib.get(attribute, default)
