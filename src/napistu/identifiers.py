"""
Systematic identifiers for species, reactions, compartments, etc.

Classes
-------
Identifiers
    Identifiers for a single entity or relationship.

Public Functions
----------------
check_reactome_identifier_compatibility
    Check whether two sets of Reactome identifiers are from the same species.
construct_cspecies_identifiers
    Construct compartmentalized species identifiers by adding sc_id to species_identifiers.
create_uri_url
    Convert from an identifier and ontology to a URL reference for the identifier.
cv_to_Identifiers
    Convert an SBML controlled vocabulary element into an Identifiers object.
df_to_identifiers
    Convert a DataFrame of identifier information to a Series of Identifiers objects.
ensembl_id_to_url_regex
    Map an ensembl ID to a validation regex and its canonical url on ensembl.
format_uri
    Convert a RDF URI into an identifier list.
format_uri_url
    Convert a URI into an identifier dictionary.
format_uri_url_identifiers_dot_org
    Parse identifiers.org identifiers from a split URL path.
is_known_unsupported_uri
    Check if a URI is known to be unsupported/pathological.
parse_ensembl_id
    Extract the molecule type and species name from an ensembl identifier.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Optional, Union
from urllib.parse import urlparse

import libsbml
import pandas as pd
from pydantic import BaseModel

from napistu import sbml_dfs_core, utils
from napistu.constants import (
    BIOLOGICAL_QUALIFIER_CODES,
    BQB,
    BQB_PRIORITIES,
    ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY,
    ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY,
    ENSEMBL_SPECIES_FROM_CODE,
    ENSEMBL_SPECIES_TO_CODE,
    IDENTIFIERS,
    IDENTIFIERS_REQUIRED_VARS,
    ONTOLOGIES,
    SBML_DFS,
    SBML_DFS_SCHEMA,
    SCHEMA_DEFS,
    SPECIES_IDENTIFIERS_REQUIRED_VARS,
)
from napistu.ingestion.constants import LATIN_SPECIES_NAMES
from napistu.utils.pd_utils import match_pd_vars

logger = logging.getLogger(__name__)


class Identifiers:
    """
    Identifiers for a single entity or relationship.

    Attributes
    ----------
    ids : list
        a list of identifiers which are each a dict containing an ontology and identifier
    verbose : bool
        extra reporting, defaults to False

    Properties
    ----------
    ids : list
        (deprecated) a list of identifiers which are each a dict containing an ontology and identifier

    Public Methods
    -------
    get_all_bqbs()
        Returns a set of all BQB entries
    get_all_ontologies()
        Returns a set of all ontology entries
    has_ontology(ontologies)
        Returns a bool of whether 1+ of the ontologies was represented
    hoist(ontology)
        Returns value(s) from an ontology
    print
        Print a table of identifiers
    """

    def __init__(self, id_list: list, verbose: bool = False) -> None:
        """
        Tracks a set of identifiers and the ontologies they belong to.

        Parameters
        ----------
        id_list : list
            a list of identifier dictionaries containing ontology, identifier, and optionally url

        Returns
        -------
        None.

        """

        # read list and validate format
        validated_id_list = _IdentifiersValidator(id_list=id_list).model_dump()[
            "id_list"
        ]

        if validated_id_list:
            df = _deduplicate_identifiers_by_priority(
                pd.DataFrame(validated_id_list),
                [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER],
            )
        else:
            # Empty DataFrame with expected schema
            df = pd.DataFrame(
                columns=[
                    IDENTIFIERS.ONTOLOGY,
                    IDENTIFIERS.IDENTIFIER,
                    IDENTIFIERS.URL,
                    IDENTIFIERS.BQB,
                ]
            )

        self.df = df.astype(
            {
                IDENTIFIERS.ONTOLOGY: "string",
                IDENTIFIERS.IDENTIFIER: "string",
                IDENTIFIERS.URL: "string",
                IDENTIFIERS.BQB: "string",
            }
        )

    def get_all_bqbs(self) -> set[str]:
        """Returns a set of all BQB entries

        Returns:
            set[str]: A set containing all unique BQB values from the identifiers
        """
        return set(self.df[IDENTIFIERS.BQB].dropna().unique())

    def get_all_ontologies(self, bqb_terms: list[str] = None) -> set[str]:
        """Returns a set of all ontology entries

        Returns:
            set[str]: A set containing all unique ontology names from the identifiers
        """

        if bqb_terms is not None:
            return set(
                self.df[self.df[IDENTIFIERS.BQB].isin(bqb_terms)][IDENTIFIERS.ONTOLOGY]
            )
        else:
            return set(self.df[IDENTIFIERS.ONTOLOGY])

    def has_ontology(self, ontologies: str | list[str]) -> bool:
        """
        Check if specified ontologies are present in the identifiers.

        Parameters
        ----------
        ontologies : str or list of str
            Ontology name(s) to search for

        Returns
        -------
        bool
            True if any specified ontologies are present
        """

        if isinstance(ontologies, str):
            ontologies = [ontologies]

        if self.df.empty:
            return False

        # Check if any rows have matching ontologies
        return bool(self.df[IDENTIFIERS.ONTOLOGY].isin(ontologies).any())

    def hoist(self, ontology: str, squeeze: bool = True) -> str | list[str] | None:
        """Returns value(s) from an ontology

        Args:
            ontology (str): the ontology of interest
            squeeze (bool): if True, return a single value if possible

        Returns:
            str or list: the value(s) of an ontology of interest

        """

        if not isinstance(ontology, str):
            raise TypeError(f"{ontology} must be a str")

        # return the value(s) of an ontology of interest
        matches = self.df[self.df[IDENTIFIERS.ONTOLOGY] == ontology]
        ontology_ids = matches[IDENTIFIERS.IDENTIFIER].tolist()

        if squeeze:
            if len(ontology_ids) == 0:
                return None
            elif len(ontology_ids) == 1:
                return ontology_ids[0]
        return ontology_ids

    @property
    def ids(self) -> list[dict]:

        logger.warning("Identifiers.ids is deprecated. Use Identifiers.df instead.")
        return self.df.to_dict("records") if self.df is not None else []

    @classmethod
    def merge(cls, identifier_series: pd.Series) -> "Identifiers":
        """
        Merge multiple Identifiers objects into a single Identifiers object.

        Parameters
        ----------
        identifier_series : pd.Series
            Series of Identifiers objects to merge

        Returns
        -------
        Identifiers
            New Identifiers object containing all unique identifiers
        """

        if len(identifier_series) == 1:
            return identifier_series.iloc[0]

        # Concatenate all DataFrames and let __init__ handle deduplication
        all_dfs = [
            identifiers.df
            for identifiers in identifier_series
            if not identifiers.df.empty
        ]

        if not all_dfs:
            return cls([])  # Return empty Identifiers

        merged_df = pd.concat(all_dfs, ignore_index=True)

        # Convert back to list format for __init__ to handle deduplication and validation
        merged_ids = merged_df.to_dict("records")

        return cls(merged_ids)

    def print(self):
        """Print a table of identifiers"""

        utils.show(self.df, hide_index=True)

    def __repr__(self) -> str:
        """Return a string representation of the Identifiers object"""
        return f"Identifiers({self.df.shape[0]} identifiers)"


def check_reactome_identifier_compatibility(
    reactome_series_a: pd.Series,
    reactome_series_b: pd.Series,
) -> None:
    """
    Check Reactome Identifier Compatibility

    Determine whether two sets of Reactome identifiers are from the same species.

    Args:
        reactome_series_a: pd.Series
            a Series containing Reactome identifiers
        reactome_series_b: pd.Series
            a Series containing Reactome identifiers

    Returns:
        None

    """

    a_species, a_species_counts = _infer_primary_reactome_species(reactome_series_a)
    b_species, b_species_counts = _infer_primary_reactome_species(reactome_series_b)

    if a_species != b_species:
        a_name = reactome_series_a.name
        if a_name is None:
            a_name = "unnamed"

        b_name = reactome_series_b.name
        if b_name is None:
            b_name = "unnamed"

        raise ValueError(
            "The two provided pd.Series containing Reactome identifiers appear to be from different species. "
            f"The pd.Series named {a_name} appears to be {a_species} with {a_species_counts} examples of this code. "
            f"The pd.Series named {b_name} appears to be {b_species} with {b_species_counts} examples of this code."
        )

    return None


def construct_cspecies_identifiers(
    species_identifiers: pd.DataFrame,
    cspecies_references: Union[sbml_dfs_core.SBML_dfs, pd.DataFrame],
) -> pd.DataFrame:
    """
    Construct compartmentalized species identifiers by adding sc_id to species_identifiers.

    This function merges compartmentalized species IDs (sc_id) into a species_identifiers
    table, allowing you to work with compartmentalized species without loading the full
    sbml_dfs object.

    Parameters
    ----------
    species_identifiers : pd.DataFrame
        A species identifiers table with columns including s_id, ontology, identifier.
        Must satisfy SPECIES_IDENTIFIERS_REQUIRED_VARS.
    cspecies_references : Union[sbml_dfs_core.SBML_dfs, pd.DataFrame]
        Either an sbml_dfs object from which compartmentalized_species will be extracted,
        or a 2-column DataFrame with s_id and sc_id columns.

    Returns
    -------
    pd.DataFrame
        The species_identifiers table with an additional sc_id column. Each row
        in the original table will be expanded to include all corresponding sc_ids
        for that s_id.
    """

    # Validate input species_identifiers table
    _check_species_identifiers_table(species_identifiers)

    # Extract sid_to_scids table based on type of cspecies_references
    if isinstance(cspecies_references, sbml_dfs_core.SBML_dfs):
        sid_to_scids = cspecies_references.compartmentalized_species.reset_index()[
            [SBML_DFS.S_ID, SBML_DFS.SC_ID]
        ]
    elif isinstance(cspecies_references, pd.DataFrame):
        sid_to_scids = cspecies_references
        match_pd_vars(
            sid_to_scids,
            req_vars={SBML_DFS.S_ID, SBML_DFS.SC_ID},
            allow_series=False,
        ).assert_present()
    else:
        raise TypeError(
            f"cspecies_references must be either an SBML_dfs object or a pandas DataFrame, "
            f"got {type(cspecies_references)}"
        )

    species_identifiers_w_scids = species_identifiers.merge(
        sid_to_scids,
        on=SBML_DFS.S_ID,
        how="left",
    )

    if any(species_identifiers_w_scids[SBML_DFS.SC_ID].isna()):
        raise ValueError(
            "Some species identifiers were not found in the cspecies_references table"
        )

    return species_identifiers_w_scids


def create_uri_url(ontology: str, identifier: str, strict: bool = True) -> str:
    """
    Create URI URL

    Convert from an identifier and ontology to a URL reference for the identifier

    Parameters
    ----------
    ontology: str
        An ontology for organizing genes, metabolites, etc.
    identifier: str
        A systematic identifier from the \"ontology\" ontology.
    strict: bool
        if strict then throw errors for invalid IDs otherwise return None

    Returns
    -------
    url: str
        A url representing a unique identifier
    """

    # default to no id_regex
    id_regex = None

    if ontology in ["ensembl_gene", "ensembl_transcript", "ensembl_protein"]:
        id_regex, url = ensembl_id_to_url_regex(identifier, ontology)
    elif ontology == "bigg.metabolite":
        url = f"http://identifiers.org/bigg.metabolite/{identifier}"
    elif ontology == "chebi":
        id_regex = "^[0-9]+$"
        url = f"http://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{identifier}"
    elif ontology == "ec-code":
        id_regex = "^[0-9]+\\.[0-9]+\\.[0-9]+(\\.[0-9]+)?$"
        url = f"https://identifiers.org/ec-code/{identifier}"
    elif ontology == "envipath":
        url = f"http://identifiers.org/envipath/{identifier}"
    elif ontology == "go":
        id_regex = "^GO:[0-9]{7}$"
        url = f"https://www.ebi.ac.uk/QuickGO/term/{identifier}"
    elif ontology == "ncbi_entrez_gene":
        url = f"https://www.ncbi.nlm.nih.gov/gene/{identifier}"
    elif ontology == "ncbi_entrez_pccompound":
        id_regex = "^[A-Z]{14}\\-[A-Z]{10}\\-[A-Z]{1}$"
        url = f"http://www.ncbi.nlm.nih.gov/sites/entrez?cmd=search&db=pccompound&term={identifier}"
    elif ontology == "pubchem":
        id_regex = "^[0-9]+$"
        url = f"http://pubchem.ncbi.nlm.nih.gov/compound/{identifier}"
    elif ontology == "pubmed":
        id_regex = "^[0-9]+$"
        url = f"http://www.ncbi.nlm.nih.gov/pubmed/{identifier}"
    elif ontology == "reactome":
        id_regex = "^R\\-[A-Z]{3}\\-[0-9]{7}$"
        url = f"https://reactome.org/content/detail/{identifier}"
    elif ontology == "uniprot":
        id_regex = "^[A-Z0-9]+$"
        url = f"https://purl.uniprot.org/uniprot/{identifier}"
    elif ontology == "sgc":
        id_regex = "^[0-9A-Z]+$"
        url = f"https://www.thesgc.org/structures/structure_description/{identifier}/"
    elif ontology == "mdpi":
        id_regex = None
        url = f"https://www.mdpi.com/{identifier}"
    elif ontology == "mirbase":
        id_regex = None
        if re.match("MIMAT[0-9]", identifier):
            url = f"https://www.mirbase.org/mature/{identifier}"
        elif re.match("MI[0-9]", identifier):
            url = f"https://www.mirbase.org/hairpin/{identifier}"
        else:
            raise NotImplementedError(f"url not defined for this MiRBase {identifier}")
    elif ontology == "rnacentral":
        id_regex = None
        url = f"https://rnacentral.org/rna/{identifier}"
    elif ontology == "chemspider":
        id_regex = "^[0-9]+$"
        url = f"https://www.chemspider.com/{identifier}"

    elif ontology == "dx_doi":
        id_regex = r"^[0-9]+\.[0-9]+$"
        url = f"https://dx.doi.org/{identifier}"
    elif ontology == "doi":
        id_regex = None
        url = f"https://doi.org/{identifier}"

    elif ontology == "ncbi_books":
        id_regex = "^[0-9A-Z]+$"
        url = f"http://www.ncbi.nlm.nih.gov/books/{identifier}/"

    elif ontology == "ncbi_entrez_gene":
        id_regex = "^[0-9]+$"
        url = f"https://www.ncbi.nlm.nih.gov/gene/{identifier}"
    elif ontology == "phosphosite":
        id_regex = "^[0-9]+$"
        url = f"https://www.phosphosite.org/siteAction.action?id={identifier}"
    elif ontology == "NCI_Thesaurus":
        id_regex = "^[A-Z][0-9]+$"
        url = f"https://ncithesaurus.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&code={identifier}"
    elif ontology == "matrixdb_biomolecule":
        id_regex = "^[0-9A-Za-z]+$"
        url = f"http://matrixdb.univ-lyon1.fr/cgi-bin/current/newPort?type=biomolecule&value={identifier}"
    else:
        raise NotImplementedError(
            f"No identifier -> url logic exists for the {ontology} ontology in create_uri_url()"
        )

    # validate identifier with regex if one exists
    if id_regex is not None:
        if re.search(id_regex, identifier) is None:
            failure_msg = f"{identifier} is not a valid {ontology} id, it did not match the regex: {id_regex}"
            if strict:
                raise TypeError(failure_msg)
            else:
                logger.warning(failure_msg + " returning None")
                return None

    return url


def cv_to_Identifiers(entity, strict: bool = False):
    """
    Convert an SBML controlled vocabulary element into a cpr Identifiers object.

    Parameters
    ----------
    entity: libsbml.Species
        An entity (species, reaction, compartment, ...) with attached CV terms
    strict: bool, default True
        If True, log full tracebacks for parsing failures.
        If False, use simple warning messages.

    Returns
    -------
    Identifiers
        An Identifiers object containing the CV terms
    """

    cv_list = list()
    for cv in entity.getCVTerms():
        if cv.getQualifierType() != libsbml.BIOLOGICAL_QUALIFIER:
            # only care about biological annotations
            continue

        biological_qualifier_type = BIOLOGICAL_QUALIFIER_CODES[
            cv.getBiologicalQualifierType()
        ]
        out_list = list()
        for i in range(cv.getNumResources()):
            uri = cv.getResourceURI(i)

            # Pre-check for known unsupported URIs
            if is_known_unsupported_uri(uri):
                logger.warning(f"Skipping known unsupported URI: {uri}")
                continue

            try:
                out_list.append(
                    format_uri(uri, biological_qualifier_type, strict=strict)
                )
            except NotImplementedError:
                if strict:
                    logger.warning("Not all identifiers resolved: ", exc_info=True)
                else:
                    logger.warning(f"Could not parse URI (not implemented): {uri}")

        cv_list.extend(out_list)
    return Identifiers(cv_list)


def df_to_identifiers(df: pd.DataFrame) -> pd.Series:
    """
    Convert a DataFrame of identifier information to a Series of Identifiers objects.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing identifier information with required columns:
        ontology, identifier, url, bqb

    Returns
    -------
    pd.Series
        Series indexed by index_col containing Identifiers objects
    """

    entity_type = utils.infer_entity_type(df)
    table_schema = SBML_DFS_SCHEMA.SCHEMA[entity_type]
    if SCHEMA_DEFS.ID not in table_schema:
        raise ValueError(f"The entity type {entity_type} does not have an id column")

    table_pk_var = table_schema[SCHEMA_DEFS.PK]
    required_vars = {table_pk_var} | IDENTIFIERS_REQUIRED_VARS
    utils.match_pd_vars(df, required_vars).assert_present()

    identifiers_dict = {}
    for pk_value in df[table_pk_var].unique():
        pk_rows = df[df[table_pk_var] == pk_value]
        # Convert to list of dicts format for Identifiers constructor
        id_list = pk_rows.drop(columns=[table_pk_var]).to_dict("records")
        identifiers_dict[pk_value] = Identifiers(
            id_list
        )  # Handles deduplication internally

    output = pd.Series(identifiers_dict, name=table_schema[SCHEMA_DEFS.ID])
    output.index.name = table_pk_var

    return output


def ensembl_id_to_url_regex(identifier: str, ontology: str) -> tuple[str, str]:
    """
    Ensembl ID to URL and Regex

    Map an ensembl ID to a validation regex and its canonical url on ensembl

    Args:
        identifier: str
            A standard identifier from ensembl genes, transcripts, or proteins
        ontology: str
            The standard ontology (ensembl_gene, ensembl_transcript, or ensembl_protein)

    Returns:
        id_regex: a regex which should match a valid entry in this ontology
        url: the id's url on ensembl
    """

    # extract the species name from the 3 letter species code in the id
    # (these letters are not present for humans)
    identifier, implied_ontology, species = parse_ensembl_id(identifier)  # type: ignore
    if implied_ontology != ontology:
        raise ValueError(
            f"Implied ontology mismatch: expected {ontology}, got {implied_ontology}"
        )

    # create an appropriate regex for validating input
    # this provides testing for other identifiers even if it is redundant with other
    # validation of ensembl ids

    if species == "Homo sapiens":
        species_code = ""
    else:
        species_code = ENSEMBL_SPECIES_TO_CODE[species]
    molecule_type_code = ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY[ontology]

    id_regex = "ENS" + species_code + molecule_type_code + "[0-9]{11}"

    # convert to species format in ensembl urls
    species_url_field = re.sub(" ", "_", species)

    if ontology == "ensembl_gene":
        url = f"http://www.ensembl.org/{species_url_field}/geneview?gene={identifier}"
    elif ontology == "ensembl_transcript":
        url = f"http://www.ensembl.org/{species_url_field}/Transcript?t={identifier}"
    elif ontology == "ensembl_protein":
        url = f"https://www.ensembl.org/{species_url_field}/Transcript/ProteinSummary?t={identifier}"
    else:
        ValueError(f"{ontology} not defined")

    return id_regex, url


def format_uri(uri: str, bqb: str, strict: bool = True) -> list[dict]:
    """
    Convert a RDF URI into an identifier list

    Parameters
    ----------
    uri : str
        The RDF URI to convert
    bqb : str
        The BQB to add to the identifier
    strict : bool
        Whether to raise an error if the URI is not valid

    Returns
    -------
    list[dict]
        The identifier list
    """

    identifier = format_uri_url(uri, strict=strict)

    if identifier is None:
        if strict:
            raise NotImplementedError(f"{uri} is not a valid way of specifying a uri")
        else:
            # Return empty list for non-strict mode
            return list()

    _validate_bqb(bqb)
    identifier[IDENTIFIERS.BQB] = bqb

    return identifier


def is_known_unsupported_uri(uri: str) -> bool:
    """
    Check if a URI is known to be unsupported/pathological.

    This prevents throwing exceptions for URIs we know we can't parse,
    allowing for cleaner logging and batch processing.

    Parameters
    ----------
    uri : str
        The URI to check

    Returns
    -------
    bool
        True if the URI is known to be unsupported
    """
    parsed = urlparse(uri)
    netloc = parsed.netloc
    path_parts = parsed.path.split("/")

    # Known problematic patterns
    if netloc == "www.proteinatlas.org":
        return True

    # Specific Ensembl pattern: /id/EBT... (not supported)
    if (
        netloc == "www.ensembl.org"
        and len(path_parts) >= 3
        and path_parts[1] == "id"
        and path_parts[2].startswith("EBT")
    ):
        return True

    return False


def format_uri_url(uri: str, strict: bool = True) -> dict:
    """
    Convert a URI into an identifier dictionary

    Parameters
    ----------
    uri : str
        The URI to convert
    strict : bool
        Whether to raise an error if the URI is not valid

    Returns
    -------
    dict
        The identifier dictionary

    Raises
    ------
    NotImplementedError
        If a parsing precedure has not been implemented for the netloc
    TypeError
        If the URI is not valid
    ValueError
        If there is a pathological identifier within ontology-specific parsing
    """

    # check whether the uri is specified using a url
    result = urlparse(uri)
    if not all([result.scheme, result.netloc, result.path]):
        return None

    # valid url

    netloc = result.netloc
    split_path = result.path.split("/")

    try:
        if netloc == "identifiers.org":
            ontology, identifier = format_uri_url_identifiers_dot_org(split_path)
        elif netloc == "reactome.org":
            ontology = ONTOLOGIES.REACTOME
            identifier = split_path[-1]
        # genes and gene products
        elif netloc == "www.ensembl.org" and split_path[-1] == "geneview":
            ontology = ONTOLOGIES.ENSEMBL_GENE
            identifier, id_ontology, _ = parse_ensembl_id(result.query)  # type: ignore
            if ontology != id_ontology:
                raise ValueError(
                    f"Ontology mismatch: expected {ontology}, got {id_ontology}"
                )
        elif netloc == "www.ensembl.org" and split_path[-1] in [
            "transview",
            "Transcript",
        ]:
            ontology = ONTOLOGIES.ENSEMBL_TRANSCRIPT
            identifier, id_ontology, _ = parse_ensembl_id(result.query)  # type: ignore
            if ontology != id_ontology:
                raise ValueError(
                    f"Ontology mismatch: expected {ontology}, got {id_ontology}"
                )
        elif netloc == "www.ensembl.org" and split_path[-1] == "ProteinSummary":
            ontology = ONTOLOGIES.ENSEMBL_PROTEIN
            identifier, id_ontology, _ = parse_ensembl_id(result.query)  # type: ignore
            if ontology != id_ontology:
                raise ValueError(
                    f"Ontology mismatch: expected {ontology}, got {id_ontology}"
                )
        elif netloc == "www.ensembl.org" and (
            re.search("ENS[GTP]", split_path[-1])
            or re.search("ENS[A-Z]{3}[GTP]", split_path[-1])
        ):
            # format ensembl IDs which lack gene/transview
            identifier, ontology, _ = parse_ensembl_id(split_path[-1])

        elif netloc == "www.mirbase.org" or netloc == "mirbase.org":
            ontology = ONTOLOGIES.MIRBASE
            if re.search("MI[0-9]+", split_path[-1]):
                identifier = utils.extract_regex_search("MI[0-9]+", split_path[-1])
            elif re.search("MIMAT[0-9]+", split_path[-1]):
                identifier = utils.extract_regex_search("MIMAT[0-9]+", split_path[-1])
            elif re.search("MI[0-9]+", result.query):
                identifier = utils.extract_regex_search("MI[0-9]+", result.query)
            elif re.search("MIMAT[0-9]+", result.query):
                identifier = utils.extract_regex_search("MIMAT[0-9]+", result.query)
            else:
                raise TypeError(
                    f"{result.query} does not appear to match MiRBase identifiers"
                )
        elif netloc == "purl.uniprot.org":
            ontology = ONTOLOGIES.UNIPROT
            identifier = split_path[-1]
        elif netloc == "rnacentral.org":
            ontology = ONTOLOGIES.RNACENTRAL
            identifier = split_path[-1]
        # chemicals
        elif split_path[1] == "chebi":
            ontology = ONTOLOGIES.CHEBI
            identifier = utils.extract_regex_search("[0-9]+$", result.query)
        elif netloc == "pubchem.ncbi.nlm.nih.gov":
            ontology = ONTOLOGIES.PUBCHEM
            if result.query != "":
                identifier = utils.extract_regex_search("[0-9]+$", result.query)
            else:
                identifier = utils.extract_regex_search("[0-9]+$", split_path[-1])
        elif netloc == "www.genome.ad.jp":
            ontology = "genome_net"
            identifier = utils.extract_regex_search("[A-Za-z]+:[0-9]+$", uri)
        elif (
            netloc == "www.guidetopharmacology.org"
            and split_path[-1] == "LigandDisplayForward"
        ):
            ontology = "grac"
            identifier = utils.extract_regex_search("[0-9]+$", result.query)
        elif netloc == "www.chemspider.com" or netloc == "chemspider.com":
            ontology = "chemspider"
            identifier = split_path[-1]
        # reactions
        elif split_path[1] == "ec-code":
            ontology = ONTOLOGIES.EC_CODE
            identifier = split_path[-1]
        elif netloc == "www.rhea-db.org":
            ontology = "rhea"
            identifier = utils.extract_regex_search("[0-9]+$", result.query)
        # misc
        elif split_path[1] == "ols":
            ontology = "ols"
            identifier = split_path[-1]
        elif split_path[1] == "QuickGO":
            ontology = ONTOLOGIES.GO
            identifier = split_path[-1]
        elif split_path[1] == "pubmed":
            ontology = ONTOLOGIES.PUBMED
            identifier = split_path[-1]
        # DNA sequences
        elif netloc == "www.ncbi.nlm.nih.gov" and split_path[1] == "nuccore":
            ontology = "ncbi_refseq"
            identifier = split_path[-1]
        elif netloc == "www.ncbi.nlm.nih.gov" and split_path[1] == "sites":
            ontology = "ncbi_entrez_" + utils.extract_regex_search(
                "db=([A-Za-z0-9]+)\\&", result.query, 1
            )
            identifier = utils.extract_regex_search(
                r"term=([A-Za-z0-9\-]+)$", result.query, 1
            )
        elif netloc == "www.ebi.ac.uk" and split_path[1] == "ena":
            ontology = "ebi_refseq"
            identifier = split_path[-1]
        elif netloc == "www.thesgc.org" and split_path[1] == "structures":
            ontology = "sgc"
            identifier = split_path[-2]
        elif netloc == "www.mdpi.com":
            ontology = "mdpi"
            identifier = "/".join([i for i in split_path[1:] if i != ""])
        elif netloc == "dx.doi.org":
            ontology = "dx_doi"
            identifier = "/".join(split_path[1:])
        elif netloc == "doi.org":
            ontology = "doi"
            identifier = "/".join(split_path[1:])
        elif netloc == "www.ncbi.nlm.nih.gov" and split_path[1] == "books":
            ontology = "ncbi_books"
            identifier = split_path[2]
        elif netloc == "www.ncbi.nlm.nih.gov" and split_path[1] == "gene":
            ontology = "ncbi_gene"
            identifier = split_path[2]
        elif netloc == "www.phosphosite.org":
            ontology = "phosphosite"
            identifier = utils.extract_regex_match(".*id=([0-9]+).*", uri)
        elif netloc == "ncithesaurus.nci.nih.gov":
            ontology = "NCI_Thesaurus"
            identifier = utils.extract_regex_match(".*code=([0-9A-Z]+).*", uri)
        elif netloc == "matrixdb.ibcp.fr":
            molecule_class = utils.extract_regex_match(
                ".*class=([a-zA-Z]+).*", uri
            ).lower()
            ontology = f"matrixdb_{molecule_class}"
            identifier = utils.extract_regex_match(".*name=([0-9A-Za-z]+).*", uri)
        elif netloc == "matrixdb.univ-lyon1.fr":
            molecule_class = utils.extract_regex_match(
                ".*type=([a-zA-Z]+).*", uri
            ).lower()
            ontology = f"matrixdb_{molecule_class}"
            identifier = utils.extract_regex_match(".*value=([0-9A-Za-z]+).*", uri)
        elif netloc == "users.rcn.com":
            # Handle users.rcn.com URLs as generic web references
            ontology = ONTOLOGIES.URL
            identifier = uri  # Use the full URI as the identifier
        elif netloc == "www.biorxiv.org":
            ontology = ONTOLOGIES.BIORXIV
            identifier = split_path[-1]
        else:
            error_msg = f"{netloc} in the {uri} url has not been associated with a known ontology"
            if strict:
                raise NotImplementedError(error_msg)
            else:
                logger.warning(error_msg)
                return None
    except (TypeError, AttributeError):
        if strict:
            logger.warning(
                f"An identifier could not be found using the specified regex for {uri} based on the {ontology} ontology"
            )
            logger.warning(result)
            logger.warning("ERROR")
            sys.exit(1)
        else:
            logger.warning(f"Could not extract identifier from URI using regex: {uri}")
            return None

    # rename some entries

    if ontology == "ncbi_gene":
        ontology = ONTOLOGIES.NCBI_ENTREZ_GENE

    id_dict = {"ontology": ontology, "identifier": identifier, "url": uri}

    return id_dict


def format_uri_url_identifiers_dot_org(split_path: list[str]):
    """Parse identifiers.org identifiers

    The identifiers.org identifier have two different formats:
    1. http://identifiers.org/<ontology>/<id>
    2. http://identifiers.org/<ontology>:<id>

    Currently we are identifying the newer format 2. by
    looking for the `:` in the second element of the split path.

    Also the ontology is converted to lower case letters.

    Args:
        split_path (list[str]): split url path

    Returns:
        tuple[str, str]: ontology, identifier
    """

    # formatting for the identifiers.org meta ontology

    # meta ontologies

    # identify old versions without `:`
    V2_SEPARATOR = ":"
    if V2_SEPARATOR in split_path[1]:
        # identifiers.org switched to format <ontology>:<id>
        path = "/".join(split_path[1:])
        if path.count(V2_SEPARATOR) != 1:
            raise ValueError(
                "The assumption is that there is only one ':'"
                f"in an identifiers.org url. Found more in: {path}"
            )
        ontology, identifier = path.split(":")
        ontology = ontology.lower()
    else:
        ontology = split_path[1]

        if ontology in [ONTOLOGIES.CHEBI]:
            identifier = utils.extract_regex_search("[0-9]+$", split_path[-1])
        elif len(split_path) != 3:
            identifier = "/".join(split_path[2:])
        else:
            identifier = split_path[-1]

    return ontology, identifier


def parse_ensembl_id(input_str: str) -> tuple[str, str, str]:
    """
    Parse Ensembl ID

    Extract the molecule type and species name from a string containing an ensembl identifier.

    Parameters
    ----------
    input_str (str):
        A string containing an ensembl gene, transcript, or protein identifier

    Returns
    -------
    tuple[str, str, str]
        identifier (str):
            The substring matching the full identifier
        molecule_type (str):
            The ontology the identifier belongs to:
                - G -> ensembl_gene
                - T -> ensembl_transcript
                - P -> ensembl_protein
        organismal_species (str):
            The species name the identifier belongs to
    """

    # validate that input is an ensembl ID
    if not re.search("ENS[GTP][0-9]+", input_str) and not re.search(
        "ENS[A-Z]{3}[GTP][0-9]+", input_str
    ):
        ValueError(
            f"{input_str} did not match the expected formats of an ensembl identifier:",
            "ENS[GTP][0-9]+ or ENS[A-Z]{3}[GTP][0-9]+",
        )

    # extract the species code (three letters after ENS if non-human)
    species_code_search = re.compile("ENS([A-Z]{3})?[GTP]").search(input_str)

    if species_code_search.group(1) is None:  # type: ignore
        organismal_species = LATIN_SPECIES_NAMES.HOMO_SAPIENS
        molecule_type_regex = "ENS([GTP])"
        id_regex = "ENS[GTP][0-9]+"
    else:
        species_code = species_code_search.group(1)  # type: ignore

        if species_code not in ENSEMBL_SPECIES_FROM_CODE.keys():
            raise ValueError(
                f"The species code for {input_str}: {species_code} did not "
                "match any of the entries in ENSEMBL_SPECIES_CODE_LOOKUPS."
            )

        organismal_species = ENSEMBL_SPECIES_FROM_CODE[species_code]
        molecule_type_regex = "ENS[A-Z]{3}([GTP])"
        id_regex = "ENS[A-Z]{3}[GTP][0-9]+"

    # extract the molecule type (genes, transcripts or proteins)
    molecule_type_code_search = re.compile(molecule_type_regex).search(input_str)
    if not molecule_type_code_search:
        raise ValueError(
            "The ensembl molecule code (i.e., G, T or P) could not be extracted from {input_str}"
        )
    else:
        molecule_type_code = molecule_type_code_search.group(1)  # type: str

    if molecule_type_code not in ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY.keys():
        raise ValueError(
            f"The molecule type code for {input_str}: {molecule_type_code} did not "
            "match ensembl genes (G), transcripts (T), or proteins (P)."
        )

    molecule_type = ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY[molecule_type_code]  # type: str

    identifier = utils.extract_regex_search(id_regex, input_str)  # type: str

    return identifier, molecule_type, organismal_species


# private utility functions


def _check_species_identifiers_table(
    species_identifiers: pd.DataFrame,
    required_vars: set = SPECIES_IDENTIFIERS_REQUIRED_VARS,
):
    missing_required_vars = required_vars.difference(
        set(species_identifiers.columns.tolist())
    )
    if len(missing_required_vars) > 0:
        raise ValueError(
            f"{len(missing_required_vars)} required variables "
            "were missing from the species_identifiers table: "
            f"{', '.join(missing_required_vars)}"
        )

    return None


def _count_reactome_species(reactome_series: pd.Series) -> pd.Series:
    """Count the number of species tags in a set of reactome IDs"""

    return (
        reactome_series.drop_duplicates().transform(_reactome_id_species).value_counts()
    )


def _deduplicate_identifiers_by_priority(
    df: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """
    Deduplicate identifiers by prioritizing BQB terms and URL presence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing identifier information with BQB and URL columns
    group_cols : list[str]
        Columns to group by for deduplication (e.g., [ontology, identifier] or [pk, ontology, identifier])

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with highest priority entries retained
    """
    return (
        df.merge(BQB_PRIORITIES, how="left")
        .assign(_has_url=lambda x: x[IDENTIFIERS.URL].notna().astype(int))
        .sort_values(["bqb_rank", "_has_url"], ascending=[True, False])
        .drop_duplicates(subset=group_cols)
        .drop(columns=["bqb_rank", "_has_url"])
    )


def _format_Identifiers_pubmed(pubmed_id: str) -> Identifiers:
    """
    Format Identifiers for a single PubMed ID.

    These will generally be used in an r_Identifiers field.
    """

    # create a url for lookup and validate the pubmed id
    url = create_uri_url(ontology=ONTOLOGIES.PUBMED, identifier=pubmed_id, strict=False)
    id_entry = format_uri(uri=url, bqb=BQB.IS_DESCRIBED_BY)

    return Identifiers([id_entry])


def _infer_primary_reactome_species(reactome_series: pd.Series) -> tuple[str, int]:
    """Infer the best supported species based on a set of Reactome identifiers"""

    series_counts = _count_reactome_species(reactome_series)

    if "ALL" in series_counts.index:
        series_counts = series_counts.drop("ALL", axis=0)

    return series_counts.index[0], series_counts.iloc[0]


def _prepare_species_identifiers(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    dogmatic: bool = False,
    species_identifiers: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Accepts and validates species_identifiers, or extracts a fresh table if None."""

    if species_identifiers is None:
        species_identifiers = sbml_dfs.get_characteristic_species_ids(dogmatic=dogmatic)
    else:
        # check for compatibility
        try:
            # check species_identifiers format

            _check_species_identifiers_table(species_identifiers)
            # quick check for compatibility between sbml_dfs and species_identifiers
            _validate_assets_sbml_ids(sbml_dfs, species_identifiers)
        except ValueError as e:
            logger.warning(
                f"The provided identifiers are not compatible with your `sbml_dfs` object. Extracting a fresh species identifier table. {e}"
            )
            species_identifiers = sbml_dfs.get_characteristic_species_ids(
                dogmatic=dogmatic
            )

    return species_identifiers


def _reactome_id_species(reactome_id: str) -> str:
    """Extract the species code from a Reactome ID"""

    reactome_match = re.match("^R\\-([A-Z]{3})\\-[0-9]+", reactome_id)
    if reactome_match:
        try:
            value = reactome_match[1]
        except ValueError:
            raise ValueError(f"{reactome_id} is not a valid reactome ID")
    else:
        raise ValueError(f"{reactome_id} is not a valid reactome ID")

    return value


def _validate_assets_sbml_ids(
    sbml_dfs: sbml_dfs_core.SBML_dfs, identifiers_df: pd.DataFrame
) -> None:
    """Check an sbml_dfs file and identifiers table for inconsistencies."""

    joined_species_w_ids = sbml_dfs.species.merge(
        identifiers_df[["s_id", "s_name"]].drop_duplicates(),
        left_index=True,
        right_on="s_id",
    )

    inconsistent_names_df = joined_species_w_ids.query("s_name_x != s_name_y").dropna()
    inconsistent_names_list = [
        f"{x} != {y}"
        for x, y in zip(
            inconsistent_names_df["s_name_x"], inconsistent_names_df["s_name_y"]
        )
    ]

    if len(inconsistent_names_list):
        example_inconsistent_names = inconsistent_names_list[
            0 : min(10, len(inconsistent_names_list))
        ]

        raise ValueError(
            f"{len(inconsistent_names_list)} species names do not match between "
            f"sbml_dfs and identifiers_df including: {', '.join(example_inconsistent_names)}"
        )

    return None


def _validate_bqb(bqb: str) -> None:
    """
    Validate a BQB code

    Parameters
    ----------
    bqb : str
        The BQB code to validate

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the BQB code is not a string
    ValueError
        If the BQB code does not start with 'BQB'
    """

    if not isinstance(bqb, str):
        raise TypeError(
            f"biological_qualifier_type was a {type(bqb)} and must be a str or None"
        )

    if not bqb.startswith("BQB"):
        raise ValueError(
            f"The provided BQB code was {bqb} and all BQB codes start with "
            'start with "BQB". Please either use a valid BQB code (see '
            '"BQB" in constansts.py) or use None'
        )

    return None


# validators


class _IdentifierValidator(BaseModel):
    ontology: str
    identifier: str
    bqb: str
    url: Optional[str] = None


class _IdentifiersValidator(BaseModel):
    id_list: list[_IdentifierValidator]
