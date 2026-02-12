"""
Standardization of ontologies for creating Identifiers and URLs

Public Functions
----------------
check_reactome_identifier_compatibility
    Check whether two sets of Reactome identifiers are from the same species.
create_uri_url
    Convert from an identifier and ontology to a URL reference for the identifier.
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

import logging
import re
from typing import Optional
from urllib.parse import urlparse

from pandas import Series

from napistu.constants import (
    BQB,
    IDENTIFIERS,
)
from napistu.identifiers import Identifiers
from napistu.ingestion.constants import LATIN_SPECIES_NAMES
from napistu.ontologies.constants import (
    ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY,
    ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY,
    ENSEMBL_SPECIES_FROM_CODE,
    ENSEMBL_SPECIES_TO_CODE,
    ONTOLOGIES,
    ONTOLOGIES_LIST,
    ONTOLOGY_TO_URL_MAP,
)
from napistu.utils.string_utils import extract_regex_match, extract_regex_search

logger = logging.getLogger(__name__)


def check_reactome_identifier_compatibility(
    reactome_series_a: Series,
    reactome_series_b: Series,
) -> None:
    """
    Check Reactome Identifier Compatibility

    Determine whether two sets of Reactome identifiers are from the same organismal species.

    Parameters
    ----------
    reactome_series_a: pd.Series
        a Series containing Reactome identifiers
    reactome_series_b: pd.Series
        a Series containing Reactome identifiers

    Returns
    -------
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

    if ontology in [
        ONTOLOGIES.ENSEMBL_GENE,
        ONTOLOGIES.ENSEMBL_TRANSCRIPT,
        ONTOLOGIES.ENSEMBL_PROTEIN,
    ]:
        id_regex, url = ensembl_id_to_url_regex(identifier, ontology)
    elif ontology == ONTOLOGIES.MIRBASE:
        id_regex = None
        if re.match("MIMAT[0-9]", identifier):
            url = f"https://www.mirbase.org/mature/{identifier}"
        elif re.match("MI[0-9]", identifier):
            url = f"https://www.mirbase.org/hairpin/{identifier}"
        else:
            raise NotImplementedError(f"url not defined for this MiRBase {identifier}")
    elif ontology in ONTOLOGY_TO_URL_MAP.keys():
        id_regex = ONTOLOGY_TO_URL_MAP[ontology]["id_regex"]
        url = ONTOLOGY_TO_URL_MAP[ontology]["url"].format(identifier=identifier)
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


def ensembl_id_to_url_regex(identifier: str, ontology: str) -> tuple[str, str]:
    """
    Ensembl ID to URL and Regex

    Map an ensembl ID to a validation regex and its canonical url on ensembl

    Parameters
    ----------
    identifier : str
        A standard identifier from ensembl genes, transcripts, or proteins
    ontology : str
        The standard ontology (ensembl_gene, ensembl_transcript, or ensembl_protein)

    Returns
    -------
    tuple[str, str]
        id_regex : a regex which should match a valid entry in this ontology
        url : the id's url on ensembl
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

    if species == LATIN_SPECIES_NAMES.HOMO_SAPIENS:
        species_code = ""
    else:
        species_code = ENSEMBL_SPECIES_TO_CODE[species]
    molecule_type_code = ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY[ontology]

    id_regex = "ENS" + species_code + molecule_type_code + "[0-9]{11}"

    # convert to species format in ensembl urls
    species_url_field = re.sub(" ", "_", species)

    if ontology == ONTOLOGIES.ENSEMBL_GENE:
        url = f"http://www.ensembl.org/{species_url_field}/geneview?gene={identifier}"
    elif ontology == ONTOLOGIES.ENSEMBL_TRANSCRIPT:
        url = f"http://www.ensembl.org/{species_url_field}/Transcript?t={identifier}"
    elif ontology == ONTOLOGIES.ENSEMBL_PROTEIN:
        url = f"https://www.ensembl.org/{species_url_field}/Transcript/ProteinSummary?t={identifier}"
    else:
        ValueError(f"{ontology} not defined")

    return id_regex, url


def format_uri(uri: str, bqb: str, strict: bool = True) -> Optional[list[dict]]:
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
    Optional[list[dict]]
        The identifier list or None if the URI is not valid
    """

    if not isinstance(uri, str):
        raise TypeError(f"uri must be a string, got {type(uri)}")
    if not isinstance(bqb, str):
        raise TypeError(f"bqb must be a string, got {type(bqb)}")

    identifier = format_uri_url(uri, strict=strict)

    if identifier is None:
        if strict:
            raise NotImplementedError(f"{uri} is not a valid way of specifying a uri")
        else:
            # Return empty list for non-strict mode
            return None

    _validate_bqb(bqb)
    identifier[IDENTIFIERS.BQB] = bqb

    return identifier


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
    split_one = split_path[1]  # used for ontology identification
    url_suffix = split_path[-1]  # used for ontology identification

    try:
        # identifiers define by just netloc
        if netloc in NETLOC_TO_IDENTIFIERS_MAP:
            ontology, identifier = NETLOC_TO_IDENTIFIERS_MAP[netloc](
                split_path, result, uri
            )
        # identifiers define by netloc + url_suffix
        elif (netloc, url_suffix) in NETLOC_W_URL_SUFFIX_TO_IDENTIFIERS_MAP:
            ontology, identifier = NETLOC_W_URL_SUFFIX_TO_IDENTIFIERS_MAP[
                (netloc, url_suffix)
            ](split_path, result)
        # identifiers define by split_one
        elif split_one in SPLIT_ONE_TO_IDENTIFIERS_MAP:
            ontology, identifier = SPLIT_ONE_TO_IDENTIFIERS_MAP[split_one](
                split_path, result
            )
        # identifiers define by netloc + split_one
        elif (netloc, split_one) in NETLOC_W_URL_ONE_TO_IDENTIFIERS_MAP:
            ontology, identifier = NETLOC_W_URL_ONE_TO_IDENTIFIERS_MAP[
                (netloc, split_one)
            ](split_path, result, uri)
        elif netloc == "www.ensembl.org" and (
            re.search("ENS[GTP]", split_path[-1])
            or re.search("ENS[A-Z]{3}[GTP]", split_path[-1])
        ):
            # format ensembl IDs which lack gene/transview
            identifier, ontology, _ = parse_ensembl_id(split_path[-1])
        else:
            error_msg = f"{netloc} in the {uri} url has not been associated with a known ontology"
            if strict:
                raise NotImplementedError(error_msg)
            else:
                logger.warning(error_msg)
                return None
    except (TypeError, AttributeError, ValueError) as e:
        if strict:
            logger.warning(
                f"An identifier could not be found using the specified regex for {uri} based on the ontology"
            )
            raise e
        else:
            logger.warning(f"Could not extract identifier from URI using regex: {uri}")
            return None

    # rename some entries

    if ontology == "ncbi_gene":
        logger.warning("Renaming ncbi_gene to ncbi_entrez_gene")
        ontology = ONTOLOGIES.NCBI_ENTREZ_GENE

    id_dict = {
        IDENTIFIERS.ONTOLOGY: ontology,
        IDENTIFIERS.IDENTIFIER: identifier,
        IDENTIFIERS.URL: uri,
    }

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
            identifier = extract_regex_search("[0-9]+$", split_path[-1])
        elif len(split_path) != 3:
            identifier = "/".join(split_path[2:])
        else:
            identifier = split_path[-1]

    # rename some entires
    if ontology == "bigg.metabolite":
        ontology = ONTOLOGIES.BIGG_METABOLITE

    return ontology, identifier


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

    if species_code_search.group(1) is None:
        organismal_species = LATIN_SPECIES_NAMES.HOMO_SAPIENS
        molecule_type_regex = "ENS([GTP])"
        id_regex = "ENS[GTP][0-9]+"
    else:
        species_code = species_code_search.group(1)

        if species_code in ENSEMBL_SPECIES_FROM_CODE.keys():
            organismal_species = ENSEMBL_SPECIES_FROM_CODE[species_code]
        else:
            logger.warning(
                f"The species code for {input_str}: {species_code} did not "
                "match any of the entries in ENSEMBL_SPECIES_CODE_LOOKUPS. Replacing with 'unknown'."
            )
            organismal_species = "unknown"

        molecule_type_regex = "ENS[A-Z]{3}([GTP])"
        id_regex = "ENS[A-Z]{3}[GTP][0-9]+"

    # extract the molecule type (genes, transcripts or proteins)
    molecule_type_code_search = re.compile(molecule_type_regex).search(input_str)
    if not molecule_type_code_search:
        raise ValueError(
            "The ensembl molecule code (i.e., G, T or P) could not be extracted from {input_str}"
        )
    else:
        molecule_type_code = molecule_type_code_search.group(1)

    if molecule_type_code not in ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY.keys():
        raise ValueError(
            f"The molecule type code for {input_str}: {molecule_type_code} did not "
            "match ensembl genes (G), transcripts (T), or proteins (P)."
        )

    molecule_type = ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY[molecule_type_code]

    identifier = extract_regex_search(id_regex, input_str)

    return identifier, molecule_type, organismal_species


def _count_reactome_species(reactome_series: Series) -> Series:
    """Count the number of species tags in a set of reactome IDs"""

    return (
        reactome_series.drop_duplicates().transform(_reactome_id_species).value_counts()
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


def _infer_primary_reactome_species(reactome_series: Series) -> tuple[str, int]:
    """Infer the best supported species based on a set of Reactome identifiers"""

    series_counts = _count_reactome_species(reactome_series)

    if "ALL" in series_counts.index:
        series_counts = series_counts.drop("ALL", axis=0)

    return series_counts.index[0], series_counts.iloc[0]


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


# ID mapping adapters and lambda functions


def _netloc_to_identifiers_mirbase_adaptor(split_path, result):
    ontology = ONTOLOGIES.MIRBASE
    if re.search("MI[0-9]+", split_path[-1]):
        identifier = extract_regex_search("MI[0-9]+", split_path[-1])
    elif re.search("MIMAT[0-9]+", split_path[-1]):
        identifier = extract_regex_search("MIMAT[0-9]+", split_path[-1])
    elif re.search("MI[0-9]+", result.query):
        identifier = extract_regex_search("MI[0-9]+", result.query)
    elif re.search("MIMAT[0-9]+", result.query):
        identifier = extract_regex_search("MIMAT[0-9]+", result.query)
    else:
        raise TypeError(f"{result.query} does not appear to match MiRBase identifiers")
    return ontology, identifier


def _netloc_to_identifiers_pubchem_adaptor(split_path, result):
    ontology = ONTOLOGIES.PUBCHEM
    if result.query != "":
        identifier = extract_regex_search("[0-9]+$", result.query)
    else:
        identifier = extract_regex_search("[0-9]+$", split_path[-1])
    return ontology, identifier


def _netloc_to_identifiers_matrixdb_adaptor(uri, class_regex, id_regex):
    molecule_class = extract_regex_match(class_regex, uri).lower()
    ontology = f"matrixdb_{molecule_class}"
    if ontology not in ONTOLOGIES_LIST:
        logger.warning(
            f"Ontology {ontology} is not a recognized ontology. Extracted from {uri}"
        )
        return None
    identifier = extract_regex_match(id_regex, uri)
    return ontology, identifier


def _netloc_w_url_suffix_to_identifiers_phosphosite_adaptor(split_path, result):
    # phosphosite ligands (i.e., sites)
    if split_path[1] == "siteAction.action":
        ontology = ONTOLOGIES.PHOSPHOSITE
        identifier = extract_regex_match("id=([0-9]+)", result.query)
    # phosphosite kinases
    elif split_path[1] == "proteinAction.do":
        ontology = ONTOLOGIES.PHOSPHOSITE_KINASE
        identifier = extract_regex_match("id=([0-9]+)", result.query)
    else:
        raise ValueError(f"Unknown phosphosite URL: {split_path[1]}")

    return ontology, identifier


# netloc + split_path[1] -> extraction function
NETLOC_TO_IDENTIFIERS_MAP = {
    "www.biorxiv.org": lambda split_path, result, uri: (
        ONTOLOGIES.BIORXIV,
        split_path[-1],
    ),
    "chemspider.com": lambda split_path, result, uri: (
        ONTOLOGIES.CHEMSPIDER,
        split_path[-1],
    ),
    "www.chemspider.com": lambda split_path, result, uri: (
        ONTOLOGIES.CHEMSPIDER,
        split_path[-1],
    ),
    "dx.doi.org": lambda split_path, result, uri: (
        ONTOLOGIES.DX_DOI,
        "/".join(split_path[1:]),
    ),
    "doi.org": lambda split_path, result, uri: (
        ONTOLOGIES.DOI,
        "/".join(split_path[1:]),
    ),
    "ec-code": lambda split_path, result, uri: (ONTOLOGIES.EC_CODE, split_path[-1]),
    "www.genome.ad.jp": lambda split_path, result, uri: (
        ONTOLOGIES.GENOME_NET,
        extract_regex_search("[A-Za-z]+:[0-9]+$", uri),
    ),
    "identifiers.org": lambda split_path, result, uri: format_uri_url_identifiers_dot_org(
        split_path
    ),
    "matrixdb.ibcp.fr": lambda split_path, result, uri: _netloc_to_identifiers_matrixdb_adaptor(
        uri, ".*class=([a-zA-Z]+).*", ".*name=([0-9A-Za-z]+).*"
    ),
    "matrixdb.univ-lyon1.fr": lambda split_path, result, uri: _netloc_to_identifiers_matrixdb_adaptor(
        uri, ".*type=([a-zA-Z]+).*", ".*value=([0-9A-Za-z]+).*"
    ),
    "www.mdpi.com": lambda split_path, result, uri: (
        ONTOLOGIES.MDPI,
        "/".join([i for i in split_path[1:] if i != ""]),
    ),
    "mirbase.org": lambda split_path, result, uri: _netloc_to_identifiers_mirbase_adaptor(
        split_path, result
    ),
    "www.mirbase.org": lambda split_path, result, uri: _netloc_to_identifiers_mirbase_adaptor(
        split_path, result
    ),
    "ncithesaurus.nci.nih.gov": lambda split_path, result, uri: (
        ONTOLOGIES.NCI_THESAURUS,
        extract_regex_match(".*code=([0-9A-Z]+).*", uri),
    ),
    "ols": lambda split_path, result, uri: (ONTOLOGIES.OLS, split_path[-1]),
    "reactome.org": lambda split_path, result, uri: (
        ONTOLOGIES.REACTOME,
        split_path[-1],
    ),
    "rhea-db.org": lambda split_path, result, uri: (
        ONTOLOGIES.RHEA,
        extract_regex_search("[0-9]+$", result.query),
    ),
    "rnacentral.org": lambda split_path, result, uri: (
        ONTOLOGIES.RNACENTRAL,
        split_path[-1],
    ),
    "www.phosphosite.org": lambda split_path, result, uri: _netloc_w_url_suffix_to_identifiers_phosphosite_adaptor(
        split_path, result
    ),
    "pubchem.ncbi.nlm.nih.gov": lambda split_path, result, uri: _netloc_to_identifiers_pubchem_adaptor(
        split_path, result
    ),
    "purl.uniprot.org": lambda split_path, result, uri: (
        ONTOLOGIES.UNIPROT,
        split_path[-1],
    ),
    "users.rcn.com": lambda split_path, result, uri: (ONTOLOGIES.URL, uri),
}


def _netloc_w_url_suffix_to_identifiers_ensembl_adaptor(result, ontology: str):
    identifier, id_ontology, _ = parse_ensembl_id(result.query)  # type: ignore
    if ontology != id_ontology:
        raise ValueError(f"Ontology mismatch: expected {ontology}, got {id_ontology}")
    return ontology, identifier


NETLOC_W_URL_SUFFIX_TO_IDENTIFIERS_MAP = {
    (
        "www.ensembl.org",
        "geneview",
    ): lambda split_path, result: _netloc_w_url_suffix_to_identifiers_ensembl_adaptor(
        result, ONTOLOGIES.ENSEMBL_GENE
    ),
    (
        "www.ensembl.org",
        "ProteinSummary",
    ): lambda split_path, result: _netloc_w_url_suffix_to_identifiers_ensembl_adaptor(
        result, ONTOLOGIES.ENSEMBL_PROTEIN
    ),
    (
        "www.ensembl.org",
        "transview",
    ): lambda split_path, result: _netloc_w_url_suffix_to_identifiers_ensembl_adaptor(
        result, ONTOLOGIES.ENSEMBL_TRANSCRIPT
    ),
    (
        "www.ensembl.org",
        "Transcript",
    ): lambda split_path, result: _netloc_w_url_suffix_to_identifiers_ensembl_adaptor(
        result, ONTOLOGIES.ENSEMBL_TRANSCRIPT
    ),
    (
        "www.guidetopharmacology.org",
        "LigandDisplayForward",
    ): lambda split_path, result: (
        ONTOLOGIES.GUIDETOPHARMACOLOGY,
        extract_regex_search("[0-9]+$", result.query),
    ),
}


def _netloc_w_url_prefix_to_identifiers_ncbi_adaptor(result, uri):
    ontology = "ncbi_entrez_" + extract_regex_search(
        "db=([A-Za-z0-9]+)\\&", result.query, 1
    )

    if ontology not in ONTOLOGIES_LIST:
        logger.warning(
            f"Ontology {ontology} is not a recognized ontology. Extracted from {uri}"
        )
        return None

    identifier = extract_regex_search(r"term=([A-Za-z0-9\-]+)$", result.query, 1)

    return ontology, identifier


NETLOC_W_URL_ONE_TO_IDENTIFIERS_MAP = {
    ("www.ncbi.nlm.nih.gov", "nuccore"): lambda split_path, result, uri: (
        ONTOLOGIES.NCBI_REFSEQ,
        split_path[-1],
    ),
    (
        "www.ncbi.nlm.nih.gov",
        "sites",
    ): lambda split_path, result, uri: _netloc_w_url_prefix_to_identifiers_ncbi_adaptor(
        result, uri
    ),
    ("www.ncbi.nlm.nih.gov", "books"): lambda split_path, result, uri: (
        ONTOLOGIES.NCBI_BOOKS,
        split_path[2],
    ),
    ("www.ncbi.nlm.nih.gov", "gene"): lambda split_path, result, uri: (
        ONTOLOGIES.NCBI_ENTREZ_GENE,
        split_path[2],
    ),
    ("www.ebi.ac.uk", "ena"): lambda split_path, result, uri: (
        ONTOLOGIES.EBI_REFSEQ,
        split_path[-1],
    ),
    ("www.thesgc.org", "structures"): lambda split_path, result, uri: (
        ONTOLOGIES.SGC,
        split_path[-2],
    ),
}


def _split_one_to_identifiers_chebi_adaptor(split_path, result):
    if re.match("CHEBI:[0-9]+", split_path[-1]):
        identifier = extract_regex_search("[0-9]+$", split_path[-1])
    else:
        identifier = extract_regex_search("[0-9]+$", result.query)

    return ONTOLOGIES.CHEBI, identifier


SPLIT_ONE_TO_IDENTIFIERS_MAP = {
    "chebi": lambda split_path, result: _split_one_to_identifiers_chebi_adaptor(
        split_path, result
    ),
    "ols": lambda split_path, result: (ONTOLOGIES.OLS, split_path[-1]),
    "pubmed": lambda split_path, result: (ONTOLOGIES.PUBMED, split_path[-1]),
    "QuickGO": lambda split_path, result: (ONTOLOGIES.GO, split_path[-1]),
}
