from __future__ import annotations

import itertools
import logging
import re
import sys
from typing import Optional
from urllib.parse import urlparse

import libsbml
import pandas as pd
from pydantic import BaseModel

from napistu import sbml_dfs_core
from napistu import sbml_dfs_utils
from napistu import utils

from napistu.constants import IDENTIFIERS
from napistu.constants import BIOLOGICAL_QUALIFIER_CODES
from napistu.constants import ENSEMBL_MOLECULE_TYPES_TO_ONTOLOGY
from napistu.constants import ENSEMBL_MOLECULE_TYPES_FROM_ONTOLOGY
from napistu.constants import ENSEMBL_SPECIES_FROM_CODE
from napistu.constants import ENSEMBL_SPECIES_TO_CODE
from napistu.constants import SPECIES_IDENTIFIERS_REQUIRED_VARS

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

    Methods
    -------
    print
        Print a table of identifiers
    filter(ontologies, summarize)
        Returns a bool of whether 1+ of the ontologies was represented
    hoist(ontology)
        Returns value(s) from an ontology

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

        if (len(id_list) == 0) and verbose:
            logger.debug('zero identifiers in "id_list"')

        if len(id_list) != 0:
            # de-duplicate {identifier, ontology} tuples

            coded_ids = [
                x[IDENTIFIERS.ONTOLOGY] + "_" + x[IDENTIFIERS.IDENTIFIER]
                for x in validated_id_list
            ]
            unique_cids = []
            unique_cid_indices = []
            i = 0
            for cid in coded_ids:
                if cid not in unique_cids:
                    unique_cids.append(cid)
                    unique_cid_indices.append(i)
                i += 1
            validated_id_list = [validated_id_list[i] for i in unique_cid_indices]

        self.ids = validated_id_list

    def print(self):
        """Print a table of identifiers"""

        utils.style_df(pd.DataFrame(self.ids), hide_index=True)

    def filter(self, ontologies, summarize=True):
        """Returns a bool of whether 1+ of the ontologies was represented"""

        if isinstance(ontologies, str):
            ontologies = [ontologies]

        # filter based on whether any ontology of interest are present
        # identifier_matches = [x['ontology'] == y for x in self.ids for y in ontologies]

        identifier_matches = []
        for an_id in self.ids:
            identifier_matches.append(
                any([an_id[IDENTIFIERS.ONTOLOGY] == y for y in ontologies])
            )

        if summarize:
            return any(identifier_matches)
        else:
            return identifier_matches

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
        ontology_matches = [
            x for x, y in zip(self.ids, self.filter(ontology, summarize=False)) if y
        ]
        ontology_ids = [x[IDENTIFIERS.IDENTIFIER] for x in ontology_matches]

        if squeeze:
            if len(ontology_ids) == 0:
                return None
            elif len(ontology_ids) == 1:
                return ontology_ids[0]
        return ontology_ids


def merge_identifiers(identifier_series: pd.Series) -> Identifiers:
    """
    Aggregate Identifiers

    Merge a pd.Series of Identifiers objects into a single Identifiers object

    Args:
    identifier_series: pd.Series
        A pd.Series of of identifiers.Identifiers objects


    Returns:
    An identifiers.Identifiers object

    """

    if len(identifier_series) == 1:
        # if there is only a single entry then just return it because no merge is needed
        return identifier_series.iloc[0]
    else:
        # merge a list of identifiers objects into a single identifers object
        # Identifiers will remove redundancy
        merged_ids = list(
            itertools.chain.from_iterable(identifier_series.map(lambda x: x.ids))
        )
        return Identifiers(merged_ids)


def format_uri(uri: str, biological_qualifier_type: str | None = None) -> Identifiers:
    """
    Convert a RDF URI into an Identifier object
    """

    identifier = format_uri_url(uri)

    if identifier is None:
        raise NotImplementedError(f"{uri} is not a valid way of specifying a uri")

    _validate_bqb(biological_qualifier_type)
    identifier[IDENTIFIERS.BQB] = biological_qualifier_type

    return identifier


def _validate_bqb(bqb):
    if bqb is None:
        logger.warning(
            '"biological_qualifier_type" is None; consider adding a valid '
            'BQB code. For a list of BQB codes see "BQB" in constants.py'
        )
    else:
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


def format_uri_url(uri: str) -> dict:
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
            ontology = "reactome"
            identifier = split_path[-1]
        # genes and gene products
        elif netloc == "www.ensembl.org" and split_path[-1] == "geneview":
            ontology = "ensembl_gene"
            identifier, id_ontology, _ = parse_ensembl_id(result.query)  # type: ignore
            if ontology != id_ontology:
                raise ValueError(
                    f"Ontology mismatch: expected {ontology}, got {id_ontology}"
                )
        elif netloc == "www.ensembl.org" and split_path[-1] in [
            "transview",
            "Transcript",
        ]:
            ontology = "ensembl_transcript"
            identifier, id_ontology, _ = parse_ensembl_id(result.query)  # type: ignore
            if ontology != id_ontology:
                raise ValueError(
                    f"Ontology mismatch: expected {ontology}, got {id_ontology}"
                )
        elif netloc == "www.ensembl.org" and split_path[-1] == "ProteinSummary":
            ontology = "ensembl_protein"
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
            identifier, implied_ontology, _ = parse_ensembl_id(split_path[-1])
            if implied_ontology != ontology:
                raise ValueError(
                    f"Implied ontology mismatch: expected {ontology}, got {implied_ontology}"
                )
        elif netloc == "www.mirbase.org" or netloc == "mirbase.org":
            ontology = "mirbase"
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
            ontology = "uniprot"
            identifier = split_path[-1]
        elif netloc == "rnacentral.org":
            ontology = "rnacentral"
            identifier = split_path[-1]
        # chemicals
        elif split_path[1] == "chebi":
            ontology = "chebi"
            identifier = utils.extract_regex_search("[0-9]+$", result.query)
        elif netloc == "pubchem.ncbi.nlm.nih.gov":
            ontology = "pubchem"
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
            ontology = "ec-code"
            identifier = split_path[-1]
        elif netloc == "www.rhea-db.org":
            ontology = "rhea"
            identifier = utils.extract_regex_search("[0-9]+$", result.query)
        # misc
        elif split_path[1] == "ols":
            ontology = "ols"
            identifier = split_path[-1]
        elif split_path[1] == "QuickGO":
            ontology = "go"
            identifier = split_path[-1]
        elif split_path[1] == "pubmed":
            ontology = "pubmed"
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
        else:
            raise NotImplementedError(
                f"{netloc} in the {uri} url has not been associated with a known ontology"
            )
    except TypeError:
        logger.warning(
            f"An identifier could not be found using the specified regex for {uri} based on the {ontology} ontology"
        )
        logger.warning(result)
        logger.warning("ERROR")
        sys.exit(1)

    # rename some entries

    if ontology == "ncbi_gene":
        ontology = "ncbi_entrez_gene"

    id_dict = {"ontology": ontology, "identifier": identifier, "url": uri}

    return id_dict


def parse_ensembl_id(input_str: str) -> tuple[str, str, str]:
    """
    Parse Ensembl ID

    Extract the molecule type and species name from a string containing an ensembl identifier.

    Args:
        input_str (str):
            A string containing an ensembl gene, transcript, or protein identifier

    Returns:
        identifier (str):
            The substring matching the full identifier
        molecule_type (str):
            The ontology the identifier belongs to:
                - G -> ensembl_gene
                - T -> ensembl_transcript
                - P -> ensembl_protein
        species (str):
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

    if species_code_search.group(1) is None:  # type:ignore
        species = "Homo sapiens"
        molecule_type_regex = "ENS([GTP])"
        id_regex = "ENS[GTP][0-9]+"
    else:
        species_code = species_code_search.group(1)  # type:ignore

        if species_code not in ENSEMBL_SPECIES_FROM_CODE.keys():
            raise ValueError(
                f"The species code for {input_str}: {species_code} did not "
                "match any of the entries in ENSEMBL_SPECIES_CODE_LOOKUPS."
            )

        species = ENSEMBL_SPECIES_FROM_CODE[species_code]
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

    return identifier, molecule_type, species


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

        if ontology in ["chebi"]:
            identifier = utils.extract_regex_search("[0-9]+$", split_path[-1])
        elif len(split_path) != 3:
            identifier = "/".join(split_path[2:])
        else:
            identifier = split_path[-1]

    return ontology, identifier


def cv_to_Identifiers(entity):
    """
    Convert an SBML controlled vocabulary element into a cpr Identifiers object.

    Parameters:
    entity: libsbml.Species
        An entity (species, reaction, compartment, ...) with attached CV terms

    Returns:


    """

    # TO DO: add qualifier type http://sbml.org/Software/libSBML/5.18.0/docs/python-api/classlibsbml_1_1_c_v_term.html#a6a613cc17c6f853cf1c68da59286b373

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
            try:
                out_list.append(
                    format_uri(cv.getResourceURI(i), biological_qualifier_type)
                )
            except NotImplementedError:
                logger.warning("Not all identifiers resolved: ", exc_info=True)

        cv_list.extend(out_list)
    return Identifiers(cv_list)


def create_uri_url(ontology: str, identifier: str, strict: bool = True) -> str:
    """
    Create URI URL

    Convert from an identifier and ontology to a URL reference for the identifier

    Parameters:
    ontology (str): An ontology for organizing genes, metabolites, etc.
    identifier (str): A systematic identifier from the \"ontology\" ontology.
    strict (bool): if strict then throw errors for invalid IDs otherwise return None

    Returns:
    url (str): A url representing a unique identifier

    """

    # check input types
    if not isinstance(ontology, str):
        raise TypeError(f"ontology was an {type(ontology).__name__} and must be a str")
    if not isinstance(identifier, str):
        raise TypeError(
            f"identifier was an {type(identifier).__name__} and must be a str"
        )
    if not isinstance(strict, bool):
        raise TypeError(f"strict was an {type(strict).__name__} and must be a bool")

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
                print(failure_msg + " returning None")
                return None

    return url


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


def _infer_primary_reactome_species(reactome_series: pd.Series) -> tuple[str, int]:
    """Infer the best supported species based on a set of Reactome identifiers"""

    series_counts = _count_reactome_species(reactome_series)

    if "ALL" in series_counts.index:
        series_counts = series_counts.drop("ALL", axis=0)

    return series_counts.index[0], series_counts.iloc[0]


def _count_reactome_species(reactome_series: pd.Series) -> pd.Series:
    """Count the number of species tags in a set of reactome IDs"""

    return (
        reactome_series.drop_duplicates().transform(_reactome_id_species).value_counts()
    )


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


def _format_Identifiers_pubmed(pubmed_id: str) -> Identifiers:
    """
    Format Identifiers for a single PubMed ID.

    These will generally be used in an r_Identifiers field.
    """

    # create a url for lookup and validate the pubmed id
    url = create_uri_url(ontology="pubmed", identifier=pubmed_id, strict=False)
    id_entry = format_uri(uri=url, biological_qualifier_type="BQB_IS_DESCRIBED_BY")

    return Identifiers([id_entry])


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


def _prepare_species_identifiers(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    dogmatic: bool = False,
    species_identifiers: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Accepts and validates species_identifiers, or extracts a fresh table if None."""

    if species_identifiers is None:
        species_identifiers = sbml_dfs_utils.get_characteristic_species_ids(
            sbml_dfs, dogmatic=dogmatic
        )
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
            species_identifiers = sbml_dfs_utils.get_characteristic_species_ids(
                sbml_dfs, dogmatic=dogmatic
            )

    return species_identifiers


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


class _IdentifierValidator(BaseModel):
    ontology: str
    identifier: str
    url: Optional[str] = None
    bqb: Optional[str] = None


class _IdentifiersValidator(BaseModel):
    id_list: list[_IdentifierValidator]
