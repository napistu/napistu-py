from types import SimpleNamespace

from napistu.constants import ONTOLOGIES

ONTOLOGY_MAP = SimpleNamespace(
    URL="url",
    ID_REGEX="id_regex",
)

ONTOLOGY_TO_URL_MAP = {
    ONTOLOGIES.BIGG_METABOLITE: {
        ONTOLOGY_MAP.URL: "http://identifiers.org/bigg.metabolite/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.CHEBI: {
        ONTOLOGY_MAP.URL: "http://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.CHEMSPIDER: {
        ONTOLOGY_MAP.URL: "https://www.chemspider.com/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.DOI: {
        ONTOLOGY_MAP.URL: "https://doi.org/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.DX_DOI: {
        ONTOLOGY_MAP.URL: "https://dx.doi.org/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: r"^[0-9]+\.[0-9]+$",
    },
    ONTOLOGIES.EC_CODE: {
        ONTOLOGY_MAP.URL: "https://identifiers.org/ec-code/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+\\.[0-9]+\\.[0-9]+(\\.[0-9]+)?$",
    },
    ONTOLOGIES.ENVIPATH: {
        ONTOLOGY_MAP.URL: "http://identifiers.org/envipath/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.GO: {
        ONTOLOGY_MAP.URL: "https://www.ebi.ac.uk/QuickGO/term/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^GO:[0-9]{7}$",
    },
    ONTOLOGIES.MATRIXDB_BIOMOLECULE: {
        ONTOLOGY_MAP.URL: "http://matrixdb.univ-lyon1.fr/cgi-bin/current/newPort?type=biomolecule&value={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9A-Za-z]+$",
    },
    ONTOLOGIES.MATRIXDB_MOLECULE_CLASS: {
        ONTOLOGY_MAP.URL: "http://matrixdb.univ-lyon1.fr/cgi-bin/current/newPort?type=biomolecule&value={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9A-Za-z]+$",
    },
    ONTOLOGIES.MDPI: {
        ONTOLOGY_MAP.URL: "https://www.mdpi.com/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.NCBI_BOOKS: {
        ONTOLOGY_MAP.URL: "http://www.ncbi.nlm.nih.gov/books/{identifier}/",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9A-Z]+$",
    },
    ONTOLOGIES.NCBI_ENTREZ_GENE: {
        ONTOLOGY_MAP.URL: "https://www.ncbi.nlm.nih.gov/gene/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.NCBI_ENTREZ_PCCOMPOUND: {
        ONTOLOGY_MAP.URL: "http://www.ncbi.nlm.nih.gov/sites/entrez?cmd=search&db=pccompound&term={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[A-Z]{14}\\-[A-Z]{10}\\-[A-Z]{1}$",
    },
    ONTOLOGIES.NCI_THESAURUS: {
        ONTOLOGY_MAP.URL: "https://ncithesaurus.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&code={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[A-Z][0-9]+$",
    },
    ONTOLOGIES.PHOSPHOSITE: {
        ONTOLOGY_MAP.URL: "https://www.phosphosite.org/siteAction.action?id={identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.PUBCHEM: {
        ONTOLOGY_MAP.URL: "http://pubchem.ncbi.nlm.nih.gov/compound/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.PUBMED: {
        ONTOLOGY_MAP.URL: "http://www.ncbi.nlm.nih.gov/pubmed/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9]+$",
    },
    ONTOLOGIES.REACTOME: {
        ONTOLOGY_MAP.URL: "https://reactome.org/content/detail/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^R\\-[A-Z]{3}\\-[0-9]{7}$",
    },
    ONTOLOGIES.RNACENTRAL: {
        ONTOLOGY_MAP.URL: "https://rnacentral.org/rna/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: None,
    },
    ONTOLOGIES.SGC: {
        ONTOLOGY_MAP.URL: "https://www.thesgc.org/structures/structure_description/{identifier}/",
        ONTOLOGY_MAP.ID_REGEX: "^[0-9A-Z]+$",
    },
    ONTOLOGIES.UNIPROT: {
        ONTOLOGY_MAP.URL: "https://purl.uniprot.org/uniprot/{identifier}",
        ONTOLOGY_MAP.ID_REGEX: "^[A-Z0-9]+$",
    },
    ONTOLOGIES.URL: {ONTOLOGY_MAP.URL: "{identifier}", ONTOLOGY_MAP.ID_REGEX: None},
}
