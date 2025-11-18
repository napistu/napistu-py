from __future__ import annotations

import datetime
import logging
import warnings
from typing import Union

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs
import pandas as pd

from napistu import identifiers, sbml_dfs_utils, utils
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    SBML_DFS,
    SBOTERM_NAMES,
)
from napistu.ingestion import napistu_edgelist
from napistu.ingestion.constants import (
    DATA_SOURCE_DESCRIPTIONS,
    DATA_SOURCES,
    INTERACTION_EDGELIST_DEFS,
    STRING_PROTEIN_ID,
    STRING_PROTEIN_ID_RAW,
    STRING_SOURCE,
    STRING_TARGET,
    STRING_TAX_IDS,
    STRING_URL_EXPRESSIONS,
    STRING_VERSION,
)
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator
from napistu.sbml_dfs_core import SBML_dfs
from napistu.source import Source

logger = logging.getLogger(__name__)


def get_string_species_url(
    organismal_species: Union[str, OrganismalSpeciesValidator],
    asset: str,
    version: float = STRING_VERSION,
) -> str:
    """
    STRING Species URL

    Construct urls for downloading specific STRING tables

    Parameters
    ----------
    organismal_species : str | OrganismalSpeciesValidator
        A species name: e.g., Homo sapiens.
    asset : str
        The type of table to be downloaded. Currently "interactions" or "aliases".
    version : float
        The version of STRING to work with.

    Returns
    -------
        str: The download url
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)
    organismal_species.assert_supported(STRING_TAX_IDS.keys())
    tax_id = STRING_TAX_IDS[organismal_species.latin_name]

    if asset not in STRING_URL_EXPRESSIONS.keys():
        raise ValueError(
            f"{asset} is not a valid value for a STRING asset, valid assets are: "
            f"{', '.join(STRING_URL_EXPRESSIONS.keys())}"
        )

    url_fstring = STRING_URL_EXPRESSIONS[asset]

    return eval(f'f"{url_fstring}"', {"taxid": tax_id, "version": version})


def download_string(
    target_uri: str, organismal_species: Union[str, OrganismalSpeciesValidator]
) -> None:
    """Downloads string to the target uri

    Parameters
    ----------
    target_uri : str
        target url
    organismal_species : str | OrganismalSpeciesValidator
        A species name: e.g., Homo sapiens

    Returns
    -------
        None
    """
    string_url = get_string_species_url(
        organismal_species, asset="interactions", version=STRING_VERSION
    )
    logger.info("Start downloading string db %s to %s", string_url, target_uri)

    utils.download_wget(string_url, target_uri)

    return None


def download_string_aliases(
    target_uri: str, organismal_species: Union[str, OrganismalSpeciesValidator]
) -> None:
    """Downloads string aliases to the target uri

    Parameters
    ----------
    target_uri : str
        target url
    organismal_species : str | OrganismalSpeciesValidator
        A species name: e.g., Homo sapiens

    Returns
    -------
        None
    """
    string_aliases_url = get_string_species_url(
        organismal_species, asset="aliases", version=STRING_VERSION
    )
    logger.info(
        "Start downloading string aliases %s to %s", string_aliases_url, target_uri
    )
    utils.download_wget(string_aliases_url, target_uri)

    return None


def convert_string_to_sbml_dfs(
    string_uri: str,
    string_aliases_uri: str,
    organismal_species: Union[str, OrganismalSpeciesValidator],
) -> SBML_dfs:
    """Ingests string to sbml dfs

    Parameters
    ----------
    string_uri : str
        URI for the string interactions file
    string_aliases_uri : str
        URI for the string aliases file
    organismal_species : str | OrganismalSpeciesValidator
        A species name: e.g., Homo sapiens

    Returns
    -------
    SBML_dfs
        A STRING pathway representation as an SBML_dfs object
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)
    organismal_species.assert_supported(STRING_TAX_IDS.keys())

    # Read string raw data
    string_edgelist = _read_string(string_uri)
    string_aliases = _read_string_aliases(string_aliases_uri)

    # Start building new sbml dfs

    # remove one edge since reciprocal edges are present; i.e., A-B and B-A
    # and attributes (e.g., combined_score are the same across both reciprocal
    # interactions
    uq_string_edgelist = napistu_edgelist.remove_reciprocal_interactions(
        string_edgelist, extra_defining_vars=["combined_score"]
    )

    # define model-level metadata
    model_source = Source.single_entry(
        model=DATA_SOURCES.STRING,
        pathway_id=DATA_SOURCES.STRING,
        data_source=DATA_SOURCES.STRING,
        organismal_species=organismal_species.latin_name,
        name=DATA_SOURCE_DESCRIPTIONS[DATA_SOURCES.STRING],
        date=datetime.date.today().strftime("%Y%m%d"),
    )

    # define identifier mapping from aliases to use:
    alias_to_identifier = {
        "Ensembl_gene": (ONTOLOGIES.ENSEMBL_GENE, BQB.IS_ENCODED_BY),
        "Ensembl_transcript": (ONTOLOGIES.ENSEMBL_TRANSCRIPT, BQB.IS_ENCODED_BY),
        "Ensembl_translation": (ONTOLOGIES.ENSEMBL_PROTEIN, BQB.IS),
        "Ensembl_UniProt_AC": (ONTOLOGIES.UNIPROT, BQB.IS),
    }

    # filter aliases to only keep required ones
    string_aliases_fil = string_aliases.query(
        "source in @alias_to_identifier.keys()"
    ).set_index(STRING_PROTEIN_ID)

    # to save on memory
    del string_aliases

    # define species
    species_df = _build_species_df(
        uq_string_edgelist, string_aliases_fil, alias_to_identifier
    )

    # Define compartments
    # Currently we are mapping everything to the `CELLULAR_COMPONENT`
    # which is a catch-all go: for unknown localisation
    compartments_df = sbml_dfs_utils.stub_compartments()

    # define interactions
    interaction_edgelist = _build_interactor_edgelist(uq_string_edgelist)

    interaction_edgelist_defaults = {
        INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: SBOTERM_NAMES.INTERACTOR,
        INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: SBOTERM_NAMES.INTERACTOR,
        SBML_DFS.R_ISREVERSIBLE: True,
    }

    # build the final object
    sbml_dfs = SBML_dfs.from_edgelist(
        interaction_edgelist=interaction_edgelist,
        species_df=species_df,
        compartments_df=compartments_df,
        model_source=model_source,
        interaction_edgelist_defaults=interaction_edgelist_defaults,
        keep_reactions_data=DATA_SOURCES.STRING,
    )
    return sbml_dfs


def _read_string(string_uri: str) -> pd.DataFrame:
    """Reads string from uri

    Args:
        string_uri (str): string uri

    Returns:
        pd.DataFrame: string edgelist
    """
    base_path, file_name = utils.get_source_base_and_path(string_uri)
    # TODO: test on gz versus txt
    with open_fs(base_path) as base_fs:
        with base_fs.open(file_name, "rb") as f:
            string_edges = pd.read_csv(f, sep=" ")
    return string_edges


def _read_string_aliases(string_aliases_uri: str) -> pd.DataFrame:
    """Reads string from uri

    Args:
        string_aliases_uri (str): string aliases uri

    Returns:
        pd.DataFrame: string aliases
    """
    base_path, file_name = utils.get_source_base_and_path(string_aliases_uri)
    # TODO: test on gz versus txt
    with open_fs(base_path) as base_fs:
        with base_fs.open(file_name, "rb") as f:
            string_aliases = (
                pd.read_csv(f, sep="\t")
                # Rename column with #
                .rename(columns={STRING_PROTEIN_ID_RAW: STRING_PROTEIN_ID})
            )
    return string_aliases


def _get_identifiers(
    row: pd.DataFrame,
    alias_to_identifier: dict[str, tuple[str, str]],
    dat_alias: pd.DataFrame,
) -> identifiers.Identifiers:
    """Helper function to get identifiers from a row of the string alias file

    Args:
        row (pd.DataFrame): grouped dataframe
        alias_to_identifier (dict[str, tuple[str, str]]):
            map from an alias source to an ontology and a qualifier
        dat_alias (pd.DataFrame): Helper dataframe with index=string_protein_id
            and columns=source (the source name), alias (the identifier)

    Returns:
        identifiers.Identifiers: An Identifiers object containing all identifiers
    """
    if row.shape[0] == 0:
        return identifiers.Identifiers([])
    d = dat_alias.loc[row.s_name]
    ids = []
    for source_name, (ontology, qualifier) in alias_to_identifier.items():
        for identifier in d.query(f"source == '{source_name}'")["alias"]:
            # Here we creating an uri
            uri = identifiers.create_uri_url(ontology=ontology, identifier=identifier)
            # This is exactly the output format from: identifiers.format_uri
            # We are doing it manually here to avoid the overhead of parsing
            # the uri again
            id_dict = {
                IDENTIFIERS.ONTOLOGY: ontology,
                IDENTIFIERS.IDENTIFIER: identifier,
                IDENTIFIERS.BQB: qualifier,
                IDENTIFIERS.URL: uri,
            }
            ids.append(id_dict)
    identifier = identifiers.Identifiers(ids)
    return identifier


def _build_species_df(
    edgelist: pd.DataFrame,
    aliases: pd.DataFrame,
    alias_to_identifier: dict,
    source_col: str = STRING_SOURCE,
    target_col: str = STRING_TARGET,
) -> pd.DataFrame:
    """Builds the species dataframe from the edgelist and aliases

    Args:
        edgelist (pd.DataFrame): edgelist
        aliases (pd.DataFrame): aliases
        alias_to_identifier (dict[str, tuple[str, str]]):
            map from an alias source to an ontology and a qualifier

    Returns:
        pd.DataFrame: species dataframe
    """
    species_df = (
        pd.Series(
            list(set(edgelist[source_col]).union(edgelist[target_col])),
            name=SBML_DFS.S_NAME,
        )
        .to_frame()
        .set_index(SBML_DFS.S_NAME, drop=False)
        .apply(
            _get_identifiers,
            alias_to_identifier=alias_to_identifier,
            dat_alias=aliases,
            axis=1,
        )
        .rename(SBML_DFS.S_IDENTIFIERS)
        .reset_index()
    )
    return species_df


def _build_interactor_edgelist(
    edgelist: pd.DataFrame,
    upstream_col_name: str = STRING_SOURCE,
    downstream_col_name: str = STRING_TARGET,
    add_reverse_interactions: bool = False,
) -> pd.DataFrame:
    """Format STRING interactions as reactions."""

    dat = edgelist.rename(
        columns={
            upstream_col_name: INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            downstream_col_name: INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
        }
    ).assign(
        **{
            SBML_DFS.R_IDENTIFIERS: lambda x: identifiers.Identifiers([]),
        }
    )
    if add_reverse_interactions:
        dat = (
            dat
            # Add the reverse interactions
            .pipe(
                lambda d: pd.concat(
                    [
                        d,
                        d.rename(
                            columns={
                                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                            }
                        ),
                    ]
                )
            )
        )

    interaction_edgelist = dat
    interaction_edgelist[SBML_DFS.R_NAME] = _build_string_reaction_name(
        dat[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM],
        dat[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM],
    )

    return interaction_edgelist


def _build_string_reaction_name(from_col: pd.Series, to_col: pd.Series) -> pd.Series:
    """Helper to build the reaction name for string reactions

    Args:
        from_col (pd.Series): from species
        to_col (pd.Series): to species

    Returns:
        pd.Series: new name column
    """
    return from_col + " - " + to_col
