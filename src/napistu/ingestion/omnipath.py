import datetime
import itertools
import logging
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from napistu import sbml_dfs_core, sbml_dfs_utils
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    ONTOLOGIES_LIST,  # noqa: F401
    SBML_DFS,
    SBOTERM_NAMES,
)
from napistu.identifiers import Identifiers
from napistu.ingestion.constants import (
    DATA_SOURCE_DESCRIPTIONS,
    DATA_SOURCES,
    INTERACTION_EDGELIST_DEFS,
    OMNIPATH_ANNOTATIONS,
    OMNIPATH_COMPLEXES,
    OMNIPATH_INTERACTIONS,
    OMNIPATH_ONTOLOGY_ALIASES,
    REACTOME_FI,
    VALID_OMNIPATH_SPECIES,
)
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator
from napistu.ontologies import renaming
from napistu.ontologies.constants import GENODEXITO_DEFS, MIRBASE_TABLES, PUBCHEM_DEFS
from napistu.ontologies.genodexito import Genodexito
from napistu.ontologies.mirbase import load_mirbase_xrefs
from napistu.ontologies.pubchem import map_pubchem_ids
from napistu.source import Source

# import lazy-loaded omnipath and interactions
from napistu.utils.optional import (
    import_omnipath,
    import_omnipath_interactions,
    require_omnipath,
)

logger = logging.getLogger(__name__)


@require_omnipath
def format_omnipath_as_sbml_dfs(
    organismal_species: Union[str, OrganismalSpeciesValidator],
    preferred_method: str,
    allow_fallback: bool,
    **kwargs: Any,
) -> sbml_dfs_core.SBML_dfs:
    """
    Format OmniPath interaction data as SBML_dfs object.

    This function processes OmniPath interaction data and converts it into a structured
    SBML_dfs format suitable for network analysis and modeling. It handles various
    types of molecular interactions including proteins, small molecules, miRNAs, and complexes.

    Parameters
    ----------
    organismal_species : str | OrganismalSpeciesValidator
        The species name (e.g., "human", "mouse", "rat") for which to retrieve interactions.
    preferred_method : str
        Preferred method for identifier mapping (e.g., "bioconductor", "ensembl").
    allow_fallback : bool
        Whether to allow fallback to alternative mapping methods if preferred method fails.
    **kwargs : Any
        Additional keyword arguments passed to `get_interactions()`.

    Returns
    -------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        SBML_dfs object containing the formatted interaction data with species and reactions.

    Raises
    ------
    ValueError
        If organismal_species is not supported by OmniPath.
        If duplicated s_names are found after processing.
    ConnectionError
        If unable to connect to OmniPath or external databases.

    Notes
    -----
    This function performs the following steps:
    1. Retrieves interactions from OmniPath
    2. Maps interactor IDs to systematic identifiers using multiple databases
        - PubChem
        - UniProt
        - miRBase
        - Complexes
    3. Aggregates molecular species across all sources
    4. Creates complex formation reactions where applicable
    5. Formats all interactions into a standardized edgelist
    6. Creates an SBML_dfs object with proper structure

    Examples
    --------
    >>> # Format human interactions using bioconductor mapping
    >>> sbml_dfs = format_omnipath_as_sbml_dfs(
    ...     organismal_species="human",
    ...     preferred_method="bioconductor",
    ...     allow_fallback=True
    ... )
    >>> print(f"Species: {len(sbml_dfs.species)}")
    >>> print(f"Reactions: {len(sbml_dfs.reactions)}")
    """

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)
    organismal_species.assert_supported(VALID_OMNIPATH_SPECIES)

    # format model-level metadata
    model_source = Source.single_entry(
        model=DATA_SOURCES.OMNIPATH,
        pathway_id=DATA_SOURCES.OMNIPATH,
        data_source=DATA_SOURCES.OMNIPATH,
        organismal_species=organismal_species.latin_name,
        name=DATA_SOURCE_DESCRIPTIONS[DATA_SOURCES.OMNIPATH],
        date=datetime.date.today().strftime("%Y%m%d"),
    )

    interactions = get_interactions(organismal_species=organismal_species, **kwargs)

    interactor_ids = set(interactions[OMNIPATH_INTERACTIONS.SOURCE]) | set(
        interactions[OMNIPATH_INTERACTIONS.TARGET]
    )
    interactor_int_ids = [x for x in interactor_ids if x.isdigit()]
    interactor_string_ids = [x for x in interactor_ids if not x.isdigit()]

    # connect interactors to systematic identifeirs and create readable names
    if len(interactor_int_ids) > 0:
        pubchem_species = _prepare_integer_based_ids(interactor_int_ids)
    else:
        logger.info("No integger-based IDs from pubchem found")
        pubchem_species = pd.DataFrame()

    uniprot_species = _prepare_omnipath_ids_uniprot(
        interactor_string_ids,
        organismal_species=organismal_species,
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    )

    mirbase_species = _prepare_omnipath_ids_mirbase(interactor_string_ids)

    complex_species, complex_formation_edgelist = _prepare_omnipath_ids_complexes(
        interactor_string_ids
    )

    # aggregate all the different sources of interactors
    interactor_df = pd.concat(
        [
            pubchem_species.assign(species_type="small_molecule"),
            uniprot_species.assign(species_type="protein"),
            mirbase_species.assign(species_type="miRNA"),
            complex_species.assign(species_type="complex"),
        ]
    ).reset_index(drop=True)

    unmapped_ids = interactor_ids - set(
        interactor_df[OMNIPATH_INTERACTIONS.INTERACTOR_ID]
    )
    sample_unmapped_ids = (
        pd.Series(list(unmapped_ids)).sample(n=min(10, len(unmapped_ids))).tolist()
    )
    logger.info(
        f"{len(unmapped_ids)} interactor ids could not be matched to an ontology and will be dropped. Example unmapped ids: {sample_unmapped_ids}"
    )

    # ensure that s_names are unique
    nondegenerate_species_df = _patch_degenerate_s_names(interactor_df)

    if len(nondegenerate_species_df[SBML_DFS.S_NAME].unique()) != len(
        nondegenerate_species_df
    ):
        raise ValueError(
            "Duplicated s_names found following `_patch_degenerate_s_names`"
        )

    # format all distinct interactions into a single edgelist
    interaction_edgelist_list = [
        _format_edgelist_interactions(
            interactions=interactions, nondegenerate_species_df=nondegenerate_species_df
        )
    ]

    if complex_formation_edgelist.shape[0] > 0:
        interaction_edgelist_list.append(
            _format_complex_interactions(
                complex_formation_edgelist=complex_formation_edgelist,
                nondegenerate_species_df=nondegenerate_species_df,
            )
        )

    interaction_edgelist = pd.concat(interaction_edgelist_list).reset_index(drop=True)

    interaction_edgelist[SBML_DFS.R_NAME] = interaction_edgelist.apply(
        lambda row: sbml_dfs_utils._name_interaction(
            row[INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM],
            row[INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM],
            row[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM],
        ),
        axis=1,
    )

    # create an SBML_dfs object
    sbml_dfs = sbml_dfs_core.SBML_dfs.from_edgelist(
        interaction_edgelist,
        nondegenerate_species_df.drop(columns=[OMNIPATH_INTERACTIONS.INTERACTOR_ID]),
        compartments_df=sbml_dfs_utils.stub_compartments(),
        model_source=model_source,
        keep_reactions_data=DATA_SOURCES.OMNIPATH,
        keep_species_data=DATA_SOURCES.OMNIPATH,
        force_edgelist_consistency=True,
    )

    return sbml_dfs


@require_omnipath
def get_interactions(
    dataset: Union[str, object] = "all",
    organismal_species: Union[str, OrganismalSpeciesValidator] = "human",
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve interaction data from Omnipath with corrected evidence processing.

    This function wraps the underlying Omnipath interaction classes and applies
    strict evidence filtering with fixes for known consensus logic bugs.

    Parameters
    ----------
    dataset : str or interaction class, default "all"
        Which interaction dataset to retrieve. Options:
        - "all": AllInteractions (all datasets)
        - "omnipath": OmniPath (literature-supported only)
        - "dorothea": Dorothea (TF-target from DoRothEA)
        - "collectri": CollecTRI (TF-target from CollecTRI)
        - "tf_target": TFtarget (TF-target interactions)
        - "transcriptional": Transcriptional (all TF-target)
        - "post_translational": PostTranslational (protein-protein)
        - "pathway_extra": PathwayExtra (activity flow, no literature)
        - "kinase_extra": KinaseExtra (enzyme-substrate, no literature)
        - "ligrec_extra": LigRecExtra (ligand-receptor, no literature)
        - "tf_mirna": TFmiRNA (TF-miRNA interactions)
        - "mirna": miRNA (miRNA-target interactions)
        - "lncrna_mrna": lncRNAmRNA (lncRNA-mRNA interactions)
        Or pass an interaction class directly.
    **kwargs
        Additional parameters passed to the underlying interaction class.

    Returns
    -------
    pd.DataFrame
        Interaction data with columns including:
        - source, target: Interacting proteins
        - is_directed, is_stimulation, is_inhibition: Evidence presence flags
        - consensus_direction, consensus_stimulation, consensus_inhibition: Consensus flags
        - curation_effort: Evidence quality score
        - sources, references: Supporting data

    Notes
    -----
    **Evidence Processing:**

    This function uses `strict_evidences=True`, which recomputes all evidence-derived
    attributes from the raw evidence data rather than using server pre-computed values.
    This ensures transparency about which evidence supports each interaction property.

    **How Evidence Flags Are Calculated:**

    The `is_*` flags indicate presence of evidence of each type:
    ```python
    is_directed = bool(any evidence in "directed" category)
    is_stimulation = bool(any evidence in "positive" category)
    is_inhibition = bool(any evidence in "negative" category)
    ```
    These are simple "any evidence exists" boolean flags.

    **How Consensus Flags Are Calculated:**

    Consensus flags compare weighted evidence (curation effort) between categories:
    ```python
    curation_effort = sum(len(evidence.references) + 1 for evidence in category)
    consensus_stimulation = curation_effort_positive >= curation_effort_negative
    consensus_inhibition = curation_effort_positive <= curation_effort_negative
    consensus_direction = curation_effort_directed >= curation_effort_undirected
    ```

    **Important:** When evidence is tied (equal curation effort), both consensus flags
    can be True. When no evidence exists, both would incorrectly be True due to
    0 >= 0 and 0 <= 0, but this function fixes that edge case.

    **Consensus Logic Bug Fix:**

    The original Omnipath logic has a bug where interactions with no stimulation or
    inhibition evidence get consensus_stimulation=True and consensus_inhibition=True
    because both 0 >= 0 and 0 <= 0 evaluate to True. This function fixes such cases
    by setting both consensus flags to False when no evidence exists.

    **Evidence Categories Explained:**

    - **positive**: Evidence supporting stimulation/activation
    - **negative**: Evidence supporting inhibition/repression
    - **directed**: Evidence that the interaction has a specific direction
    - **undirected**: Evidence that interaction exists but direction is unclear

    **Interpreting Results:**

    Common patterns and their meanings:
    - `is_stimulation=True, consensus_stimulation=True`: Strong positive evidence
    - `is_stimulation=True, is_inhibition=True`: Conflicting evidence exists
    - `consensus_stimulation=True, consensus_inhibition=True`: Tied evidence
    - `is_stimulation=False, is_inhibition=False`: No directional evidence

    **Why Use Strict Evidence Mode:**

    - Transparency: Know exactly which evidence supports each attribute
    - Filtering: Can restrict to specific datasets/resources in query
    - Consistency: All attributes computed from same evidence base
    - Reproducibility: Results don't depend on server-side integration

    Examples
    --------
    >>> # Get all interactions with corrected evidence processing
    >>> df = get_interactions("all")
    >>>
    >>> # Get only DoRothEA TF-target interactions
    >>> tf_interactions = get_interactions("dorothea")
    >>>
    >>> # Filter to specific resources
    >>> filtered = get_interactions("all", resources=["IntAct", "BioGRID"])
    >>>
    >>> # Check for conflicting evidence
    >>> conflicted = df[(df.is_stimulation) & (df.is_inhibition)]
    >>> print(f"Found {len(conflicted)} interactions with conflicting evidence")
    >>>
    >>> # Look at evidence quality
    >>> high_quality = df[df.curation_effort >= 10]
    """

    if "organism" in kwargs:
        raise ValueError(
            "Please don't specify 'organism' directly. Use the 'organismal_species' argument instead."
        )

    if isinstance(organismal_species, str):
        organismal_species = OrganismalSpeciesValidator(organismal_species)
    if not isinstance(organismal_species, OrganismalSpeciesValidator):
        raise ValueError(f"Invalid organismal_species: {organismal_species}")
    organismal_species.assert_supported(VALID_OMNIPATH_SPECIES)

    # Get the interaction class
    OMNIPATH_FXN_MAP = _get_omnipath_fxn_map()

    if isinstance(dataset, str):
        if dataset not in OMNIPATH_FXN_MAP:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Options: {list(OMNIPATH_FXN_MAP.keys())}"
            )
        interaction_class = OMNIPATH_FXN_MAP[dataset]
    else:
        interaction_class = dataset

    # Get the data with strict evidence processing
    df = interaction_class.get(
        organism=organismal_species.common_name, strict_evidences=True, **kwargs
    )

    # Fix consensus logic bug
    df = _fix_consensus_logic(df)

    return df


@require_omnipath
def _get_omnipath_fxn_map():
    """Get a map of dataset names to interaction classes"""
    interactions = import_omnipath_interactions()

    # Map dataset names to interaction classes
    return {
        "all": interactions.AllInteractions,
        "omnipath": interactions.OmniPath,
        "dorothea": interactions.Dorothea,
        "collectri": interactions.CollecTRI,
        "tf_target": interactions.TFtarget,
        "transcriptional": interactions.Transcriptional,
        "post_translational": interactions.PostTranslational,
        "pathway_extra": interactions.PathwayExtra,
        "kinase_extra": interactions.KinaseExtra,
        "ligrec_extra": interactions.LigRecExtra,
        "tf_mirna": interactions.TFmiRNA,
        "mirna": interactions.miRNA,
        "lncrna_mrna": interactions.lncRNAmRNA,
    }


def _fix_consensus_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the consensus logic bug for interactions with no evidence.

    When strict_evidences=True, interactions with no stimulation or inhibition
    evidence incorrectly get consensus_stimulation=True and consensus_inhibition=True
    due to the 0>=0 and 0<=0 comparisons both being True.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing OmniPath interaction data with consensus flags.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected consensus logic flags.

    Notes
    -----
    This function identifies interactions where:
    - No stimulation evidence exists (is_stimulation=False)
    - No inhibition evidence exists (is_inhibition=False)
    - Both consensus flags are True (consensus_stimulation=True, consensus_inhibition=True)

    For such interactions, both consensus flags are set to False.
    """
    # Identify problematic records: no evidence but consensus for both
    no_evidence = (~df.is_stimulation) & (~df.is_inhibition)
    both_consensus = df.consensus_stimulation & df.consensus_inhibition
    problematic = no_evidence & both_consensus

    if problematic.sum() > 0:
        # Fix the problematic records
        df.loc[problematic, OMNIPATH_INTERACTIONS.CONSENSUS_STIMULATION] = False
        df.loc[problematic, OMNIPATH_INTERACTIONS.CONSENSUS_INHIBITION] = False

    return df


def _prepare_integer_based_ids(interactor_int_ids: List[str]) -> pd.DataFrame:
    """
    Prepare PubChem identifiers for integer-based interactor IDs.

    Parameters
    ----------
    interactor_int_ids : List[str]
        List of integer-based interactor IDs to map to PubChem.

    Returns
    -------
    pd.DataFrame
        DataFrame containing PubChem species data with columns:
        - interactor_id: Original interactor ID
        - s_name: PubChem compound name
        - s_Identifiers: Identifiers object with PubChem and SMILES identifiers
    """

    logger.info(f"Searching PubChem for {len(interactor_int_ids)} integer-based IDs")

    pubchem_ids = map_pubchem_ids(interactor_int_ids, verbose=False)

    pubchem_species = (
        pd.DataFrame(pubchem_ids)
        .T.rename_axis(OMNIPATH_INTERACTIONS.INTERACTOR_ID)
        .reset_index()
        .rename(columns={PUBCHEM_DEFS.NAME: SBML_DFS.S_NAME})
    )

    pubchem_species[SBML_DFS.S_IDENTIFIERS] = [
        Identifiers(
            [
                {
                    IDENTIFIERS.ONTOLOGY: ONTOLOGIES.PUBCHEM,
                    IDENTIFIERS.IDENTIFIER: row[OMNIPATH_INTERACTIONS.INTERACTOR_ID],
                    IDENTIFIERS.BQB: BQB.IS,
                },
                {
                    IDENTIFIERS.ONTOLOGY: ONTOLOGIES.SMILES,
                    IDENTIFIERS.IDENTIFIER: row[PUBCHEM_DEFS.SMILES],
                    IDENTIFIERS.BQB: BQB.IS,
                },
            ]
        )
        for _, row in pubchem_species.iterrows()
    ]

    return pubchem_species.drop(columns=PUBCHEM_DEFS.SMILES)


def _prepare_omnipath_ids_uniprot(
    interactor_string_ids: List[str],
    organismal_species: Union[str, OrganismalSpeciesValidator],
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """
    Prepare UniProt identifiers for string-based interactor IDs.

    Parameters
    ----------
    interactor_string_ids : List[str]
        List of string-based interactor IDs to cross-reference with UniProt.
    organismal_species : Union[str, OrganismalSpeciesValidator]
        The species for which to retrieve UniProt annotations.
    preferred_method : str, optional
        Preferred method for identifier mapping, by default GENODEXITO_DEFS.BIOCONDUCTOR.
    allow_fallback : bool, optional
        Whether to allow fallback to alternative mapping methods, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing UniProt species data with columns:
        - interactor_id: Original interactor ID
        - s_name: Gene name
        - s_Identifiers: Identifiers object with UniProt identifiers

    Raises
    ------
    ValueError
        If organismal_species is invalid or not supported.
    """

    if isinstance(organismal_species, str):
        organismal_species = OrganismalSpeciesValidator(organismal_species)
    if not isinstance(organismal_species, OrganismalSpeciesValidator):
        raise ValueError(f"Invalid organismal_species: {organismal_species}")

    logger.info(
        f"Searching Genodexito for {organismal_species.common_name} ({organismal_species.latin_name}) Uniprot annotations"
    )

    genodexito = Genodexito(
        organismal_species=organismal_species.latin_name,
        preferred_method=preferred_method,
        allow_fallback=allow_fallback,
    )

    genodexito.create_mapping_tables({ONTOLOGIES.UNIPROT, ONTOLOGIES.GENE_NAME})
    genodexito.merge_mappings()

    # remove entries missing Uniprot IDs
    uniprot_species = (
        genodexito.merged_mappings.dropna(subset=[ONTOLOGIES.UNIPROT])
        .drop(columns=[ONTOLOGIES.NCBI_ENTREZ_GENE])
        .reset_index(drop=True)
        .rename(
            columns={
                ONTOLOGIES.GENE_NAME: SBML_DFS.S_NAME,
                ONTOLOGIES.UNIPROT: OMNIPATH_INTERACTIONS.INTERACTOR_ID,
            }
        )
        .query(f"{OMNIPATH_INTERACTIONS.INTERACTOR_ID} in @interactor_string_ids")
        # just retain one name
        .groupby(OMNIPATH_INTERACTIONS.INTERACTOR_ID)
        .head(1)
    )

    logger.info(f"Found {len(uniprot_species)} Uniprot species")

    uniprot_species[SBML_DFS.S_IDENTIFIERS] = [
        Identifiers(
            [
                {
                    IDENTIFIERS.ONTOLOGY: ONTOLOGIES.UNIPROT,
                    IDENTIFIERS.IDENTIFIER: x,
                    IDENTIFIERS.BQB: BQB.IS,
                }
            ]
        )
        for x in uniprot_species[OMNIPATH_INTERACTIONS.INTERACTOR_ID]
    ]

    return uniprot_species


def _prepare_omnipath_ids_mirbase(
    interactor_string_ids: List[str],
) -> pd.DataFrame:
    """
    Prepare miRBase identifiers for string-based interactor IDs.

    Parameters
    ----------
    interactor_string_ids : List[str]
        List of string-based interactor IDs to map to miRBase.

    Returns
    -------
    pd.DataFrame
        DataFrame containing miRBase species data with columns:
        - interactor_id: Original interactor ID
        - s_name: miRNA name
        - s_Identifiers: Identifiers object with miRBase identifiers
    """

    logger.info("Loading miRBase for cross references")

    mirbase_xrefs = load_mirbase_xrefs()

    mirbase_species = (
        mirbase_xrefs.query(f"{MIRBASE_TABLES.PRIMARY_ID} in @interactor_string_ids")[
            [MIRBASE_TABLES.PRIMARY_ID, MIRBASE_TABLES.SECONDARY_ID]
        ]
        .drop_duplicates()
        .rename(
            columns={
                MIRBASE_TABLES.PRIMARY_ID: OMNIPATH_INTERACTIONS.INTERACTOR_ID,
                MIRBASE_TABLES.SECONDARY_ID: SBML_DFS.S_NAME,
            }
        )
    )

    logger.info(f"Found {len(mirbase_species)} miRBase species")

    mirbase_species[SBML_DFS.S_IDENTIFIERS] = [
        Identifiers(
            [
                {
                    IDENTIFIERS.ONTOLOGY: ONTOLOGIES.MIRBASE,
                    IDENTIFIERS.IDENTIFIER: x,
                    IDENTIFIERS.BQB: BQB.IS,
                }
            ]
        )
        for x in mirbase_species[OMNIPATH_INTERACTIONS.INTERACTOR_ID]
    ]

    return mirbase_species


@require_omnipath
def _prepare_omnipath_ids_complexes(
    interactor_string_ids: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare complex identifiers and formation reactions for string-based interactor IDs.

    Parameters
    ----------
    interactor_string_ids : List[str]
        List of string-based interactor IDs to check for complexes.

    Returns
    -------
    complex_species : pd.DataFrame
        DataFrame containing complex species data with columns:
        - interactor_id: Complex identifier
        - s_name: Complex name
        - s_Identifiers: Identifiers object with complex identifiers
    complex_formation_edgelist : pd.DataFrame
        DataFrame containing complex formation reactions with columns:
        - source: Component interactor ID
        - target: Complex interactor ID
        - upstream_stoichiometry: Stoichiometry of component
        - downstream_stoichiometry: Stoichiometry of complex (typically 1)
    """

    omnipath = import_omnipath()

    complexes = omnipath.requests.Complexes().get()
    complexes[OMNIPATH_INTERACTIONS.INTERACTOR_ID] = [
        OMNIPATH_COMPLEXES.COMPLEX_FSTRING.format(x=x)
        for x in complexes[OMNIPATH_COMPLEXES.COMPONENTS]
    ]
    observed_complexes = complexes.query(
        f"{OMNIPATH_INTERACTIONS.INTERACTOR_ID} in @interactor_string_ids"
    )

    if observed_complexes.shape[0] == 0:
        logger.info("No complexes present")
        return pd.DataFrame(), pd.DataFrame()

    # add a default name
    missing_names_mask = [
        x is None for x in observed_complexes[OMNIPATH_COMPLEXES.NAME]
    ]
    observed_complexes.loc[missing_names_mask, OMNIPATH_COMPLEXES.NAME] = (
        observed_complexes.loc[missing_names_mask, OMNIPATH_INTERACTIONS.INTERACTOR_ID]
    )

    # create a species table
    complex_identifiers = _extract_omnipath_identifiers(
        observed_complexes[OMNIPATH_COMPLEXES.IDENTIFIERS],
        OMNIPATH_COMPLEXES.IDENTIFIERS,
    )

    complex_species = (
        observed_complexes.merge(
            complex_identifiers, on=OMNIPATH_COMPLEXES.IDENTIFIERS, how="left"
        )
        .fillna({SBML_DFS.S_IDENTIFIERS: Identifiers([])})
        .rename(columns={OMNIPATH_COMPLEXES.NAME: SBML_DFS.S_NAME})[
            [
                OMNIPATH_INTERACTIONS.INTERACTOR_ID,
                SBML_DFS.S_NAME,
                SBML_DFS.S_IDENTIFIERS,
            ]
        ]
    )

    # complex formation reactions
    complex_formation_edgelist = list()
    for _, row in observed_complexes.iterrows():
        members = row[OMNIPATH_COMPLEXES.COMPONENTS].split("_")
        upstream_stoi = [
            0 if float(x) == 0 else -1 * float(x)
            for x in row[OMNIPATH_COMPLEXES.STOICHIOMETRY].split(":")
        ]

        if all([x == 0 for x in upstream_stoi]):
            downstream_stoi = 0
        else:
            downstream_stoi = 1

        complex_formation_edgelist.append(
            pd.DataFrame(
                {
                    OMNIPATH_INTERACTIONS.SOURCE: members,
                    OMNIPATH_INTERACTIONS.TARGET: [
                        row[OMNIPATH_INTERACTIONS.INTERACTOR_ID]
                    ]
                    * len(members),
                    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM: upstream_stoi,
                    INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM: [
                        downstream_stoi
                    ]
                    * len(members),
                }
            )
        )

    complex_formation_edgelist = pd.concat(complex_formation_edgelist)

    return complex_species, complex_formation_edgelist


def _extract_omnipath_identifiers(
    identifier_series: pd.Series,
    entity_name: str,
    ontology_aliases: Dict[str, List[str]] = OMNIPATH_ONTOLOGY_ALIASES,
) -> pd.DataFrame:
    """
    Extract and standardize OmniPath identifiers from a series of annotation strings.

    Parameters
    ----------
    identifier_series : pd.Series
        Series containing OmniPath annotation strings.
    entity_name : str
        Name of the entity column in the output DataFrame.
    ontology_aliases : Dict[str, List[str]], optional
        Mapping from OmniPath ontology names to Napistu ontology names, by default OMNIPATH_ONTOLOGY_ALIASES.

    Returns
    -------
    pd.DataFrame
        DataFrame with entity_name as index and s_identifiers as column containing
        Identifiers objects for each entity.
    """

    # map from omnipath controlled vocabulary to Napistu controlled vocabulary
    aliases = renaming.OntologySet(ontologies=ontology_aliases).ontologies
    alias_mapping = renaming._create_alias_mapping(aliases)

    # pull out identifiers, rename them based on the Napistu CV and remove unrecognized ontologies
    complex_identifiers = pd.concat(
        [
            _parse_omnipath_named_annotation(x)
            for x in identifier_series
            if x is not None
        ]
    )
    complex_identifiers[IDENTIFIERS.ONTOLOGY] = complex_identifiers[
        OMNIPATH_ANNOTATIONS.NAME
    ].replace(alias_mapping)
    valid_complex_identifiers = complex_identifiers.query(
        f"{IDENTIFIERS.ONTOLOGY} in @ONTOLOGIES_LIST"
    )

    # create Identifiers objects
    identifiers_map = {}
    for annotation_str, group in valid_complex_identifiers.groupby(
        OMNIPATH_ANNOTATIONS.ANNOTATION_STR
    ):
        identifier_dicts = []
        for _, row in group.iterrows():
            identifier_dict = {
                IDENTIFIERS.ONTOLOGY: row[IDENTIFIERS.ONTOLOGY],
                IDENTIFIERS.IDENTIFIER: row[OMNIPATH_ANNOTATIONS.ANNOTATION],
                IDENTIFIERS.BQB: BQB.IS,
            }
            identifier_dicts.append(identifier_dict)

        identifiers_map[annotation_str] = Identifiers(identifier_dicts)

    return (
        pd.DataFrame(identifiers_map, index=[SBML_DFS.S_IDENTIFIERS])
        .T.rename_axis(entity_name)
        .reset_index()
    )


def _load_omnipath_attribute_mapper() -> pd.DataFrame:
    """
    Create a mapping table for OmniPath interaction attributes to SBO terms.

    Based on OmniPath interaction's consensus reversibility, stimulation, and inhibition,
    assign them to SBO terms and expand based on reversibility to forward and reverse directions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all valid combinations of OmniPath attributes mapped to
        SBO terms, with columns for consensus flags, SBO terms, and reversibility.
    """

    combinations = list(itertools.product([True, False], repeat=4))
    df = pd.DataFrame(
        combinations,
        columns=[
            OMNIPATH_INTERACTIONS.CONSENSUS_DIRECTION,
            OMNIPATH_INTERACTIONS.CONSENSUS_STIMULATION,
            OMNIPATH_INTERACTIONS.CONSENSUS_INHIBITION,
            REACTOME_FI.DIRECTION,
        ],
    )
    df[REACTOME_FI.DIRECTION] = df[REACTOME_FI.DIRECTION].replace(
        {True: REACTOME_FI.FORWARD, False: REACTOME_FI.REVERSE}
    )

    df[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM] = None
    df[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM] = None
    df[SBML_DFS.R_ISREVERSIBLE] = None
    df["is_valid"] = True

    for i, row in df.iterrows():

        direction = row[OMNIPATH_INTERACTIONS.CONSENSUS_DIRECTION]
        stimulation = row[OMNIPATH_INTERACTIONS.CONSENSUS_STIMULATION]
        inhibition = row[OMNIPATH_INTERACTIONS.CONSENSUS_INHIBITION]
        rxn_direction = row[REACTOME_FI.DIRECTION]

        if (not direction) and (rxn_direction == REACTOME_FI.REVERSE):
            df.loc[i, "is_valid"] = False
            continue

        # interactors
        if (not stimulation) and (not inhibition) and (not direction):
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM] = (
                SBOTERM_NAMES.INTERACTOR
            )
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM] = (
                SBOTERM_NAMES.INTERACTOR
            )
            df.loc[i, SBML_DFS.R_ISREVERSIBLE] = True
            continue

        if (not stimulation) and (not inhibition):
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM] = (
                SBOTERM_NAMES.MODIFIER
            )
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM] = (
                SBOTERM_NAMES.MODIFIED
            )

        if inhibition and (not stimulation):
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM] = (
                SBOTERM_NAMES.INHIBITOR
            )
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM] = (
                SBOTERM_NAMES.MODIFIED
            )

        if stimulation and (not inhibition):
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM] = (
                SBOTERM_NAMES.STIMULATOR
            )
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM] = (
                SBOTERM_NAMES.MODIFIED
            )

        if stimulation and inhibition:
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM] = (
                SBOTERM_NAMES.MODIFIER
            )
            df.loc[i, INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM] = (
                SBOTERM_NAMES.MODIFIED
            )

        df.loc[i, SBML_DFS.R_ISREVERSIBLE] = False

    df = df.query("is_valid")

    # flip attributes for reverse direction
    return pd.concat(
        [
            df.query(f"{REACTOME_FI.DIRECTION} == '{REACTOME_FI.FORWARD}'"),
            df.query(f"{REACTOME_FI.DIRECTION} == '{REACTOME_FI.REVERSE}'").rename(
                {
                    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM,
                    INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM,
                },
                axis=1,
            ),
        ]
    )


def _format_reaction_references(reference_series: pd.Series) -> pd.DataFrame:
    """
    Format reaction references as Identifiers objects.

    Parameters
    ----------
    reference_series : pd.Series
        Series containing reference strings for reactions.

    Returns
    -------
    pd.DataFrame
        DataFrame with reference strings as index and r_Identifiers as column containing
        Identifiers objects for each reference.
    """

    references_df = pd.concat(
        [_parse_omnipath_annotation(x) for x in reference_series.unique()]
    )

    identifiers_map = {}
    for annotation_str, group in references_df.groupby(
        OMNIPATH_ANNOTATIONS.ANNOTATION_STR
    ):
        identifier_dicts = []
        for _, row in group.iterrows():
            identifier_dict = {
                IDENTIFIERS.ONTOLOGY: ONTOLOGIES.PUBMED,
                IDENTIFIERS.IDENTIFIER: row[OMNIPATH_ANNOTATIONS.ANNOTATION],
                IDENTIFIERS.BQB: BQB.IS_DESCRIBED_BY,
            }
            identifier_dicts.append(identifier_dict)

        identifiers_map[annotation_str] = Identifiers(identifier_dicts)

    reaction_identifiers = (
        pd.DataFrame(identifiers_map, index=[SBML_DFS.R_IDENTIFIERS])
        .T.rename_axis(OMNIPATH_INTERACTIONS.REFERENCES_STRIPPED)
        .reset_index()
    )

    return reaction_identifiers


def _parse_omnipath_named_annotation(annotation_str: str) -> pd.DataFrame:
    """
    Convert a semicolon-separated named annotation string to a pandas DataFrame.

    Parses strings in the format 'name:annotation;name:annotation;...' into a DataFrame
    with columns for name, annotation, and the original annotation string.

    Parameters
    ----------
    annotation_str : str
        String containing named annotations separated by semicolons,
        with each annotation in the format 'name:annotation'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'name', 'annotation', and 'annotation_str'.

    Examples
    --------
    >>> s = 'CORUM:4478;Compleat:HC1449;PDB:4awl'
    >>> df = _parse_omnipath_named_annotation(s)
    >>> print(df)
         name annotation                    annotation_str
    0   CORUM       4478  CORUM:4478;Compleat:HC1449;PDB:4awl
    1 Compleat    HC1449  CORUM:4478;Compleat:HC1449;PDB:4awl
    2      PDB       4awl  CORUM:4478;Compleat:HC1449;PDB:4awl
    """
    # Split by semicolon and then by colon
    pairs = [item.split(":", 1) for item in annotation_str.split(";") if item.strip()]

    # Create DataFrame
    df = pd.DataFrame(
        pairs, columns=[OMNIPATH_ANNOTATIONS.NAME, OMNIPATH_ANNOTATIONS.ANNOTATION]
    )

    # Add the original annotation string as a column
    df[OMNIPATH_ANNOTATIONS.ANNOTATION_STR] = annotation_str

    return df


def _parse_omnipath_annotation(annotation_str: str) -> pd.DataFrame:
    """
    Convert a semicolon-separated annotation string to a pandas DataFrame.

    Parses strings in the format 'annotation;annotation;...' into a DataFrame
    with columns for annotation and the original annotation string.

    Parameters
    ----------
    annotation_str : str
        String containing annotations separated by semicolons.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'annotation' and 'annotation_str'.

    Examples
    --------
    >>> s = '11290752;11983166;12601176'
    >>> df = _parse_omnipath_annotation(s)
    >>> print(df)
      annotation            annotation_str
    0   11290752  11290752;11983166;12601176
    1   11983166  11290752;11983166;12601176
    2   12601176  11290752;11983166;12601176
    """
    # Split by semicolon and filter out empty strings
    annotations = [item.strip() for item in annotation_str.split(";") if item.strip()]

    # Create DataFrame
    df = pd.DataFrame(annotations, columns=[OMNIPATH_ANNOTATIONS.ANNOTATION])

    # Add the original annotation string as a column
    df[OMNIPATH_ANNOTATIONS.ANNOTATION_STR] = annotation_str

    return df


def _patch_degenerate_s_names(
    df: pd.DataFrame,
    id_col: str = OMNIPATH_INTERACTIONS.INTERACTOR_ID,
    name_col: str = SBML_DFS.S_NAME,
) -> pd.DataFrame:
    """
    Patch degenerate s_names by appending interactor IDs to non-unique names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing species data.
    id_col : str, optional
        Column name containing interactor IDs, by default OMNIPATH_INTERACTIONS.INTERACTOR_ID.
    name_col : str, optional
        Column name containing species names, by default SBML_DFS.S_NAME.

    Returns
    -------
    pd.DataFrame
        DataFrame with patched s_names where non-unique names have been made unique
        by appending the interactor ID.

    Notes
    -----
    This function ensures that all s_names are unique by appending the interactor ID
    to any name that appears multiple times in the dataset.
    """

    working_df = df.copy()
    name_counts = working_df[name_col].value_counts()

    # Find non-unique names (count > 1)
    non_unique_names = name_counts[name_counts > 1].index
    mask = working_df[name_col].isin(non_unique_names)

    if mask.sum() > 0:
        logger.info(
            f"Patching {mask.sum()} degenerate species names so they are unique"
        )

        working_df.loc[mask, name_col] = (
            working_df.loc[mask, id_col].astype(str)
            + ":"
            + working_df.loc[mask, name_col].astype(str)
        )
    else:
        logger.debug("No degenerate species names found")

    return working_df


def _format_edgelist_interactions(
    interactions: pd.DataFrame, nondegenerate_species_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Format OmniPath interactions into a standardized edgelist format.

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame containing OmniPath interaction data.
    nondegenerate_species_df : pd.DataFrame
        DataFrame containing species data with unique s_names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing formatted interactions in edgelist format with columns
        for upstream/downstream names, SBO terms, stoichiometry, and metadata.
    """

    # format reaction references as Identifiers objects
    reaction_references_df = _format_reaction_references(
        interactions[OMNIPATH_INTERACTIONS.REFERENCES_STRIPPED]
    )

    # interaction attributes to SBO terms and half reactions (reversibile reactions go to forward
    # and reverese irreversible unless it is an "interactor" reaction)
    omnipath_attribute_mapper = _load_omnipath_attribute_mapper()

    edgelist_interactions_df = (
        _name_interactions(interactions, nondegenerate_species_df)
        .merge(reaction_references_df, how="left")
        .fillna({SBML_DFS.R_IDENTIFIERS: Identifiers([])})
        .merge(
            omnipath_attribute_mapper,
            on=[
                OMNIPATH_INTERACTIONS.CONSENSUS_DIRECTION,
                OMNIPATH_INTERACTIONS.CONSENSUS_STIMULATION,
                OMNIPATH_INTERACTIONS.CONSENSUS_INHIBITION,
            ],
            how="left",
        )
        .assign(
            **{
                INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM: 0,
                INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM: 0,
            }
        )[
            [
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM,
                SBML_DFS.R_ISREVERSIBLE,
                INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM,
                INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM,
                SBML_DFS.R_IDENTIFIERS,
                OMNIPATH_INTERACTIONS.IS_DIRECTED,
                OMNIPATH_INTERACTIONS.IS_STIMULATION,
                OMNIPATH_INTERACTIONS.IS_INHIBITION,
                OMNIPATH_INTERACTIONS.CONSENSUS_DIRECTION,
                OMNIPATH_INTERACTIONS.CONSENSUS_STIMULATION,
                OMNIPATH_INTERACTIONS.CONSENSUS_INHIBITION,
                OMNIPATH_INTERACTIONS.N_PRIMARY_SOURCES,
                OMNIPATH_INTERACTIONS.N_REFERENCES,
                OMNIPATH_INTERACTIONS.N_SOURCES,
            ]
        ]
    )

    return edgelist_interactions_df


def _format_complex_interactions(
    complex_formation_edgelist: pd.DataFrame, nondegenerate_species_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Format complex formation interactions into a standardized edgelist format.

    Parameters
    ----------
    complex_formation_edgelist : pd.DataFrame
        DataFrame containing complex formation reactions.
    nondegenerate_species_df : pd.DataFrame
        DataFrame containing species data with unique s_names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing formatted complex formation reactions in edgelist format.
    """

    return _name_interactions(
        complex_formation_edgelist, nondegenerate_species_df
    ).assign(
        **{
            SBML_DFS.R_ISREVERSIBLE: False,
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: SBOTERM_NAMES.REACTANT,
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: SBOTERM_NAMES.PRODUCT,
            SBML_DFS.R_IDENTIFIERS: Identifiers([]),
        }
    )[
        [
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM,
            SBML_DFS.R_ISREVERSIBLE,
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM,
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM,
            SBML_DFS.R_IDENTIFIERS,
        ]
    ]


def _name_interactions(
    interactions: pd.DataFrame, nondegenerate_species_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add systematic names to interactions by merging with species data.

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame containing interactions with source and target interactor IDs.
    nondegenerate_species_df : pd.DataFrame
        DataFrame containing species data with interactor IDs and s_names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing interactions with upstream_name and downstream_name
        columns added by merging with species data.
    """

    return interactions.merge(
        (
            nondegenerate_species_df[
                [OMNIPATH_INTERACTIONS.INTERACTOR_ID, SBML_DFS.S_NAME]
            ].rename(
                columns={
                    OMNIPATH_INTERACTIONS.INTERACTOR_ID: OMNIPATH_INTERACTIONS.SOURCE,
                    SBML_DFS.S_NAME: INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM,
                }
            )
        ),
        on=OMNIPATH_INTERACTIONS.SOURCE,
        how="inner",
    ).merge(
        (
            nondegenerate_species_df[
                [OMNIPATH_INTERACTIONS.INTERACTOR_ID, SBML_DFS.S_NAME]
            ].rename(
                columns={
                    OMNIPATH_INTERACTIONS.INTERACTOR_ID: OMNIPATH_INTERACTIONS.TARGET,
                    SBML_DFS.S_NAME: INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM,
                }
            )
        ),
        on=OMNIPATH_INTERACTIONS.TARGET,
        how="inner",
    )
