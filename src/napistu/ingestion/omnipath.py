import logging
from typing import Union
import pandas as pd

from omnipath import interactions

from napistu.identifiers import Identifiers
from napistu.ontologies.genodexito import Genodexito
from napistu.ingestion.species import SpeciesValidator
from napistu.ontologies.pubchem import map_pubchem_ids
from napistu.ontologies.mirbase import load_mirbase_xrefs
from napistu.constants import SBML_DFS, IDENTIFIERS, BQB, ONTOLOGIES
from napistu.ingestion.constants import (
    OMNIPATH_ANNOTATIONS,
    OMNIPATH_INTERACTIONS,
    VALID_OMNIPATH_SPECIES,
)
from napistu.ontologies.constants import GENODEXITO_DEFS, PUBCHEM_DEFS, MIRBASE_TABLES

logger = logging.getLogger(__name__)

# Map dataset names to interaction classes
OMNIPATH_FXN_MAP = {
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


def get_interactions(
    dataset: Union[str, object] = "all",
    organismal_species: Union[str, SpeciesValidator] = "human",
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
        species = SpeciesValidator(organismal_species)
    species.assert_supported(VALID_OMNIPATH_SPECIES)

    # Get the interaction class
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
        organism=species.common_name, strict_evidences=True, **kwargs
    )

    # Fix consensus logic bug
    df = _fix_consensus_logic(df)

    return df


def _fix_consensus_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the consensus logic bug for interactions with no evidence.

    When strict_evidences=True, interactions with no stimulation or inhibition
    evidence incorrectly get consensus_stimulation=True and consensus_inhibition=True
    due to the 0>=0 and 0<=0 comparisons both being True.
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


def _prepare_integer_based_ids(interactor_int_ids: list[str]) -> pd.DataFrame:

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
    interactor_string_ids: list[str],
    organismal_species: Union[str, SpeciesValidator],
    preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
    allow_fallback: bool = True,
) -> pd.DataFrame:

    if isinstance(organismal_species, str):
        species = SpeciesValidator(organismal_species)

    logger.info(
        f"Searching Genodexito for {species.common_name} ({species.latin_name}) Uniprot annotations"
    )

    genodexito = Genodexito(
        species=species.latin_name,
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
    interactor_string_ids: list[str],  # noqa: F401
) -> pd.DataFrame:

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
    >>> df = parse_omnipath_named_annotation(s)
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
    >>> df = parse_omnipath_annotation(s)
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
