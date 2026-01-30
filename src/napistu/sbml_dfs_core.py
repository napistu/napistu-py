"""
The core SBML DataFrame class for representing a SBML model as a collection of pandas DataFrames.

Classes
-------
SBML_dfs
    A class representing a SBML model as a collection of pandas DataFrames.
"""

from __future__ import annotations

import copy
import logging
import re
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Union,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs
import pandas as pd

from napistu import identifiers, sbml_dfs_utils, source, utils
from napistu.ontologies import id_tables

if TYPE_CHECKING:
    from napistu.ingestion import sbml
from napistu.constants import (
    BQB,
    BQB_DEFINING_ATTRS,
    BQB_DEFINING_ATTRS_LOOSE,
    BQB_PRIORITIES,
    CONSENSUS_CHECKS,
    CONSENSUS_CHECKS_LIST,
    ENTITIES_TO_ENTITY_DATA,
    ENTITIES_W_DATA,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    MINI_SBO_TO_NAME,
    NAPISTU_STANDARD_OUTPUTS,
    ONTOLOGY_PRIORITIES,
    SBML_DFS,
    SBML_DFS_CLEANUP_ORDER,
    SBML_DFS_METADATA,
    SBML_DFS_METHOD_DEFS,
    SBML_DFS_SCHEMA,
    SBOTERM_NAMES,
    SCHEMA_DEFS,
)
from napistu.ingestion.constants import (
    DEFAULT_PRIORITIZED_PATHWAYS,
    INTERACTION_EDGELIST_DEFAULTS,
)

logger = logging.getLogger(__name__)


class SBML_dfs:
    """
    System Biology Markup Language Model Data Frames.

    A class representing a SBML model as a collection of pandas DataFrames.
    This class provides methods for manipulating and analyzing biological pathway models
    with support for species, reactions, compartments, and their relationships.

    Attributes
    ----------
    compartments : pd.DataFrame
        Sub-cellular compartments in the model, indexed by compartment ID (c_id)
    species : pd.DataFrame
        Molecular species in the model, indexed by species ID (s_id)
    species_data : Dict[str, pd.DataFrame]
        Additional data for species. Each DataFrame is indexed by species_id (s_id)
    reactions : pd.DataFrame
        Reactions in the model, indexed by reaction ID (r_id)
    reactions_data : Dict[str, pd.DataFrame]
        Additional data for reactions. Each DataFrame is indexed by reaction_id (r_id)
    reaction_species : pd.DataFrame
        One entry per species participating in a reaction, indexed by reaction-species ID (rsc_id)
    schema : dict
        Dictionary representing the structure of the other attributes and meaning of their variables

    Public Methods (alphabetical)
    ----------------------------
    add_reactions_data(label, data)
        Add a new reactions data table to the model with validation.
    add_species_data(label, data)
        Add a new species data table to the model with validation.
    copy()
        Return a deep copy of the SBML_dfs object.
    export_sbml_dfs(model_prefix, outdir, overwrite=False, dogmatic=True)
        Export the SBML_dfs model and its tables to files in a specified directory.
    find_entity_references(entity_type, entity_ids, reference_type, reference_ids)
        Find entities that reference specified entities through a given reference type.
    from_edgelist(interaction_edgelist, species_df, compartments_df, interaction_source=source.Source(init=True), interaction_edgelist_defaults=INTERACTION_EDGELIST_DEFAULTS, keep_species_data=False, keep_reactions_data=False)
        Create SBML_dfs from interaction edgelist.
    from_pickle(path)
        Load an SBML_dfs from a pickle file.
    get_characteristic_species_ids(dogmatic=True)
        Return characteristic systematic identifiers for molecular species, optionally using a strict or loose definition.
    get_cspecies_features()
        Compute and return additional features for compartmentalized species, such as degree and type.
    get_identifiers(id_type)
        Retrieve a table of identifiers for a specified entity type (e.g., species or reactions).
    get_ontology_cooccurrence(entity_type, stratify_by_bqb=True, allow_col_multindex=False)
        Get ontology co-occurrence matrix for a specific entity type.
    get_ontology_occurrence(entity_type, stratify_by_bqb=True, allow_col_multindex=False)
        Get ontology occurrence summary for a specific entity type.
    get_ontology_x_source_cooccurrence(entity_type, stratify_by_bqb=True, allow_col_multindex=False, characteristic_only=False, dogmatic=True, priority_pathways=DEFAULT_PRIORITIZED_PATHWAYS)
        Get ontology × source co-occurrence matrix for a specific entity type.
    get_sbo_term_occurrence(entity_type, stratify_by_bqb=True, allow_col_multindex=False)
        Get SBO term occurrence summary for a specific entity type.
    get_sbo_term_x_source_cooccurrence(entity_type, stratify_by_bqb=True, allow_col_multindex=False, characteristic_only=False, dogmatic=True, priority_pathways=DEFAULT_PRIORITIZED_PATHWAYS)
        Get SBO term × source co-occurrence matrix for a specific entity type.
    get_source_cooccurrence(entity_type, priority_pathways=DEFAULT_PRIORITIZED_PATHWAYS)
        Get pathway co-occurrence matrix for a specific entity type.
    get_source_occurrence(entity_type, priority_pathways=DEFAULT_PRIORITIZED_PATHWAYS)
        Get pathway occurrence summary for a specific entity type.
    get_sources(entity_type)
        Get the unnest sources table for a given entity type.
    get_source_total_counts(entity_type)
        Get the total counts of each source for a given entity type.
    get_species_features()
        Compute and return additional features for species, such as species type.
    get_summary()
        Return a dictionary of diagnostic statistics summarizing the SBML_dfs structure.
    get_table(entity_type, required_attributes=None)
        Retrieve a table for a given entity type, optionally validating required attributes.
    get_uri_urls(entity_type, entity_ids=None, required_ontology=None)
        Return reference URLs for specified entities, optionally filtered by ontology.
    infer_sbo_terms()
        Infer and fill in missing SBO terms for reaction species based on stoichiometry.
    infer_uncompartmentalized_species_location()
        Infer and assign compartments for compartmentalized species with missing compartment information.
    name_compartmentalized_species()
        Rename compartmentalized species to include compartment information if needed.
    post_consensus_checks(entity_types=[SBML_DFS.SPECIES, SBML_DFS.COMPARTMENTS], check_types=[CONSENSUS_CHECKS.SOURCE_COOCCURRENCE, CONSENSUS_CHECKS.ONTOLOGY_X_SOURCE_COOCCURRENCE])
        Perform checks on the SBML_dfs object after consensus building.
    reaction_formulas(r_ids=None)
        Generate human-readable reaction formulas for specified reactions.
    reaction_summaries(r_ids=None)
        Return a summary DataFrame for specified reactions, including names and formulas.
    remove_entities(entity_type, entity_ids, remove_species=False)
        Remove specified entities and optionally remove unused species.
    remove_reactions_data(label)
        Remove a reactions data table by label.
    remove_species_data(label)
        Remove a species data table by label.
    remove_unused()
        Find and remove unused entities from the model with cascading cleanup.
    search_by_ids(id_table, identifiers=None, ontologies=None, bqbs=None)
        Find entities and identifiers matching a set of query IDs.
    search_by_name(name, entity_type, partial_match=True)
        Find entities by exact or partial name match.
    select_species_data(species_data_table)
        Select a species data table from the SBML_dfs object by name.
    show_summary()
        Display a formatted summary of the SBML_dfs model.
    species_status(s_id)
        Return all reactions a species participates in, with stoichiometry and formula information.
    to_dict()
        Return the 5 major SBML_dfs tables as a dictionary.
    to_pickle(path)
        Save the SBML_dfs to a pickle file.
    validate()
        Validate the SBML_dfs structure and relationships.
    validate_and_resolve()
        Validate and attempt to automatically fix common issues.

    Private/Hidden Methods (alphabetical, appear after public methods)
    -----------------------------------------------------------------
    _attempt_resolve(e)
    _edgelist_assemble_sbml_model(compartments, species, comp_species, reactions, reaction_species, species_data, reactions_data, keep_species_data, keep_reactions_data, extra_columns)
    _find_invalid_entities_by_reference(entity_type, reference_type, reference_ids)
    _find_underspecified_reactions_by_reference(reference_type, reference_ids)
    _get_entity_data(entity_type, label)
    _get_identifiers_table_for_ontology_occurrence(entity_type, characteristic_only=False, dogmatic=True)
    _get_non_interactor_reactions()
    _remove_entities_direct(entity_type, entity_ids)
    _remove_entity_data(entity_type, label)
    _validate_entity_data_access(entity_type, label)
    _validate_identifiers()
    _validate_pk_fk_correspondence()
    _validate_r_ids(r_ids)
    _validate_reaction_species()
    _validate_reactions_data(reactions_data_table)
    _validate_sources()
    _validate_species_data(species_data_table)
    _validate_table(table_name)
    """

    compartments: pd.DataFrame
    species: pd.DataFrame
    species_data: dict[str, pd.DataFrame]
    reactions: pd.DataFrame
    reactions_data: dict[str, pd.DataFrame]
    reaction_species: pd.DataFrame
    schema: dict
    _required_entities: set[str]
    _optional_entities: set[str]

    def __init__(
        self,
        sbml_model: (
            sbml.SBML | MutableMapping[str, pd.DataFrame | dict[str, pd.DataFrame]]
        ),
        model_source: source.Source,
        validate: bool = True,
        resolve: bool = True,
    ) -> None:
        """
        Initialize a SBML_dfs object from a SBML model or dictionary of tables.

        Parameters
        ----------
        sbml_model : Union[sbml.SBML, MutableMapping[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]
            Either a SBML model produced by sbml.SBML() or a dictionary containing tables
            following the sbml_dfs schema
        validate : bool, optional
            Whether to validate the model structure and relationships, by default True
        resolve : bool, optional
            Whether to attempt automatic resolution of common issues, by default True

        Raises
        ------
        ValueError
            If the model structure is invalid and cannot be resolved
        """

        self.schema = SBML_DFS_SCHEMA.SCHEMA
        self._required_entities = SBML_DFS_SCHEMA.REQUIRED_ENTITIES
        self._optional_entities = SBML_DFS_SCHEMA.OPTIONAL_ENTITIES

        # Initialize the dynamic attributes for type checking
        if TYPE_CHECKING:
            self.compartments = pd.DataFrame()
            self.species = pd.DataFrame()
            self.compartmentalized_species = pd.DataFrame()
            self.reactions = pd.DataFrame()
            self.reaction_species = pd.DataFrame()

        # initialize the model's metadata attribute
        self.metadata = dict()
        # validate the model-level source data

        if not isinstance(model_source, source.Source):
            raise ValueError(
                f"model_source was a {type(model_source)} and must be a source.Source"
            )

        model_source.validate_single_source()
        self.metadata[SBML_DFS_METADATA.SBML_DFS_SOURCE] = model_source

        # create a model from dictionary entries
        if isinstance(sbml_model, dict):
            for ent in SBML_DFS_SCHEMA.REQUIRED_ENTITIES:
                setattr(self, ent, sbml_model[ent])
            for ent in SBML_DFS_SCHEMA.OPTIONAL_ENTITIES:
                if ent in sbml_model:
                    setattr(self, ent, sbml_model[ent])
        else:
            from napistu.ingestion import sbml

            self = sbml.sbml_dfs_from_sbml(self, sbml_model)

        for ent in SBML_DFS_SCHEMA.OPTIONAL_ENTITIES:
            # Initialize optional entities if not set
            if not hasattr(self, ent):
                setattr(self, ent, {})

        if validate:
            if resolve:
                self.validate_and_resolve()
            else:
                self.validate()
        else:
            if resolve:
                logger.warning(
                    '"validate" = False so "resolve" will be ignored (eventhough it was True)'
                )

    # =============================================================================
    # PUBLIC METHODS (ALPHABETICAL ORDER)
    # =============================================================================

    def add_reactions_data(self, label: str, data: pd.DataFrame):
        """
        Add additional reaction data with validation.

        Parameters
        ----------
        label : str
            Label for the new data
        data : pd.DataFrame
            Data to add, must be indexed by reaction_id

        Raises
        ------
        ValueError
            If the data is invalid or label already exists
        """
        self._validate_reactions_data(data)
        if label in self.reactions_data:
            raise ValueError(
                f"{label} already exists in reactions_data. " "Drop it first."
            )
        self.reactions_data[label] = data

    def add_species_data(self, label: str, data: pd.DataFrame):
        """
        Add additional species data with validation.

        Parameters
        ----------
        label : str
            Label for the new data
        data : pd.DataFrame
            Data to add, must be indexed by species_id

        Raises
        ------
        ValueError
            If the data is invalid or label already exists
        """
        self._validate_species_data(data)
        if label in self.species_data:
            raise ValueError(
                f"{label} already exists in species_data. " "Drop it first."
            )
        self.species_data[label] = data

    def copy(self):
        """
        Return a deep copy of the SBML_dfs object.

        Returns
        -------
        SBML_dfs
            A deep copy of the current SBML_dfs object.
        """
        return copy.deepcopy(self)

    def export_sbml_dfs(
        self,
        model_prefix: str,
        outdir: str,
        overwrite: bool = False,
        dogmatic: bool = True,
    ) -> None:
        """
        Export SBML_dfs

        Export summaries of species identifiers and each table underlying
        an SBML_dfs pathway model

        Params
        ------
        model_prefix: str
            Label to prepend to all exported files
        outdir: str
            Path to an existing directory where results should be saved
        overwrite: bool
            Should the directory be overwritten if it already exists?
        dogmatic: bool
            If True then treat genes, transcript, and proteins as separate species. If False
            then treat them interchangeably.

        Returns
        -------
        None
        """
        if not isinstance(model_prefix, str):
            raise TypeError(
                f"model_prefix was a {type(model_prefix)} " "and must be a str"
            )
        if not isinstance(self, SBML_dfs):
            raise TypeError(f"sbml_dfs was a {type(self)} and must" " be an SBML_dfs")

        # filter to identifiers which make sense when mapping from ids -> species
        species_identifiers = self.get_characteristic_species_ids(dogmatic=dogmatic)

        # get reactions' source total counts
        reactions_source_total_counts = self.get_source_total_counts(SBML_DFS.REACTIONS)

        try:
            utils.initialize_dir(outdir, overwrite=overwrite)
        except FileExistsError:
            logger.warning(
                f"Directory {outdir} already exists and overwrite is False. "
                "Files will be added to the existing directory."
            )
        with open_fs(outdir, writeable=True) as fs:
            species_identifiers_path = (
                model_prefix + NAPISTU_STANDARD_OUTPUTS.SPECIES_IDENTIFIERS
            )
            with fs.openbin(species_identifiers_path, "w") as f:
                species_identifiers.to_csv(f, sep="\t", index=False)

            # export reactions' source total counts
            reactions_source_total_counts_path = (
                model_prefix + NAPISTU_STANDARD_OUTPUTS.REACTIONS_SOURCE_TOTAL_COUNTS
            )
            with fs.openbin(reactions_source_total_counts_path, "w") as f:
                reactions_source_total_counts.to_csv(f, sep="\t", index=True)

            # export s_id to sc_id lookup table
            sid_to_scids = self.compartmentalized_species.reset_index()[
                [SBML_DFS.S_ID, SBML_DFS.SC_ID]
            ]
            sid_to_scids_path = model_prefix + NAPISTU_STANDARD_OUTPUTS.SID_TO_SCIDS
            with fs.openbin(sid_to_scids_path, "w") as f:
                sid_to_scids.to_csv(f, sep="\t", index=False)

            # export jsons
            species_path = model_prefix + NAPISTU_STANDARD_OUTPUTS.SPECIES
            reactions_path = model_prefix + NAPISTU_STANDARD_OUTPUTS.REACTIONS
            reation_species_path = (
                model_prefix + NAPISTU_STANDARD_OUTPUTS.REACTION_SPECIES
            )
            compartments_path = model_prefix + NAPISTU_STANDARD_OUTPUTS.COMPARTMENTS
            compartmentalized_species_path = (
                model_prefix + NAPISTU_STANDARD_OUTPUTS.COMPARTMENTALIZED_SPECIES
            )
            with fs.openbin(species_path, "w") as f:
                self.species[[SBML_DFS.S_NAME]].to_json(f)

            with fs.openbin(reactions_path, "w") as f:
                self.reactions[[SBML_DFS.R_NAME]].to_json(f)

            with fs.openbin(reation_species_path, "w") as f:
                self.reaction_species.to_json(f)

            with fs.openbin(compartments_path, "w") as f:
                self.compartments[[SBML_DFS.C_NAME]].to_json(f)

            with fs.openbin(compartmentalized_species_path, "w") as f:
                self.compartmentalized_species.drop(SBML_DFS.SC_SOURCE, axis=1).to_json(
                    f
                )

        return None

    def find_entity_references(
        self, entity_type: str, entity_ids: list[str]
    ) -> dict[str, set[str]]:
        """Find all entities that directly depend on the set of requested entities.

        Parameters
        ----------
        entity_type : str
            The initial entity type to remove
        entity_ids : list[str]
            IDs of entities to remove

        Returns
        -------
        dict[str, set[str]]
            Dictionary mapping entity types to sets of IDs that directly depend on the requested entities
        """

        dependents = {
            SBML_DFS.COMPARTMENTS: set(),
            SBML_DFS.SPECIES: set(),
            SBML_DFS.COMPARTMENTALIZED_SPECIES: set(),
            SBML_DFS.REACTIONS: set(),
            SBML_DFS.REACTION_SPECIES: set(),
        }

        if entity_type == "cofactors":
            entity_type = SBML_DFS.REACTION_SPECIES
            literal_cleanup = True
        else:
            literal_cleanup = False

        # Start with the directly requested entities
        dependents[entity_type] = set(entity_ids)
        logger.debug(
            f"Starting find_entity_references from {entity_type} with {len(entity_ids)} entities"
        )

        # Iterate through cleanup order to find cascading removals
        cleanup_order = SBML_DFS_CLEANUP_ORDER[entity_type]
        logger.debug(f"Cleanup order for {entity_type}: {cleanup_order}")

        for updated_table, reference_table in cleanup_order:
            # Get orphaned entities based on type
            if (updated_table == SBML_DFS.REACTIONS) and not literal_cleanup:
                # Special handling for reactions - check for underspecified reactions
                invalid_reactions, invalid_reaction_species = (
                    self._find_underspecified_reactions_by_reference(
                        reference_table, dependents[reference_table]
                    )
                )

                new_invalid_reactions = (
                    invalid_reactions - dependents[SBML_DFS.REACTIONS]
                )
                new_invalid_reaction_species = (
                    invalid_reaction_species - dependents[SBML_DFS.REACTION_SPECIES]
                )
                logger.debug(
                    f"Registering {len(new_invalid_reactions)} new invalid reactions and {len(new_invalid_reaction_species)} new invalid reaction species"
                )
                logger.debug(f"Invalid reactions: {new_invalid_reactions}")
                logger.debug(
                    f"Invalid reaction species: {new_invalid_reaction_species}"
                )
                dependents[SBML_DFS.REACTIONS] = (
                    dependents[SBML_DFS.REACTIONS] | new_invalid_reactions
                )
                dependents[SBML_DFS.REACTION_SPECIES] = (
                    dependents[SBML_DFS.REACTION_SPECIES] | new_invalid_reaction_species
                )
            else:
                # Standard orphaned entity removal - find all orphaned entities
                invalid_entities = self._find_invalid_entities_by_reference(
                    updated_table, reference_table, dependents[reference_table]
                )

                new_invalid_entities = invalid_entities - dependents[updated_table]
                logger.debug(
                    f"Registering {len(new_invalid_entities)} new invalid {updated_table}"
                )
                logger.debug(f"Invalid {updated_table}: {new_invalid_entities}")
                dependents[updated_table] = (
                    dependents[updated_table] | new_invalid_entities
                )

        return dependents

    @classmethod
    def from_edgelist(
        cls,
        interaction_edgelist: pd.DataFrame,
        species_df: pd.DataFrame,
        compartments_df: pd.DataFrame,
        model_source: source.Source,
        interaction_edgelist_defaults: dict[str, Any] = INTERACTION_EDGELIST_DEFAULTS,
        keep_species_data: bool | str = False,
        keep_reactions_data: bool | str = False,
        force_edgelist_consistency: bool = False,
    ) -> "SBML_dfs":
        """
        Create SBML_dfs from interaction edgelist.

        Combines a set of molecular interactions into a mechanistic SBML_dfs model
        by processing interaction data, species information, and compartment definitions.

        Parameters
        ----------
        interaction_edgelist : pd.DataFrame
            Table containing molecular interactions with columns:
            - name_upstream : str, matches "s_name" from species_df
            - name_downstream : str, matches "s_name" from species_df
            - r_name : str, name for the interaction
            - r_Identifiers : identifiers.Identifiers, supporting identifiers
            - compartment_upstream : str, matches "c_name" from compartments_df
            - compartment_downstream : str, matches "c_name" from compartments_df
            - sbo_term_name_upstream : str, SBO term defining interaction type
            - sbo_term_name_downstream : str, SBO term defining interaction type
            - stoichiometry_upstream : float, stoichiometry of upstream species
            - stoichiometry_downstream : float, stoichiometry of downstream species
            - r_isreversible : bool, whether reaction is reversible
        species_df : pd.DataFrame
            Table defining molecular species with columns:
            - s_name : str, name of molecular species
            - s_Identifiers : identifiers.Identifiers, species identifiers
        compartments_df : pd.DataFrame
            Table defining compartments with columns:
            - c_name : str, name of compartment
            - c_Identifiers : identifiers.Identifiers, compartment identifiers
        model_source : source.Source
            Source annotations for the data source
        interaction_edgelist_defaults : dict[str, Any], default INTERACTION_EDGELIST_DEFAULTS
            Default values for interaction edgelist columns
        keep_species_data : bool or str, default False
            Whether to preserve extra species columns. If True, saves as 'source' label.
            If string, uses as custom label. If False, discards extra data.
        keep_reactions_data : bool or str, default False
            Whether to preserve extra reaction columns. If True, saves as 'source' label.
            If string, uses as custom label. If False, discards extra data.
        force_edgelist_consistency : bool, default False
            Whether to force the edgelist to be consistent with the species and compartments dataframes
            This is useful for cases where there may be reasonable departures between the edgelist and
            the species and compartments dataframes but the user wants to create an SBML_dfs model anyway


        Returns
        -------
        SBML_dfs
            Validated SBML data structure containing compartments, species,
            compartmentalized species, reactions, and reaction species tables.
        """

        # organize source info
        if not isinstance(model_source, source.Source):
            raise ValueError("model_source must be a source.Source object")

        # Validate that interaction_edgelist_defaults is a dictionary
        if not isinstance(interaction_edgelist_defaults, dict):
            raise TypeError(
                f"interaction_edgelist_defaults must be a dictionary, got {type(interaction_edgelist_defaults)}"
            )

        # set default entity-level source
        interaction_source = source.Source.empty()

        # 0. Add defaults to interaction edgelist
        interaction_edgelist_with_defaults = sbml_dfs_utils._add_edgelist_defaults(
            interaction_edgelist, interaction_edgelist_defaults
        )

        if force_edgelist_consistency:
            interaction_edgelist_with_defaults, species_df, compartments_df = (
                sbml_dfs_utils.force_edgelist_consistency(
                    interaction_edgelist_with_defaults, species_df, compartments_df
                )
            )

        # 1. Validate inputs
        sbml_dfs_utils._edgelist_validate_inputs(
            interaction_edgelist_with_defaults, species_df, compartments_df
        )

        # 2. Identify which extra columns to preserve
        extra_columns = sbml_dfs_utils._edgelist_identify_extra_columns(
            interaction_edgelist_with_defaults,
            species_df,
            keep_reactions_data,
            keep_species_data,
        )

        # 3. Process compartments and species tables
        processed_compartments = sbml_dfs_utils._edgelist_process_compartments(
            compartments_df, interaction_source
        )
        processed_species, species_data = sbml_dfs_utils._edgelist_process_species(
            species_df, interaction_source, extra_columns[SBML_DFS.SPECIES]
        )

        # drop extra species and compartments and warn

        # 4. Create compartmentalized species
        comp_species = sbml_dfs_utils._edgelist_create_compartmentalized_species(
            interaction_edgelist_with_defaults,
            processed_species,
            processed_compartments,
            interaction_source,
        )

        # 5. Create reactions and reaction species
        reactions, reaction_species, reactions_data = (
            sbml_dfs_utils._edgelist_create_reactions_and_species(
                interaction_edgelist_with_defaults,
                comp_species,
                processed_species,
                processed_compartments,
                interaction_source,
                extra_columns[SBML_DFS.REACTIONS],
            )
        )

        # 6. Assemble final SBML_dfs object
        sbml_dfs = cls._edgelist_assemble_sbml_model(
            compartments=processed_compartments,
            species=processed_species,
            comp_species=comp_species,
            reactions=reactions,
            reaction_species=reaction_species,
            species_data=species_data,
            reactions_data=reactions_data,
            keep_species_data=keep_species_data,
            keep_reactions_data=keep_reactions_data,
            extra_columns=extra_columns,
            model_source=model_source,
        )

        sbml_dfs.validate()
        return sbml_dfs

    @classmethod
    def from_pickle(cls, path: str) -> "SBML_dfs":
        """
        Load an SBML_dfs from a pickle file.

        Parameters
        ----------
        path : str
            Path to the pickle file

        Returns
        -------
        SBML_dfs
            The loaded SBML_dfs object
        """
        sbml_dfs = utils.load_pickle(path)
        if not isinstance(sbml_dfs, cls):
            raise ValueError(
                f"Pickled input is not an SBML_dfs object but {type(sbml_dfs)}: {path}"
            )

        return sbml_dfs

    def get_characteristic_species_ids(self, dogmatic: bool = True) -> pd.DataFrame:
        """
        Get Characteristic Species IDs

        List the systematic identifiers which are characteristic of molecular species, e.g.,
        excluding subcomponents, and optionally, treating proteins, transcripts, and genes equiavlently.

        Characteristic identifiers include:
        - the defining IDs (BQB_IS) if dogmatic is True, and BQB_IS, BQB_IS_ENCODED_BY, BQB_ENCODES if dogmatic = False.
        - small complexes (BQB_HAS_PART)

        This function is useful for pulling out the species which are closely associated with a specific proteins, metabolites, etc.

        Parameters
        ----------
        dogmatic : bool, default=True
            Whether to use the dogmatic flag to determine which BQB attributes are valid.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the systematic identifiers which are characteristic of molecular species.
        """

        # select valid BQB attributes based on dogmatic flag
        defining_biological_qualifiers = sbml_dfs_utils._dogmatic_to_defining_bqbs(
            dogmatic
        )

        # pre-summarize ontologies
        species_identifiers = self.get_identifiers(SBML_DFS.SPECIES)

        # drop some BQB_HAS_PART annotations
        species_identifiers = sbml_dfs_utils.filter_to_characteristic_species_ids(
            species_identifiers,
            defining_biological_qualifiers=defining_biological_qualifiers,
        )

        return species_identifiers

    def get_cspecies_features(self) -> pd.DataFrame:
        """
        Get additional attributes of compartmentalized species.

        Returns
        -------
        pd.DataFrame
            Compartmentalized species with additional features including:
            - sc_degree: Number of reactions the species participates in
            - sc_children: Number of reactions where species is consumed
            - sc_parents: Number of reactions where species is produced
            - species_type: Classification of the species
        """
        cspecies_n_connections = (
            self.reaction_species[SBML_DFS.SC_ID]
            .value_counts()
            .rename(SBML_DFS_METHOD_DEFS.SC_DEGREE)
        )

        cspecies_n_children = (
            self.reaction_species.loc[
                self.reaction_species[SBML_DFS.STOICHIOMETRY] <= 0, SBML_DFS.SC_ID
            ]
            .value_counts()
            .rename(SBML_DFS_METHOD_DEFS.SC_CHILDREN)
        )

        cspecies_n_parents = (
            self.reaction_species.loc[
                self.reaction_species[SBML_DFS.STOICHIOMETRY] > 0, SBML_DFS.SC_ID
            ]
            .value_counts()
            .rename(SBML_DFS_METHOD_DEFS.SC_PARENTS)
        )

        species_features = self.get_species_features()[
            SBML_DFS_METHOD_DEFS.SPECIES_TYPE
        ]

        return (
            self.compartmentalized_species.join(cspecies_n_connections)
            .join(cspecies_n_children)
            .join(cspecies_n_parents)
            .fillna(int(0))  # Explicitly fill with int(0) to avoid downcasting warning
            .merge(species_features, left_on=SBML_DFS.S_ID, right_index=True)
            .drop(
                columns=[
                    SBML_DFS.SC_NAME,
                    SBML_DFS.SC_SOURCE,
                    SBML_DFS.S_ID,
                    SBML_DFS.C_ID,
                ]
            )
        )

    def get_identifiers(
        self, id_type, filter_by_bqb=None, add_names=True, keep_source=False
    ) -> pd.DataFrame:
        """
        Get identifiers from a specified entity type.

        Parameters
        ----------
        id_type : str
            Type of entity to get identifiers for (e.g., 'species', 'reactions')
        filter_by_bqb : None, list, or str, optional
            Filter identifiers by biological qualifier (BQB) terms:
            - None: No filtering, return all identifiers (default)
            - list: List of specific BQB terms to include
            - "defining": Use BQB_DEFINING_ATTRS (strict defining identifiers)
            - "loose": Use BQB_DEFINING_ATTRS_LOOSE (includes encoded/encodes relationships)
        add_names : bool, optional
            Whether to add entity names and other metadata from the entity table, by default True
        keep_source : bool, optional
            Whether to include the source column in the output, by default False.
            Only applies when add_names=True. The source column is excluded by default
            as it's typically not needed for identifier lookups.

        Returns
        -------
        pd.DataFrame
            Table of identifiers for the specified entity type. If add_names=True, includes
            entity metadata; if add_names=False, returns only the core identifier data.

        Raises
        ------
        ValueError
            If id_type is invalid, identifiers are malformed, or filter_by_bqb is invalid
        """

        if id_type == SBML_DFS.REACTIONS:
            selected_table = self._get_non_interactor_reactions()
        else:
            selected_table = self.get_table(id_type, {SCHEMA_DEFS.ID})
        schema = SBML_DFS_SCHEMA.SCHEMA

        identifiers_dict = dict()
        for sysid in selected_table.index:
            id_entry = selected_table[schema[id_type][SCHEMA_DEFS.ID]][sysid]

            if isinstance(id_entry, identifiers.Identifiers):
                identifiers_dict[sysid] = pd.DataFrame(id_entry.df)
            elif pd.isna(id_entry):
                continue
            else:
                raise ValueError(
                    f"id_entry was a {type(id_entry)} and must either be"
                    " an identifiers.Identifiers object or a missing value (None, np.nan, pd.NA)"
                )
        # Get the source column name if it exists for this entity type
        source_col = None
        if SCHEMA_DEFS.SOURCE in schema[id_type]:
            source_col = schema[id_type][SCHEMA_DEFS.SOURCE]

        # Determine columns to retain from selected_table when add_names=True
        # Exclude the ID column, and source column (if present and keep_source=False)
        id_col = schema[id_type][SCHEMA_DEFS.ID]
        columns_to_retain = []
        for col in selected_table.columns:
            # Always exclude the ID column
            if col == id_col:
                continue
            # Exclude source column unless keep_source=True
            if source_col is not None and col == source_col and not keep_source:
                continue
            # Keep all other columns
            columns_to_retain.append(col)

        if not identifiers_dict:
            # Return empty DataFrame with expected columns if nothing found
            base_columns = [schema[id_type][SCHEMA_DEFS.PK], "entry"]
            if add_names:
                return pd.DataFrame(columns=base_columns + columns_to_retain)
            else:
                return pd.DataFrame(columns=base_columns)

        identifiers_tbl = pd.concat(identifiers_dict)
        identifiers_tbl.index.names = [schema[id_type][SCHEMA_DEFS.PK], "entry"]
        identifiers_tbl = identifiers_tbl.reset_index()

        # Conditionally add names and metadata based on add_names parameter
        if add_names:
            result_identifiers = identifiers_tbl.merge(
                selected_table[columns_to_retain],
                left_on=schema[id_type][SCHEMA_DEFS.PK],
                right_index=True,
            )
        else:
            result_identifiers = identifiers_tbl

        # Apply BQB filtering if specified
        if filter_by_bqb is not None:
            if isinstance(filter_by_bqb, str):
                if filter_by_bqb == "defining":
                    bqb_terms = BQB_DEFINING_ATTRS
                elif filter_by_bqb == "loose":
                    bqb_terms = BQB_DEFINING_ATTRS_LOOSE
                else:
                    raise ValueError(
                        f"Invalid filter_by_bqb string: '{filter_by_bqb}'. "
                        "Must be 'defining', 'loose', or a list of BQB terms."
                    )
            elif isinstance(filter_by_bqb, (list, tuple)):
                bqb_terms = list(filter_by_bqb)
            else:
                raise ValueError(
                    f"filter_by_bqb must be None, a list, or a string ('defining'/'loose'). "
                    f"Got: {type(filter_by_bqb)}"
                )

            # Filter the identifiers by BQB terms
            result_identifiers = result_identifiers[
                result_identifiers[IDENTIFIERS.BQB].isin(bqb_terms)
            ]

        return result_identifiers

    def get_ontology_cooccurrence(
        self,
        entity_type: str,
        stratify_by_bqb: bool = True,
        allow_col_multindex: bool = False,
        characteristic_only: bool = False,
        dogmatic: bool = True,
    ) -> pd.DataFrame:
        """
        Get ontology co-occurrence matrix for a specific entity type.

        This method creates a co-occurrence matrix showing which ontologies share entities
        of the specified type, indicating ontology relationships and overlaps.

        Note: When entity_type is 'reactions', reactions that consist entirely of interactor
        species will be excluded from the analysis.

        Parameters
        ----------
        entity_type : str
            The type of entity to analyze (e.g., 'species', 'reactions', 'compartments')
        stratify_by_bqb : bool, optional
            Whether to stratify by BQB (Biological Qualifier) terms, by default True
        allow_col_multindex : bool, optional
            Whether to allow column multi-index, by default False
        characteristic_only : bool, optional
            Whether to use only characteristic identifiers (only supported for species), by default False
        dogmatic : bool, optional
            Whether to use dogmatic identifier filtering, by default True

        Returns
        -------
        pd.DataFrame
            Co-occurrence matrix with ontologies as both rows and columns

        Raises
        ------
        ValueError
            If the entity type is invalid or identifiers are malformed
        """
        identifiers_table = self._get_identifiers_table_for_ontology_occurrence(
            entity_type, characteristic_only, dogmatic
        )

        return sbml_dfs_utils._summarize_ontology_cooccurrence(
            identifiers_table, stratify_by_bqb, allow_col_multindex
        )

    def get_ontology_occurrence(
        self,
        entity_type: str,
        stratify_by_bqb: bool = True,
        allow_col_multindex: bool = False,
        characteristic_only: bool = False,
        dogmatic: bool = True,
        include_missing: bool = False,
        binarize: bool = False,
    ) -> pd.DataFrame:
        """
        Get ontology occurrence summary for a specific entity type.

        This method analyzes which ontologies are associated with entities of the specified type,
        providing a summary of ontology occurrence patterns.

        Note: When entity_type is 'reactions', reactions that consist entirely of interactor
        species will be excluded from the analysis.

        Parameters
        ----------
        entity_type : str
            The type of entity to analyze (e.g., 'species', 'reactions', 'compartments')
        stratify_by_bqb : bool, optional
            Whether to stratify by BQB (Biological Qualifier) terms, by default True
        allow_col_multindex : bool, optional
            Whether to allow column multi-index, by default False
        characteristic_only: bool, optional
            Whether to only include characteristic identifiers. Only supported for species. If,
                true - returns only the characteristic identifiers (BQB_IS, and small complex BQB_HAS_PART annotations)
                false - returns all identifiers
        dogmatic: bool, optional
            Whether to use a strict or loose definition of characteristic identifiers. Only applicable if `characteristic_only` is True and `entity_type` is SBML_DFS.SPECIES.
        include_missing: bool, optional
            Whether to include missing entities in the result using add_missing_ids_column, by default False
        binarize: bool, optional
            Whether to convert the result to binary values (0 vs 1+), by default False

        Returns
        -------
        pd.DataFrame
            Summary of ontology occurrence patterns with entities as rows and ontologies as columns.
            If binarize=True, values are 0 or 1.

        Raises
        ------
        ValueError
            If the entity type is invalid or identifiers are malformed
        """

        identifiers_table = self._get_identifiers_table_for_ontology_occurrence(
            entity_type, characteristic_only, dogmatic
        )

        result = sbml_dfs_utils._summarize_ontology_occurrence(
            identifiers_table, stratify_by_bqb, allow_col_multindex, binarize
        )

        if include_missing:
            # Get the reference table (all entities of this type)
            if entity_type == SBML_DFS.REACTIONS:
                reference_table = self._get_non_interactor_reactions()
            else:
                reference_table = self.get_table(entity_type, {SCHEMA_DEFS.ID})

            result = sbml_dfs_utils.add_missing_ids_column(result, reference_table)

        return result

    def get_ontology_x_source_cooccurrence(
        self,
        entity_type: str,
        # Parameters from get_ontology_occurrence
        stratify_by_bqb: bool = True,
        allow_col_multindex: bool = False,
        characteristic_only: bool = False,
        dogmatic: bool = True,
        # Parameters from get_source_occurrence
        priority_pathways: list[str] = DEFAULT_PRIORITIZED_PATHWAYS,
    ) -> pd.DataFrame:
        """
        Get ontology × source co-occurrence matrix for a specific entity type.

        This method creates a co-occurrence matrix showing the relationship between
        ontologies and sources (pathways) by calculating how many entities of the
        specified type are shared between each ontology-source pair.

        The method combines ontology occurrence data with source occurrence data to
        create a cross-tabulation matrix where:
        - Rows represent ontologies
        - Columns represent sources/pathways
        - Values represent the number of entities shared between each ontology-source pair

        Note: When entity_type is 'reactions', reactions that consist entirely of interactor
        species will be excluded from the analysis.

        Parameters
        ----------
        entity_type : str
            The type of entity to analyze (e.g., 'species', 'reactions', 'compartments')
        stratify_by_bqb : bool, optional
            Whether to stratify by BQB (Biological Qualifier) terms in ontology analysis, by default True
        allow_col_multindex : bool, optional
            Whether to allow column multi-index in ontology analysis, by default False
        characteristic_only : bool, optional
            Whether to use only characteristic identifiers in ontology analysis (only supported for species), by default False
        dogmatic : bool, optional
            Whether to use dogmatic identifier filtering in ontology analysis, by default True
        priority_pathways : list[str], optional
            List of pathway IDs to prioritize in the source analysis, by default DEFAULT_PRIORITIZED_PATHWAYS

        Returns
        -------
        pd.DataFrame
            Co-occurrence matrix with ontologies as rows and sources as columns.
            Values represent the number of entities shared between each ontology-source pair.

        Raises
        ------
        ValueError
            If the entity type is invalid, identifiers are malformed, or source tables are empty

        Examples
        --------
        >>> # Get ontology × source co-occurrence for species
        >>> cooccurrence_matrix = sbml_dfs.get_ontology_x_source_cooccurrence(SBML_DFS.SPECIES)
        >>>
        >>> # Use characteristic species only
        >>> char_cooccurrence = sbml_dfs.get_ontology_x_source_cooccurrence(
        ...     SBML_DFS.SPECIES, characteristic_only=True
        ... )
        >>>
        >>> # Custom pathway priority
        >>> custom_cooccurrence = sbml_dfs.get_ontology_x_source_cooccurrence(
        ...     SBML_DFS.SPECIES, priority_pathways=['reactome', 'kegg']
        ... )
        """
        sources = self.get_source_occurrence(
            entity_type, priority_pathways=priority_pathways, include_missing=True
        )
        ontologies = self.get_ontology_occurrence(
            entity_type,
            stratify_by_bqb=stratify_by_bqb,
            allow_col_multindex=allow_col_multindex,
            characteristic_only=characteristic_only,
            dogmatic=dogmatic,
            include_missing=True,
        )

        ontologies_matrix = (ontologies > 0).astype(int)
        sources_matrix = (sources > 0).astype(int)

        # Calculate co-occurrence matrix: ontologies × sources
        # This gives us the number of entities shared between each ontology-source pair
        cooccurrences = ontologies_matrix.T @ sources_matrix

        return cooccurrences

    def get_summary(self) -> Mapping[str, Any]:
        """
        Get diagnostic statistics about the SBML_dfs.

        Returns
        -------
        Mapping[str, Any]
            Dictionary of diagnostic statistics including:
            - n_species_types: Number of species types
            - n_species_per_type: Number of species per type
            - n_entity_types: Dictionary of entity counts by type
            - dict_n_species_per_compartment: Number of species per compartment
            - stats_species_per_reactions: Statistics on reactands per reaction
            - top10_species_per_reactions: Top 10 reactions by number of reactands
            - sbo_name_counts: Count of reaction species by SBO term name
            - stats_degree: Statistics on species connectivity
            - top10_degree: Top 10 species by connectivity
            - species_ontology_counts: Count of species by ontology identifiers
            - data_summary: Summary of species and reaction data
        """
        stats: MutableMapping[str, Any] = {}

        # species_summaries
        species_features = self.get_species_features()
        stats["n_species_types"] = species_features["species_type"].nunique()
        stats["n_species_per_type"] = (
            species_features.groupby(by="species_type").size().to_dict()
        )

        # schema summaries
        stats["n_entity_types"] = {
            SBML_DFS.SPECIES: self.species.shape[0],
            SBML_DFS.REACTIONS: self.reactions.shape[0],
            SBML_DFS.COMPARTMENTS: self.compartments.shape[0],
            SBML_DFS.COMPARTMENTALIZED_SPECIES: self.compartmentalized_species.shape[0],
            SBML_DFS.REACTION_SPECIES: self.reaction_species.shape[0],
        }
        stats["dict_n_species_per_compartment"] = (
            self.compartmentalized_species.groupby(SBML_DFS.C_ID)
            .size()
            .rename("n_species")  # type: ignore
            .to_frame()
            .join(self.compartments[[SBML_DFS.C_NAME]])
            .reset_index(drop=False)
            .to_dict(orient="records")
        )

        # reaction summaries
        per_reaction_stats = self.reaction_species.groupby(SBML_DFS.R_ID).size()
        stats["stats_species_per_reactions"] = per_reaction_stats.describe().to_dict()
        stats["top10_species_per_reactions"] = (
            per_reaction_stats.sort_values(ascending=False)  # type: ignore
            .head(10)
            .rename("n_species")
            .to_frame()
            .join(self.reactions[[SBML_DFS.R_NAME]])
            .reset_index(drop=False)
            .to_dict(orient="records")
        )
        sbo_name_counts = self.reaction_species.value_counts(SBML_DFS.SBO_TERM)
        sbo_name_counts.index = sbo_name_counts.index.map(MINI_SBO_TO_NAME)
        stats["sbo_name_counts"] = sbo_name_counts.to_dict()

        # cspecies summaries
        cspecies_features = self.get_cspecies_features()
        stats["stats_degree"] = (
            cspecies_features[SBML_DFS_METHOD_DEFS.SC_DEGREE].describe().to_dict()
        )
        stats["top10_degree"] = (
            cspecies_features.sort_values(
                SBML_DFS_METHOD_DEFS.SC_DEGREE, ascending=False
            )
            .head(10)[
                [
                    SBML_DFS_METHOD_DEFS.SC_DEGREE,
                    SBML_DFS_METHOD_DEFS.SC_CHILDREN,
                    SBML_DFS_METHOD_DEFS.SC_PARENTS,
                    SBML_DFS_METHOD_DEFS.SPECIES_TYPE,
                ]
            ]
            .merge(
                self.compartmentalized_species[[SBML_DFS.S_ID, SBML_DFS.C_ID]],
                on=SBML_DFS.SC_ID,
            )
            .merge(self.compartments[[SBML_DFS.C_NAME]], on=SBML_DFS.C_ID)
            .merge(self.species[[SBML_DFS.S_NAME]], on=SBML_DFS.S_ID)
            .reset_index(drop=False)
            .to_dict(orient="records")
        )
        s_identifiers = sbml_dfs_utils.unnest_identifiers(
            self.species, SBML_DFS.S_IDENTIFIERS
        )
        stats["species_ontology_counts"] = s_identifiers.value_counts(
            IDENTIFIERS.ONTOLOGY
        ).to_dict()

        # data summaries
        stats["data_summary"] = self._get_data_summary()

        return stats

    def get_sbo_term_occurrence(
        self, name_terms=True, include_interactor_reactions=False
    ) -> pd.DataFrame:
        """
        Get the occurrence of SBO terms for reactions.

        Note: By default, reactions that consist entirely of interactor species will
        be excluded from the analysis. This is mandatory for most of the other occurrence and
        co-occurrence methods.

        Parameters
        ----------
        name_terms : bool, optional
            Whether to name the SBO terms, by default True
        include_interactor_reactions : bool, optional
            Whether to exclude interactor reactions, by default True
        """

        reaction_species = self.reaction_species
        if not include_interactor_reactions:
            # ignore reactions which are all interactors
            valid_reactions = self._get_non_interactor_reactions()
            reaction_species = reaction_species[
                reaction_species[SBML_DFS.R_ID].isin(valid_reactions.index.values)
            ]

        rxn_sbo_term_counts = reaction_species.pivot_table(
            index=SBML_DFS.R_ID, columns=SBML_DFS.SBO_TERM, aggfunc="size", fill_value=0
        )

        if name_terms:
            # map columns sbo_terms to sbo_term_names
            rxn_sbo_term_counts.columns = rxn_sbo_term_counts.columns.map(
                MINI_SBO_TO_NAME
            )

        return rxn_sbo_term_counts

    def get_sbo_term_x_source_cooccurrence(
        self,
        # Parameters from get_sbo_term_occurrence
        name_terms: bool = True,
        # Parameters from get_source_occurrence
        priority_pathways: list[str] = DEFAULT_PRIORITIZED_PATHWAYS,
    ) -> pd.DataFrame:
        """
        Get SBO term × source co-occurrence matrix for reactions.

        This method creates a co-occurrence matrix showing the relationship between
        SBO terms and sources (pathways) by calculating how many reactions are shared
        between each SBO term-source pair.

        The method combines SBO term occurrence data with source occurrence data to
        create a cross-tabulation matrix where:
        - Rows represent SBO terms
        - Columns represent sources/pathways
        - Values represent the number of reactions shared between each SBO term-source pair

        Note: Reactions that consist entirely of interactor species will be excluded
        from the analysis.

        Parameters
        ----------
        name_terms : bool, optional
            Whether to name the SBO terms using human-readable names, by default True
        priority_pathways : list[str], optional
            List of pathway IDs to prioritize in the source analysis, by default DEFAULT_PRIORITIZED_PATHWAYS

        Returns
        -------
        pd.DataFrame
            Co-occurrence matrix with SBO terms as rows and sources as columns.
            Values represent the number of reactions shared between each SBO term-source pair.

        Raises
        ------
        ValueError
            If source tables are empty

        Examples
        --------
        >>> # Get SBO term × source co-occurrence for reactions
        >>> cooccurrence_matrix = sbml_dfs.get_sbo_term_x_source_cooccurrence()
        >>>
        >>> # Use numeric SBO term codes instead of names
        >>> numeric_cooccurrence = sbml_dfs.get_sbo_term_x_source_cooccurrence(name_terms=False)
        """
        sources = self.get_source_occurrence(
            SBML_DFS.REACTIONS,
            priority_pathways=priority_pathways,
            include_missing=True,
        )
        sbo_terms = self.get_sbo_term_occurrence(
            name_terms=name_terms,
            include_interactor_reactions=False,
        )

        sbo_terms_matrix = (sbo_terms > 0).astype(int)
        sources_matrix = (sources > 0).astype(int)

        # Calculate co-occurrence matrix: SBO terms × sources
        # This gives us the number of reactions shared between each SBO term-source pair
        cooccurrences = sbo_terms_matrix.T @ sources_matrix

        return cooccurrences

    def get_source_cooccurrence(
        self,
        entity_type: str,
        priority_pathways: Optional[list[str]] = DEFAULT_PRIORITIZED_PATHWAYS,
    ) -> pd.DataFrame:
        """
        Get pathway co-occurrence matrix for a specific entity type.

        This method creates a co-occurrence matrix showing which pathways share entities
        of the specified type, indicating pathway relationships and overlaps.

        Note: When entity_type is 'reactions', reactions that consist entirely of interactor
        species will be excluded from the analysis.

        Parameters
        ----------
        entity_type : str
            The type of entity to analyze (e.g., 'species', 'reactions', 'compartments')
        priority_pathways : Optional[list[str]], default DEFAULT_PRIORITIZED_PATHWAYS
            List of pathway IDs to prioritize in the analysis. If None, uses all pathways
            without filtering or warnings.

        Returns
        -------
        pd.DataFrame
            Co-occurrence matrix with pathways as both rows and columns

        Raises
        ------
        ValueError
            If the source tables for the entity type are empty (indicating single-source model)
        """
        source_table = self.get_sources(entity_type)
        if source_table is None:
            raise ValueError(
                f"The Source tables for {entity_type} were empty, this indicates that the sbml_dfs is from a single source. "
                "Only sbml_dfs which have been merged with consensus should use this method."
            )

        filtered_sources = sbml_dfs_utils._select_priority_pathway_sources(
            source_table, priority_pathways
        )

        return sbml_dfs_utils._summarize_source_cooccurrence(filtered_sources)

    def get_source_occurrence(
        self,
        entity_type: str,
        priority_pathways: Optional[list[str]] = DEFAULT_PRIORITIZED_PATHWAYS,
        include_missing: bool = False,
        binarize: bool = False,
    ) -> pd.DataFrame:
        """
        Get pathway occurrence summary for a specific entity type.

        This method analyzes which pathways contain entities of the specified type,
        providing a summary of pathway occurrence patterns.

        Note: When entity_type is 'reactions', reactions that consist entirely of interactor
        species will be excluded from the analysis.

        Parameters
        ----------
        entity_type : str
            The type of entity to analyze (e.g., 'species', 'reactions', 'compartments')
        priority_pathways : Optional[list[str]], default DEFAULT_PRIORITIZED_PATHWAYS
            List of pathway IDs to prioritize in the analysis. If None, uses all pathways
            without filtering or warnings.
        include_missing: bool, optional
            Whether to include missing entities in the result using add_missing_ids_column, by default False
        binarize: bool, optional
            Whether to convert the result to binary values (0 vs 1+), by default False

        Returns
        -------
        pd.DataFrame
            Summary of pathway occurrence patterns. If binarize=True, values are 0 or 1.

        Raises
        ------
        ValueError
            If the source tables for the entity type are empty (indicating single-source model)
        """
        source_table = self.get_sources(entity_type)
        if source_table is None:
            raise ValueError(
                f"The Source tables for {entity_type} were empty, this indicates that the sbml_dfs is from a single source. "
                "Only sbml_dfs which have been merged with consensus should use this method."
            )

        filtered_sources = sbml_dfs_utils._select_priority_pathway_sources(
            source_table, priority_pathways
        )

        result = sbml_dfs_utils._summarize_source_occurrence(filtered_sources, binarize)

        if include_missing:
            # Get the reference table (all entities of this type)
            if entity_type == SBML_DFS.REACTIONS:
                reference_table = self._get_non_interactor_reactions()
            else:
                reference_table = self.get_table(entity_type, {SCHEMA_DEFS.ID})

            result = sbml_dfs_utils.add_missing_ids_column(result, reference_table)

        return result

    def get_sources(self, entity_type: str) -> pd.DataFrame | None:
        """
        Get the unnest sources table for a given entity type.

        Parameters
        ----------
        entity_type : str
            The type of entity to get sources for (e.g., 'species', 'reactions')

        Returns
        -------
        pd.DataFrame | None
            DataFrame containing the unnest sources table, or None if no sources found

        Raises
        ------
        ValueError
            If entity_type is invalid or does not have a source attribute
        """
        # Validate that the entity_type exists in the schema
        if entity_type not in SBML_DFS_SCHEMA.SCHEMA:
            raise ValueError(
                f"{entity_type} is not a valid entity type. "
                f"Valid types are: {', '.join(SBML_DFS_SCHEMA.SCHEMA.keys())}"
            )

        # Check if the entity_type has a source attribute
        entity_schema = SBML_DFS_SCHEMA.SCHEMA[entity_type]
        if SCHEMA_DEFS.SOURCE not in entity_schema:
            raise ValueError(f"{entity_type} does not have a source attribute")

        if entity_type == SBML_DFS.REACTIONS:
            entity_table = self._get_non_interactor_reactions()
        else:
            entity_table = self.get_table(entity_type)

        all_sources_table = source.unnest_sources(entity_table)

        return all_sources_table

    def get_source_total_counts(self, entity_type: str) -> pd.Series:
        """
        Get the total counts of each source for a given entity type.

        Parameters
        ----------
        entity_type : str
            The type of entity to get the total counts of (e.g., 'species', 'reactions')

        Returns
        -------
        pd.Series
            Series containing the total counts of each source, indexed by pathway_id

        Raises
        ------
        ValueError
            If entity_type is invalid
        """
        all_sources_table = self.get_sources(entity_type)

        if all_sources_table is None:
            logger.warning(
                f"No sources found for {entity_type} in sbml_dfs. Returning an empty series."
            )
            return pd.Series([], name="total_counts")

        source_total_counts = all_sources_table.value_counts(
            source.SOURCE_SPEC.PATHWAY_ID
        ).rename("total_counts")

        return source_total_counts

    def get_species_features(self) -> pd.DataFrame:
        """
        Get additional attributes of species.

        Returns
        -------
        pd.DataFrame
            Species with additional features including:
            - species_type: Classification of the species (e.g., metabolite, protein)
        """
        species = self.species
        augmented_species = species.assign(
            **{
                SBML_DFS_METHOD_DEFS.SPECIES_TYPE: lambda d: d[
                    SBML_DFS.S_IDENTIFIERS
                ].apply(sbml_dfs_utils.species_type_types)
            }
        )

        return augmented_species

    def get_table(
        self, entity_type: str, required_attributes: None | set[str] = None
    ) -> pd.DataFrame:
        """
        Get a table from the SBML_dfs object with optional attribute validation.

        Parameters
        ----------
        entity_type : str
            The type of entity table to retrieve (e.g., 'species', 'reactions')
        required_attributes : Optional[Set[str]], optional
            Set of attributes that must be present in the table, by default None.
            Must be passed as a set, e.g. {'id'}, not a string.

        Returns
        -------
        pd.DataFrame
            The requested table

        Raises
        ------
        ValueError
            If entity_type is invalid or required attributes are missing
        TypeError
            If required_attributes is not a set
        """

        schema = self.schema

        if entity_type not in schema.keys():
            raise ValueError(
                f"{entity_type} does not match a table in the SBML_dfs object. The tables "
                f"which are present are {', '.join(schema.keys())}"
            )

        if required_attributes is not None:
            if not isinstance(required_attributes, set):
                raise TypeError(
                    f"required_attributes must be a set (e.g. {{'id'}}), but got {type(required_attributes).__name__}. "
                    "Did you pass a string instead of a set?"
                )

            # determine whether required_attributes are appropriate
            VALID_REQUIRED_ATTRIBUTES = {
                SCHEMA_DEFS.ID,
                SCHEMA_DEFS.SOURCE,
                SCHEMA_DEFS.LABEL,
            }
            invalid_required_attributes = required_attributes.difference(
                VALID_REQUIRED_ATTRIBUTES
            )

            if len(invalid_required_attributes) > 0:
                raise ValueError(
                    f"The following required attributes are not valid: {', '.join(invalid_required_attributes)}. "
                    f"Requiered attributes must be a subset of {', '.join(VALID_REQUIRED_ATTRIBUTES)}"
                )

            # determine if required_attributes are satisified
            invalid_attrs = [
                s for s in required_attributes if s not in schema[entity_type].keys()
            ]
            if len(invalid_attrs) > 0:
                raise ValueError(
                    f"The following required attributes are not present for the {entity_type} table: "
                    f"{', '.join(invalid_attrs)}."
                )

        return getattr(self, entity_type)

    def get_uri_urls(
        self,
        entity_type: str,
        entity_ids: Iterable[str] | None = None,
        required_ontology: str | None = None,
    ) -> pd.Series:
        """
        Get reference URLs for specified entities.

        Parameters
        ----------
        entity_type : str
            Type of entity to get URLs for (e.g., 'species', 'reactions')
        entity_ids : Optional[Iterable[str]], optional
            Specific entities to get URLs for, by default None (all entities)
        required_ontology : Optional[str], optional
            Specific ontology to get URLs from, by default None

        Returns
        -------
        pd.Series
            Series mapping entity IDs to their reference URLs

        Raises
        ------
        ValueError
            If entity_type is invalid
        """
        schema = self.schema

        # valid entities and their identifier variables
        valid_entity_types = [
            SBML_DFS.COMPARTMENTS,
            SBML_DFS.SPECIES,
            SBML_DFS.REACTIONS,
        ]

        if entity_type not in valid_entity_types:
            raise ValueError(
                f"{entity_type} is an invalid entity_type; valid types "
                f"are {', '.join(valid_entity_types)}"
            )

        entity_table = getattr(self, entity_type)

        if entity_ids is not None:
            # ensure that entity_ids are unique and then convert back to list
            # to support pandas indexing
            entity_ids = list(set(entity_ids))

            # filter to a subset of identifiers if one is provided
            entity_table = entity_table.loc[entity_ids]

        # create a dataframe of all identifiers for the select entities
        # Use unnest_identifiers for efficient vectorized operation
        all_ids = sbml_dfs_utils.unnest_identifiers(
            entity_table, schema[entity_type][SCHEMA_DEFS.ID]
        ).reset_index()

        # Rename the entity ID column to match the schema
        entity_id_col = entity_table.index.name or entity_table.index.names[0]
        all_ids = all_ids.rename(
            columns={entity_id_col: schema[entity_type][SCHEMA_DEFS.PK]}
        )

        # set priorities for ontologies and bqb terms

        if required_ontology is None:
            all_ids = all_ids.merge(BQB_PRIORITIES, how="left").merge(
                ONTOLOGY_PRIORITIES, how="left"
            )
        else:
            ontology_priorities = pd.DataFrame(
                [{IDENTIFIERS.ONTOLOGY: required_ontology, "ontology_rank": 1}]
            )
            # if only a single ontology is sought then just return matching entries
            all_ids = all_ids.merge(BQB_PRIORITIES, how="left").merge(
                ontology_priorities, how="inner"
            )

        uri_urls = (
            all_ids.sort_values(["bqb_rank", "ontology_rank", IDENTIFIERS.URL])
            .groupby(schema[entity_type][SCHEMA_DEFS.PK])
            .first()[IDENTIFIERS.URL]
        )
        return uri_urls

    def infer_sbo_terms(self):
        """
        Infer SBO Terms

        Define SBO terms based on stoichiometry for reaction_species with missing terms.
        Modifies the SBML_dfs object in-place.

        Returns
        -------
        None (modifies SBML_dfs object in-place)
        """
        valid_sbo_terms = self.reaction_species[
            self.reaction_species[SBML_DFS.SBO_TERM].isin(MINI_SBO_TO_NAME.keys())
        ]

        invalid_sbo_terms = self.reaction_species[
            ~self.reaction_species[SBML_DFS.SBO_TERM].isin(MINI_SBO_TO_NAME.keys())
        ]

        if not all(self.reaction_species[SBML_DFS.SBO_TERM].notnull()):
            raise ValueError("All reaction_species[SBML_DFS.SBO_TERM] must be not null")
        if invalid_sbo_terms.shape[0] == 0:
            logger.info("All sbo_terms were valid; nothing to update.")
            return

        logger.info(f"Updating {invalid_sbo_terms.shape[0]} reaction_species' sbo_term")

        # add missing/invalid terms based on stoichiometry
        invalid_sbo_terms.loc[
            invalid_sbo_terms[SBML_DFS.STOICHIOMETRY] < 0, SBML_DFS.SBO_TERM
        ] = MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT]

        invalid_sbo_terms.loc[
            invalid_sbo_terms[SBML_DFS.STOICHIOMETRY] > 0, SBML_DFS.SBO_TERM
        ] = MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT]

        invalid_sbo_terms.loc[
            invalid_sbo_terms[SBML_DFS.STOICHIOMETRY] == 0, SBML_DFS.SBO_TERM
        ] = MINI_SBO_FROM_NAME[SBOTERM_NAMES.STIMULATOR]

        updated_reaction_species = pd.concat(
            [valid_sbo_terms, invalid_sbo_terms]
        ).sort_index()

        if self.reaction_species.shape[0] != updated_reaction_species.shape[0]:
            raise ValueError(
                f"Trying to overwrite {self.reaction_species.shape[0]} reaction_species with {updated_reaction_species.shape[0]}"
            )
        self.reaction_species = updated_reaction_species
        return

    def infer_uncompartmentalized_species_location(self):
        """
        Infer Uncompartmentalized Species Location

        If the compartment of a subset of compartmentalized species
        was not specified, infer an appropriate compartment from
        other members of reactions they participate in.

        This method modifies the SBML_dfs object in-place.

        Returns
        -------
        None (modifies SBML_dfs object in-place)
        """
        default_compartment = (
            self.compartmentalized_species.value_counts(SBML_DFS.C_ID)
            .rename("N")
            .reset_index()
            .sort_values("N", ascending=False)[SBML_DFS.C_ID][0]
        )
        if not isinstance(default_compartment, str):
            raise ValueError(
                "No default compartment could be found - compartment "
                "information may not be present"
            )

        # infer the compartments of species missing compartments
        missing_compartment_scids = self.compartmentalized_species[
            self.compartmentalized_species[SBML_DFS.C_ID].isnull()
        ].index.tolist()
        if len(missing_compartment_scids) == 0:
            logger.info(
                "All compartmentalized species have compartments, "
                "returning input SBML_dfs"
            )
            return self

        participating_reactions = (
            self.reaction_species[
                self.reaction_species[SBML_DFS.SC_ID].isin(missing_compartment_scids)
            ][SBML_DFS.R_ID]
            .unique()
            .tolist()
        )
        reaction_participants = self.reaction_species[
            self.reaction_species[SBML_DFS.R_ID].isin(participating_reactions)
        ].reset_index(drop=True)[[SBML_DFS.SC_ID, SBML_DFS.R_ID]]
        reaction_participants = reaction_participants.merge(
            self.compartmentalized_species[SBML_DFS.C_ID],
            left_on=SBML_DFS.SC_ID,
            right_index=True,
        )

        # find a default compartment to fall back on if all compartmental information is missing
        primary_reaction_compartment = (
            reaction_participants.value_counts([SBML_DFS.R_ID, SBML_DFS.C_ID])
            .rename("N")
            .reset_index()
            .sort_values("N", ascending=False)
            .groupby(SBML_DFS.R_ID)
            .first()[SBML_DFS.C_ID]
            .reset_index()
        )

        inferred_compartmentalization = (
            self.reaction_species[
                self.reaction_species[SBML_DFS.SC_ID].isin(missing_compartment_scids)
            ]
            .merge(primary_reaction_compartment)
            .value_counts([SBML_DFS.SC_ID, SBML_DFS.C_ID])
            .rename("N")
            .reset_index()
            .sort_values("N", ascending=False)
            .groupby(SBML_DFS.SC_ID)
            .first()
            .reset_index()[[SBML_DFS.SC_ID, SBML_DFS.C_ID]]
        )
        logger.info(
            f"{inferred_compartmentalization.shape[0]} species' compartmentalization inferred"
        )

        # define where a reaction is most likely to occur based on the compartmentalization of its participants
        species_with_unknown_compartmentalization = set(
            missing_compartment_scids
        ).difference(set(inferred_compartmentalization[SBML_DFS.SC_ID].tolist()))
        if len(species_with_unknown_compartmentalization) != 0:
            logger.warning(
                f"{len(species_with_unknown_compartmentalization)} "
                "species compartmentalization could not be inferred"
                " from other reaction participants. Their compartmentalization "
                f"will be set to the default of {default_compartment}"
            )

            inferred_compartmentalization = pd.concat(
                [
                    inferred_compartmentalization,
                    pd.DataFrame(
                        {
                            SBML_DFS.SC_ID: list(
                                species_with_unknown_compartmentalization
                            )
                        }
                    ).assign(c_id=default_compartment),
                ]
            )

        if len(missing_compartment_scids) != inferred_compartmentalization.shape[0]:
            raise ValueError(
                f"{inferred_compartmentalization.shape[0]} were inferred but {len(missing_compartment_scids)} are required"
            )

        updated_compartmentalized_species = pd.concat(
            [
                self.compartmentalized_species[
                    ~self.compartmentalized_species[SBML_DFS.C_ID].isnull()
                ],
                self.compartmentalized_species[
                    self.compartmentalized_species[SBML_DFS.C_ID].isnull()
                ]
                .drop(SBML_DFS.C_ID, axis=1)
                .merge(
                    inferred_compartmentalization,
                    left_index=True,
                    right_on=SBML_DFS.SC_ID,
                )
                .set_index(SBML_DFS.SC_ID),
            ]
        )

        if (
            updated_compartmentalized_species.shape[0]
            != self.compartmentalized_species.shape[0]
        ):
            raise ValueError(
                f"Trying to overwrite {self.compartmentalized_species.shape[0]}"
                " compartmentalized species with "
                f"{updated_compartmentalized_species.shape[0]}"
            )

        if any(updated_compartmentalized_species[SBML_DFS.C_ID].isnull()):
            raise ValueError("Some species compartments are still missing")

        self.compartmentalized_species = updated_compartmentalized_species
        return

    def name_compartmentalized_species(self):
        """
        Name Compartmentalized Species

        Rename compartmentalized species if they have the same
        name as their species. Modifies the SBML_dfs object in-place.

        Returns
        -------
        None (modifies SBML_dfs object in-place)
        """
        augmented_cspecies = self.compartmentalized_species.merge(
            self.species[SBML_DFS.S_NAME], left_on=SBML_DFS.S_ID, right_index=True
        ).merge(
            self.compartments[SBML_DFS.C_NAME], left_on=SBML_DFS.C_ID, right_index=True
        )
        augmented_cspecies[SBML_DFS.SC_NAME] = [
            f"{s} [{c}]" if sc == s else sc
            for sc, c, s in zip(
                augmented_cspecies[SBML_DFS.SC_NAME],
                augmented_cspecies[SBML_DFS.C_NAME],
                augmented_cspecies[SBML_DFS.S_NAME],
            )
        ]

        self.compartmentalized_species = augmented_cspecies.loc[
            :, self.schema[SBML_DFS.COMPARTMENTALIZED_SPECIES][SCHEMA_DEFS.VARS]
        ]
        return

    def post_consensus_checks(
        self,
        entity_types: list[str] = [SBML_DFS.SPECIES, SBML_DFS.COMPARTMENTS],
        check_types: list[str] = [
            CONSENSUS_CHECKS.SOURCE_COOCCURRENCE,
            CONSENSUS_CHECKS.ONTOLOGY_X_SOURCE_COOCCURRENCE,
        ],
    ) -> None:
        """
        Post-consensus checks

        Perform checks on the SBML_dfs object after consensus building.

        Parameters
        ----------
        entity_types: list[str], optional
            Entity types to check
        check_types: list[str], optional
            Check types to perform

        Returns
        -------
        None
        """

        invalid_checks = set(check_types).difference(CONSENSUS_CHECKS_LIST)
        if len(invalid_checks) > 0:
            raise ValueError(f"Invalid check types: {invalid_checks}")

        checks_results = {}
        for entity_type in entity_types:
            checks_results[entity_type] = {}
            for check_type in check_types:
                if check_type == CONSENSUS_CHECKS.SOURCE_COOCCURRENCE:
                    cooccurrences = self.get_source_cooccurrence(entity_type)
                elif check_type == CONSENSUS_CHECKS.ONTOLOGY_X_SOURCE_COOCCURRENCE:
                    cooccurrences = self.get_ontology_x_source_cooccurrence(entity_type)

                checks_results[entity_type][check_type] = cooccurrences

        return checks_results

    def reaction_formulas(
        self, r_ids: Optional[Union[str, list[str]]] = None
    ) -> pd.Series:
        """
        Reaction Summary

        Return human-readable formulas for reactions.

        Parameters:
        ----------
        r_ids: [str], str or None
            Reaction IDs or None for all reactions

        Returns
        ----------
        formula_strs: pd.Series
        """

        validated_rids = self._validate_r_ids(r_ids)

        matching_reaction_species = self.reaction_species[
            self.reaction_species.r_id.isin(validated_rids)
        ].merge(
            self.compartmentalized_species, left_on=SBML_DFS.SC_ID, right_index=True
        )

        # split into within compartment and cross-compartment reactions
        r_id_compartment_counts = matching_reaction_species.groupby(SBML_DFS.R_ID)[
            SBML_DFS.C_ID
        ].nunique()

        # Process cross-compartment reactions (use sc_name for species identification)
        cross_compartment_rids = r_id_compartment_counts[
            r_id_compartment_counts > 1
        ].index
        cross_compartment_data = matching_reaction_species[
            matching_reaction_species[SBML_DFS.R_ID].isin(cross_compartment_rids)
        ]

        rxn_eqtn_cross_compartment = sbml_dfs_utils.create_reaction_formula_series(
            cross_compartment_data,
            self.reactions,
            species_name_col=SBML_DFS.SC_NAME,
            sort_cols=[SBML_DFS.SC_NAME],
            group_cols=[SBML_DFS.R_ID],
            r_id_col=SBML_DFS.R_ID,
            c_name_col=SBML_DFS.C_NAME,
        )

        # Process within-compartment reactions (use s_name and add compartment prefix)
        within_compartment_rids = r_id_compartment_counts[
            r_id_compartment_counts == 1
        ].index
        within_compartment_data = matching_reaction_species[
            matching_reaction_species[SBML_DFS.R_ID].isin(within_compartment_rids)
        ]

        if within_compartment_data.shape[0] > 0:
            # Augment with additional data needed for s_name
            within_compartment_data = within_compartment_data.merge(
                self.compartments, left_on=SBML_DFS.C_ID, right_index=True
            ).merge(self.species, left_on=SBML_DFS.S_ID, right_index=True)

            rxn_eqtn_within_compartment = sbml_dfs_utils.create_reaction_formula_series(
                within_compartment_data,
                self.reactions,
                species_name_col=SBML_DFS.S_NAME,
                sort_cols=[SBML_DFS.S_NAME],
                group_cols=[SBML_DFS.R_ID, SBML_DFS.C_NAME],
                add_compartment_prefix=True,
                r_id_col=SBML_DFS.R_ID,
                c_name_col=SBML_DFS.C_NAME,
            )
        else:
            rxn_eqtn_within_compartment = None

        # Combine results
        formula_strs = pd.concat(
            [rxn_eqtn_cross_compartment, rxn_eqtn_within_compartment]
        )

        return formula_strs

    def reaction_summaries(
        self, r_ids: Optional[Union[str, list[str]]] = None
    ) -> pd.DataFrame:
        """
        Reaction Summary

        Return a summary of reactions.

        Parameters:
        ----------
        r_ids: [str], str or None
            Reaction IDs or None for all reactions

        Returns
        ----------
        reaction_summaries_df: pd.DataFrame
            A table with r_id as an index and columns:
            - r_name: str, name of the reaction
            - r_formula_str: str, human-readable formula of the reaction
        """

        validated_rids = self._validate_r_ids(r_ids)

        participating_r_names = self.reactions.loc[validated_rids, SBML_DFS.R_NAME]
        participating_r_formulas = self.reaction_formulas(r_ids=validated_rids)
        reaction_summareis_df = pd.concat(
            [participating_r_names, participating_r_formulas], axis=1
        )

        return reaction_summareis_df

    def remove_entities(
        self,
        entity_type: str,
        entity_ids: Iterable[str],
        remove_references: bool = True,
    ):
        """Public method to remove entities and optionally clean up orphaned references.

        Special handling for "cofactors" where literal cleanup of reactions based on reaction_species is allowed
        normally, removing substrates/products would remove the reaction.

        Parameters
        ----------
        entity_type : str
            The entity type (e.g., 'reactions', 'compartmentalized_species', 'species', 'compartments', or "cofactors")
        entity_ids : Iterable[str]
            IDs of entities to remove
        remove_references : bool, default True
            Whether to remove orphaned references after entity removal
        """

        entity_ids = list(entity_ids)
        if not entity_ids:
            return

        # Find all entities that need to be removed (including cascading references)
        if remove_references:
            to_be_removed_entities = self.find_entity_references(
                entity_type, entity_ids
            )
        else:
            logger.warning(
                "Removing entities without removing references, because remove_references=False. This may result in a validation error."
            )
            # Only remove the directly requested entities
            to_be_removed_entities = {entity_type: set(entity_ids)}

        # now treat entity_type as actual table to remove if this was "cofactors"
        if entity_type == "cofactors":
            entity_type = SBML_DFS.REACTION_SPECIES

        # iterate through to-be-removed and remove
        logger.info(f"Removing the requested {len(entity_ids)} {entity_type} entities")
        self._remove_entities_direct(entity_type, entity_ids)

        # these would be entries of the requested table which are indirectly removed
        # e.g., if we request to remove the substrate of a reaction, it removes the reaction and the products
        # in turn which may result in removal of the cspecies if it isn't used elsewhere
        to_be_removed_entities[entity_type] = to_be_removed_entities[entity_type] - set(
            entity_ids
        )
        for k, v in to_be_removed_entities.items():
            if not v:
                continue
            logger.info(f"Removing {len(v)} orphaned {k}")
            self._remove_entities_direct(k, list(v))

    def remove_reactions_data(self, label: str):
        """
        Remove reactions data by label.
        """
        self._remove_entity_data(SBML_DFS.REACTIONS, label)

    def remove_species_data(self, label: str):
        """
        Remove species data by label.
        """
        self._remove_entity_data(SBML_DFS.SPECIES, label)

    def remove_unused(self) -> None:
        """
        Find and remove unused entities from the model.

        This method identifies unused entities using find_unused_entities and
        then cleans them up using the existing remove_entities method which
        properly handles cleanup of species_data and reactions_data as needed.

        Returns
        -------
        None
            Modifies the SBML_dfs object in-place
        """

        unused_entities = sbml_dfs_utils.find_unused_entities(self)

        for k, v in unused_entities.items():
            self.remove_entities(k, v, remove_references=False)

        return None

    def search_by_ids(
        self,
        id_table: pd.DataFrame,
        identifiers: Optional[Union[str, list, set]] = None,
        ontologies: Optional[Union[str, list, set]] = None,
        bqbs: Optional[Union[str, list, set]] = BQB_DEFINING_ATTRS_LOOSE
        + [BQB.HAS_PART],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Find entities and identifiers matching a set of query IDs.

        Parameters
        ----------
        id_table : pd.DataFrame
            DataFrame containing identifier mappings
        identifiers : Optional[Union[str, list, set]], optional
            Identifiers to filter by, by default None
        ontologies : Optional[Union[str, list, set]], optional
            Ontologies to filter by, by default None
        bqbs : Optional[Union[str, list, set]], optional
            BQB terms to filter by, by default [BQB.IS, BQB.HAS_PART]

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            - Matching entities
            - Matching identifiers

        Raises
        ------
        ValueError
            If entity_type is invalid or ontologies are invalid
        TypeError
            If ontologies is not a set
        """
        # validate inputs

        entity_type = utils.infer_entity_type(id_table)
        entity_table = self.get_table(entity_type, required_attributes={SCHEMA_DEFS.ID})
        entity_pk = self.schema[entity_type][SCHEMA_DEFS.PK]

        matching_identifiers = id_tables.filter_id_table(
            id_table=id_table, identifiers=identifiers, ontologies=ontologies, bqbs=bqbs
        )

        matching_keys = matching_identifiers[entity_pk].tolist()
        entity_subset = entity_table.loc[matching_keys]

        if matching_identifiers.shape[0] != entity_subset.shape[0]:
            raise ValueError(
                f"Some identifiers did not match to an entity for {entity_type}. "
                "This suggests that the identifiers and sbml_dfs are not in sync. "
                "Please create new identifiers with sbml_dfs.get_characteristic_species_ids() "
                "or sbml_dfs.get_identifiers()."
            )

        return entity_subset, matching_identifiers

    def search_by_name(
        self, name: str, entity_type: str, partial_match: bool = True
    ) -> pd.DataFrame:
        """
        Find entities by exact or partial name match.

        Parameters
        ----------
        name : str
            Name to search for
        entity_type : str
            Type of entity to search (e.g., 'species', 'reactions')
        partial_match : bool, optional
            Whether to allow partial string matches, by default True

        Returns
        -------
        pd.DataFrame
            Matching entities
        """
        entity_table = self.get_table(
            entity_type, required_attributes={SCHEMA_DEFS.LABEL}
        )
        label_attr = self.schema[entity_type][SCHEMA_DEFS.LABEL]

        if partial_match:
            matches = entity_table.loc[
                entity_table[label_attr].str.contains(name, case=False)
            ]
        else:
            matches = entity_table.loc[entity_table[label_attr].str.lower() == name]
        return matches

    def select_species_data(self, species_data_table: str) -> pd.DataFrame:
        """
        Select a species data table from the SBML_dfs object.

        Parameters
        ----------
        species_data_table : str
            Name of the species data table to select

        Returns
        -------
        pd.DataFrame
            The selected species data table

        Raises
        ------
        ValueError
            If species_data_table is not found
        """
        # Check if species_data_table exists in sbml_dfs.species_data
        if species_data_table not in self.species_data:
            raise ValueError(
                f"species_data_table {species_data_table} not found in sbml_dfs.species_data. "
                f"Available tables: {self.species_data.keys()}"
            )

        # Get the species data
        return self.species_data[species_data_table]

    def show_summary(self) -> None:
        """
        Display a formatted summary of the SBML_dfs model.

        This method chains together get_summary(), format_sbml_dfs_summary(),
        and utils.show() to provide a convenient way to display network statistics.

        Returns
        -------
        None
            Displays the formatted summary table to console

        Examples
        --------
        >>> sbml_dfs.show_network_summary()
        """
        summary_stats = self.get_summary()
        summary_table = sbml_dfs_utils.format_sbml_dfs_summary(summary_stats)
        utils.show(summary_table, max_rows=50)

    def species_status(self, s_id: str) -> pd.DataFrame:
        """
        Species Status

        Return all of the reactions a species participates in.

        Parameters:
        s_id: str
            A species ID

        Returns:
        pd.DataFrame, one row per reaction the species participates in
        with columns:
        - sc_name: str, name of the compartment the species participates in
        - stoichiometry: float, stoichiometry of the species in the reaction
        - r_name: str, name of the reaction
        - r_formula_str: str, human-readable formula of the reaction
        """

        if s_id not in self.species.index:
            raise ValueError(f"{s_id} not found in species table")

        matching_species = self.species.loc[s_id]

        if not isinstance(matching_species, pd.Series):
            raise ValueError(f"{s_id} did not match a single species")

        # find all rxns species participate in
        matching_compartmentalized_species = self.compartmentalized_species[
            self.compartmentalized_species.s_id.isin([s_id])
        ]

        rxns_participating = self.reaction_species[
            self.reaction_species.sc_id.isin(matching_compartmentalized_species.index)
        ]

        # find all participants in these rxns
        full_rxns_participating = self.reaction_species[
            self.reaction_species.r_id.isin(rxns_participating[SBML_DFS.R_ID])
        ].merge(
            self.compartmentalized_species, left_on=SBML_DFS.SC_ID, right_index=True
        )

        participating_rids = full_rxns_participating[SBML_DFS.R_ID].unique()
        reaction_descriptions = self.reaction_summaries(r_ids=participating_rids)

        status = (
            full_rxns_participating.loc[
                full_rxns_participating[SBML_DFS.SC_ID].isin(
                    matching_compartmentalized_species.index.values.tolist()
                ),
                [SBML_DFS.SC_NAME, SBML_DFS.STOICHIOMETRY, SBML_DFS.R_ID],
            ]
            .merge(reaction_descriptions, left_on=SBML_DFS.R_ID, right_index=True)
            .reset_index(drop=True)
            .drop(SBML_DFS.R_ID, axis=1)
        )

        return status

    def to_dict(self) -> dict[str, pd.DataFrame]:
        """
        Return the 5 major SBML_dfs tables as a dictionary.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary containing the core SBML_dfs tables:
            - 'compartments': Compartments table
            - 'species': Species table
            - 'compartmentalized_species': Compartmentalized species table
            - 'reactions': Reactions table
            - 'reaction_species': Reaction species table
        """
        return {
            SBML_DFS.COMPARTMENTS: self.compartments,
            SBML_DFS.SPECIES: self.species,
            SBML_DFS.COMPARTMENTALIZED_SPECIES: self.compartmentalized_species,
            SBML_DFS.REACTIONS: self.reactions,
            SBML_DFS.REACTION_SPECIES: self.reaction_species,
        }

    def to_pickle(self, path: str) -> None:
        """
        Save the SBML_dfs to a pickle file.

        Parameters
        ----------
        path : str
            Path where to save the pickle file
        """
        utils.save_pickle(path, self)

    def validate(self):
        """
        Validate the SBML_dfs structure and relationships.

        Checks:
        - Schema existence
        - Required tables presence
        - Individual table structure
        - Primary key uniqueness
        - Foreign key relationships
        - Optional data table validity
        - Reaction species validity

        Raises
        ------
        ValueError
            If any validation check fails
        """

        if not hasattr(self, "schema"):
            raise ValueError("No schema found")

        required_tables = self._required_entities
        schema_tables = set(self.schema.keys())

        extra_tables = schema_tables.difference(required_tables)
        if len(extra_tables) != 0:
            logger.debug(
                f"{len(extra_tables)} unexpected tables found: "
                f"{', '.join(extra_tables)}"
            )

        missing_tables = required_tables.difference(schema_tables)
        if len(missing_tables) != 0:
            raise ValueError(
                f"Missing {len(missing_tables)} required tables: "
                f"{', '.join(missing_tables)}"
            )

        # check individual tables
        for table in required_tables:
            self._validate_table(table)

        # check whether pks and fks agree (bidirectional)
        self._validate_pk_fk_correspondence()

        # check optional data tables:
        for k, v in self.species_data.items():
            try:
                self._validate_species_data(v)
            except ValueError as e:
                raise ValueError(f"species data {k} was invalid.") from e

        for k, v in self.reactions_data.items():
            try:
                self._validate_reactions_data(v)
            except ValueError as e:
                raise ValueError(f"reactions data {k} was invalid.") from e

        # validate reaction_species sbo_terms and stoi
        self._validate_reaction_species()

        # validate identifiers and sources
        self._validate_identifiers()
        self._validate_sources()

    def validate_and_resolve(self):
        """
        Validate and attempt to automatically fix common issues.

        This method iteratively:
        1. Attempts validation
        2. If validation fails, tries to resolve the issue
        3. Repeats until validation passes or issue cannot be resolved

        Raises
        ------
        ValueError
            If validation fails and cannot be automatically resolved
        """

        current_exception = None
        validated = False

        while not validated:
            try:
                self.validate()
                validated = True
            except Exception as e:
                e_str = str(e)
                if e_str == current_exception:
                    logger.warning(
                        "Automated resolution of an Exception was attempted but failed"
                    )
                    raise e

                # try to resolve
                self._attempt_resolve(e)

    # =============================================================================
    # PRIVATE METHODS (ALPHABETICAL ORDER)
    # =============================================================================

    def _attempt_resolve(self, e):
        str_e = str(e)
        if str_e == "compartmentalized_species included missing c_id values":
            logger.warning(str_e)
            logger.warning(
                "Attempting to resolve with infer_uncompartmentalized_species_location()"
            )
            self.infer_uncompartmentalized_species_location()
        elif re.search("sbo_terms were not defined", str_e):
            logger.warning(str_e)
            logger.warning("Attempting to resolve with infer_sbo_terms()")
            self.infer_sbo_terms()
        elif re.search("Referential completeness violation", str_e):
            logger.warning(str_e)
            logger.warning("Attempting to resolve with remove_unused()")
            self.remove_unused()
        else:
            logger.warning(
                "An error occurred which could not be automatically resolved"
            )
            raise e

    @classmethod
    def _edgelist_assemble_sbml_model(
        cls,
        compartments: pd.DataFrame,
        species: pd.DataFrame,
        comp_species: pd.DataFrame,
        reactions: pd.DataFrame,
        reaction_species: pd.DataFrame,
        species_data: pd.DataFrame,
        reactions_data: pd.DataFrame,
        keep_species_data: Union[bool, str],
        keep_reactions_data: Union[bool, str],
        extra_columns: dict[str, list[str]],
        model_source: source.Source,
    ) -> "SBML_dfs":
        """
        Assemble the final SBML_dfs object.

        Parameters
        ----------
        compartments : pd.DataFrame
            Processed compartments data
        species : pd.DataFrame
            Processed species data
        comp_species : pd.DataFrame
            Compartmentalized species data
        reactions : pd.DataFrame
            Reactions data
        reaction_species : pd.DataFrame
            Reaction species relationships
        species_data : pd.DataFrame
            Extra species data to include
        reactions_data : pd.DataFrame
            Extra reactions data to include
        keep_species_data : bool or str
            Label for species extra data
        keep_reactions_data : bool or str
            Label for reactions extra data
        extra_columns : dict
            Dictionary containing lists of extra column names

        Returns
        -------
        SBML_dfs
            Validated SBML data structure
        """
        sbml_tbl_dict = {
            SBML_DFS.COMPARTMENTS: compartments,
            SBML_DFS.SPECIES: species,
            SBML_DFS.COMPARTMENTALIZED_SPECIES: comp_species,
            SBML_DFS.REACTIONS: reactions,
            SBML_DFS.REACTION_SPECIES: reaction_species,
        }

        # Add extra data if requested
        if len(extra_columns[SBML_DFS.REACTIONS]) > 0:
            data_label = (
                keep_reactions_data
                if isinstance(keep_reactions_data, str)
                else "source"
            )
            sbml_tbl_dict[SBML_DFS.REACTIONS_DATA] = {data_label: reactions_data}

        if len(extra_columns[SBML_DFS.SPECIES]) > 0:
            data_label = (
                keep_species_data if isinstance(keep_species_data, str) else "source"
            )
            sbml_tbl_dict[SBML_DFS.SPECIES_DATA] = {data_label: species_data}

        sbml_dfs = cls(sbml_tbl_dict, model_source)
        sbml_dfs.validate()

        return sbml_dfs

    def _find_underspecified_reactions_by_reference(
        self, reference_type: str, reference_ids: set[str]
    ) -> set[str]:
        """Find reactions that would become underspecified after removing species.

        Parameters
        ----------
        reference_type : str
            The type of foreign key reference to check
        reference_ids : set[str]
            Specific reference IDs that were removed

        Returns
        -------
        set[str]
            Set of reaction IDs that were orphaned and removed
        set[str]
            Set of reaction-species IDs that were orphaned and removed
        """

        updated_reaction_species = self.reaction_species.copy()

        if reference_type == SBML_DFS.COMPARTMENTALIZED_SPECIES:
            updated_reaction_species["new"] = ~updated_reaction_species[
                SBML_DFS.SC_ID
            ].isin(reference_ids)
        elif reference_type == SBML_DFS.REACTION_SPECIES:
            updated_reaction_species["new"] = ~updated_reaction_species.index.isin(
                reference_ids
            )
        else:
            raise ValueError(f"Invalid reference type: {reference_type}")

        updated_reaction_species = sbml_dfs_utils.add_sbo_role(updated_reaction_species)
        underspecified_reactions = sbml_dfs_utils.find_underspecified_reactions(
            updated_reaction_species
        )

        # either directly removed or indirectly removed by the reactions
        invalid_reaction_species_direct = set(
            updated_reaction_species.index[~updated_reaction_species["new"]].tolist()
        )
        invalid_reaction_species_by_rxn = set(
            updated_reaction_species.loc[
                updated_reaction_species[SBML_DFS.R_ID].isin(underspecified_reactions)
            ].index.tolist()
        )
        invalid_reaction_species = (
            invalid_reaction_species_direct | invalid_reaction_species_by_rxn
        )

        return set(underspecified_reactions), invalid_reaction_species

    def _get_data_summary(self):
        """Summarize the data tables in the SBML_dfs object"""

        data_types = {}
        for entity in ENTITIES_W_DATA:
            entity_data = getattr(self, ENTITIES_TO_ENTITY_DATA[entity])
            data_types[entity] = {
                k: {
                    "entity_type": entity,
                    "columns": v.columns.tolist(),
                    "n_rows": v.shape[0],
                }
                for k, v in entity_data.items()
            }
        return data_types

    def _get_entity_data(self, entity_type: str, label: str) -> pd.DataFrame:
        """
        Get data from species_data or reactions_data by table name and label.

        Parameters
        ----------
        entity_type : str
            Name of the table to get data from ('species' or 'reactions')
        label : str
            Label of the data to retrieve

        Returns
        -------
        pd.DataFrame
            The requested data as a DataFrame

        Raises
        ------
        ValueError
            If entity_type is not 'species' or 'reactions', or if label doesn't exist
        """
        data_dict = self._validate_entity_data_access(entity_type, label)
        return data_dict[label]

    def _get_identifiers_table_for_ontology_occurrence(
        self, entity_type: str, characteristic_only: bool = False, dogmatic: bool = True
    ) -> pd.DataFrame:
        """
        Get the appropriate identifiers table for ontology analysis.

        This method handles the common logic for determining which identifiers
        table to use based on the characteristic_only and dogmatic parameters.

        Parameters
        ----------
        entity_type : str
            The type of entity to analyze (e.g., 'species', 'reactions', 'compartments')
        characteristic_only : bool, optional
            Whether to use only characteristic identifiers (only supported for species), by default False
        dogmatic : bool, optional
            Whether to use dogmatic identifier filtering, by default True

        Returns
        -------
        pd.DataFrame
            The appropriate identifiers table for ontology analysis

        Raises
        ------
        ValueError
            If the entity type is invalid
        """
        import logging

        from napistu.constants import SBML_DFS

        logger = logging.getLogger(__name__)

        if characteristic_only and entity_type == SBML_DFS.SPECIES:
            logger.debug("loading characteristic species ids")
            identifiers_table = self.get_characteristic_species_ids(dogmatic)
        else:
            logger.debug("loading all identifiers")
            if characteristic_only:
                logger.warning(
                    f"Characteristic only is only supported for species. Returning all ontologies for {entity_type}."
                )
            identifiers_table = self.get_identifiers(entity_type)

        return identifiers_table

    def _get_non_interactor_reactions(self) -> pd.DataFrame:
        """
        Get reactions table filtered to exclude reactions that are all interactors.

        Returns
        -------
        pd.DataFrame
            Reactions table with non-interactor reactions only
        """
        entity_table = self.get_table(SBML_DFS.REACTIONS)
        interactor_sbo_term = MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR]
        reaction_species = self.get_table(SBML_DFS.REACTION_SPECIES)
        valid_reactions = reaction_species[
            ~reaction_species[SBML_DFS.SBO_TERM].isin([interactor_sbo_term])
        ][SBML_DFS.R_ID].unique()

        if valid_reactions.shape[0] != entity_table.shape[0]:
            logger.info(
                f"Dropped {entity_table.shape[0] - valid_reactions.shape[0]} reactions which are all interactors from the reactions table"
            )

        valid_reactions_df = entity_table.loc[valid_reactions]

        if valid_reactions_df.shape[0] == 0:
            logger.warning("Zero reactions left after removing interactors")

        return valid_reactions_df

    def _find_invalid_entities_by_reference(
        self, entity_type: str, reference_type: str, reference_ids: set[str]
    ) -> set[str]:
        """Find and return orphaned entities based on broken foreign key references.

        Parameters
        ----------
        entity_type : str
            The entity type to check for orphans (the table with primary keys)
        reference_type : str
            The type of foreign key reference to check
        reference_ids : set[str]
            Specific reference IDs that were removed

        Returns
        -------
        set[str]
            Set of primary keys that are orphaned and should be removed
        """

        # Get the entity table and its schema
        entity_schema = SBML_DFS_SCHEMA.SCHEMA[entity_type]
        entity_pk = entity_schema[SCHEMA_DEFS.PK]
        reference_schema = SBML_DFS_SCHEMA.SCHEMA[reference_type]
        reference_pk = reference_schema[SCHEMA_DEFS.PK]

        # figure out whether the entity_type or reference type is primary or foreign key
        if SCHEMA_DEFS.FK in entity_schema and (
            reference_pk in entity_schema[SCHEMA_DEFS.FK]
        ):
            # this is the daughter table remove all reverences to the reference ids
            entity_df = getattr(self, entity_type)
            should_be_removed_ids = entity_df[
                entity_df[reference_pk].isin(reference_ids)
            ].index.tolist()
        elif SCHEMA_DEFS.FK in reference_schema and (
            entity_pk in reference_schema[SCHEMA_DEFS.FK]
        ):
            # this is the parent table, look at the reference table and see what entries
            # would be totally removed once the reference ids are removed
            # e.g., are any compartments removed becasue we removed all the relevant cspecies
            reference_df = getattr(self, reference_type).copy()

            # add the mask of to-be-removed reference ids
            reference_df["to_be_removed"] = reference_df.index.isin(reference_ids)
            is_orphaned = reference_df.groupby(entity_pk)["to_be_removed"].apply(
                lambda x: x.all()
            )
            should_be_removed_ids = is_orphaned[is_orphaned].index.tolist()
        else:
            raise ValueError(
                f"Entity type {entity_type} does not have a foreign key to {reference_type}"
            )

        return set(should_be_removed_ids)

    def _remove_entities_direct(self, entity_type: str, entity_ids: list[str]):
        """Directly remove entities without cascading cleanup.

        Parameters
        ----------
        entity_type : str
            The entity type to remove
        entity_ids : list[str]
            IDs of entities to remove
        """
        # Get the DataFrame for this entity type
        entity_df = getattr(self, entity_type)

        # Use set operations to find existing and missing IDs
        entity_ids_set = set(entity_ids)
        existing_ids_set = entity_ids_set & set(entity_df.index)
        missing_ids_set = entity_ids_set - existing_ids_set

        if existing_ids_set:
            # Remove from main entity table
            setattr(self, entity_type, entity_df.drop(index=list(existing_ids_set)))
            logger.debug(f"Removed {len(existing_ids_set)} {entity_type}")

            # Handle associated data tables using query for efficiency
            if entity_type == SBML_DFS.REACTIONS and hasattr(
                self, SBML_DFS.REACTIONS_DATA
            ):
                for k, data in self.reactions_data.items():
                    self.reactions_data[k] = data.query(
                        "index not in @existing_ids_set"
                    )

            if entity_type == SBML_DFS.SPECIES and hasattr(self, SBML_DFS.SPECIES_DATA):
                for k, data in self.species_data.items():
                    self.species_data[k] = data.query("index not in @existing_ids_set")

        # Log if some entities weren't found (might indicate logic issues)
        if missing_ids_set:
            missing_list = sorted(list(missing_ids_set))
            logger.warning(
                f"Attempted to remove {len(missing_ids_set)} {entity_type} that don't exist: {missing_list[:5]}{'...' if len(missing_list) > 5 else ''}"
            )

    def _remove_entity_data(self, entity_type: str, label: str) -> None:
        """
        Remove data from species_data or reactions_data by table name and label.

        Parameters
        ----------
        entity_type : str
            Name of the table to remove data from ('species' or 'reactions')
        label : str
            Label of the data to remove

        Raises
        ------
        ValueError
            If entity_type is not 'species' or 'reactions', or if label doesn't exist
        """
        data_dict = self._validate_entity_data_access(entity_type, label)
        del data_dict[label]

    def _validate_entity_data_access(
        self, entity_type: str, label: str
    ) -> Optional[MutableMapping[str, pd.DataFrame]]:
        """
        Validate entity type and label, and return the data dictionary if valid.

        Parameters
        ----------
        entity_type : str
            Name of the table to access ('species' or 'reactions')
        label : str
            Label of the data to access

        Returns
        -------
        MutableMapping[str, pd.DataFrame]
            The data dictionary if entity_type and label are valid

        Raises
        ------
        ValueError
            If entity_type is not 'species' or 'reactions', or if label doesn't exist
        """
        if entity_type not in ENTITIES_W_DATA:
            raise ValueError("entity_type must be either 'species' or 'reactions'")

        data_dict = getattr(self, ENTITIES_TO_ENTITY_DATA[entity_type])
        if label not in data_dict:
            existing_labels = list(data_dict.keys())
            raise ValueError(
                f"Label '{label}' not found in {ENTITIES_TO_ENTITY_DATA[entity_type]}. "
                f"Existing labels: {existing_labels}"
            )

        return data_dict

    def _validate_identifiers(self):
        """
        Validate identifiers in the model

        Iterates through all tables and checks if the identifier columns are valid.

        Raises
        ------
        ValueError
            missing identifiers in the table
        """

        SCHEMA = SBML_DFS_SCHEMA.SCHEMA
        for table in SBML_DFS_SCHEMA.SCHEMA.keys():
            if SCHEMA_DEFS.ID not in SCHEMA[table].keys():
                continue
            id_series = self.get_table(table)[SCHEMA[table][SCHEMA_DEFS.ID]]

            # Check for missing identifiers
            if id_series.isna().sum() > 0:
                missing_ids = id_series[id_series.isna()].index
                raise ValueError(
                    f"{table} has {len(missing_ids)} missing ids: {missing_ids}"
                )

            # Check that all Identifiers objects have a 'df' attribute
            for idx, identifiers_obj in id_series.items():
                if not hasattr(identifiers_obj, "df"):
                    raise ValueError(
                        f"{table} row {idx}: Identifiers object is missing 'df' attribute"
                    )
                if not hasattr(identifiers_obj.df, "empty"):
                    raise ValueError(
                        f"{table} row {idx}: Identifiers.df is not a valid DataFrame"
                    )

    def _validate_pk_fk_correspondence(self):
        """
        Check bidirectional primary key and foreign key correspondence for all tables in the schema.

        Validates:
        1. All foreign keys exist as primary keys (standard FK constraint)
        2. All primary keys are referenced as foreign keys (referential completeness)

        Raises ValueError if any FK constraint or referential completeness violations are found.
        """

        pk_df = pd.DataFrame(
            [{"pk_table": k, "key": v[SCHEMA_DEFS.PK]} for k, v in self.schema.items()]
        )

        fk_df = (
            pd.DataFrame(
                [
                    {"fk_table": k, SCHEMA_DEFS.FK: v[SCHEMA_DEFS.FK]}
                    for k, v in self.schema.items()
                    if SCHEMA_DEFS.FK in v.keys()
                ]
            )
            .set_index("fk_table")[SCHEMA_DEFS.FK]
            .apply(pd.Series)
            .reset_index()
            .melt(id_vars="fk_table")
            .drop(["variable"], axis=1)
            .rename(columns={"value": "key"})
        )

        pk_fk_correspondences = pk_df.merge(fk_df)

        for i in range(0, pk_fk_correspondences.shape[0]):
            pk_table_keys = set(
                getattr(self, pk_fk_correspondences["pk_table"][i]).index.tolist()
            )
            if None in pk_table_keys:
                raise ValueError(
                    f"{pk_fk_correspondences['pk_table'][i]} had "
                    "missing values in its index"
                )

            fk_table_keys = set(
                getattr(self, pk_fk_correspondences["fk_table"][i]).loc[
                    :, pk_fk_correspondences["key"][i]
                ]
            )
            if None in fk_table_keys:
                raise ValueError(
                    f"{pk_fk_correspondences['fk_table'][i]} included "
                    f"missing {pk_fk_correspondences['key'][i]} values"
                )

            # Check 1: All foreign keys need to match a primary key (standard FK constraint)
            extra_fks = fk_table_keys.difference(pk_table_keys)
            if len(extra_fks) != 0:
                raise ValueError(
                    f"{len(extra_fks)} distinct "
                    f"{pk_fk_correspondences['key'][i]} values were"
                    f" found in {pk_fk_correspondences['fk_table'][i]} "
                    f"but missing from {pk_fk_correspondences['pk_table'][i]}."
                    " All foreign keys must have a matching primary key.\n\n"
                    f"Extra key are: {', '.join(extra_fks)}"
                )

            # Check 2: All primary keys should be referenced as foreign keys (referential completeness)
            unused_pks = pk_table_keys.difference(fk_table_keys)
            if len(unused_pks) != 0:
                raise ValueError(
                    f"Referential completeness violation: {len(unused_pks)} "
                    f"{pk_fk_correspondences['key'][i]} values in "
                    f"{pk_fk_correspondences['pk_table'][i]} are not referenced by "
                    f"{pk_fk_correspondences['fk_table'][i]}. "
                    f"All primary keys must be referenced as foreign keys.\n\n"
                    f"Unused keys are: {sorted(list(unused_pks))[:10]}"
                )

    def _validate_r_ids(self, r_ids: Optional[Union[str, list[str]]]) -> list[str]:

        if isinstance(r_ids, str):
            r_ids = [r_ids]

        if r_ids is None:
            return self.reactions.index.tolist()
        else:
            if not all(r_id in self.reactions.index for r_id in r_ids):
                raise ValueError(f"Reaction IDs {r_ids} not found in reactions table")

            return r_ids

    def _validate_reaction_species(self):
        if not all(self.reaction_species[SBML_DFS.STOICHIOMETRY].notnull()):
            raise ValueError(
                "All reaction_species[SBML_DFS.STOICHIOMETRY] must be not null"
            )

        # test for null SBO terms
        n_null_sbo_terms = sum(self.reaction_species[SBML_DFS.SBO_TERM].isnull())
        if n_null_sbo_terms != 0:
            raise ValueError(
                f"{n_null_sbo_terms} sbo_terms were None; all terms should be defined"
            )

        # find invalid SBO terms
        sbo_counts = self.reaction_species.value_counts(SBML_DFS.SBO_TERM)
        invalid_sbo_term_counts = sbo_counts[
            ~sbo_counts.index.isin(MINI_SBO_TO_NAME.keys())
        ]

        if invalid_sbo_term_counts.shape[0] != 0:
            invalid_sbo_counts_str = ", ".join(
                [f"{k} (N={v})" for k, v in invalid_sbo_term_counts.to_dict().items()]
            )
            raise ValueError(
                f"{invalid_sbo_term_counts.shape[0]} sbo_terms were not "
                f"defined {invalid_sbo_counts_str}"
            )

    def _validate_reactions_data(self, reactions_data_table: pd.DataFrame):
        """Validates reactions data attribute

        Parameters
        ----------
        reactions_data_table : pd.DataFrame
            a reactions data table

        Raises
        ------
        ValueError
            r_id not index name
            r_id index contains duplicates
            r_id not in reactions table
        """
        sbml_dfs_utils._validate_matching_data(reactions_data_table, self.reactions)

    def _validate_sources(self):
        """
        Validate sources in the model

        Iterates through all tables and checks if the source columns are valid.

        Raises:
            ValueError: missing sources in the table
        """

        SCHEMA = SBML_DFS_SCHEMA.SCHEMA
        for table in SBML_DFS_SCHEMA.SCHEMA.keys():
            if "source" not in SCHEMA[table].keys():
                continue
            source_series = self.get_table(table)[SCHEMA[table]["source"]]
            if source_series.isna().sum() > 0:
                missing_sources = source_series[source_series.isna()].index
                raise ValueError(
                    f"{table} has {len(missing_sources)} missing sources: {missing_sources}"
                )

    def _validate_species_data(self, species_data_table: pd.DataFrame):
        """Validates species data attribute

        Parameters
        ----------
        species_data_table : pd.DataFrame
            a species data table

        Raises
        ------
        ValueError
            s_id not index name
            s_id index contains duplicates
            s_id not in species table
        """
        sbml_dfs_utils._validate_matching_data(species_data_table, self.species)

    def _validate_table(self, table_name: str) -> None:
        """
        Validate a table in this SBML_dfs object against its schema.

        This is an internal method that validates a table that is part of this SBML_dfs
        object against the schema stored in self.schema.

        Parameters
        ----------
        table : str
            Name of the table to validate

        Raises
        ------
        ValueError
            If the table does not conform to its schema
        """
        table_data = getattr(self, table_name)

        sbml_dfs_utils.validate_sbml_dfs_table(table_data, table_name)
