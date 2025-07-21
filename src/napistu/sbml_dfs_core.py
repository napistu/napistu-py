from __future__ import annotations

import copy
import logging
import re
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import MutableMapping
from typing import TYPE_CHECKING
from typing import Optional
from typing import Union

from fs import open_fs
import pandas as pd

from napistu import identifiers
from napistu import sbml_dfs_utils
from napistu import source
from napistu import utils
from napistu.ingestion import sbml
from napistu.ontologies import id_tables
from napistu.constants import (
    BQB,
    BQB_DEFINING_ATTRS_LOOSE,
    BQB_PRIORITIES,
    ENTITIES_W_DATA,
    ENTITIES_TO_ENTITY_DATA,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    MINI_SBO_TO_NAME,
    NAPISTU_STANDARD_OUTPUTS,
    ONTOLOGY_PRIORITIES,
    SBML_DFS,
    SBML_DFS_SCHEMA,
    SBOTERM_NAMES,
    SCHEMA_DEFS,
)
from napistu.ingestion.constants import INTERACTION_EDGELIST_DEFAULTS

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
    get_characteristic_species_ids(dogmatic=True)
        Return characteristic systematic identifiers for molecular species, optionally using a strict or loose definition.
    get_cspecies_features()
        Compute and return additional features for compartmentalized species, such as degree and type.
    get_identifiers(id_type)
        Retrieve a table of identifiers for a specified entity type (e.g., species or reactions).
    get_network_summary()
        Return a dictionary of diagnostic statistics summarizing the network structure.
    get_source_total_counts(entity_type)
        Get the total counts of each source for a given entity type.
    get_species_features()
        Compute and return additional features for species, such as species type.
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
    reaction_formulas(r_ids=None)
        Generate human-readable reaction formulas for specified reactions.
    reaction_summaries(r_ids=None)
        Return a summary DataFrame for specified reactions, including names and formulas.
    remove_compartmentalized_species(sc_ids)
        Remove specified compartmentalized species and associated reactions from the model.
    remove_reactions(r_ids, remove_species=False)
        Remove specified reactions and optionally remove unused species.
    remove_reactions_data(label)
        Remove a reactions data table by label.
    remove_species_data(label)
        Remove a species data table by label.
    search_by_ids(id_table, identifiers=None, ontologies=None, bqbs=None)
        Find entities and identifiers matching a set of query IDs.
    search_by_name(name, entity_type, partial_match=True)
        Find entities by exact or partial name match.
    select_species_data(species_data_table)
        Select a species data table from the SBML_dfs object by name.
    species_status(s_id)
        Return all reactions a species participates in, with stoichiometry and formula information.
    validate()
        Validate the SBML_dfs structure and relationships.
    validate_and_resolve()
        Validate and attempt to automatically fix common issues.

    Private/Hidden Methods (alphabetical, appear after public methods)
    -----------------------------------------------------------------
    _attempt_resolve(e)
    _find_underspecified_reactions_by_scids(sc_ids)
    _get_unused_cspecies()
    _get_unused_species()
    _remove_compartmentalized_species(sc_ids)
    _remove_entity_data(entity_type, label)
    _remove_species(s_ids)
    _remove_unused_cspecies()
    _remove_unused_species()
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

        # create a model from dictionary entries
        if isinstance(sbml_model, dict):
            for ent in SBML_DFS_SCHEMA.REQUIRED_ENTITIES:
                setattr(self, ent, sbml_model[ent])
            for ent in SBML_DFS_SCHEMA.OPTIONAL_ENTITIES:
                if ent in sbml_model:
                    setattr(self, ent, sbml_model[ent])
        else:
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
            raise TypeError(
                f"sbml_dfs was a {type(self)} and must" " be an sbml.SBML_dfs"
            )

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
                species_identifiers.drop([SBML_DFS.S_SOURCE], axis=1).to_csv(
                    f, sep="\t", index=False
                )

            # export reactions' source total counts
            reactions_source_total_counts_path = (
                model_prefix + NAPISTU_STANDARD_OUTPUTS.REACTIONS_SOURCE_TOTAL_COUNTS
            )
            with fs.openbin(reactions_source_total_counts_path, "w") as f:
                reactions_source_total_counts.to_csv(f, sep="\t", index=True)

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

    def get_characteristic_species_ids(self, dogmatic: bool = True) -> pd.DataFrame:
        """
        Get Characteristic Species IDs

        List the systematic identifiers which are characteristic of molecular species, e.g., excluding subcomponents, and optionally, treating proteins, transcripts, and genes equiavlently.

        Parameters
        ----------
        sbml_dfs : sbml_dfs_core.SBML_dfs
            The SBML_dfs object.
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
            self.reaction_species["sc_id"].value_counts().rename("sc_degree")
        )

        cspecies_n_children = (
            self.reaction_species.loc[
                self.reaction_species[SBML_DFS.STOICHIOMETRY] <= 0, "sc_id"
            ]
            .value_counts()
            .rename("sc_children")
        )

        cspecies_n_parents = (
            self.reaction_species.loc[
                self.reaction_species[SBML_DFS.STOICHIOMETRY] > 0, "sc_id"
            ]
            .value_counts()
            .rename("sc_parents")
        )

        species_features = self.get_species_features()["species_type"]

        return (
            self.compartmentalized_species.join(cspecies_n_connections)
            .join(cspecies_n_children)
            .join(cspecies_n_parents)
            .fillna(int(0))  # Explicitly fill with int(0) to avoid downcasting warning
            .merge(species_features, left_on="s_id", right_index=True)
            .drop(columns=["sc_name", "s_id", "c_id"])
        )

    def get_identifiers(self, id_type) -> pd.DataFrame:
        """
        Get identifiers from a specified entity type.

        Parameters
        ----------
        id_type : str
            Type of entity to get identifiers for (e.g., 'species', 'reactions')

        Returns
        -------
        pd.DataFrame
            Table of identifiers for the specified entity type

        Raises
        ------
        ValueError
            If id_type is invalid or identifiers are malformed
        """
        selected_table = self.get_table(id_type, {SCHEMA_DEFS.ID})
        schema = SBML_DFS_SCHEMA.SCHEMA

        identifiers_dict = dict()
        for sysid in selected_table.index:
            id_entry = selected_table[schema[id_type][SCHEMA_DEFS.ID]][sysid]

            if isinstance(id_entry, identifiers.Identifiers):
                identifiers_dict[sysid] = pd.DataFrame(id_entry.ids)
            elif pd.isna(id_entry):
                continue
            else:
                raise ValueError(
                    f"id_entry was a {type(id_entry)} and must either be"
                    " an identifiers.Identifiers object or a missing value (None, np.nan, pd.NA)"
                )
        if not identifiers_dict:
            # Return empty DataFrame with expected columns if nothing found
            return pd.DataFrame(columns=[schema[id_type][SCHEMA_DEFS.PK], "entry"])

        identifiers_tbl = pd.concat(identifiers_dict)

        identifiers_tbl.index.names = [schema[id_type][SCHEMA_DEFS.PK], "entry"]
        identifiers_tbl = identifiers_tbl.reset_index()

        named_identifiers = identifiers_tbl.merge(
            selected_table.drop(schema[id_type][SCHEMA_DEFS.ID], axis=1),
            left_on=schema[id_type][SCHEMA_DEFS.PK],
            right_index=True,
        )

        return named_identifiers

    def get_network_summary(self) -> Mapping[str, Any]:
        """
        Get diagnostic statistics about the network.

        Returns
        -------
        Mapping[str, Any]
            Dictionary of diagnostic statistics including:
            - n_species_types: Number of species types
            - dict_n_species_per_type: Number of species per type
            - n_species: Number of species
            - n_cspecies: Number of compartmentalized species
            - n_reaction_species: Number of reaction species
            - n_reactions: Number of reactions
            - n_compartments: Number of compartments
            - dict_n_species_per_compartment: Number of species per compartment
            - stats_species_per_reaction: Statistics on reactands per reaction
            - top10_species_per_reaction: Top 10 reactions by number of reactands
            - stats_degree: Statistics on species connectivity
            - top10_degree: Top 10 species by connectivity
            - stats_identifiers_per_species: Statistics on identifiers per species
            - top10_identifiers_per_species: Top 10 species by number of identifiers
        """
        stats: MutableMapping[str, Any] = {}
        species_features = self.get_species_features()
        stats["n_species_types"] = species_features["species_type"].nunique()
        stats["dict_n_species_per_type"] = (
            species_features.groupby(by="species_type").size().to_dict()
        )
        stats["n_species"] = self.species.shape[0]
        stats["n_cspecies"] = self.compartmentalized_species.shape[0]
        stats["n_reaction_species"] = self.reaction_species.shape[0]
        stats["n_reactions"] = self.reactions.shape[0]
        stats["n_compartments"] = self.compartments.shape[0]
        stats["dict_n_species_per_compartment"] = (
            self.compartmentalized_species.groupby(SBML_DFS.C_ID)
            .size()
            .rename("n_species")  # type:ignore
            .to_frame()
            .join(self.compartments[[SBML_DFS.C_NAME]])
            .reset_index(drop=False)
            .to_dict(orient="records")
        )
        per_reaction_stats = self.reaction_species.groupby(SBML_DFS.R_ID).size()
        stats["stats_species_per_reactions"] = per_reaction_stats.describe().to_dict()
        stats["top10_species_per_reactions"] = (
            per_reaction_stats.sort_values(ascending=False)  # type:ignore
            .head(10)
            .rename("n_species")
            .to_frame()
            .join(self.reactions[[SBML_DFS.R_NAME]])
            .reset_index(drop=False)
            .to_dict(orient="records")
        )

        cspecies_features = self.get_cspecies_features()
        stats["stats_degree"] = cspecies_features["sc_degree"].describe().to_dict()
        stats["top10_degree"] = (
            cspecies_features.sort_values("sc_degree", ascending=False)
            .head(10)[["sc_degree", "sc_children", "sc_parents", "species_type"]]
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
        identifiers_stats = s_identifiers.groupby("s_id").size()
        stats["stats_identifiers_per_species"] = identifiers_stats.describe().to_dict()
        stats["top10_identifiers_per_species"] = (
            identifiers_stats.sort_values(ascending=False)
            .head(10)
            .rename("n_identifiers")
            .to_frame()
            .join(species_features[[SBML_DFS.S_NAME, "species_type"]])
            .reset_index(drop=False)
            .to_dict(orient="records")
        )

        return stats

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
        entity_table = self.get_table(entity_type)

        if entity_type == SBML_DFS.REACTIONS:
            logger.info(
                "Excluding reactions which are all interactors from reaction source counts"
            )

            interactor_sbo_term = MINI_SBO_FROM_NAME[SBOTERM_NAMES.INTERACTOR]
            reaction_species = self.get_table(SBML_DFS.REACTION_SPECIES)
            valid_reactions = reaction_species[
                ~reaction_species[SBML_DFS.SBO_TERM].isin([interactor_sbo_term])
            ][SBML_DFS.R_ID].unique()

            entity_table = entity_table.loc[valid_reactions]

        all_sources_table = source.unnest_sources(entity_table)

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
                "species_type": lambda d: d["s_Identifiers"].apply(
                    sbml_dfs_utils.species_type_types
                )
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
            VALID_REQUIRED_ATTRIBUTES = {"id", "source", "label"}
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
        all_ids = pd.concat(
            [
                sbml_dfs_utils._id_dict_to_df(
                    entity_table[schema[entity_type]["id"]].iloc[i].ids
                ).assign(id=entity_table.index[i])
                for i in range(0, entity_table.shape[0])
            ]
        ).rename(columns={"id": schema[entity_type]["pk"]})

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
            .groupby(schema[entity_type]["pk"])
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
            :, self.schema[SBML_DFS.COMPARTMENTALIZED_SPECIES]["vars"]
        ]
        return

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

        # identify reactions which work across compartments
        r_id_cross_compartment = r_id_compartment_counts[r_id_compartment_counts > 1]
        # there species must be labelled with the sc_name to specify where a species exists
        if r_id_cross_compartment.shape[0] > 0:
            rxn_eqtn_cross_compartment = (
                matching_reaction_species[
                    matching_reaction_species[SBML_DFS.R_ID].isin(
                        r_id_cross_compartment.index
                    )
                ]
                .sort_values([SBML_DFS.SC_NAME])
                .groupby(SBML_DFS.R_ID)
                .apply(
                    lambda x: sbml_dfs_utils.construct_formula_string(
                        x, self.reactions, SBML_DFS.SC_NAME
                    )
                )
                .rename("r_formula_str")
            )
        else:
            rxn_eqtn_cross_compartment = None

        # identify reactions which occur within a single compartment; for these the reaction
        # can be labelled with the compartment and individual species can receive a more readable s_name
        r_id_within_compartment = r_id_compartment_counts[r_id_compartment_counts == 1]
        if r_id_within_compartment.shape[0] > 0:
            # add s_name
            augmented_matching_reaction_species = (
                matching_reaction_species[
                    matching_reaction_species[SBML_DFS.R_ID].isin(
                        r_id_within_compartment.index
                    )
                ]
                .merge(self.compartments, left_on=SBML_DFS.C_ID, right_index=True)
                .merge(self.species, left_on=SBML_DFS.S_ID, right_index=True)
                .sort_values([SBML_DFS.S_NAME])
            )
            # create formulas based on s_names of components
            rxn_eqtn_within_compartment = augmented_matching_reaction_species.groupby(
                [SBML_DFS.R_ID, SBML_DFS.C_NAME]
            ).apply(
                lambda x: sbml_dfs_utils.construct_formula_string(
                    x, self.reactions, SBML_DFS.S_NAME
                )
            )
            # add compartment for each reaction
            rxn_eqtn_within_compartment = pd.Series(
                [
                    y + ": " + x
                    for x, y in zip(
                        rxn_eqtn_within_compartment,
                        rxn_eqtn_within_compartment.index.get_level_values(
                            SBML_DFS.C_NAME
                        ),
                    )
                ],
                index=rxn_eqtn_within_compartment.index.get_level_values(SBML_DFS.R_ID),
            ).rename("r_formula_str")
        else:
            rxn_eqtn_within_compartment = None

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

    def remove_compartmentalized_species(self, sc_ids: Iterable[str]):
        """
        Remove compartmentalized species and associated reactions.

        Starting with a set of compartmentalized species, determine which reactions
        should be removed based on their removal. Then remove these reactions,
        compartmentalized species, and species.

        Parameters
        ----------
        sc_ids : Iterable[str]
            IDs of compartmentalized species to remove
        """

        # find reactions which should be totally removed since they are losing critical species
        removed_reactions = self._find_underspecified_reactions_by_scids(sc_ids)
        self.remove_reactions(removed_reactions)

        self._remove_compartmentalized_species(sc_ids)

        # remove species (and their associated species data if all their cspecies have been lost)
        self._remove_unused_species()

    def remove_reactions(self, r_ids: Iterable[str], remove_species: bool = False):
        """
        Remove reactions from the model.

        Parameters
        ----------
        r_ids : Iterable[str]
            IDs of reactions to remove
        remove_species : bool, optional
            Whether to remove species that are no longer part of any reactions,
            by default False
        """
        # remove corresponding reactions_species
        self.reaction_species = self.reaction_species.query("r_id not in @r_ids")
        # remove reactions
        self.reactions = self.reactions.drop(index=list(r_ids))
        # remove reactions_data
        if hasattr(self, "reactions_data"):
            for k, data in self.reactions_data.items():
                self.reactions_data[k] = data.drop(index=list(r_ids))
        # remove species if requested
        if remove_species:
            self._remove_unused_cspecies()
            self._remove_unused_species()

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

        entity_type = sbml_dfs_utils.infer_entity_type(id_table)
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
        entity_table = self.get_table(entity_type, required_attributes={"label"})
        label_attr = self.schema[entity_type]["label"]

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

        # check whether pks and fks agree
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
        else:
            logger.warning(
                "An error occurred which could not be automatically resolved"
            )
            raise e

    def _find_underspecified_reactions_by_scids(
        self, sc_ids: Iterable[str]
    ) -> set[str]:
        """
        Find Underspecified reactions

        Identify reactions which should be removed if a set of molecular species are removed
        from the system.

        Parameters
        ----------
        sc_ids : list[str]
            A list of compartmentalized species ids (sc_ids) which will be removed.

        Returns
        -------
        underspecified_reactions : set[str]
            A set of reactions which should be removed because they will not occur once
            "sc_ids" are removed.
        """
        updated_reaction_species = self.reaction_species.copy()
        updated_reaction_species["new"] = ~updated_reaction_species[
            SBML_DFS.SC_ID
        ].isin(sc_ids)
        updated_reaction_species = sbml_dfs_utils.add_sbo_role(updated_reaction_species)
        underspecified_reactions = sbml_dfs_utils.find_underspecified_reactions(
            updated_reaction_species
        )
        return underspecified_reactions

    def _get_unused_cspecies(self) -> set[str]:
        """Returns a set of compartmentalized species
        that are not part of any reactions"""
        sc_ids = set(self.compartmentalized_species.index) - set(
            self.reaction_species[SBML_DFS.SC_ID]
        )
        return sc_ids  # type: ignore

    def _get_unused_species(self) -> set[str]:
        """Returns a list of species that are not part of any reactions"""
        s_ids = set(self.species.index) - set(
            self.compartmentalized_species[SBML_DFS.S_ID]
        )
        return s_ids  # type: ignore

    def _remove_compartmentalized_species(self, sc_ids: Iterable[str]):
        """Removes compartmentalized species from the model

        This should not be directly used by the user, as it can lead to
        invalid reactions when removing species without a logic to decide
        if the reaction needs to be removed as well.

        Args:
            sc_ids (Iterable[str]): the compartmentalized species to remove
        """
        # Remove compartmentalized species
        self.compartmentalized_species = self.compartmentalized_species.drop(
            index=list(sc_ids)
        )
        # remove corresponding reactions_species
        self.reaction_species = self.reaction_species.query("sc_id not in @sc_ids")

    def _remove_entity_data(self, entity_type: str, label: str) -> None:
        """
        Remove data from species_data or reactions_data by table name and label.

        Parameters
        ----------
        entity_type : str
            Name of the table to remove data from ('species' or 'reactions')
        label : str
            Label of the data to remove

        Notes
        -----
        If the label does not exist, a warning will be logged that includes the existing labels.
        """
        if entity_type not in ENTITIES_W_DATA:
            raise ValueError("table_name must be either 'species' or 'reactions'")

        data_dict = getattr(self, ENTITIES_TO_ENTITY_DATA[entity_type])
        if label not in data_dict:
            existing_labels = list(data_dict.keys())
            logger.warning(
                f"Label '{label}' not found in {ENTITIES_TO_ENTITY_DATA[entity_type]}. "
                f"Existing labels: {existing_labels}"
            )
            return

        del data_dict[label]

    def _remove_species(self, s_ids: Iterable[str]):
        """Removes species from the model

        This should not be directly used by the user, as it can lead to
        invalid reactions when removing species without a logic to decide
        if the reaction needs to be removed as well.

        This removes the species and corresponding compartmentalized species and
        reactions_species.

        Args:
            s_ids (Iterable[str]): the species to remove
        """
        sc_ids = self.compartmentalized_species.query("s_id in @s_ids").index.tolist()
        self._remove_compartmentalized_species(sc_ids)
        # Remove species
        self.species = self.species.drop(index=list(s_ids))
        # remove data
        for k, data in self.species_data.items():
            self.species_data[k] = data.drop(index=list(s_ids))

    def _remove_unused_cspecies(self):
        """Removes compartmentalized species that are no
        longer part of any reactions"""
        sc_ids = self._get_unused_cspecies()
        self._remove_compartmentalized_species(sc_ids)

    def _remove_unused_species(self):
        """Removes species that are no longer part of any
        compartmentalized species"""
        s_ids = self._get_unused_species()
        self._remove_species(s_ids)

    def _validate_identifiers(self):
        """
        Validate identifiers in the model

        Iterates through all tables and checks if the identifier columns are valid.

        Raises:
            ValueError: missing identifiers in the table
        """

        SCHEMA = SBML_DFS_SCHEMA.SCHEMA
        for table in SBML_DFS_SCHEMA.SCHEMA.keys():
            if "id" not in SCHEMA[table].keys():
                continue
            id_series = self.get_table(table)[SCHEMA[table]["id"]]
            if id_series.isna().sum() > 0:
                missing_ids = id_series[id_series.isna()].index
                raise ValueError(
                    f"{table} has {len(missing_ids)} missing ids: {missing_ids}"
                )

    def _validate_pk_fk_correspondence(self):
        """
        Check whether primary keys and foreign keys agree for all tables in the schema.
        Raises ValueError if any correspondence fails.
        """

        pk_df = pd.DataFrame(
            [{"pk_table": k, "key": v["pk"]} for k, v in self.schema.items()]
        )

        fk_df = (
            pd.DataFrame(
                [
                    {"fk_table": k, "fk": v["fk"]}
                    for k, v in self.schema.items()
                    if "fk" in v.keys()
                ]
            )
            .set_index("fk_table")["fk"]
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

            # all foreign keys need to match a primary key
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

        Args:
            reactions_data_table (pd.DataFrame): a reactions data table

        Raises:
            ValueError: r_id not index name
            ValueError: r_id index contains duplicates
            ValueError: r_id not in reactions table
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

        Args:
            species_data_table (pd.DataFrame): a species data table

        Raises:
            ValueError: s_id not index name
            ValueError: s_id index contains duplicates
            ValueError: s_id not in species table
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


def sbml_dfs_from_edgelist(
    interaction_edgelist: pd.DataFrame,
    species_df: pd.DataFrame,
    compartments_df: pd.DataFrame,
    interaction_source: source.Source = source.Source(init=True),
    interaction_edgelist_defaults: dict[str, Any] = INTERACTION_EDGELIST_DEFAULTS,
    keep_species_data: bool | str = False,
    keep_reactions_data: bool | str = False,
) -> SBML_dfs:
    """
    Create SBML_dfs from interaction edgelist.

    Combines a set of molecular interactions into a mechanistic SBML_dfs model
    by processing interaction data, species information, and compartment definitions.

    Parameters
    ----------
    interaction_edgelist : pd.DataFrame
        Table containing molecular interactions with columns:
        - upstream_name : str, matches "s_name" from species_df
        - downstream_name : str, matches "s_name" from species_df
        - r_name : str, name for the interaction
        - r_Identifiers : identifiers.Identifiers, supporting identifiers
        - upstream_compartment : str, matches "c_name" from compartments_df
        - downstream_compartment : str, matches "c_name" from compartments_df
        - sbo_term_upstream : str, SBO term defining interaction type
        - r_isreversible : bool, whether reaction is reversible
    species_df : pd.DataFrame
        Table defining molecular species with columns:
        - s_name : str, name of molecular species
        - s_Identifiers : identifiers.Identifiers, species identifiers
    compartments_df : pd.DataFrame
        Table defining compartments with columns:
        - c_name : str, name of compartment
        - c_Identifiers : identifiers.Identifiers, compartment identifiers
    interaction_source : source.Source
        Source object linking model entities to interaction source
    interaction_edgelist_defaults : dict[str, Any], default INTERACTION_EDGELIST_DEFAULTS
        Default values for interaction edgelist columns
    keep_species_data : bool or str, default False
        Whether to preserve extra species columns. If True, saves as 'source' label.
        If string, uses as custom label. If False, discards extra data.
    keep_reactions_data : bool or str, default False
        Whether to preserve extra reaction columns. If True, saves as 'source' label.
        If string, uses as custom label. If False, discards extra data.

    Returns
    -------
    SBML_dfs
        Validated SBML data structure containing compartments, species,
        compartmentalized species, reactions, and reaction species tables.
    """

    # 0. Add defaults to interaction edgelist
    interaction_edgelist_with_defaults = sbml_dfs_utils._add_edgelist_defaults(
        interaction_edgelist, interaction_edgelist_defaults
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
    sbml_dfs = _edgelist_assemble_sbml_model(
        processed_compartments,
        processed_species,
        comp_species,
        reactions,
        reaction_species,
        species_data,
        reactions_data,
        keep_species_data,
        keep_reactions_data,
        extra_columns,
    )

    return sbml_dfs


def _edgelist_assemble_sbml_model(
    compartments: pd.DataFrame,
    species: pd.DataFrame,
    comp_species: pd.DataFrame,
    reactions: pd.DataFrame,
    reaction_species: pd.DataFrame,
    species_data,
    reactions_data,
    keep_species_data,
    keep_reactions_data,
    extra_columns: dict[str, list[str]],
) -> SBML_dfs:
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
            keep_reactions_data if isinstance(keep_reactions_data, str) else "source"
        )
        sbml_tbl_dict[SBML_DFS.REACTIONS_DATA] = {data_label: reactions_data}

    if len(extra_columns[SBML_DFS.SPECIES]) > 0:
        data_label = (
            keep_species_data if isinstance(keep_species_data, str) else "source"
        )
        sbml_tbl_dict[SBML_DFS.SPECIES_DATA] = {data_label: species_data}

    sbml_model = SBML_dfs(sbml_tbl_dict)
    sbml_model.validate()

    return sbml_model
