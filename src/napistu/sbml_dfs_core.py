from __future__ import annotations

import logging
import re
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import MutableMapping
from typing import TYPE_CHECKING

import pandas as pd
from napistu import identifiers
from napistu import sbml_dfs_utils
from napistu import source
from napistu import utils
from napistu.constants import SBML_DFS
from napistu.constants import SBML_DFS_SCHEMA
from napistu.constants import IDENTIFIERS
from napistu.constants import REQUIRED_REACTION_FROMEDGELIST_COLUMNS
from napistu.constants import CPR_STANDARD_OUTPUTS
from napistu.constants import INTERACTION_EDGELIST_EXPECTED_VARS
from napistu.constants import BQB_PRIORITIES
from napistu.constants import ONTOLOGY_PRIORITIES
from napistu.constants import BQB
from napistu.constants import BQB_DEFINING_ATTRS
from napistu.constants import COMPARTMENTS
from napistu.constants import COMPARTMENT_ALIASES
from napistu.constants import COMPARTMENTS_GO_TERMS
from napistu.constants import MINI_SBO_FROM_NAME
from napistu.constants import MINI_SBO_TO_NAME
from napistu.constants import ONTOLOGIES
from napistu.constants import SBO_NAME_TO_ROLE
from napistu.constants import SBOTERM_NAMES
from napistu.constants import CHARACTERISTIC_COMPLEX_ONTOLOGIES
from napistu.ingestion import sbml
from fs import open_fs

logger = logging.getLogger(__name__)


class SBML_dfs:
    """
    System Biology Markup Language Model Data Frames.

    Attributes
    ----------
    compartments: pd.DataFrame
        sub-cellular compartments in the model
    species: pd.DataFrame
        molecular species in the model
    species_data: Dict[str, pd.DataFrame]: Additional data for species.
        DataFrames with additional data and index = species_id
    reactions: pd.DataFrame
        reactions in the model
    reactions_data: Dict[str, pd.DataFrame]: Additional data for reactions.
        DataFrames with additional data and index = reaction_id
    reaction_species: pd.DataFrame
        One entry per species participating in a reaction
    schema: dict
        dictionary reprenting the structure of the other attributes and meaning of their variables

    Methods
    -------
    get_table(entity_type, required_attributes)
        Get a table from the SBML_dfs object and optionally validate that it contains a set of required attributes
    search_by_ids(ids, entity_type, identifiers_df, ontologies)
        Pull out identifiers and entities matching a set of query ids which optionally match a set of ontologies
    search_by_name(name, entity_type, partial_match)
        Pull out a set of entities by name or partial string match [default]
    get_cspecies_features()
        Returns additional attributes of compartmentalized species
    get_species_features()
        Returns additional attributes of species
    get_identifiers(id_type)
        Returns a DataFrame containing identifiers from the id_type table
    get_uri_urls(entity_type, entity_ids = None)
        Returns a Series containing reference urls for each entity
    validate()
        Validate that the sbml_dfs follows the schema and identify clear pathologies
    validate_and_rec()
        Validate the sbml_dfs and attempt to automatically resolve common issues
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
        Creates a pathway

        Parameters
        ----------
        sbml_model : cpr.SBML or a dict containing tables following the sbml_dfs schema
            A SBML model produced by cpr.SBML().
        validate (bool): if True then call self.validate() to identify formatting issues
        resolve (bool): if True then try to automatically resolve common problems

        Returns
        -------
        None.
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
            self = sbml.sbml_df_from_sbml(self, sbml_model)

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

    def get_table(
        self, entity_type: str, required_attributes: None | set[str] = None
    ) -> pd.DataFrame:
        """
        Get Table

        Get a table from the SBML_dfs object and optionally validate that it contains a set of required attributes.
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
                    f"required_attributes must be a set, but got {type(required_attributes).__name__}"
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

    def search_by_ids(
        self,
        ids: list[str],
        entity_type: str,
        identifiers_df: pd.DataFrame,
        ontologies: None | set[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # validate inputs
        entity_table = self.get_table(entity_type, required_attributes={"id"})
        entity_pk = self.schema[entity_type]["pk"]

        utils.match_pd_vars(
            identifiers_df,
            req_vars={
                entity_pk,
                IDENTIFIERS.ONTOLOGY,
                IDENTIFIERS.IDENTIFIER,
                IDENTIFIERS.URL,
                IDENTIFIERS.BQB,
            },
            allow_series=False,
        ).assert_present()

        if ontologies is not None:
            if not isinstance(ontologies, set):
                # for clarity this should not be reachable based on type hints
                raise TypeError(
                    f"ontologies must be a set, but got {type(ontologies).__name__}"
                )
            ALL_VALID_ONTOLOGIES = identifiers_df["ontology"].unique()
            invalid_ontologies = ontologies.difference(ALL_VALID_ONTOLOGIES)
            if len(invalid_ontologies) > 0:
                raise ValueError(
                    f"The following ontologies are not valid: {', '.join(invalid_ontologies)}.\n"
                    f"Valid ontologies are {', '.join(ALL_VALID_ONTOLOGIES)}"
                )

            # fitler to just to identifiers matchign the ontologies of interest
            identifiers_df = identifiers_df.query("ontology in @ontologies")

        matching_identifiers = identifiers_df.loc[
            identifiers_df["identifier"].isin(ids)
        ]
        entity_subset = entity_table.loc[matching_identifiers[entity_pk].tolist()]

        return entity_subset, matching_identifiers

    def search_by_name(
        self, name: str, entity_type: str, partial_match: bool = True
    ) -> pd.DataFrame:
        entity_table = self.get_table(entity_type, required_attributes={"label"})
        label_attr = self.schema[entity_type]["label"]

        if partial_match:
            matches = entity_table.loc[
                entity_table[label_attr].str.contains(name, case=False)
            ]
        else:
            matches = entity_table.loc[entity_table[label_attr].str.lower() == name]
        return matches

    def get_species_features(self) -> pd.DataFrame:
        species = self.species
        augmented_species = species.assign(
            **{"species_type": lambda d: d["s_Identifiers"].apply(species_type_types)}
        )

        return augmented_species

    def get_cspecies_features(self) -> pd.DataFrame:
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
        selected_table = self.get_table(id_type, {"id"})
        schema = self.schema

        identifiers_dict = dict()
        for sysid in selected_table.index:
            id_entry = selected_table[schema[id_type]["id"]][sysid]

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
            return pd.DataFrame(columns=[schema[id_type]["pk"], "entry"])
        identifiers_tbl = pd.concat(identifiers_dict)

        identifiers_tbl.index.names = [schema[id_type]["pk"], "entry"]
        identifiers_tbl = identifiers_tbl.reset_index()

        named_identifiers = identifiers_tbl.merge(
            selected_table.drop(schema[id_type]["id"], axis=1),
            left_on=schema[id_type]["pk"],
            right_index=True,
        )

        return named_identifiers

    def get_uri_urls(
        self,
        entity_type: str,
        entity_ids: Iterable[str] | None = None,
        required_ontology: str | None = None,
    ) -> pd.Series:
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
                sbml_dfs_utils._stub_ids(
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

    def get_network_summary(self) -> Mapping[str, Any]:
        """Return diagnostic statistics about the network

        Returns:
            Mapping[str, Any]: A dictionary of diagnostic statistics with entries:
                n_species_types [int]: Number of species types
                dict_n_species_per_type [dict[str, int]]: Number of
                    species per species type
                n_species [int]: Number of species
                n_cspecies [int]: Number of compartmentalized species
                n_reaction_species [int]: Number of reaction species
                n_reactions [int]: Number of reactions
                n_compartments [int]: Number of compartments
                dict_n_species_per_compartment [dict[str, int]]:
                    Number of species per compartment
                stats_species_per_reaction [dict[str, float]]:
                    Statistics on the number of reactands per reaction
                top10_species_per_reaction [list[dict[str, Any]]]:
                    Top 10 reactions with highest number of reactands
                stats_degree [dict[str, float]]: Statistics on the degree
                    of a species (number of reactions it is involved in)
                top10_degree [list[dict[str, Any]]]:
                    Top 10 species with highest degree
                stats_identifiers_per_species [dict[str, float]]:
                    Statistics on the number of identifiers per species
                top10_identifiers_per_species [list[dict[str, Any]]]:
                    Top 10 species with highest number of identifiers
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

    def add_species_data(self, label: str, data: pd.DataFrame):
        """Adds additional species_data with validation

        Args:
            label (str): the label for the new data
            data (pd.DataFrame): the data

        Raises:
            ValueError: if the data is not valid, ie does not match with `species`
        """
        self._validate_species_data(data)
        if label in self.species_data:
            raise ValueError(
                f"{label} already exists in species_data. " "Drop it first."
            )
        self.species_data[label] = data

    def add_reactions_data(self, label: str, data: pd.DataFrame):
        """Adds additional reaction_data with validation

        Args:
            label (str): the label for the new data
            data (pd.DataFrame): the data

        Raises:
            ValueError: if the data is not valid, ie does not match with `reactions`
        """
        self._validate_reactions_data(data)
        if label in self.reactions_data:
            raise ValueError(
                f"{label} already exists in reactions_data. Drop it first."
            )
        self.reactions_data[label] = data

    def remove_compartmentalized_species(self, sc_ids: Iterable[str]):
        """
        Starting with a set of compartmentalized species determine which reactions should be removed
        based on there removal. Then remove these reactions, compartmentalized species, and species.

        """

        # find reactions which should be totally removed since they are losing critical species
        removed_reactions = find_underspecified_reactions(self, sc_ids)
        self.remove_reactions(removed_reactions)

        self._remove_compartmentalized_species(sc_ids)

        # remove species (and their associated species data if all their cspecies have been lost)
        self._remove_unused_species()

    def remove_reactions(self, r_ids: Iterable[str], remove_species: bool = False):
        """Removes reactions from the model

        Args:
            r_ids (List[str]): the reactions to remove
            remove_species (bool, optional): whether to remove species that are no longer
                part of any reactions. Defaults to False.
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

    def validate(self):
        """Validates the object for obvious errors"""

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
            table_schema = self.schema[table]
            table_data = getattr(self, table)

            if not isinstance(table_data, pd.DataFrame):
                raise ValueError(
                    f"{table} must be a pd.DataFrame, but was a " f"{type(table_data)}"
                )

            # check index
            expected_index_name = table_schema["pk"]
            if table_data.index.name != expected_index_name:
                raise ValueError(
                    f"the index name for {table} was not the pk: "
                    f"{expected_index_name}"
                )

            # check that all entries in the index are unique
            if len(set(table_data.index.tolist())) != table_data.shape[0]:
                duplicated_pks = table_data.index.value_counts()
                duplicated_pks = duplicated_pks[duplicated_pks > 1]

                example_duplicates = duplicated_pks.index[
                    0 : min(duplicated_pks.shape[0], 5)
                ]
                raise ValueError(
                    f"{duplicated_pks.shape[0]} primary keys were "
                    f"duplicated including {', '.join(example_duplicates)}"
                )

            # check variables
            expected_vars = set(table_schema["vars"])
            table_vars = set(list(table_data.columns))

            extra_vars = table_vars.difference(expected_vars)
            if len(extra_vars) != 0:
                logger.debug(
                    f"{len(extra_vars)} extra variables were found"
                    f" for {table}: {', '.join(extra_vars)}"
                )

            missing_vars = expected_vars.difference(table_vars)
            if len(missing_vars) != 0:
                raise ValueError(
                    f"Missing {len(missing_vars)} required variables"
                    f" for {table}: {', '.join(missing_vars)}"
                )

            # check
            if table_data.shape[0] == 0:
                raise ValueError(f"{table} contained no entries")

        # check whether pks and fks agree

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

    def validate_and_resolve(self):
        """Call validate and try to iteratively resolve common validation errors"""

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

    def _remove_unused_cspecies(self):
        """Removes compartmentalized species that are no
        longer part of any reactions"""
        sc_ids = self._get_unused_cspecies()
        self._remove_compartmentalized_species(sc_ids)

    def _get_unused_cspecies(self) -> set[str]:
        """Returns a set of compartmentalized species
        that are not part of any reactions"""
        sc_ids = set(self.compartmentalized_species.index) - set(
            self.reaction_species[SBML_DFS.SC_ID]
        )
        return sc_ids  # type: ignore

    def _remove_unused_species(self):
        """Removes species that are no longer part of any
        compartmentalized species"""
        s_ids = self._get_unused_species()
        self._remove_species(s_ids)

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

    def _validate_species_data(self, species_data_table: pd.DataFrame):
        """Validates species data attribute

        Args:
            species_data_table (pd.DataFrame): a species data table

        Raises:
            ValueError: s_id not index name
            ValueError: s_id index contains duplicates
            ValueError: s_id not in species table
        """
        _validate_matching_data(species_data_table, self.species)

    def _validate_reactions_data(self, reactions_data_table: pd.DataFrame):
        """Validates reactions data attribute

        Args:
            reactions_data_table (pd.DataFrame): a reactions data table

        Raises:
            ValueError: r_id not index name
            ValueError: r_id index contains duplicates
            ValueError: r_id not in reactions table
        """
        _validate_matching_data(reactions_data_table, self.reactions)

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

    def _attempt_resolve(self, e):
        str_e = str(e)
        if str_e == "compartmentalized_species included missing c_id values":
            logger.warning(str_e)
            logger.warning(
                "Attempting to resolve with infer_uncompartmentalized_species_location()"
            )
            self = infer_uncompartmentalized_species_location(self)
        elif re.search("sbo_terms were not defined", str_e):
            logger.warning(str_e)
            logger.warning("Attempting to resolve with infer_sbo_terms()")
            self = infer_sbo_terms(self)
        else:
            logger.warning(
                "An error occurred which could not be automatically resolved"
            )
            raise e


def species_status(s_id: str, sbml_dfs: SBML_dfs) -> pd.DataFrame:
    """
    Species Status

    Return all of the reaction's a species particpates in.

    Parameters:
    s_id: str
      A species ID
    sbml_dfs: SBML_dfs

    Returns:
    pd.DataFrame, one row reaction
    """

    matching_species = sbml_dfs.species.loc[s_id]

    if not isinstance(matching_species, pd.Series):
        raise ValueError(f"{s_id} did not match a single species")

    # find all rxns species particpate in

    matching_compartmentalized_species = sbml_dfs.compartmentalized_species[
        sbml_dfs.compartmentalized_species.s_id.isin([s_id])
    ]

    rxns_participating = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species.sc_id.isin(matching_compartmentalized_species.index)
    ]

    # find all participants in these rxns

    full_rxns_participating = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species.r_id.isin(rxns_participating[SBML_DFS.R_ID])
    ].merge(
        sbml_dfs.compartmentalized_species, left_on=SBML_DFS.SC_ID, right_index=True
    )

    reaction_descriptions = pd.concat(
        [
            reaction_summary(x, sbml_dfs)
            for x in set(full_rxns_participating[SBML_DFS.R_ID].tolist())
        ]
    )

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


def reaction_summary(r_id: str, sbml_dfs: SBML_dfs) -> pd.DataFrame:
    """
    Reaction Summary

    Return a reaction's name and a human-readable formula.

    Parameters:
    r_id: str
      A reaction ID
    sbml_dfs: SBML_dfs

    Returns:
    one row pd.DataFrame
    """

    logger.warning(
        "reaction_summary is deprecated and will be removed in a future version of rcpr; "
        "please use reaction_summaries() instead"
    )

    matching_reaction = sbml_dfs.reactions.loc[r_id]

    if not isinstance(matching_reaction, pd.Series):
        raise ValueError(f"{r_id} did not match a single reaction")

    matching_reaction = sbml_dfs.reactions.loc[r_id]

    matching_reaction_species = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species.r_id.isin([r_id])
    ].merge(
        sbml_dfs.compartmentalized_species, left_on=SBML_DFS.SC_ID, right_index=True
    )

    # collapse all reaction species to a formula string

    if len(matching_reaction_species[SBML_DFS.C_ID].unique()) == 1:
        augmented_matching_reaction_species = matching_reaction_species.merge(
            sbml_dfs.compartments, left_on=SBML_DFS.C_ID, right_index=True
        ).merge(sbml_dfs.species, left_on=SBML_DFS.S_ID, right_index=True)
        str_formula = (
            construct_formula_string(
                augmented_matching_reaction_species, sbml_dfs.reactions, SBML_DFS.S_NAME
            )
            + " ["
            + augmented_matching_reaction_species[SBML_DFS.C_NAME].iloc[0]
            + "]"
        )
    else:
        str_formula = construct_formula_string(
            matching_reaction_species, sbml_dfs.reactions, SBML_DFS.SC_NAME
        )

    output = pd.DataFrame(
        {
            SBML_DFS.R_NAME: matching_reaction[SBML_DFS.R_NAME],
            "r_formula_str": str_formula,
        },
        index=[r_id],
    )

    output.index.name = SBML_DFS.R_ID

    return output


def reaction_summaries(sbml_dfs: SBML_dfs, r_ids=None) -> pd.Series:
    """
    Reaction Summary

    Return human-readable formulas for reactions.

    Parameters:
    ----------
    sbml_dfs: sbml.SBML_dfs
        A relational mechanistic model
    r_ids: [str], str or None
        Reaction IDs or None for all reactions

    Returns:
    ----------
    formula_strs: pd.Series
    """

    if isinstance(r_ids, str):
        r_ids = [r_ids]

    if r_ids is None:
        matching_reactions = sbml_dfs.reactions
    else:
        matching_reactions = sbml_dfs.reactions.loc[r_ids]

    matching_reaction_species = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species.r_id.isin(matching_reactions.index)
    ].merge(
        sbml_dfs.compartmentalized_species, left_on=SBML_DFS.SC_ID, right_index=True
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
                lambda x: construct_formula_string(
                    x, sbml_dfs.reactions, SBML_DFS.SC_NAME
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
            .merge(sbml_dfs.compartments, left_on=SBML_DFS.C_ID, right_index=True)
            .merge(sbml_dfs.species, left_on=SBML_DFS.S_ID, right_index=True)
            .sort_values([SBML_DFS.S_NAME])
        )
        # create formulas based on s_names of components
        rxn_eqtn_within_compartment = augmented_matching_reaction_species.groupby(
            [SBML_DFS.R_ID, SBML_DFS.C_NAME]
        ).apply(
            lambda x: construct_formula_string(x, sbml_dfs.reactions, SBML_DFS.S_NAME)
        )
        # add compartment for each reaction
        rxn_eqtn_within_compartment = pd.Series(
            [
                y + ": " + x
                for x, y in zip(
                    rxn_eqtn_within_compartment,
                    rxn_eqtn_within_compartment.index.get_level_values(SBML_DFS.C_NAME),
                )
            ],
            index=rxn_eqtn_within_compartment.index.get_level_values(SBML_DFS.R_ID),
        ).rename("r_formula_str")
    else:
        rxn_eqtn_within_compartment = None

    formula_strs = pd.concat([rxn_eqtn_cross_compartment, rxn_eqtn_within_compartment])

    return formula_strs


def construct_formula_string(
    reaction_species_df: pd.DataFrame,
    reactions_df: pd.DataFrame,
    name_var: str,
) -> str:
    """
    Construct Formula String

    Convert a table of reaction species into a formula string

    Parameters:
    ----------
    reaction_species_df: pd.DataFrame
        Table containing a reactions' species
    reactions_df: pd.DataFrame
        smbl.reactions
    name_var: str
        Name used to label species

    Returns:
    ----------
    formula_str: str
        String representation of a reactions substrates, products and
        modifiers

    """

    reaction_species_df["label"] = [
        add_stoi_to_species_name(x, y)
        for x, y in zip(
            reaction_species_df[SBML_DFS.STOICHIOMETRY], reaction_species_df[name_var]
        )
    ]

    rxn_reversible = bool(
        reactions_df.loc[
            reaction_species_df[SBML_DFS.R_ID].iloc[0], SBML_DFS.R_ISREVERSIBLE
        ]
    )  # convert from a np.bool_ to bool if needed
    if not isinstance(rxn_reversible, bool):
        raise TypeError(
            f"rxn_reversible must be a bool, but got {type(rxn_reversible).__name__}"
        )

    if rxn_reversible:
        arrow_type = " <-> "
    else:
        arrow_type = " -> "

    substrates = " + ".join(
        reaction_species_df["label"][
            reaction_species_df[SBML_DFS.STOICHIOMETRY] < 0
        ].tolist()
    )
    products = " + ".join(
        reaction_species_df["label"][
            reaction_species_df[SBML_DFS.STOICHIOMETRY] > 0
        ].tolist()
    )
    modifiers = " + ".join(
        reaction_species_df["label"][
            reaction_species_df[SBML_DFS.STOICHIOMETRY] == 0
        ].tolist()
    )
    if modifiers != "":
        modifiers = f" ---- modifiers: {modifiers}]"

    return f"{substrates}{arrow_type}{products}{modifiers}"


def add_stoi_to_species_name(stoi: float | int, name: str) -> str:
    """
    Add Stoi To Species Name

    Add # of molecules to a species name

    Parameters:
    ----------
    stoi: float or int
        Number of molecules
    name: str
        Name of species

    Returns:
    ----------
    name: str
        Name containing number of species

    """

    if stoi in [-1, 0, 1]:
        return name
    else:
        return str(abs(stoi)) + " " + name


def filter_to_characteristic_species_ids(
    species_ids: pd.DataFrame,
    max_complex_size: int = 4,
    max_promiscuity: int = 20,
    defining_biological_qualifiers: list[str] = BQB_DEFINING_ATTRS,
) -> pd.DataFrame:
    """
    Filter to Characteristic Species IDs

    Remove identifiers corresponding to one component within a large protein
    complexes and non-characteristic annotations such as pubmed references and
    homologues.

    Parameters
    ----------
    species_ids: pd.DataFrame
        A table of identifiers produced by sdbml_dfs.get_identifiers("species")
    max_complex_size: int
        The largest size of a complex, where BQB_HAS_PART terms will be retained.
        In most cases, complexes are handled with specific formation and
        dissolutation reactions,but these identifiers will be pulled in when
        searching by identifiers or searching the identifiers associated with a
        species against an external resource such as Open Targets.
    max_promiscuity: int
        Maximum number of species where a single molecule can act as a
        BQB_HAS_PART component associated with a single identifier (and common ontology).
    defining_biological_qualifiers (list[str]):
        BQB codes which define distinct entities. Narrowly this would be BQB_IS, while more
        permissive settings would include homologs, different forms of the same gene.

    Returns:
    --------
    species_id: pd.DataFrame
        Input species filtered to characteristic identifiers

    """

    if not isinstance(species_ids, pd.DataFrame):
        raise TypeError(
            f"species_ids was a {type(species_ids)} but must be a pd.DataFrame"
        )

    if not isinstance(max_complex_size, int):
        raise TypeError(
            f"max_complex_size was a {type(max_complex_size)} but must be an int"
        )

    if not isinstance(max_promiscuity, int):
        raise TypeError(
            f"max_promiscuity was a {type(max_promiscuity)} but must be an int"
        )

    if not isinstance(defining_biological_qualifiers, list):
        raise TypeError(
            f"defining_biological_qualifiers was a {type(defining_biological_qualifiers)} but must be a list"
        )

    # primary annotations of a species
    bqb_is_species = species_ids.query("bqb in @defining_biological_qualifiers")

    # add components within modestly sized protein complexes
    # look at HAS_PART IDs
    bqb_has_parts_species = species_ids[species_ids[IDENTIFIERS.BQB] == BQB.HAS_PART]
    # filter to genes
    bqb_has_parts_species = bqb_has_parts_species[
        bqb_has_parts_species[IDENTIFIERS.ONTOLOGY].isin(
            CHARACTERISTIC_COMPLEX_ONTOLOGIES
        )
    ]

    # number of species in a complex
    n_species_components = bqb_has_parts_species.value_counts(
        [IDENTIFIERS.ONTOLOGY, SBML_DFS.S_ID]
    )
    big_complex_sids = set(
        n_species_components[
            n_species_components > max_complex_size
        ].index.get_level_values(SBML_DFS.S_ID)
    )

    # number of complexes a species is part of
    n_complexes_involvedin = bqb_has_parts_species.value_counts(
        [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]
    )
    promiscuous_component_identifiers_index = n_complexes_involvedin[
        n_complexes_involvedin > max_promiscuity
    ].index
    promiscuous_component_identifiers = pd.Series(
        data=[True] * len(promiscuous_component_identifiers_index),
        index=promiscuous_component_identifiers_index,
        name="is_shared_component",
        dtype=bool,
    )

    if len(promiscuous_component_identifiers) == 0:
        # no complexes to filter
        return species_ids

    filtered_bqb_has_parts = bqb_has_parts_species.merge(
        promiscuous_component_identifiers,
        left_on=[IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER],
        right_index=True,
        how="left",
    )

    filtered_bqb_has_parts["is_shared_component"] = filtered_bqb_has_parts[
        "is_shared_component"
    ].fillna(False)
    # drop identifiers shared as components across many species
    filtered_bqb_has_parts = filtered_bqb_has_parts[
        ~filtered_bqb_has_parts["is_shared_component"]
    ].drop(["is_shared_component"], axis=1)
    # drop species parts if there are many components
    filtered_bqb_has_parts = filtered_bqb_has_parts[
        ~filtered_bqb_has_parts[SBML_DFS.S_ID].isin(big_complex_sids)
    ]

    # combine primary identifiers and rare components
    characteristic_species_ids = pd.concat(
        [
            bqb_is_species,
            filtered_bqb_has_parts,
        ]
    )

    return characteristic_species_ids


def infer_uncompartmentalized_species_location(sbml_dfs: SBML_dfs) -> SBML_dfs:
    """
    Infer Uncompartmentalized Species Location

    If the compartment of a subset of compartmentalized species
    was not specified, infer an appropriate compartment from
    other members of reactions they particpate in

    Parameters:
    ----------
    sbml_dfs: sbml.SBML_dfs
        A relational pathway model

    Returns:
    ----------
    sbml_dfs: sbml.SBML_dfs
        A relational pathway model (with filled in species compartments)

    """

    default_compartment = (
        sbml_dfs.compartmentalized_species.value_counts(SBML_DFS.C_ID)
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

    missing_compartment_scids = sbml_dfs.compartmentalized_species[
        sbml_dfs.compartmentalized_species[SBML_DFS.C_ID].isnull()
    ].index.tolist()
    if len(missing_compartment_scids) == 0:
        logger.info(
            "All compartmentalized species have compartments, "
            "returning input sbml_dfs"
        )
        return sbml_dfs

    participating_reactions = (
        sbml_dfs.reaction_species[
            sbml_dfs.reaction_species[SBML_DFS.SC_ID].isin(missing_compartment_scids)
        ][SBML_DFS.R_ID]
        .unique()
        .tolist()
    )
    reaction_participants = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species[SBML_DFS.R_ID].isin(participating_reactions)
    ].reset_index(drop=True)[[SBML_DFS.SC_ID, SBML_DFS.R_ID]]
    reaction_participants = reaction_participants.merge(
        sbml_dfs.compartmentalized_species[SBML_DFS.C_ID],
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
        sbml_dfs.reaction_species[
            sbml_dfs.reaction_species[SBML_DFS.SC_ID].isin(missing_compartment_scids)
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

    # define where a reaction is most likely to occur based on the compartmentalization of its particpants
    species_with_unknown_compartmentalization = set(
        missing_compartment_scids
    ).difference(set(inferred_compartmentalization[SBML_DFS.SC_ID].tolist()))
    if len(species_with_unknown_compartmentalization) != 0:
        logger.warning(
            f"{len(species_with_unknown_compartmentalization)} "
            "species compartmentalization could not be inferred"
            " from other reaction particpants. Their compartmentalization "
            f"will be set to the default of {default_compartment}"
        )

        inferred_compartmentalization = pd.concat(
            [
                inferred_compartmentalization,
                pd.DataFrame(
                    {SBML_DFS.SC_ID: list(species_with_unknown_compartmentalization)}
                ).assign(c_id=default_compartment),
            ]
        )

    if len(missing_compartment_scids) != inferred_compartmentalization.shape[0]:
        raise ValueError(
            f"{inferred_compartmentalization.shape[0]} were inferred but {len(missing_compartment_scids)} are required"
        )

    updated_compartmentalized_species = pd.concat(
        [
            sbml_dfs.compartmentalized_species[
                ~sbml_dfs.compartmentalized_species[SBML_DFS.C_ID].isnull()
            ],
            sbml_dfs.compartmentalized_species[
                sbml_dfs.compartmentalized_species[SBML_DFS.C_ID].isnull()
            ]
            .drop(SBML_DFS.C_ID, axis=1)
            .merge(
                inferred_compartmentalization, left_index=True, right_on=SBML_DFS.SC_ID
            )
            .set_index(SBML_DFS.SC_ID),
        ]
    )

    if (
        updated_compartmentalized_species.shape[0]
        != sbml_dfs.compartmentalized_species.shape[0]
    ):
        raise ValueError(
            f"Trying to overwrite {sbml_dfs.compartmentalized_species.shape[0]}"
            " compartmentalized species with "
            f"{updated_compartmentalized_species.shape[0]}"
        )

    if any(updated_compartmentalized_species[SBML_DFS.C_ID].isnull()):
        raise ValueError("Some species compartments are still missing")

    sbml_dfs.compartmentalized_species = updated_compartmentalized_species

    return sbml_dfs


def infer_sbo_terms(sbml_dfs: SBML_dfs) -> SBML_dfs:
    """
    Infer SBO Terms

    Define SBO terms based on stoichiometry for reaction_species with missing terms

    Parameters:
    ----------
    sbml_dfs: sbml.SBML_dfs
        A relational pathway model

    Returns:
    ----------
    sbml_dfs: sbml.SBML_dfs
        A relational pathway model (with missing/invalid reaction species sbo_terms resolved)

    """

    valid_sbo_terms = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species[SBML_DFS.SBO_TERM].isin(MINI_SBO_TO_NAME.keys())
    ]

    invalid_sbo_terms = sbml_dfs.reaction_species[
        ~sbml_dfs.reaction_species[SBML_DFS.SBO_TERM].isin(MINI_SBO_TO_NAME.keys())
    ]

    if not all(sbml_dfs.reaction_species[SBML_DFS.SBO_TERM].notnull()):
        raise ValueError(
            "All sbml_dfs.reaction_species[SBML_DFS.SBO_TERM] must be not null"
        )
    if invalid_sbo_terms.shape[0] == 0:
        logger.info("All sbo_terms were valid; returning input sbml_dfs")
        return sbml_dfs

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

    if sbml_dfs.reaction_species.shape[0] != updated_reaction_species.shape[0]:
        raise ValueError(
            f"Trying to overwrite {sbml_dfs.reaction_species.shape[0]} reaction_species with {updated_reaction_species.shape[0]}"
        )
    sbml_dfs.reaction_species = updated_reaction_species

    return sbml_dfs


def name_compartmentalized_species(sbml_dfs):
    """
    Name Compartmentalized Species

    Rename compartmentalized species if they have the same
    name as their species

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        A model formed by aggregating pathways

    Returns:
    ----------
    sbml_dfs
    """

    augmented_cspecies = sbml_dfs.compartmentalized_species.merge(
        sbml_dfs.species[SBML_DFS.S_NAME], left_on=SBML_DFS.S_ID, right_index=True
    ).merge(
        sbml_dfs.compartments[SBML_DFS.C_NAME], left_on=SBML_DFS.C_ID, right_index=True
    )
    augmented_cspecies[SBML_DFS.SC_NAME] = [
        f"{s} [{c}]" if sc == s else sc
        for sc, c, s in zip(
            augmented_cspecies[SBML_DFS.SC_NAME],
            augmented_cspecies[SBML_DFS.C_NAME],
            augmented_cspecies[SBML_DFS.S_NAME],
        )
    ]

    sbml_dfs.compartmentalized_species = augmented_cspecies.loc[
        :, sbml_dfs.schema[SBML_DFS.COMPARTMENTALIZED_SPECIES]["vars"]
    ]

    return sbml_dfs


def export_sbml_dfs(
    model_prefix: str,
    sbml_dfs: SBML_dfs,
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
    sbml_dfs: sbml.SBML_dfs
        A pathway model
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
        raise TypeError(f"model_prefix was a {type(model_prefix)} " "and must be a str")
    if not isinstance(sbml_dfs, SBML_dfs):
        raise TypeError(
            f"sbml_dfs was a {type(sbml_dfs)} and must" " be an sbml.SBML_dfs"
        )

    # filter to identifiers which make sense when mapping from ids -> species
    species_identifiers = sbml_dfs_utils.get_characteristic_species_ids(
        sbml_dfs,
        dogmatic=dogmatic,
    )

    try:
        utils.initialize_dir(outdir, overwrite=overwrite)
    except FileExistsError:
        logger.warning(
            f"Directory {outdir} already exists and overwrite is False. "
            "Files will be added to the existing directory."
        )
    with open_fs(outdir, writeable=True) as fs:
        species_identifiers_path = (
            model_prefix + CPR_STANDARD_OUTPUTS.SPECIES_IDENTIFIERS
        )
        with fs.openbin(species_identifiers_path, "w") as f:
            species_identifiers.drop([SBML_DFS.S_SOURCE], axis=1).to_csv(
                f, sep="\t", index=False
            )

        # export jsons
        species_path = model_prefix + CPR_STANDARD_OUTPUTS.SPECIES
        reactions_path = model_prefix + CPR_STANDARD_OUTPUTS.REACTIONS
        reation_species_path = model_prefix + CPR_STANDARD_OUTPUTS.REACTION_SPECIES
        compartments_path = model_prefix + CPR_STANDARD_OUTPUTS.COMPARTMENTS
        compartmentalized_species_path = (
            model_prefix + CPR_STANDARD_OUTPUTS.COMPARTMENTALIZED_SPECIES
        )
        with fs.openbin(species_path, "w") as f:
            sbml_dfs.species[[SBML_DFS.S_NAME]].to_json(f)

        with fs.openbin(reactions_path, "w") as f:
            sbml_dfs.reactions[[SBML_DFS.R_NAME]].to_json(f)

        with fs.openbin(reation_species_path, "w") as f:
            sbml_dfs.reaction_species.to_json(f)

        with fs.openbin(compartments_path, "w") as f:
            sbml_dfs.compartments[[SBML_DFS.C_NAME]].to_json(f)

        with fs.openbin(compartmentalized_species_path, "w") as f:
            sbml_dfs.compartmentalized_species.drop(SBML_DFS.SC_SOURCE, axis=1).to_json(
                f
            )

    return None


def sbml_dfs_from_edgelist(
    interaction_edgelist: pd.DataFrame,
    species_df: pd.DataFrame,
    compartments_df: pd.DataFrame,
    interaction_source: source.Source,
    upstream_stoichiometry: int = 0,
    downstream_stoichiometry: int = 1,
    downstream_sbo_name: str = SBOTERM_NAMES.PRODUCT,
    keep_species_data: bool | str = False,
    keep_reactions_data: bool | str = False,
) -> SBML_dfs:
    """
    Create SBML_dfs from Edgelist

    Combine a set of interactions into an sbml.SBML_dfs mechanistic model

    Parameters:
    interaction_edgelist (pd.DataFrame): A table containing interactions:
        - upstream_name (str): matching "s_name" from "species_df"
        - downstream_name (str): matching "s_name" from "species_df"
        - upstream_compartment (str): compartment of "upstream_name"
            with names matching "c_name" from "compartments_df"
        - downstream_compartment (str): compartment of "downstream_name"
            with names matching "c_name" from "compartments_df"
        - r_name (str): a name for the interaction
        - sbo_term (str): sbo term defining the type of
            molecular interaction (see MINI_SBO_FROM_NAME)
        - r_Identifiers (identifiers.Identifiers): identifiers
            supporting the interaction (e.g., pubmed ids)
        - r_isreversible (bool): Is this reaction reversible?
            If True, the reaction is reversible
            By default, the interactions of TRRUST networks are irreversible, and reversible for STRING networks
    species_df (pd.DataFrame): A table defining unique molecular
        species participating in "interaction_edgelist":
        - s_name (str): name of molecular species
        - s_Identifiers (identifiers.Identifiers): identifiers
            defining the species
    compartments_df (pd.DataFrame): A table defining compartments
        where interactions are occurring "interaction_edgelist":
        - c_name (str): name of compartment
        - c_Identifiers (identifiers.Identifiers):
            identifiers defining the compartment (see
            bigg.annotate_recon() for a set of names > go categories)
    interaction_source (source.Source): A source object
        which will tie model entities to the interaction source
    upstream_stoichiometry (int): stoichiometry of
        upstream species in reaction
    downstream_stoichiometry (int): stoichiometry of
        downstream species in reaction
    downstream_sbo_name (str): sbo term defining the
        type of molecular interaction for the downstream reactand
        (see MINI_SBO_FROM_NAME)
    keep_species_data (bool | str): Should species data
        be kept in the model? If True, all species data will be kept
        and saved as "species_data" in the SBML_dfs. The label will be 'source'
        If False, no species data will be kept.
        If a string: label for the species data to be kept.
    keep_reactions_data (bool | str): Should reaction data be kept in the model?
        If True, all reaction data will be kept and saved
        as "reactions_data" in the SBML_dfs. The label will be 'source'.
        If False, no reaction data will be kept.
        If a string: label for the reaction data to be kept.

    Returns:
    sbml.SBML_dfs

    """

    # check input dfs for required variables
    _sbml_dfs_from_edgelist_validate_inputs(
        interaction_edgelist, species_df, compartments_df
    )

    # Identify extra columns in the input data.
    # if keep_reactions_data is True, this will be added
    # as `reaction_data`
    interaction_edgelist_required_vars = {
        "upstream_name",
        "downstream_name",
        "upstream_compartment",
        "downstream_compartment",
        SBML_DFS.R_NAME,
        SBML_DFS.SBO_TERM,
        SBML_DFS.R_IDENTIFIERS,
        SBML_DFS.R_ISREVERSIBLE,
    }
    if keep_reactions_data is not False:
        extra_reactions_columns = [
            c
            for c in interaction_edgelist.columns
            if c not in interaction_edgelist_required_vars
        ]
    else:
        extra_reactions_columns = []
    # Extra species columns
    if keep_species_data is not False:
        extra_species_columns = [
            c
            for c in species_df.columns
            if c not in {SBML_DFS.S_NAME, SBML_DFS.S_IDENTIFIERS}
        ]
    else:
        extra_species_columns = []

    # format compartments
    compartments_df[SBML_DFS.C_SOURCE] = interaction_source
    compartments_df[SBML_DFS.C_ID] = sbml_dfs_utils.id_formatter(
        range(compartments_df.shape[0]), SBML_DFS.C_ID
    )
    compartments_df = compartments_df.set_index(SBML_DFS.C_ID)[
        [SBML_DFS.C_NAME, SBML_DFS.C_IDENTIFIERS, SBML_DFS.C_SOURCE]
    ]

    # format species
    species_df[SBML_DFS.S_SOURCE] = interaction_source
    species_df[SBML_DFS.S_ID] = sbml_dfs_utils.id_formatter(
        range(species_df.shape[0]), SBML_DFS.S_ID
    )

    required_cols = [SBML_DFS.S_NAME, SBML_DFS.S_IDENTIFIERS, SBML_DFS.S_SOURCE]
    species_df = species_df.set_index(SBML_DFS.S_ID)[
        required_cols + extra_species_columns
    ]
    # Keep extra columns to save them as extra data
    species_data = species_df[extra_species_columns]
    # Remove extra columns
    species_df = species_df[required_cols]

    # create compartmentalized species

    # define all distinct upstream and downstream compartmentalized species
    comp_species = pd.concat(
        [
            interaction_edgelist[["upstream_name", "upstream_compartment"]].rename(
                {
                    "upstream_name": SBML_DFS.S_NAME,
                    "upstream_compartment": SBML_DFS.C_NAME,
                },
                axis=1,
            ),
            interaction_edgelist[["downstream_name", "downstream_compartment"]].rename(
                {
                    "downstream_name": SBML_DFS.S_NAME,
                    "downstream_compartment": SBML_DFS.C_NAME,
                },
                axis=1,
            ),
        ]
    ).drop_duplicates()

    # merge to add species and compartments primary keys
    comp_species_w_ids = comp_species.merge(
        species_df[SBML_DFS.S_NAME].reset_index(),
        how="left",
        left_on=SBML_DFS.S_NAME,
        right_on=SBML_DFS.S_NAME,
    ).merge(
        compartments_df[SBML_DFS.C_NAME].reset_index(),
        how="left",
        left_on=SBML_DFS.C_NAME,
        right_on=SBML_DFS.C_NAME,
    )

    # check whether all species and compartments exist
    _sbml_dfs_from_edgelist_check_cspecies_merge(comp_species_w_ids, comp_species)

    # name compounds
    comp_species_w_ids[SBML_DFS.SC_NAME] = [
        f"{s} [{c}]"
        for s, c in zip(
            comp_species_w_ids[SBML_DFS.S_NAME], comp_species_w_ids[SBML_DFS.C_NAME]
        )
    ]
    # add source object
    comp_species_w_ids[SBML_DFS.SC_SOURCE] = interaction_source
    # name index
    comp_species_w_ids[SBML_DFS.SC_ID] = sbml_dfs_utils.id_formatter(
        range(comp_species_w_ids.shape[0]), SBML_DFS.SC_ID
    )
    comp_species_w_ids = comp_species_w_ids.set_index(SBML_DFS.SC_ID)[
        [SBML_DFS.SC_NAME, SBML_DFS.S_ID, SBML_DFS.C_ID, SBML_DFS.SC_SOURCE]
    ]

    # create reactions

    # create a from cs_species -> to cs_species edgelist
    # interaction_edgelist
    comp_species_w_names = (
        comp_species_w_ids.reset_index()
        .merge(species_df[SBML_DFS.S_NAME].reset_index())
        .merge(compartments_df[SBML_DFS.C_NAME].reset_index())
    )

    interaction_edgelist_w_cspecies = interaction_edgelist.merge(
        comp_species_w_names[[SBML_DFS.SC_ID, SBML_DFS.S_NAME, SBML_DFS.C_NAME]].rename(
            {
                SBML_DFS.SC_ID: "sc_id_up",
                SBML_DFS.S_NAME: "upstream_name",
                SBML_DFS.C_NAME: "upstream_compartment",
            },
            axis=1,
        ),
        how="left",
    ).merge(
        comp_species_w_names[[SBML_DFS.SC_ID, SBML_DFS.S_NAME, SBML_DFS.C_NAME]].rename(
            {
                SBML_DFS.SC_ID: "sc_id_down",
                SBML_DFS.S_NAME: "downstream_name",
                SBML_DFS.C_NAME: "downstream_compartment",
            },
            axis=1,
        ),
        how="left",
    )[
        REQUIRED_REACTION_FROMEDGELIST_COLUMNS + extra_reactions_columns
    ]

    # some extra checks
    if interaction_edgelist.shape[0] != interaction_edgelist_w_cspecies.shape[0]:
        raise ValueError(
            "Merging compartmentalized species to interaction_edgelist"
            " resulted in an increase in the tables from "
            f"{interaction_edgelist.shape[0]} to "
            f"{interaction_edgelist_w_cspecies.shape[0]} indicating"
            " a 1-many join which should have been 1-1"
        )

    # create one reaction per interaction
    interaction_edgelist_w_cspecies[SBML_DFS.R_SOURCE] = interaction_source
    interaction_edgelist_w_cspecies[SBML_DFS.R_ID] = sbml_dfs_utils.id_formatter(
        range(interaction_edgelist_w_cspecies.shape[0]), SBML_DFS.R_ID
    )

    reactions_df_columns = [
        SBML_DFS.R_NAME,
        SBML_DFS.R_IDENTIFIERS,
        SBML_DFS.R_SOURCE,
        SBML_DFS.R_ISREVERSIBLE,
    ]
    reactions_df = interaction_edgelist_w_cspecies.copy().set_index(SBML_DFS.R_ID)[
        reactions_df_columns + extra_reactions_columns
    ]
    # Keep extra columns to save them as extra data
    reactions_data = reactions_df[extra_reactions_columns]
    reactions_df = reactions_df[reactions_df_columns]

    # define upstream and downstream comp species as reaction species
    reaction_species_df = pd.concat(
        [
            # upstream interactions are defined by sbo_term and should generally
            # be modifiers/stimulator/inhibitor/interactor
            interaction_edgelist_w_cspecies[["sc_id_up", "sbo_term", "r_id"]]
            .assign(stoichiometry=upstream_stoichiometry)
            .rename({"sc_id_up": "sc_id"}, axis=1),
            # downstream interactions indicate some modification of the state
            # of the species and hence are defined as product
            interaction_edgelist_w_cspecies[["sc_id_down", "r_id"]]
            .assign(
                stoichiometry=downstream_stoichiometry,
                sbo_term=MINI_SBO_FROM_NAME[downstream_sbo_name],
            )
            .rename({"sc_id_down": "sc_id"}, axis=1),
        ]
    )
    reaction_species_df["rsc_id"] = sbml_dfs_utils.id_formatter(
        range(reaction_species_df.shape[0]), "rsc_id"
    )
    reaction_species_df = reaction_species_df.set_index("rsc_id")

    # form sbml_dfs object
    sbml_tbl_dict: MutableMapping[str, pd.DataFrame | dict[str, pd.DataFrame]] = {
        "compartments": compartments_df,
        "species": species_df,
        "compartmentalized_species": comp_species_w_ids,
        "reactions": reactions_df,
        "reaction_species": reaction_species_df,
    }
    if len(extra_reactions_columns) > 0:
        if isinstance(keep_reactions_data, str):
            reactions_data_label = keep_reactions_data
        else:
            reactions_data_label = "source"
        sbml_tbl_dict["reactions_data"] = {reactions_data_label: reactions_data}

    if len(extra_species_columns) > 0:
        if isinstance(keep_species_data, str):
            species_data_label = keep_species_data
        else:
            species_data_label = "source"
        sbml_tbl_dict["species_data"] = {species_data_label: species_data}

    sbml_model = SBML_dfs(sbml_tbl_dict)
    sbml_model.validate()

    return sbml_model


def find_underspecified_reactions(
    sbml_dfs: SBML_dfs, sc_ids: Iterable[str]
) -> set[str]:
    """
    Find Underspecified reactions

    Identity reactions which should be removed if a set of molecular species are removed
    from the system.

    Params:
    sbml_dfs (SBML_dfs):
        A pathway representation
    sc_ids (list[str])
        A list of compartmentalized species ids (sc_ids) which will be removed.

    Returns:
    underspecified_reactions (set[str]):
        A list of reactions which should be removed because they will not occur once
        \"sc_ids\" are removed.

    """

    updated_reaction_species = sbml_dfs.reaction_species.copy()
    updated_reaction_species["new"] = ~updated_reaction_species[SBML_DFS.SC_ID].isin(
        sc_ids
    )

    updated_reaction_species = (
        updated_reaction_species.assign(
            sbo_role=updated_reaction_species[SBML_DFS.SBO_TERM]
        )
        .replace({"sbo_role": MINI_SBO_TO_NAME})
        .replace({"sbo_role": SBO_NAME_TO_ROLE})
    )

    reactions_with_lost_defining_members = set(
        updated_reaction_species.query("~new")
        .query("sbo_role == 'DEFINING'")[SBML_DFS.R_ID]
        .tolist()
    )

    N_reactions_with_lost_defining_members = len(reactions_with_lost_defining_members)
    if N_reactions_with_lost_defining_members > 0:
        logger.info(
            f"Removing {N_reactions_with_lost_defining_members} reactions which have lost at least one defining species"
        )

    # for each reaction what are the required sbo_terms?
    reactions_with_requirements = (
        updated_reaction_species.query("sbo_role == 'REQUIRED'")[
            ["r_id", "sbo_term", "new"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # which required members are still present after removing some entries
    reactions_with_lost_requirements = set(
        reactions_with_requirements.query("~new")
        .merge(
            reactions_with_requirements.query("new").rename(
                {"new": "still_present"}, axis=1
            ),
            how="left",
        )
        .fillna(False)[SBML_DFS.R_ID]  # Fill boolean column with False
        .tolist()
    )

    N_reactions_with_lost_requirements = len(reactions_with_lost_requirements)
    if N_reactions_with_lost_requirements > 0:
        logger.info(
            f"Removing {N_reactions_with_lost_requirements} reactions which have lost all required members"
        )

    underspecified_reactions = reactions_with_lost_defining_members.union(
        reactions_with_lost_requirements
    )

    return underspecified_reactions


def _sbml_dfs_from_edgelist_validate_inputs(
    interaction_edgelist: pd.DataFrame,
    species_df: pd.DataFrame,
    compartments_df: pd.DataFrame,
) -> None:
    """Check that the inputs for creating an SBML_dfs from an edgelist are appropriate."""

    # check compartments
    compartments_df_expected_vars = {SBML_DFS.C_NAME, SBML_DFS.C_IDENTIFIERS}
    compartments_df_columns = set(compartments_df.columns.tolist())
    missing_required_fields = compartments_df_expected_vars.difference(
        compartments_df_columns
    )
    if len(missing_required_fields) > 0:
        raise ValueError(
            f"{', '.join(missing_required_fields)} are required variables"
            ' in "compartments_df" but were not present in the input file.'
        )

    # check species
    species_df_expected_vars = {SBML_DFS.S_NAME, SBML_DFS.S_IDENTIFIERS}
    species_df_columns = set(species_df.columns.tolist())
    missing_required_fields = species_df_expected_vars.difference(species_df_columns)
    if len(missing_required_fields) > 0:
        raise ValueError(
            f"{', '.join(missing_required_fields)} are required"
            ' variables in "species_df" but were not present '
            "in the input file."
        )

    # check interactions
    interaction_edgelist_columns = set(interaction_edgelist.columns.tolist())
    missing_required_fields = INTERACTION_EDGELIST_EXPECTED_VARS.difference(
        interaction_edgelist_columns
    )
    if len(missing_required_fields) > 0:
        raise ValueError(
            f"{', '.join(missing_required_fields)} are required "
            'variables in "interaction_edgelist" but were not '
            "present in the input file."
        )

    return None


def _sbml_dfs_from_edgelist_check_cspecies_merge(
    merged_species: pd.DataFrame, original_species: pd.DataFrame
) -> None:
    """Check for a mismatch between the provided species data and species implied by the edgelist."""

    # check for 1-many merge
    if merged_species.shape[0] != original_species.shape[0]:
        raise ValueError(
            "Merging compartmentalized species to species_df"
            " and compartments_df by names resulted in an "
            f"increase in the tables from {original_species.shape[0]}"
            f" to {merged_species.shape[0]} indicating that names were"
            " not unique"
        )

    # check for missing species and compartments
    missing_compartments = merged_species[merged_species[SBML_DFS.C_ID].isna()][
        SBML_DFS.C_NAME
    ].unique()
    if len(missing_compartments) >= 1:
        raise ValueError(
            f"{len(missing_compartments)} compartments were present in"
            ' "interaction_edgelist" but not "compartments_df":'
            f" {', '.join(missing_compartments)}"
        )

    missing_species = merged_species[merged_species[SBML_DFS.S_ID].isna()][
        SBML_DFS.S_NAME
    ].unique()
    if len(missing_species) >= 1:
        raise ValueError(
            f"{len(missing_species)} species were present in "
            '"interaction_edgelist" but not "species_df":'
            f" {', '.join(missing_species)}"
        )

    return None


def _stub_compartments(
    stubbed_compartment: str = "CELLULAR_COMPONENT",
) -> pd.DataFrame:
    """Stub Compartments

    Create a compartments table with only a single compartment

    Args:
    stubbed_compartment (str): the name of a compartment which should match the
        keys in constants.COMPARTMENTS and constants.COMPARTMENTS_GO_TERMS

    Returns:
    compartments_df (pd.DataFrame): compartments dataframe
    """

    if stubbed_compartment not in COMPARTMENT_ALIASES.keys():
        raise ValueError(
            f"{stubbed_compartment} is not defined in constants.COMPARTMENTS"
        )

    if stubbed_compartment not in COMPARTMENTS_GO_TERMS.keys():
        raise ValueError(
            f"{stubbed_compartment} is not defined in constants.COMPARTMENTS_GO_TERMS"
        )

    stubbed_compartment_name = COMPARTMENTS[stubbed_compartment]
    stubbed_compartment_id = COMPARTMENTS_GO_TERMS[stubbed_compartment]

    formatted_uri = identifiers.format_uri(
        uri=identifiers.create_uri_url(
            ontology=ONTOLOGIES.GO,
            identifier=stubbed_compartment_id,
        ),
        biological_qualifier_type=BQB.IS,
    )

    compartments_df = pd.DataFrame(
        {
            SBML_DFS.C_NAME: [stubbed_compartment_name],
            SBML_DFS.C_IDENTIFIERS: [identifiers.Identifiers([formatted_uri])],
        }
    )
    compartments_df.index = sbml_dfs_utils.id_formatter([0], SBML_DFS.C_ID)  # type: ignore
    compartments_df.index.name = SBML_DFS.C_ID

    return compartments_df


def _validate_matching_data(data_table: pd.DataFrame, ref_table: pd.DataFrame):
    """Validates a table against a reference

    This check if the table has the same index, no duplicates in the index
    and that all values in the index are in the reference table.

    Args:
        data_table (pd.DataFrame): a table with data that should
            match the reference
        ref_table (pd.DataFrame): a reference table

    Raises:
        ValueError: not same index name
        ValueError: index contains duplicates
        ValueError: index not subset of index of reactions table
    """
    ref_index_name = ref_table.index.name
    if data_table.index.name != ref_index_name:
        raise ValueError(
            "the index name for reaction data table was not"
            f" {ref_index_name}: {data_table.index.name}"
        )
    ids = data_table.index
    if any(ids.duplicated()):
        raise ValueError(
            "the index for reaction data table " "contained duplicate values"
        )
    if not all(ids.isin(ref_table.index)):
        raise ValueError(
            "the index for reaction data table contained values"
            " not found in the reactions table"
        )
    if not isinstance(data_table, pd.DataFrame):
        raise TypeError(
            f"The data table was type {type(data_table).__name__}"
            " but must be a pd.DataFrame"
        )


def species_type_types(x):
    """Assign a high-level molecule type to a molecular species"""

    if isinstance(x, identifiers.Identifiers):
        if x.filter(["chebi"]):
            return "metabolite"
        elif x.filter(["molodex"]):
            return "drug"
        else:
            return "protein"
    else:
        return "unknown"


def stub_ids(ids):
    if len(ids) == 0:
        return pd.DataFrame(
            {
                IDENTIFIERS.ONTOLOGY: [None],
                IDENTIFIERS.IDENTIFIER: [None],
                IDENTIFIERS.URL: [None],
                IDENTIFIERS.BQB: [None],
            }
        )
    else:
        return pd.DataFrame(ids)
