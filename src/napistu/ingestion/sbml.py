from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Any

import libsbml
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs
from pydantic import RootModel, field_validator

from napistu import consensus, identifiers, sbml_dfs_utils, source, utils
from napistu.constants import (
    BQB,
    ONTOLOGIES,
    SBML_DFS,
    SBML_DFS_SCHEMA,
    SCHEMA_DEFS,
)
from napistu.ingestion.constants import (
    COMPARTMENT_ALIASES,
    COMPARTMENTS_GO_TERMS,
    GENERIC_COMPARTMENT,
    SBML_DEFS,
    VALID_COMPARTMENTS,
)

logger = logging.getLogger(__name__)


class SBML:
    """A class for handling Systems Biology Markup Language (SBML) files.

    This class provides an interface to read and parse SBML files, offering
    methods to access the model, summarize its contents, and report any errors
    encountered during parsing.

    Parameters
    ----------
    sbml_path : str
        The file path to an SBML model. Supports local paths and GCS URIs.

    Attributes
    ----------
    document : libsbml.SBMLDocument
        The raw SBML document object from libsbml.
    model : libsbml.Model
        The parsed SBML model object from libsbml.
    verbose : bool, default=False
        If True, then include detailed logs.

    Methods
    -------
    summary()
        Prints a summary of the sbml model
    sbml_errors(reduced_log, return_df)
        Print a summary of all errors in the SBML file

    Raises
    ------
    ValueError
        If the SBML model is not Level 3, or if critical, unknown errors are
        found during parsing.
    """

    def __init__(
        self,
        sbml_path: str,
        verbose: bool = False,
    ) -> None:
        """Initializes the SBML object by reading and validating an SBML file."""
        reader = libsbml.SBMLReader()
        if os.path.exists(sbml_path):
            self.document = reader.readSBML(sbml_path)
        else:
            with open_fs(os.path.dirname(sbml_path)) as fs:
                txt = fs.readtext(os.path.basename(sbml_path))
                self.document = reader.readSBMLFromString(txt)

        if self.document.getLevel() < 3:
            raise ValueError(
                f"SBML model is level {self.document.getLevel()}, only SBML 3 is supported"
            )

        self.model = self.document.getModel()

        # check for critical sbml errors
        errors = self.sbml_errors(reduced_log=False, return_df=True)
        if errors is not None:
            critical_errors = errors[errors[SBML_DEFS.ERROR_SEVERITY] >= 2]
            critical_errors = set(critical_errors[SBML_DEFS.ERROR_DESCRIPTION].unique())
            known_errors = {
                "<layout> must have 'id' and may have 'name'",
                "Missing value for the 'compartment' attribute",
            }

            found_known_errors = known_errors.intersection(critical_errors)
            if len(found_known_errors) > 0:
                if verbose:
                    logger.info(
                        f"The following known errors were found: {found_known_errors}"
                    )

            unknown_critical_errors = critical_errors - known_errors
            if len(unknown_critical_errors) != 0:
                critical_errors = ", ".join(unknown_critical_errors)
                raise ValueError(
                    f"Critical errors were found when reading the sbml file: {critical_errors}"
                )

    def sbml_errors(self, reduced_log: bool = True, return_df: bool = False):
        """Formats and reports all errors found in the SBML file.

        Parameters
        ----------
        reduced_log : bool, optional
            If True, aggregates errors by category and severity. Defaults to True.
        return_df : bool, optional
            If True, returns a DataFrame of the errors. Otherwise, prints a
            styled summary. Defaults to False.

        Returns
        -------
        pd.DataFrame or None
            A DataFrame containing the error log if `return_df` is True and
            errors are present, otherwise None.
        """
        n_errors = self.document.getNumErrors()
        if n_errors == 0:
            return None

        error_log = list()
        for i in range(n_errors):
            e = self.document.getError(i)

            error_entry = {
                SBML_DEFS.ERROR_NUMBER: i,
                SBML_DEFS.ERROR_CATEGORY: e.getCategoryAsString(),
                SBML_DEFS.ERROR_SEVERITY: e.getSeverity(),
                SBML_DEFS.ERROR_DESCRIPTION: e.getShortMessage(),
                SBML_DEFS.ERROR_MESSAGE: e.getMessage(),
            }

            error_log.append(error_entry)
        error_log = pd.DataFrame(error_log)

        if reduced_log:
            error_log = (
                error_log[
                    [
                        SBML_DEFS.ERROR_CATEGORY,
                        SBML_DEFS.ERROR_SEVERITY,
                        SBML_DEFS.ERROR_MESSAGE,
                    ]
                ]
                .groupby([SBML_DEFS.ERROR_CATEGORY, SBML_DEFS.ERROR_SEVERITY])
                .count()
            )

        if return_df:
            return error_log
        else:
            if reduced_log:
                headers = [
                    f"{SBML_DEFS.ERROR_CATEGORY}, {SBML_DEFS.ERROR_SEVERITY}",
                    "count",
                ]
            else:
                headers = [
                    SBML_DEFS.ERROR_CATEGORY,
                    SBML_DEFS.ERROR_SEVERITY,
                    SBML_DEFS.ERROR_DESCRIPTION,
                ]
                error_log = error_log[headers]

            utils.show(error_log, headers=headers)

            return None

    def summary(self) -> pd.DataFrame:
        """Generates a styled summary of the SBML model.

        Returns
        -------
        pd.io.formats.style.Styler
            A styled pandas DataFrame containing a summary of the model,
            including pathway name, ID, and counts of species and reactions.
        """
        model = self.model

        model_summaries = dict()

        model_summaries[SBML_DEFS.SUMMARY_PATHWAY_NAME] = model.getName()
        model_summaries[SBML_DEFS.SUMMARY_PATHWAY_ID] = model.getId()

        model_summaries[SBML_DEFS.SUMMARY_N_SPECIES] = model.getNumSpecies()
        model_summaries[SBML_DEFS.SUMMARY_N_REACTIONS] = model.getNumReactions()

        compartments = [
            model.getCompartment(i).getName() for i in range(model.getNumCompartments())
        ]
        compartments.sort()
        model_summaries[SBML_DEFS.SUMMARY_COMPARTMENTS] = ",\n".join(compartments)

        model_summaries_dat = pd.DataFrame(model_summaries, index=[0]).T

        return utils.show(model_summaries_dat)  # type: ignore

    def _define_compartments(
        self, compartment_aliases_dict: dict | None = None
    ) -> pd.DataFrame:
        """Extracts and defines compartments from the SBML model.

        This function iterates through the compartments in the SBML model,
        extracting their IDs, names, and identifiers. It also handles cases where
        CVTerms are missing by mapping compartment names to known GO terms.

        Parameters
        ----------
        sbml_model : SBML
            The SBML model to process.
        compartment_aliases_dict : dict, optional
            A dictionary to map custom compartment names. If None, the default
            mapping from `COMPARTMENT_ALIASES` is used.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing information about each compartment, indexed by
            compartment ID.
        """
        if compartment_aliases_dict is None:
            aliases = COMPARTMENT_ALIASES
        else:
            aliases = CompartmentAliasesValidator.from_dict(compartment_aliases_dict)

        compartments = list()
        for i in range(self.model.getNumCompartments()):
            comp = self.model.getCompartment(i)

            if not comp.getCVTerms():
                logger.warning(
                    f"Compartment {comp.getId()} has empty CVterms, mapping its c_Identifiers from the Compartment dict"
                )

                compartments.append(_define_compartments_missing_cvterms(comp, aliases))

            else:
                compartments.append(
                    {
                        SBML_DFS.C_ID: comp.getId(),
                        SBML_DFS.C_NAME: comp.getName(),
                        SBML_DFS.C_IDENTIFIERS: identifiers.cv_to_Identifiers(comp),
                        SBML_DFS.C_SOURCE: source.Source.empty(),
                    }
                )

        if len(compartments) == 0:
            logger.warning(
                "No compartments were found in the SBML model. using default compartment GO CELLULAR_COMPONENT"
            )
            return sbml_dfs_utils.stub_compartments(with_source=True)

        return pd.DataFrame(compartments).set_index(SBML_DFS.C_ID)

    def _define_cspecies(self) -> pd.DataFrame:
        """Creates a DataFrame of compartmentalized species from an SBML model.

        This function extracts all species from the model and creates a
        standardized DataFrame that includes unique IDs for each compartmentalized
        species (`sc_id`), along with species and compartment IDs, and their
        corresponding identifiers.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing information about each compartmentalized species.
        """
        comp_species = list()
        for i in range(self.model.getNumSpecies()):
            spec = self.model.getSpecies(i)

            spec_dict = {
                SBML_DFS.SC_ID: spec.getId(),
                SBML_DFS.SC_NAME: spec.getName(),
                SBML_DFS.C_ID: spec.getCompartment(),
                SBML_DFS.S_IDENTIFIERS: identifiers.cv_to_Identifiers(spec),
                SBML_DFS.SC_SOURCE: source.Source.empty(),
            }

            comp_species.append(spec_dict)

        # add geneproducts defined using L3 FBC extension
        fbc_gene_products = self._define_fbc_gene_products()
        comp_species.extend(fbc_gene_products)

        comp_species_df = pd.DataFrame(comp_species).set_index(SBML_DFS.SC_ID)
        comp_species_df[SBML_DFS.SC_NAME] = utils.update_pathological_names(
            comp_species_df[SBML_DFS.SC_NAME], "SC"
        )

        # replace empty strings for c_id with None so the resolve function will properly fill these
        comp_species_df[SBML_DFS.C_ID] = comp_species_df[SBML_DFS.C_ID].replace(
            "", None
        )

        # if all entries are missing then c_id was likely stubbed with sbml_dfs_utils.stub_compartments()
        if all(comp_species_df[SBML_DFS.C_ID].isnull()):
            default_cid = sbml_dfs_utils.stub_compartments().index[0]
            comp_species_df[SBML_DFS.C_ID] = default_cid

        return comp_species_df

    def _define_fbc_gene_products(self) -> list[dict]:

        mplugin = self.model.getPlugin("fbc")

        fbc_gene_products = list()
        if mplugin is not None:
            for i in range(mplugin.getNumGeneProducts()):
                gene_product = mplugin.getGeneProduct(i)

                gene_dict = {
                    SBML_DFS.SC_ID: gene_product.getId(),
                    SBML_DFS.SC_NAME: (
                        gene_product.getName()
                        if gene_product.isSetName()
                        else gene_product.getLabel()
                    ),
                    # use getLabel() to accomendate sbml model (e.g. HumanGEM.xml) with no fbc:name attribute
                    # Recon3D.xml has both fbc:label and fbc:name attributes, with gene name in fbc:nam
                    SBML_DFS.C_ID: None,
                    SBML_DFS.S_IDENTIFIERS: identifiers.cv_to_Identifiers(gene_product),
                    SBML_DFS.SC_SOURCE: source.Source.empty(),
                }

                fbc_gene_products.append(gene_dict)

        return fbc_gene_products

    def _define_reactions(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extracts and defines reactions and their participating species.

        This function iterates through all reactions in the SBML model, creating
        a DataFrame for reaction attributes and another for all participating
        species (reactants, products, and modifiers).

        Parameters
        ----------
        sbml_model : SBML
            The SBML model to process.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing two DataFrames:
            - The first DataFrame contains reaction attributes, indexed by reaction ID.
            - The second DataFrame lists all species participating in reactions.
        """
        reactions_list = []
        reaction_species_list = []
        for i in range(self.model.getNumReactions()):
            rxn = SBML_reaction(self.model.getReaction(i))
            reactions_list.append(rxn.reaction_dict)

            rxn_specs = rxn.species
            rxn_specs[SBML_DFS.R_ID] = rxn.reaction_dict[SBML_DFS.R_ID]
            reaction_species_list.append(rxn_specs)

        reactions = pd.DataFrame(reactions_list).set_index(SBML_DFS.R_ID)

        reaction_species_df = pd.concat(reaction_species_list)
        # add an index if reaction species didn't have IDs in the .sbml
        if all([v == "" for v in reaction_species_df.index.tolist()]):
            reaction_species_df = (
                reaction_species_df.reset_index(drop=True)
                .assign(
                    rsc_id=sbml_dfs_utils.id_formatter(
                        range(reaction_species_df.shape[0]), SBML_DFS.RSC_ID
                    )
                )
                .set_index(SBML_DFS.RSC_ID)
            )

        return reactions, reaction_species_df

    def _define_species(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extracts and defines species and compartmentalized species.

        This function creates two DataFrames: one for unique molecular species
        (un-compartmentalized) and another for compartmentalized species, which
        represent a species within a specific compartment.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing two DataFrames:
            - The first DataFrame represents unique molecular species.
            - The second DataFrame represents compartmentalized species.
        """

        SPECIES_SCHEMA = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.SPECIES]
        CSPECIES_SCHEMA = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.COMPARTMENTALIZED_SPECIES]
        SPECIES_VARS = SPECIES_SCHEMA[SCHEMA_DEFS.VARS]
        CSPECIES_VARS = CSPECIES_SCHEMA[SCHEMA_DEFS.VARS]

        comp_species_df = self._define_cspecies()

        # find unique species and create a table
        consensus_species_df = comp_species_df.copy()
        consensus_species_df.index.names = [SBML_DFS.S_ID]
        consensus_species, species_lookup = consensus._reduce_to_consensus_ids(
            consensus_species_df,
            # note that this is an incomplete schema because consensus_species_df isn't a
            # normal species table
            {
                SCHEMA_DEFS.PK: SBML_DFS.S_ID,
                SCHEMA_DEFS.ID: SBML_DFS.S_IDENTIFIERS,
                SCHEMA_DEFS.TABLE: SBML_DFS.SPECIES,
            },
        )

        # create a table of unique molecular species
        consensus_species.index.name = SBML_DFS.S_ID
        consensus_species[SBML_DFS.S_NAME] = [
            re.sub("\\[.+\\]", "", x).strip()
            for x in consensus_species[SBML_DFS.SC_NAME]
        ]
        consensus_species = consensus_species.drop(
            [SBML_DFS.SC_NAME, SBML_DFS.C_ID], axis=1
        )
        consensus_species[SBML_DFS.S_SOURCE] = [
            source.Source.empty() for x in range(0, consensus_species.shape[0])
        ]

        species = consensus_species[SPECIES_VARS]
        compartmentalized_species = comp_species_df.join(species_lookup).rename(
            columns={"new_id": SBML_DFS.S_ID}
        )[CSPECIES_VARS]

        # Internal validation: Check consistency between species and compartmentalized species
        _validate_species_consistency(species, compartmentalized_species)

        return species, compartmentalized_species


class CompartmentAliasesValidator(RootModel):
    """
    A Pydantic model for validating compartment alias dictionaries.

    This model ensures that the compartment alias dictionary is a mapping
    from a string (the canonical compartment name) to a list of strings
    (the aliases for that compartment). It also validates that the keys
    of the dictionary are valid compartment names.

    Attributes
    ----------
    root : dict[str, list[str]]
        The root of the model is a dictionary where keys are strings and
        values are lists of strings.
    """

    root: dict[str, list[str]]

    @field_validator("root")
    def validate_aliases(cls, values: dict[str, list[str]]):
        """Validate the compartment alias dictionary."""
        for key, alias_list in values.items():
            if not key:
                raise ValueError("Compartment keys must be non-empty.")
            if key not in VALID_COMPARTMENTS:
                raise ValueError(
                    f"Invalid compartment key: {key}. "
                    f"Must be one of {VALID_COMPARTMENTS}"
                )
            if not alias_list:
                raise ValueError(f"Alias list for '{key}' cannot be empty.")
        return values

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "CompartmentAliasesValidator":
        """
        Create a CompartmentAliasesValidator from a dictionary.

        Parameters
        ----------
        data : dict[str, list[str]]
            A dictionary mapping canonical compartment names to their aliases.

        Returns
        -------
        CompartmentAliasesValidator
            A validated instance of the model.
        """
        return cls.model_validate(data)

    def __getitem__(self, key: str) -> list[str]:
        return self.root[key]

    def items(self):
        return self.root.items()

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)


class SBML_reaction:
    """A convenience class for processing individual SBML reactions.

    This class extracts and organizes key information about an SBML reaction,
    including its attributes and participating species (substrates, products,
    and modifiers).

    Parameters
    ----------
    sbml_reaction : libsbml.Reaction
        A libsbml Reaction object to be processed.

    Attributes
    ----------
    reaction_dict : dict
        A dictionary of reaction-level attributes, including its ID, name,
        reversibility, identifiers, and source information.
    species : pd.DataFrame
        A DataFrame listing all species participating in the reaction,
        including their roles (substrate, product, modifier), stoichiometry,
        and SBO terms.
    """

    def __init__(
        self,
        sbml_reaction: libsbml.Reaction,
    ) -> None:
        """Initializes the SBML_reaction object by parsing a libsbml Reaction."""
        reaction_dict = {
            SBML_DFS.R_ID: sbml_reaction.getId(),
            SBML_DFS.R_NAME: sbml_reaction.getName(),
            SBML_DFS.R_IDENTIFIERS: identifiers.cv_to_Identifiers(sbml_reaction),
            SBML_DFS.R_SOURCE: source.Source.empty(),
            SBML_DFS.R_ISREVERSIBLE: sbml_reaction.getReversible(),
        }

        self.reaction_dict = reaction_dict

        # process reaction species
        reaction_species = list()
        # save modifiers
        for i in range(sbml_reaction.getNumModifiers()):
            spec = sbml_reaction.getModifier(i)
            spec_dict = {
                SBML_DFS.RSC_ID: spec.getId(),
                SBML_DFS.SC_ID: spec.getSpecies(),
                SBML_DFS.STOICHIOMETRY: 0,
                SBML_DFS.SBO_TERM: spec.getSBOTermID(),
            }
            reaction_species.append(spec_dict)

        # find gene products defined using the fbc plugin
        rxn_fbc = sbml_reaction.getPlugin("fbc")
        if rxn_fbc:
            gpa = rxn_fbc.getGeneProductAssociation()
            if gpa:
                gene_products = _extract_gene_products(gpa.getAssociation())
                # de-duplicate
                gene_products = list(
                    {d[SBML_DFS.SC_ID]: d for d in gene_products}.values()
                )
                reaction_species.extend(gene_products)

        # save reactants
        for i in range(sbml_reaction.getNumReactants()):
            spec = sbml_reaction.getReactant(i)
            spec_dict = {
                SBML_DFS.RSC_ID: spec.getId(),
                SBML_DFS.SC_ID: spec.getSpecies(),
                SBML_DFS.STOICHIOMETRY: -1 * spec.getStoichiometry(),
                SBML_DFS.SBO_TERM: spec.getSBOTermID(),
            }
            reaction_species.append(spec_dict)
        # save products
        for i in range(sbml_reaction.getNumProducts()):
            spec = sbml_reaction.getProduct(i)
            spec_dict = {
                SBML_DFS.RSC_ID: spec.getId(),
                SBML_DFS.SC_ID: spec.getSpecies(),
                SBML_DFS.STOICHIOMETRY: spec.getStoichiometry(),
                SBML_DFS.SBO_TERM: spec.getSBOTermID(),
            }
            reaction_species.append(spec_dict)

        self.species = pd.DataFrame(reaction_species).set_index(SBML_DFS.RSC_ID)


def sbml_dfs_from_sbml(self, sbml_model: SBML, compartment_aliases: dict | None = None):
    """Parses an SBML model into a set of standardized DataFrames.

    This function serves as the main entry point for converting an SBML model
    into the internal DataFrame-based representation used by napistu. It
    orchestrates the processing of compartments, species, and reactions.

    Parameters
    ----------
    self : object
        The instance of the calling class, expected to have a `schema` attribute.
    sbml_model : SBML
        The SBML model to be parsed.
    compartment_aliases : dict, optional
        A dictionary to map custom compartment names to the napistu controlled
        vocabulary. If None, the default mapping (COMPARTMENT_ALIASES) is used.
        Defaults to None.

    Returns
    -------
    object
        The calling class instance, now populated with DataFrames for
        compartments, species, compartmentalized_species, reactions, and reaction_species
    """
    # 1. Process compartments from the SBML model
    self.compartments = sbml_model._define_compartments(compartment_aliases)

    # 2. Process species and compartmentalized species
    self.species, self.compartmentalized_species = sbml_model._define_species()

    # Refine compartments to only those actually used by species
    self.compartments = _refine_compartments(
        self.compartments, self.compartmentalized_species
    )

    # 3. Process reactions and their participating species
    self.reactions, self.reaction_species = sbml_model._define_reactions()

    return self


def _validate_species_consistency(species_df, compartmentalized_species_df):
    """Validate consistency between species and compartmentalized species tables.

    Parameters
    ----------
    species_df : pd.DataFrame
        DataFrame of species with s_id as index
    compartmentalized_species_df : pd.DataFrame
        DataFrame of compartmentalized species with s_id column

    Raises
    ------
    ValueError
        If there are inconsistencies between the two tables
    """
    species_ids_in_species = set(species_df.index)
    species_ids_in_cspecies = set(compartmentalized_species_df[SBML_DFS.S_ID])

    missing_in_species = species_ids_in_cspecies - species_ids_in_species
    missing_in_cspecies = species_ids_in_species - species_ids_in_cspecies

    if missing_in_species:
        raise ValueError(
            f"Species extraction validation error: {len(missing_in_species)} species IDs "
            f"found in compartmentalized_species but missing from species table. "
            f"Missing species: {sorted(list(missing_in_species))[:10]}"
        )

    if missing_in_cspecies:
        raise ValueError(
            f"Species extraction validation error: {len(missing_in_cspecies)} species IDs "
            f"found in species table but missing from compartmentalized_species table. "
            f"Missing from compartmentalized_species: {sorted(list(missing_in_cspecies))[:10]}"
        )


def _refine_compartments(compartments_df, compartmentalized_species_df):
    """Refine compartments to only those actually used by compartmentalized species.

    This function filters the compartments DataFrame to include only compartments
    that are referenced by compartmentalized species, and validates that all
    required compartments exist.

    Parameters
    ----------
    compartments_df : pd.DataFrame
        DataFrame of all extracted compartments with c_id as index
    compartmentalized_species_df : pd.DataFrame
        DataFrame of compartmentalized species with c_id column

    Returns
    -------
    pd.DataFrame
        Filtered compartments DataFrame containing only used compartments

    Raises
    ------
    ValueError
        If compartmentalized species reference compartments that don't exist
    """
    compartment_ids_in_compartments = set(compartments_df.index)
    compartment_ids_in_cspecies = set(compartmentalized_species_df[SBML_DFS.C_ID])
    # ignore None in compartment_ids_in_cspecies - this can likely be resolved downstream
    compartment_ids_in_cspecies = {
        c for c in compartment_ids_in_cspecies if c is not None
    }

    # Check that all required compartments exist
    missing_compartments = compartment_ids_in_cspecies - compartment_ids_in_compartments
    if missing_compartments:
        raise ValueError(
            f"Compartment refinement error: {len(missing_compartments)} compartment IDs "
            f"found in compartmentalized_species but missing from compartments table. "
            f"Missing compartments: {sorted(list(missing_compartments))[:10]}"
        )

    # Filter to only compartments that are actually used
    unused_compartments = compartment_ids_in_compartments - compartment_ids_in_cspecies
    if unused_compartments:
        logger.debug(
            f"Filtering out {len(unused_compartments)} unused compartments from compartments table. "
            f"Unused compartments: {sorted(list(unused_compartments))[:10]}"
        )

    # Return only the compartments that are referenced by species
    used_compartments = compartments_df.loc[
        compartments_df.index.isin(compartment_ids_in_cspecies)
    ]

    logger.debug(
        f"Refined compartments: {len(used_compartments)} of {len(compartments_df)} compartments retained"
    )

    return used_compartments


def _define_compartments_missing_cvterms(
    comp: libsbml.Compartment, aliases: dict
) -> dict[str, Any]:

    comp_name = comp.getName()
    mapped_compartment_key = [
        compkey for compkey, mappednames in aliases.items() if comp_name in mappednames
    ]

    if len(mapped_compartment_key) == 0:
        logger.warning(
            f"No GO compartment for {comp_name} is mapped, use the generic cellular_component's GO id"
        )

        compartment_entry = {
            SBML_DFS.C_ID: comp.getId(),
            SBML_DFS.C_NAME: comp.getName(),
            SBML_DFS.C_IDENTIFIERS: identifiers.Identifiers(
                [
                    identifiers.format_uri(
                        uri=identifiers.create_uri_url(
                            ontology=ONTOLOGIES.GO,
                            identifier=COMPARTMENTS_GO_TERMS[GENERIC_COMPARTMENT],
                        ),
                        bqb=BQB.BQB_IS,
                    )
                ]
            ),
            SBML_DFS.C_SOURCE: source.Source.empty(),
        }

    if len(mapped_compartment_key) > 0:
        if len(mapped_compartment_key) > 1:
            logger.warning(
                f"More than one GO compartments for {comp_name} are mapped, using the first one"
            )

        compartment_entry = {
            SBML_DFS.C_ID: comp.getId(),
            SBML_DFS.C_NAME: comp.getName(),
            SBML_DFS.C_IDENTIFIERS: identifiers.Identifiers(
                [
                    identifiers.format_uri(
                        uri=identifiers.create_uri_url(
                            ontology=ONTOLOGIES.GO,
                            identifier=COMPARTMENTS_GO_TERMS[mapped_compartment_key[0]],
                        ),
                        bqb=BQB.IS,
                    )
                ]
            ),
            SBML_DFS.C_SOURCE: source.Source.empty(),
        }

    return compartment_entry


def _get_gene_product_dict(gp):
    """Extracts attributes of a gene product from an SBML reaction object.

    Parameters
    ----------
    gp : libsbml.GeneProduct
        A libsbml GeneProduct object.

    Returns
    -------
    dict
        A dictionary containing the gene product's ID, name, and identifiers.
    """
    return {
        SBML_DFS.RSC_ID: gp.getId(),
        SBML_DFS.SC_ID: gp.getGeneProduct(),
        SBML_DFS.STOICHIOMETRY: 0,
        SBML_DFS.SBO_TERM: gp.getSBOTermID(),
    }


def _extract_gene_products(association: libsbml.Association) -> list[dict]:
    """Recursively extracts gene products from an association tree."""
    gene_products = []

    def _recursive_helper(assoc: libsbml.Association):
        if hasattr(assoc, SBML_DEFS.REACTION_ATTR_GET_GENE_PRODUCT):
            gene_products.append(_get_gene_product_dict(assoc))
        elif hasattr(assoc, "getNumAssociations"):
            for i in range(assoc.getNumAssociations()):
                _recursive_helper(assoc.getAssociation(i))

    _recursive_helper(association)
    return gene_products
