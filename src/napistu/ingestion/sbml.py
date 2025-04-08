from __future__ import annotations

import logging
import os
import re

import libsbml
import pandas as pd
from napistu import consensus
from napistu import constants
from napistu import identifiers
from napistu import sbml_dfs_utils
from napistu import source
from napistu import utils

from napistu.constants import BQB

from napistu.ingestion.constants import SBML_ANNOTATION_METHOD_GET_COMPARTMENT
from napistu.ingestion.constants import SBML_ANNOTATION_METHOD_GET_REACTION
from napistu.ingestion.constants import SBML_ANNOTATION_METHOD_GET_SPECIES
from napistu.ingestion.constants import SBML_COMPARTMENT_DICT_ID
from napistu.ingestion.constants import SBML_COMPARTMENT_DICT_IDENTIFIERS
from napistu.ingestion.constants import SBML_COMPARTMENT_DICT_NAME
from napistu.ingestion.constants import SBML_COMPARTMENT_DICT_SOURCE
from napistu.ingestion.constants import SBML_COMPARTMENTALIZED_SPECIES_DICT_NAME
from napistu.ingestion.constants import SBML_COMPARTMENTALIZED_SPECIES_DICT_SOURCE
from napistu.ingestion.constants import SBML_REACTION_ATTR_GET_GENE_PRODUCT
from napistu.ingestion.constants import SBML_SPECIES_DICT_ID
from napistu.ingestion.constants import SBML_SPECIES_DICT_IDENTIFIERS
from napistu.ingestion.constants import SBML_SPECIES_DICT_NAME
from napistu.ingestion.constants import SMBL_ERROR_CATEGORY
from napistu.ingestion.constants import SMBL_ERROR_DESCRIPTION
from napistu.ingestion.constants import SMBL_ERROR_MESSAGE
from napistu.ingestion.constants import SMBL_ERROR_NUMBER
from napistu.ingestion.constants import SMBL_ERROR_SEVERITY
from napistu.ingestion.constants import SMBL_REACTION_DICT_ID
from napistu.ingestion.constants import SMBL_REACTION_DICT_IDENTIFIERS
from napistu.ingestion.constants import SMBL_REACTION_DICT_IS_REVERSIBLE
from napistu.ingestion.constants import SMBL_REACTION_DICT_NAME
from napistu.ingestion.constants import SMBL_REACTION_DICT_SOURCE
from napistu.ingestion.constants import SMBL_REACTION_SPEC_RSC_ID
from napistu.ingestion.constants import SMBL_REACTION_SPEC_SBO_TERM
from napistu.ingestion.constants import SMBL_REACTION_SPEC_SC_ID
from napistu.ingestion.constants import SMBL_REACTION_SPEC_STOICHIOMETRY
from napistu.ingestion.constants import SMBL_SUMMARY_COMPARTMENTS
from napistu.ingestion.constants import SMBL_SUMMARY_N_REACTIONS
from napistu.ingestion.constants import SMBL_SUMMARY_N_SPECIES
from napistu.ingestion.constants import SMBL_SUMMARY_PATHWAY_ID
from napistu.ingestion.constants import SMBL_SUMMARY_PATHWAY_NAME

from fs import open_fs

logger = logging.getLogger(__name__)


class SBML:
    """
    System Biology Markup Language Connections.

    Attributes
    ----------
    document
        Connection to the SBML document
    model
        Connection to the SBML model

    Methods
    -------
    summary()
        Prints a summary of the sbml model
    sbml_errors(reduced_log, return_df)
        Print a summary of all errors in the SBML file

    """

    def __init__(
        self,
        sbml_path: str,
    ) -> None:
        """
        Connects to an SBML file

        Parameters
        ----------
        sbml_path : str
            path to a .sbml file.

        Returns
        -------
        None.
        """

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
            critical_errors = errors[errors[SMBL_ERROR_SEVERITY] >= 2]
            critical_errors = set(critical_errors[SMBL_ERROR_DESCRIPTION].unique())
            known_errors = {"<layout> must have 'id' and may have 'name'"}

            found_known_errors = known_errors.intersection(critical_errors)
            if len(found_known_errors) > 0:
                logger.warning(
                    f"The following known errors were found: {found_known_errors}"
                )

            unknown_critical_errors = critical_errors - known_errors
            if len(unknown_critical_errors) != 0:
                critical_errors = ", ".join(unknown_critical_errors)
                raise ValueError(
                    f"Critical errors were found when reading the sbml file: {critical_errors}"
                )

    def summary(self) -> pd.DataFrame:
        """Returns a pd.DataFrame summary of an SBML model."""
        model = self.model

        model_summaries = dict()

        model_summaries[SMBL_SUMMARY_PATHWAY_NAME] = model.getName()
        model_summaries[SMBL_SUMMARY_PATHWAY_ID] = model.getId()

        model_summaries[SMBL_SUMMARY_N_SPECIES] = model.getNumSpecies()
        model_summaries[SMBL_SUMMARY_N_REACTIONS] = model.getNumReactions()

        compartments = [
            model.getCompartment(i).getName() for i in range(model.getNumCompartments())
        ]
        compartments.sort()
        model_summaries[SMBL_SUMMARY_COMPARTMENTS] = ",\n".join(compartments)

        model_summaries_dat = pd.DataFrame(model_summaries, index=[0]).T

        return utils.style_df(model_summaries_dat)  # type: ignore

    def sbml_errors(self, reduced_log: bool = True, return_df: bool = False):
        """
        Format and print all SBML errors

        Parameters
        ----------
        reduced_log : bool
            Reduced log aggregates errors across categories an severity levels
        return_df: bool
            If False then print a log, if True then return a pd.DataFrame

        Returns
        -------
        None or pd.DataFrame.
        """
        n_errors = self.document.getNumErrors()
        if n_errors == 0:
            return None

        error_log = list()
        for i in range(n_errors):
            e = self.document.getError(i)

            error_entry = {
                SMBL_ERROR_NUMBER: i,
                SMBL_ERROR_CATEGORY: e.getCategoryAsString(),
                SMBL_ERROR_SEVERITY: e.getSeverity(),
                SMBL_ERROR_DESCRIPTION: e.getShortMessage(),
                SMBL_ERROR_MESSAGE: e.getMessage(),
            }

            error_log.append(error_entry)
        error_log = pd.DataFrame(error_log)

        if reduced_log:
            error_log = (
                error_log[
                    [SMBL_ERROR_CATEGORY, SMBL_ERROR_SEVERITY, SMBL_ERROR_MESSAGE]
                ]
                .groupby([SMBL_ERROR_CATEGORY, SMBL_ERROR_SEVERITY])
                .count()
            )

        if return_df:
            return error_log
        else:
            if reduced_log:
                headers = [f"{SMBL_ERROR_CATEGORY}, {SMBL_ERROR_SEVERITY}", "count"]
            else:
                headers = [
                    SMBL_ERROR_CATEGORY,
                    SMBL_ERROR_SEVERITY,
                    SMBL_ERROR_DESCRIPTION,
                ]
                error_log = error_log[headers]

            utils.style_df(error_log, headers=headers)

            return None


class SBML_reaction:
    """
    System Biology Markup Language Model Reactions.

    Attributes
    ----------
    reaction_dict: dict
        dictionary of reaction-level attributes, id, name, identifiers
    species: pd.DataFrame
        table of substrates, products, and modifiers

    """

    def __init__(
        self,
        sbml_reaction: libsbml.Reaction,
    ) -> None:
        """
        Convenience class for working with sbml reactions
        """
        reaction_dict = {
            SMBL_REACTION_DICT_ID: sbml_reaction.getId(),
            SMBL_REACTION_DICT_NAME: sbml_reaction.getName(),
            SMBL_REACTION_DICT_IDENTIFIERS: identifiers.cv_to_Identifiers(
                sbml_reaction
            ),
            SMBL_REACTION_DICT_SOURCE: source.Source(init=True),
            SMBL_REACTION_DICT_IS_REVERSIBLE: sbml_reaction.getReversible(),
        }

        self.reaction_dict = reaction_dict

        # process reaction species
        reaction_species = list()
        # save modifiers
        for i in range(sbml_reaction.getNumModifiers()):
            spec = sbml_reaction.getModifier(i)
            spec_dict = {
                SMBL_REACTION_SPEC_RSC_ID: spec.getId(),
                SMBL_REACTION_SPEC_SC_ID: spec.getSpecies(),
                SMBL_REACTION_SPEC_STOICHIOMETRY: 0,
                SMBL_REACTION_SPEC_SBO_TERM: spec.getSBOTermID(),
            }
            reaction_species.append(spec_dict)

        rxn_fbc = sbml_reaction.getPlugin("fbc")
        # check for gene products associated with the FBC L3 extension
        if rxn_fbc is not None:
            gene_products = list()
            gpa = rxn_fbc.getGeneProductAssociation()
            if gpa is not None:
                gpaa = gpa.getAssociation()
                if hasattr(gpaa, SBML_REACTION_ATTR_GET_GENE_PRODUCT):
                    gene_products.append(_get_gene_product_dict(gpaa))
                else:
                    for i in range(gpaa.getNumAssociations()):
                        gpaaa = gpaa.getAssociation(i)
                        if hasattr(gpaaa, SBML_REACTION_ATTR_GET_GENE_PRODUCT):
                            gene_products.append(_get_gene_product_dict(gpaaa))
                        else:
                            for i in range(gpaaa.getNumAssociations()):
                                gpaaaa = gpaaa.getAssociation(i)
                                if hasattr(gpaaaa, SBML_REACTION_ATTR_GET_GENE_PRODUCT):
                                    gene_products.append(_get_gene_product_dict(gpaaaa))
                                else:
                                    for i in range(gpaa.getNumAssociations()):
                                        gpaaaaa = gpaaaa.getAssociation(i)
                                        if hasattr(
                                            gpaaaaa, SBML_REACTION_ATTR_GET_GENE_PRODUCT
                                        ):
                                            gene_products.append(
                                                _get_gene_product_dict(gpaaaaa)
                                            )
                                        else:
                                            logger.warning(
                                                "gene annotations nested deeper than 4 levels, ignoring"
                                            )
                                            continue
            # de-duplicate
            gene_products = list(
                {d[SMBL_REACTION_SPEC_SC_ID]: d for d in gene_products}.values()
            )
            reaction_species = reaction_species + gene_products

        # save reactants
        for i in range(sbml_reaction.getNumReactants()):
            spec = sbml_reaction.getReactant(i)
            spec_dict = {
                SMBL_REACTION_SPEC_RSC_ID: spec.getId(),
                SMBL_REACTION_SPEC_SC_ID: spec.getSpecies(),
                SMBL_REACTION_SPEC_STOICHIOMETRY: -1 * spec.getStoichiometry(),
                SMBL_REACTION_SPEC_SBO_TERM: spec.getSBOTermID(),
            }
            reaction_species.append(spec_dict)
        # save products
        for i in range(sbml_reaction.getNumProducts()):
            spec = sbml_reaction.getProduct(i)
            spec_dict = {
                SMBL_REACTION_SPEC_RSC_ID: spec.getId(),
                SMBL_REACTION_SPEC_SC_ID: spec.getSpecies(),
                SMBL_REACTION_SPEC_STOICHIOMETRY: spec.getStoichiometry(),
                SMBL_REACTION_SPEC_SBO_TERM: spec.getSBOTermID(),
            }
            reaction_species.append(spec_dict)

        self.species = pd.DataFrame(reaction_species).set_index(
            SMBL_REACTION_SPEC_RSC_ID
        )


def sbml_df_from_sbml(self, sbml_model: SBML):
    # specify compartments

    compartments = list()
    for i in range(sbml_model.model.getNumCompartments()):
        comp = sbml_model.model.getCompartment(i)

        if not comp.getCVTerms():
            logger.warning(
                f"Compartment {comp.getId()} has empty CVterms, mapping its c_Identifiers from the Compartment dict"
            )

            comp_name = comp.getName()
            mapped_compartment_key = [
                compkey
                for compkey, mappednames in constants.COMPARTMENT_ALIASES.items()
                if comp_name in mappednames
            ]

            if len(mapped_compartment_key) == 0:
                logger.warning(
                    f"No GO compartment for {comp_name} is mapped, use the generic cellular_component's GO id"
                )
                compartments.append(
                    {
                        SBML_COMPARTMENT_DICT_ID: comp.getId(),
                        SBML_COMPARTMENT_DICT_NAME: comp.getName(),
                        SBML_COMPARTMENT_DICT_IDENTIFIERS: identifiers.Identifiers(
                            [
                                identifiers.format_uri(
                                    uri=identifiers.create_uri_url(
                                        ontology=constants.ONTOLOGIES.GO,
                                        identifier=constants.COMPARTMENTS_GO_TERMS[
                                            "CELLULAR_COMPONENT"
                                        ],
                                    ),
                                    biological_qualifier_type=BQB.BQB_IS,
                                )
                            ]
                        ),
                        SBML_COMPARTMENT_DICT_SOURCE: source.Source(init=True),
                    }
                )

            if len(mapped_compartment_key) > 0:
                if len(mapped_compartment_key) > 1:
                    logger.warning(
                        f"More than one GO compartments for {comp_name} are mapped, using the first one"
                    )
                compartments.append(
                    {
                        SBML_COMPARTMENT_DICT_ID: comp.getId(),
                        SBML_COMPARTMENT_DICT_NAME: comp.getName(),
                        SBML_COMPARTMENT_DICT_IDENTIFIERS: identifiers.Identifiers(
                            [
                                identifiers.format_uri(
                                    uri=identifiers.create_uri_url(
                                        ontology=constants.ONTOLOGIES.GO,
                                        identifier=constants.COMPARTMENTS_GO_TERMS[
                                            mapped_compartment_key[0]
                                        ],
                                    ),
                                    biological_qualifier_type=BQB.IS,
                                )
                            ]
                        ),
                        SBML_COMPARTMENT_DICT_SOURCE: source.Source(init=True),
                    }
                )

        else:
            compartments.append(
                {
                    SBML_COMPARTMENT_DICT_ID: comp.getId(),
                    SBML_COMPARTMENT_DICT_NAME: comp.getName(),
                    SBML_COMPARTMENT_DICT_IDENTIFIERS: identifiers.cv_to_Identifiers(
                        comp
                    ),
                    SBML_COMPARTMENT_DICT_SOURCE: source.Source(init=True),
                }
            )

    self.compartments = pd.DataFrame(compartments).set_index(SBML_COMPARTMENT_DICT_ID)

    # create a species df
    comp_species_df = setup_cspecies(sbml_model)

    # find unique species and create a table
    consensus_species_df = comp_species_df.copy()
    consensus_species_df.index.names = [SBML_SPECIES_DICT_ID]
    consensus_species, species_lookup = consensus.reduce_to_consensus_ids(
        consensus_species_df,
        {"pk": SBML_SPECIES_DICT_ID, "id": SBML_SPECIES_DICT_IDENTIFIERS},
    )

    # create a table of unique molecular species
    consensus_species.index.name = SBML_SPECIES_DICT_ID
    consensus_species[SBML_SPECIES_DICT_NAME] = [
        re.sub("\\[.+\\]", "", x).strip()
        for x in consensus_species[SBML_COMPARTMENTALIZED_SPECIES_DICT_NAME]
    ]
    consensus_species = consensus_species.drop(
        [SBML_COMPARTMENTALIZED_SPECIES_DICT_NAME, SBML_COMPARTMENT_DICT_ID], axis=1
    )
    consensus_species["s_Source"] = [
        source.Source(init=True) for x in range(0, consensus_species.shape[0])
    ]

    self.species = consensus_species[self.schema["species"]["vars"]]

    self.compartmentalized_species = comp_species_df.join(species_lookup).rename(
        columns={"new_id": SBML_SPECIES_DICT_ID}
    )[self.schema["compartmentalized_species"]["vars"]]

    # specify reactions

    reactions = list()
    reaction_species = list()
    for i in range(sbml_model.model.getNumReactions()):
        rxn = SBML_reaction(sbml_model.model.getReaction(i))
        reactions.append(rxn.reaction_dict)

        rxn_specs = rxn.species
        rxn_specs[SMBL_REACTION_DICT_ID] = rxn.reaction_dict[SMBL_REACTION_DICT_ID]
        reaction_species.append(rxn_specs)

    self.reactions = pd.DataFrame(reactions).set_index(SMBL_REACTION_DICT_ID)

    reaction_species_df = pd.concat(reaction_species)
    # add an index if reaction species didn't have IDs in the .sbml
    if all([v == "" for v in reaction_species_df.index.tolist()]):
        reaction_species_df = (
            reaction_species_df.reset_index(drop=True)
            .assign(
                rsc_id=sbml_dfs_utils.id_formatter(
                    range(reaction_species_df.shape[0]), SMBL_REACTION_SPEC_RSC_ID
                )
            )
            .set_index(SMBL_REACTION_SPEC_RSC_ID)
        )

    self.reaction_species = reaction_species_df

    return self


def setup_cspecies(sbml_model: SBML) -> pd.DataFrame:
    """
    Setup Compartmentalized Species

    Read all compartmentalized species from a model
    and setup as a pd.DataFrame.
    This operation is functionalized to test the subsequent call of
    consensus.reduce_to_consensus_ids()
    which collapses compartmentalized_species -> species
    based on shared identifiers.
    """
    comp_species = list()
    for i in range(sbml_model.model.getNumSpecies()):
        spec = sbml_model.model.getSpecies(i)

        spec_dict = {
            SMBL_REACTION_SPEC_SC_ID: spec.getId(),
            SBML_COMPARTMENTALIZED_SPECIES_DICT_NAME: spec.getName(),
            SBML_COMPARTMENT_DICT_ID: spec.getCompartment(),
            SBML_SPECIES_DICT_IDENTIFIERS: identifiers.cv_to_Identifiers(spec),
            SBML_COMPARTMENTALIZED_SPECIES_DICT_SOURCE: source.Source(init=True),
        }

        comp_species.append(spec_dict)

    mplugin = sbml_model.model.getPlugin("fbc")

    # add geneproducts defined using L3 FBC extension
    if mplugin is not None:
        for i in range(mplugin.getNumGeneProducts()):
            gene_product = mplugin.getGeneProduct(i)

            gene_dict = {
                SMBL_REACTION_SPEC_SC_ID: gene_product.getId(),
                SBML_COMPARTMENTALIZED_SPECIES_DICT_NAME: (
                    gene_product.getName()
                    if gene_product.isSetName()
                    else gene_product.getLabel()
                ),
                # use getLabel() to accomendate sbml model (e.g. HumanGEM.xml) with no fbc:name attribute
                # Recon3D.xml has both fbc:label and fbc:name attributes, with gene name in fbc:nam
                SBML_COMPARTMENT_DICT_ID: None,
                SBML_SPECIES_DICT_IDENTIFIERS: identifiers.cv_to_Identifiers(
                    gene_product
                ),
                SBML_COMPARTMENTALIZED_SPECIES_DICT_SOURCE: source.Source(init=True),
            }

            comp_species.append(gene_dict)

    return pd.DataFrame(comp_species).set_index(SMBL_REACTION_SPEC_SC_ID)


def add_sbml_annotations(
    sbml_model: SBML, annotations: pd.DataFrame, save_path: str
) -> None:
    """
    Add SBML Annotations

    Add additional identifiers to an sbml file and save the updated document

    Parameters:
    sbml_model: SBML
        A .sbml model
    annotations: pd.DataFrame
        A table of annotations to add containing an "id" matching the
        primary key of an entity, "type" matching the type of entity,
        and "uri" representing the annotation to add.
    save_path: str
        Path to save the model to

    Returns:
    None
    """

    logger.warning(
        "add_sbml_annotations is deprecated and may be removed in a future version of rcpr; "
        "we are now adding these annotation during ingestion by sbml.sbml_df_from_sbml() rather "
        "than directly appending them to the raw .sbml"
    )

    if not isinstance(sbml_model, SBML):
        raise TypeError("sbml_model must be an SBML object")

    if not isinstance(annotations, pd.DataFrame):
        raise TypeError("annotations must be a pd.DataFrame")

    for i in range(0, annotations.shape[0]):
        annot_type = annotations["type"][i]

        if annot_type == "species":
            entity_fxn = SBML_ANNOTATION_METHOD_GET_SPECIES
        elif annot_type == "compartment":
            entity_fxn = SBML_ANNOTATION_METHOD_GET_COMPARTMENT
        elif annot_type == "reaction":
            entity_fxn = SBML_ANNOTATION_METHOD_GET_REACTION
        else:
            raise ValueError(
                f"{annot_type} is not a valid annotation type,"
                " valid types are species, compartment, and reaction"
            )
        # access the node to modify
        entity_fxn_method = getattr(sbml_model.model, entity_fxn)
        entity_node = entity_fxn_method(annotations["id"][i])

        # TO DO - check for a valid entity_node in case id is not found

        # set meta-id if there isn't one; required to add a node
        if not entity_node.isSetMetaId():
            add_metaid_code = entity_node.setMetaId(annotations["id"][i])

            if add_metaid_code != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise ValueError(
                    f"adding metaId to {annotations['id'][i]} failed"
                    f" with return code {add_metaid_code} "
                    f"({libsbml.OperationReturnValue_toString(add_metaid_code).strip()})"
                )

        # create a controlled vocabulary term
        cv = libsbml.CVTerm()
        cv.setQualifierType(libsbml.BIOLOGICAL_QUALIFIER)
        cv.setBiologicalQualifierType(libsbml.BQB_IS_VERSION_OF)

        add_resource_code = cv.addResource(annotations["uri"][i])
        if add_resource_code != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise ValueError(
                "adding resource to CV term returned code"
                f" {add_resource_code} "
                f"({libsbml.OperationReturnValue_toString(add_resource_code).strip()})"
                f" rather than {libsbml.LIBSBML_OPERATION_SUCCESS} when "
                f"adding {annotations['uri'][i]} to {annotations['id'][i]}"
            )

        add_cv_code = entity_node.addCVTerm(cv)
        if add_cv_code != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise ValueError(
                f"adding CV to entity returned code {add_cv_code} "
                f"({libsbml.OperationReturnValue_toString(add_cv_code).strip()})"
                f" rather than {libsbml.LIBSBML_OPERATION_SUCCESS} when adding"
                f" {annotations['uri'][i]} to {annotations['id'][i]}"
            )

    libsbml.writeSBML(sbml_model.document, save_path)

    return None


def _get_gene_product_dict(gp):
    """Read a gene product node from an sbml file."""
    return {
        SMBL_REACTION_SPEC_RSC_ID: gp.getId(),
        SMBL_REACTION_SPEC_SC_ID: gp.getGeneProduct(),
        SMBL_REACTION_SPEC_STOICHIOMETRY: 0,
        SMBL_REACTION_SPEC_SBO_TERM: gp.getSBOTermID(),
    }
