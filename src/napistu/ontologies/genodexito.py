import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from napistu import identifiers, sbml_dfs_core
from napistu.constants import IDENTIFIERS, ONTOLOGIES, SBML_DFS, SBML_DFS_SCHEMA
from napistu.ontologies._validation import GenodexitoConfig
from napistu.ontologies.constants import (
    GENODEXITO_DEFS,
    INTERCONVERTIBLE_GENIC_ONTOLOGIES,
)
from napistu.ontologies.mygene import create_python_mapping_tables

logger = logging.getLogger(__name__)


class Genodexito:
    """A tool for mapping gene identifiers across ontologies.

    Genodexito provides a unified interface for mapping between different gene identifier
    ontologies (e.g. Ensembl, Entrez, UniProt). It supports both an R-centric workflow
    using Bioconductor through RPy2, as well as a Python-centric workflow using MyGene.info.

    The class automatically handles fallback between the two methods if one fails.

    Parameters
    ----------
    organismal_species : str, optional
        The organismal species to map identifiers for, by default "Homo sapiens"
    preferred_method : str, optional
        Which mapping method to try first ("bioconductor" or "python"), by default "bioconductor"
    allow_fallback : bool, optional
        Whether to allow falling back to the other method if preferred fails, by default True
    r_paths : Optional[List[str]], optional
        Optional paths to R libraries for Bioconductor, by default None
    test_mode : bool, optional
        If True, limit queries to 1000 genes for testing purposes, by default False

    Attributes
    ----------
    mappings : Optional[Dict[str, pd.DataFrame]]
        Dictionary of mapping tables between ontologies
    mapper_used : Optional[str]
        Which mapping method was successfully used ("bioconductor" or "python")
    merged_mappings : Optional[pd.DataFrame]
        Combined wide-format mapping table
    stacked_mappings : Optional[pd.DataFrame]
        Combined long-format mapping table

    Methods
    -------
    create_mapping_tables(mappings: Set[str], overwrite: bool = False)
        Create mapping tables between different ontologies. This is the primary method
        to fetch and store identifier mappings. Must be called before using other methods.

    merge_mappings(ontologies: Optional[Set[str]] = None)
        Create a wide-format table where each row is an Entrez gene ID and columns
        contain the corresponding identifiers in other ontologies.

    stack_mappings(ontologies: Optional[Set[str]] = None)
        Create a long-format table combining all mappings, with columns for
        ontology type and identifier values.

    expand_sbml_dfs_ids(sbml_dfs: sbml_dfs_core.SBML_dfs, ontologies: Optional[Set[str]] = None)
        Update the expanded identifiers for a model by adding additional related
        ontologies pulled from Bioconductor or MyGene.info.

    Examples
    --------
    >>> # Initialize mapper with Python method
    >>> geno = Genodexito(preferred_method="python")
    >>>
    >>> # Create mapping tables for specific ontologies
    >>> mappings = {'ensembl_gene', 'symbol', 'uniprot'}
    >>> geno.create_mapping_tables(mappings)
    >>>
    >>> # Create merged wide-format table
    >>> geno.merge_mappings()
    >>> print(geno.merged_mappings.head())
    >>>
    >>> # Create stacked long-format table
    >>> geno.stack_mappings()
    >>> print(geno.stacked_mappings.head())
    """

    def __init__(
        self,
        organismal_species: str = "Homo sapiens",
        preferred_method: str = GENODEXITO_DEFS.BIOCONDUCTOR,
        allow_fallback: bool = True,
        r_paths: Optional[List[str]] = None,
        test_mode: bool = False,
    ) -> None:
        """
        Initialize unified gene mapper

        Parameters
        ----------
        organismal_species : str, optional
            Species name, by default "Homo sapiens"
        preferred_method : str, optional
            Which mapping method to try first ("bioconductor" or "python"), by default "bioconductor"
        allow_fallback : bool, optional
            Whether to allow falling back to other method if preferred fails, by default True
        r_paths : Optional[List[str]], optional
            Optional paths to R libraries for Bioconductor, by default None
        test_mode : bool, optional
            If True, limit queries to 1000 genes for testing purposes, by default False
        """
        # Validate configuration using Pydantic model
        config = GenodexitoConfig(
            organismal_species=organismal_species,
            preferred_method=preferred_method,
            allow_fallback=allow_fallback,
            r_paths=r_paths,
            test_mode=test_mode,
        )

        self.organismal_species = config.organismal_species
        self.preferred_method = config.preferred_method
        self.allow_fallback = config.allow_fallback
        self.r_paths = config.r_paths
        self.test_mode = config.test_mode

        # Initialize empty attributes
        self.mappings: Optional[Dict[str, pd.DataFrame]] = None
        self.mapper_used: Optional[str] = None
        self.merged_mappings: Optional[pd.DataFrame] = None
        self.stacked_mappings: Optional[pd.DataFrame] = None

    def create_mapping_tables(
        self, mappings: Set[str], overwrite: bool = False
    ) -> None:
        """Create mapping tables between different ontologies.

        This is a drop-in replacement for create_bioconductor_mapping_tables that handles
        both Bioconductor and Python-based mapping methods.

        Parameters
        ----------
        mappings : Set[str]
            Set of ontologies to create mappings for
        overwrite : bool, optional
            Whether to overwrite existing mappings, by default False

        Returns
        -------
        None
            Updates self.mappings and self.mapper_used in place
        """

        # check for existing mappings
        if self.mappings is not None and not overwrite:
            logger.warning(
                f"Mapping tables for {self.organismal_species} already exist. Use overwrite=True to create new mappings."
            )
            return None

        if self.preferred_method == GENODEXITO_DEFS.BIOCONDUCTOR:
            try:
                # Only import R functionality when needed
                from napistu.rpy2.rids import create_bioconductor_mapping_tables

                self.mappings = create_bioconductor_mapping_tables(
                    mappings=mappings,
                    species=self.organismal_species,
                    r_paths=self.r_paths,
                )
                self.mapper_used = GENODEXITO_DEFS.BIOCONDUCTOR
            except Exception as e:
                if self.allow_fallback:
                    logger.warning(
                        f"Error creating bioconductor mapping tables for {self.organismal_species} with {mappings}. Falling back to python."
                    )
                    self.mappings = create_python_mapping_tables(
                        mappings=mappings,
                        species=self.organismal_species,
                        test_mode=self.test_mode,
                    )
                    self.mapper_used = GENODEXITO_DEFS.PYTHON
                else:
                    logger.error(
                        f"Error creating bioconductor mapping tables for {self.organismal_species} with {mappings} and fallback is disabled."
                    )
                    raise e

        elif self.preferred_method == GENODEXITO_DEFS.PYTHON:
            try:
                self.mappings = create_python_mapping_tables(
                    mappings=mappings,
                    species=self.organismal_species,
                    test_mode=self.test_mode,
                )
                self.mapper_used = GENODEXITO_DEFS.PYTHON
            except Exception as e:
                if self.allow_fallback:
                    logger.warning(
                        f"Error creating mygene Python mapping tables for {self.organismal_species} with {mappings}. Trying the bioconductor fallback."
                    )
                    # Only import R functionality when needed
                    from napistu.rpy2.rids import create_bioconductor_mapping_tables

                    self.mappings = create_bioconductor_mapping_tables(
                        mappings=mappings,
                        species=self.organismal_species,
                        r_paths=self.r_paths,
                    )
                    self.mapper_used = GENODEXITO_DEFS.BIOCONDUCTOR
                else:
                    logger.error(
                        f"Error creating Python mapping tables for {self.organismal_species} with {mappings} and fallback is disabled."
                    )
                    raise e

        else:
            raise ValueError(f"Invalid preferred_method: {self.preferred_method}")

        return None

    def merge_mappings(self, ontologies: Optional[Set[str]] = None) -> None:
        """Merge mappings into a single wide table.

        Creates a wide-format table where each row is an Entrez gene ID and
        columns contain the corresponding identifiers in other ontologies.

        Parameters
        ----------
        ontologies : Optional[Set[str]], optional
            Set of ontologies to include in merged table, by default None
            If None, uses all available ontologies

        Returns
        -------
        None
            Updates self.merged_mappings in place

        Raises
        ------
        ValueError
            If mappings don't exist or requested ontologies are invalid
        TypeError
            If any identifiers are not strings
        ValueError
            If any mapping tables contain NA values
        """

        # mappings must exist and be valid
        self._check_mappings()
        ontologies = self._use_mappings(ontologies)

        running_ids = self.mappings[ONTOLOGIES.NCBI_ENTREZ_GENE]

        for mapping in ontologies:
            logger.debug(f"adding entries for {mapping} to running_ids")
            mapping_df = self.mappings[mapping]

            running_ids = running_ids.join(mapping_df)

        running_ids = running_ids.reset_index()

        self.merged_mappings = running_ids

        return None

    def stack_mappings(self, ontologies: Optional[Set[str]] = None) -> None:
        """Stack mappings into a single long table.

        Convert a dict of mappings between Entrez identifiers and other identifiers
        into a single long-format table.

        Parameters
        ----------
        ontologies : Optional[Set[str]], optional
            Set of ontologies to include in stacked table, by default None
            If None, uses all available ontologies

        Returns
        -------
        None
            Updates self.stacked_mappings in place

        Raises
        ------
        ValueError
            If mappings don't exist or requested ontologies are invalid
        TypeError
            If any identifiers are not strings
        ValueError
            If any mapping tables contain NA values
        """

        # mappings must exist and be valid
        self._check_mappings()
        ontologies = self._use_mappings(ontologies)

        mappings_list = list()
        for ont in ontologies:
            one_mapping_df = (
                self.mappings[ont]
                .assign(ontology=ont)
                .rename({ont: IDENTIFIERS.IDENTIFIER}, axis=1)
            )

            mappings_list.append(one_mapping_df)

        self.stacked_mappings = pd.concat(mappings_list)

    def expand_sbml_dfs_ids(
        self, sbml_dfs: sbml_dfs_core.SBML_dfs, ontologies: Optional[Set[str]] = None
    ) -> sbml_dfs_core.SBML_dfs:
        """Update the expanded identifiers for a model.

        Parameters
        ----------
        sbml_dfs : sbml_dfs_core.SBML_dfs
            The SBML model to update with expanded identifiers
        ontologies : Optional[Set[str]], optional
            Set of ontologies to use for mapping. If None, uses all available ontologies
            from INTERCONVERTIBLE_GENIC_ONTOLOGIES.

        Returns
        -------
        sbml_dfs_core.SBML_dfs
            Updated SBML model with expanded identifiers
        """

        ids = getattr(sbml_dfs, "species")

        # If no ontologies specified, use all available ones
        if ontologies is None:
            ontologies = INTERCONVERTIBLE_GENIC_ONTOLOGIES
        else:
            # Ensure ncbi_entrez_gene is included in the ontologies
            ontologies = set(ontologies)
            ontologies.add(ONTOLOGIES.NCBI_ENTREZ_GENE)

            invalid_ontologies = ontologies - INTERCONVERTIBLE_GENIC_ONTOLOGIES
            if invalid_ontologies:
                raise ValueError(
                    f"Invalid ontologies: {', '.join(invalid_ontologies)}.\n"
                    f"Valid options are: {', '.join(sorted(INTERCONVERTIBLE_GENIC_ONTOLOGIES))}"
                )

        # create mapping tables if they don't exist
        if self.mappings is None:
            self.create_mapping_tables(ontologies)

        # select and validate mappings
        ontologies = self._use_mappings(ontologies)

        if self.merged_mappings is None:
            self.merge_mappings(ontologies)

        # merge existing and new identifiers
        expanded_ids = self._create_expanded_identifiers(sbml_dfs, ontologies)

        # make sure expanded_ids and original model.species have same number of s_ids
        # if a s_id only in model.species, adding it to expanded_ids.
        if ids.shape[0] != expanded_ids.shape[0]:
            matched_expanded_ids = expanded_ids.combine_first(
                ids[SBML_DFS.S_IDENTIFIERS]
            )
            logger.debug(
                f"{ids.shape[0] - expanded_ids.shape[0]} "
                "ids are not included in expanded ids. These will be filled with empty Identifiers"
            )
        else:
            matched_expanded_ids = expanded_ids

        updated_ids = ids.drop(SBML_DFS.S_IDENTIFIERS, axis=1).join(
            pd.DataFrame(matched_expanded_ids)
        )
        # fill missing attributes with empty Identifiers
        updated_ids[SBML_DFS.S_IDENTIFIERS] = updated_ids[
            SBML_DFS.S_IDENTIFIERS
        ].fillna(identifiers.Identifiers([]))

        setattr(sbml_dfs, "species", updated_ids)

        return sbml_dfs

    def _check_mappings(self) -> None:
        """Check that mappings exist and contain required ontologies.

        Raises
        ------
        ValueError
            If mappings don't exist or don't contain NCBI_ENTREZ_GENE
        TypeError
            If any identifiers are not strings
        ValueError
            If any mapping tables contain NA values
        """
        if self.mappings is None:
            raise ValueError(
                f"Mapping tables for {self.organismal_species} do not exist. Use create_mapping_tables to create new mappings."
            )

        # entrez should always be present if any mappings exist
        if ONTOLOGIES.NCBI_ENTREZ_GENE not in self.mappings.keys():
            raise ValueError(
                f"Mapping tables for {self.organismal_species} do not contain {ONTOLOGIES.NCBI_ENTREZ_GENE}. Use create_mapping_tables to create new mappings."
            )

        # Check that all identifiers are strings
        for ontology, df in self.mappings.items():
            # Check index (which should be NCBI_ENTREZ_GENE)
            if not df.index.dtype == "object":
                raise TypeError(
                    f"Index of mapping table for {ontology} contains non-string values. "
                    f"Found type: {df.index.dtype}"
                )

            # Check all columns
            for col in df.columns:
                if not df[col].dtype == "object":
                    raise TypeError(
                        f"Column {col} in mapping table for {ontology} contains non-string values. "
                        f"Found type: {df[col].dtype}"
                    )

            # Check for NA values in index
            if df.index.isna().any():
                raise ValueError(
                    f"Mapping table for {ontology} contains NA values in index (NCBI_ENTREZ_GENE). "
                    f"Found {df.index.isna().sum()} NA values."
                )

            # Check for NA values in columns
            na_counts = df.isna().sum()
            if na_counts.any():
                na_cols = na_counts[na_counts > 0].index.tolist()
                raise ValueError(
                    f"Mapping table for {ontology} contains NA values in columns: {na_cols}. "
                    f"NA counts per column: {na_counts[na_cols].to_dict()}"
                )

    def _use_mappings(self, ontologies: Optional[Set[str]]) -> Set[str]:
        """Validate and process ontologies for mapping operations.

        Parameters
        ----------
        ontologies : Optional[Set[str]]
            Set of ontologies to validate. If None, uses all available mappings.

        Returns
        -------
        Set[str]
            Set of validated ontologies to use

        Raises
        ------
        ValueError
            If mappings don't exist or ontologies are invalid
        """

        if self.mappings is None:
            raise ValueError(
                f"Mapping tables for {self.organismal_species} do not exist. Use create_mapping_tables to create new mappings."
            )

        if ontologies is None:
            return set(self.mappings.keys())

        # validate provided mappings to see if they are genic ontologies within the controlled vocabulary
        never_valid_mappings = set(ontologies) - INTERCONVERTIBLE_GENIC_ONTOLOGIES
        if never_valid_mappings:
            raise ValueError(
                f"Invalid mappings: {', '.join(never_valid_mappings)}. "
                f"Valid mappings are {', '.join(INTERCONVERTIBLE_GENIC_ONTOLOGIES)}"
            )

        # validate provided mappings against existing mappings
        missing_mappings = set(ontologies) - set(self.mappings.keys())
        if missing_mappings:
            raise ValueError(
                f"Missing mappings: {', '.join(missing_mappings)}. "
                f"Recreate mappings by calling create_mapping_tables() while including "
                f"{', '.join(missing_mappings)} and other mappings of interest."
            )

        return ontologies

    def _create_expanded_identifiers(
        self,
        sbml_dfs: sbml_dfs_core.SBML_dfs,
        ontologies: Optional[Set[str]] = None,
    ) -> pd.Series:
        """Create expanded identifiers for SBML species.

        Update a table's identifiers to include additional related ontologies.
        Ontologies are pulled from the bioconductor "org" packages or MyGene.info.

        Parameters
        ----------
        sbml_dfs : sbml_dfs_core.SBML_dfs
            A relational pathway model built around reactions interconverting
            compartmentalized species
        ontologies : Optional[Set[str]], optional
            Ontologies to add or complete, by default None
            If None, uses all available ontologies

        Returns
        -------
        pd.Series
            Series with identifiers as the index and updated Identifiers objects as values

        Raises
        ------
        ValueError
            If merged mappings don't exist or all requested ontologies already exist
        TypeError
            If identifiers are not in expected format
        """

        ontologies = self._use_mappings(ontologies)
        if self.merged_mappings is None:
            raise ValueError(
                "Merged mappings do not exist. Use merge_mappings() to create new mappings."
            )

        # pull out all identifiers as a pd.DataFrame
        all_entity_identifiers = sbml_dfs.get_identifiers("species")
        if not isinstance(all_entity_identifiers, pd.DataFrame):
            raise TypeError("all_entity_identifiers must be a pandas DataFrame")

        # find entries in valid_expanded_ontologies which are already present
        # these are the entries that will be used to expand to other ontologies
        # or fill in ontologies with incomplete annotations
        starting_ontologies = ontologies.intersection(
            set(all_entity_identifiers["ontology"])
        )

        if len(starting_ontologies) == 0:
            raise ValueError(
                f"None of the ontologies currently in the sbml_dfs match `ontologies`. The currently included ontologies are {set(all_entity_identifiers['ontology'])}. If there are major genic ontologies in this list then you may need to use ontologies.clean_ontologies() to convert from aliases to ontologies in the ONTOLOGIES controlled vocabulary."
            )

        expanded_ontologies = ontologies - starting_ontologies
        if len(expanded_ontologies) == 0:
            raise ValueError(
                "All of the requested ontologies already exist in species' s_Identifiers"
            )

        # map from existing ontologies to expanded ontologies
        ontology_mappings = list()
        # starting w/
        for start in starting_ontologies:
            # ending w/
            for end in expanded_ontologies:
                if start == end:
                    continue
                lookup = (
                    self.merged_mappings[[start, end]]
                    .rename(
                        columns={start: IDENTIFIERS.IDENTIFIER, end: "new_identifier"}
                    )
                    .assign(ontology=start)
                    .assign(new_ontology=end)
                )

                ontology_mappings.append(lookup)

        ontology_mappings_df = pd.concat(ontology_mappings).dropna()

        # old identifiers joined with new identifiers

        # first, define the names of keys and ids
        table_pk_var = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.SPECIES]["pk"]

        # retain bqb terms to define how an identifier is related to sid
        # this relation will be preserved for the new ids

        merged_identifiers = all_entity_identifiers[
            [
                table_pk_var,
                IDENTIFIERS.ONTOLOGY,
                IDENTIFIERS.IDENTIFIER,
                IDENTIFIERS.BQB,
            ]
        ].merge(ontology_mappings_df)

        # new, possibly redundant identifiers
        new_identifiers = merged_identifiers[
            [table_pk_var, "new_ontology", "new_identifier", IDENTIFIERS.BQB]
        ].rename(
            columns={
                "new_ontology": IDENTIFIERS.ONTOLOGY,
                "new_identifier": IDENTIFIERS.IDENTIFIER,
            }
        )

        expanded_identifiers_df = pd.concat(
            [
                all_entity_identifiers[
                    [
                        table_pk_var,
                        IDENTIFIERS.ONTOLOGY,
                        IDENTIFIERS.IDENTIFIER,
                        IDENTIFIERS.URL,
                        IDENTIFIERS.BQB,
                    ]
                ],
                new_identifiers,
                # ignore new identifier if it already exists
            ]
        )

        output = identifiers.df_to_identifiers(expanded_identifiers_df)

        return output
