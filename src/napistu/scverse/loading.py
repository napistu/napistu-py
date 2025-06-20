from __future__ import annotations

import logging
from typing import Optional, List, Union, Set, Dict

import anndata
import pandas as pd
import mudata
import numpy as np
from pydantic import BaseModel, Field, RootModel

from napistu.matching import species
from napistu.constants import ONTOLOGIES_LIST
from napistu.scverse.constants import (
    ADATA,
    ADATA_DICTLIKE_ATTRS,
    ADATA_IDENTITY_ATTRS,
    ADATA_FEATURELEVEL_ATTRS,
    ADATA_ARRAY_ATTRS,
    SCVERSE_DEFS,
    VALID_MUDATA_LEVELS,
)

logger = logging.getLogger(__name__)


def prepare_anndata_results_df(
    adata: Union[anndata.AnnData, mudata.MuData],
    table_type: str = ADATA.VAR,
    table_name: Optional[str] = None,
    results_attrs: Optional[List[str]] = None,
    ontologies: Optional[Union[Set[str], Dict[str, str]]] = None,
    index_which_ontology: Optional[str] = None,
    table_colnames: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Prepare a results table from an AnnData object for use in Napistu.

    This function extracts a table from an AnnData object and formats it for use in Napistu.
    The returned DataFrame will always include systematic identifiers from the var table,
    along with the requested results data.

    Parameters
    ----------
    adata : anndata.AnnData or mudata.MuData
        The AnnData or MuData object containing the results to be formatted.
    table_type : str, optional
        The type of table to extract from the AnnData object. Must be one of: "var", "varm", or "X".
    table_name : str, optional
        The name of the table to extract from the AnnData object.
    results_attrs : list of str, optional
        The attributes to extract from the table.
    index_which_ontology : str, optional
        The ontology to use for the systematic identifiers. This column will be pulled out of the
        index renamed to the ontology name, and added to the results table as a new column with
        the same name. Must not already exist in var table.
    ontologies : Optional[Union[Set[str], Dict[str, str]]], default=None
        Either:
        - Set of columns to treat as ontologies (these should be entries in ONTOLOGIES_LIST )
        - Dict mapping wide column names to ontology names in the ONTOLOGIES_LIST controlled vocabulary
        - None to automatically detect valid ontology columns based on ONTOLOGIES_LIST

        If index_which_ontology is defined, it should be represented in these ontologies.
    table_colnames : Optional[List[str]], optional
        Column names for varm tables. Required when table_type is "varm". Ignored otherwise.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the formatted results with systematic identifiers.
        The index will match the var_names of the AnnData object.

    Raises
    ------
    ValueError
        If table_type is not one of: "var", "varm", or "X"
        If index_which_ontology already exists in var table
    """

    if table_type not in ADATA_FEATURELEVEL_ATTRS:
        raise ValueError(
            f"table_type must be one of {ADATA_FEATURELEVEL_ATTRS}, got {table_type}"
        )

    # pull out the table containing results
    raw_results_table = _load_raw_table(adata, table_type, table_name)

    # convert the raw results to a pd.DataFrame with rows corresponding to vars and columns
    # being attributes of interest
    results_data_table = _select_results_attrs(
        adata, raw_results_table, table_type, results_attrs, table_colnames
    )

    # Extract and validate ontologies from var table
    var_ontologies = _extract_ontologies(adata.var, ontologies, index_which_ontology)

    # Combine ontologies with results data
    # Both should have the same index (var_names)
    results_table = pd.concat([var_ontologies, results_data_table], axis=1)

    return results_table


def prepare_mudata_results_df(
    mdata: mudata.MuData,
    mudata_ontologies: Union[
        "MultiModalityOntologyConfig",
        Dict[
            str,
            Dict[str, Union[Optional[Union[Set[str], Dict[str, str]]], Optional[str]]],
        ],
    ],
    table_type: str = ADATA.VAR,
    table_name: Optional[str] = None,
    results_attrs: Optional[List[str]] = None,
    table_colnames: Optional[List[str]] = None,
    level: str = "mdata",
) -> Dict[str, pd.DataFrame]:
    """
    Prepare results tables from a MuData object for use in Napistu, with adata-specific ontology handling.

    This function extracts tables from each adata in a MuData object and formats them for use in Napistu.
    Each adata's table will include systematic identifiers from its var table along with the requested results data.
    Ontology handling is configured per-adata using MultiModalityOntologyConfig.

    Parameters
    ----------
    mdata : mudata.MuData
        The MuData object containing the results to be formatted.
    mudata_ontologies : MultiModalityOntologyConfig or dict
        Configuration for ontology handling modality (each with a separate AnnData object). Must include an entry for each modality. Can be either:
        - A MultiModalityOntologyConfig object
        - A dictionary that can be converted to MultiModalityOntologyConfig using from_dict()
        Each modality's 'ontologies' field can be:
        - None to automatically detect valid ontology columns
        - Set of columns to treat as ontologies
        - Dict mapping wide column names to ontology names
        The 'index_which_ontology' field is optional.
    table_type : str, optional
        The type of table to extract from each modality. Must be one of: "var", "varm", or "X".
    table_name : str, optional
        The name of the table to extract from each modality.
    results_attrs : list of str, optional
        The attributes to extract from the table.
    table_colnames : list of str, optional
        Column names for varm tables. Required when table_type is "varm". Ignored otherwise.
    level : str, optional
        Whether to extract data from "mdata" (MuData-level) or "adata" (individual AnnData-level) tables.
        Default is "mdata".

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping modality names to their formatted results DataFrames.
        Each DataFrame contains the modality's results with systematic identifiers.
        The index of each DataFrame will match the var_names of that modality.

    Raises
    ------
    ValueError
        If table_type is not one of: "var", "varm", or "X"
        If mudata_ontologies contains invalid configuration
        If modality-specific ontology extraction fails
        If any modality is missing from mudata_ontologies
        If level is not "global" or "modality"
    """
    if table_type not in ADATA_FEATURELEVEL_ATTRS:
        raise ValueError(
            f"table_type must be one of {ADATA_FEATURELEVEL_ATTRS}, got {table_type}"
        )

    if level not in VALID_MUDATA_LEVELS:
        raise ValueError(
            f"level must be one of {sorted(VALID_MUDATA_LEVELS)}, got {level}"
        )

    # Convert dict config to MultiModalityOntologyConfig if needed
    if isinstance(mudata_ontologies, dict):
        mudata_ontologies = MultiModalityOntologyConfig.from_dict(mudata_ontologies)

    # Validate that all modalities have configurations
    missing_modalities = set(mdata.mod.keys()) - set(mudata_ontologies.root.keys())
    if missing_modalities:
        raise ValueError(
            f"Missing ontology configurations for modalities: {missing_modalities}. "
            "Each modality must have at least the 'ontologies' field specified."
        )

    if level == SCVERSE_DEFS.MDATA:
        # Use MuData-level tables
        # Pull out the table containing results
        raw_results_table = _load_raw_table(mdata, table_type, table_name)

        # Convert the raw results to a pd.DataFrame with rows corresponding to vars and columns
        # being attributes of interest
        results_data_table = _select_results_attrs(
            mdata, raw_results_table, table_type, results_attrs, table_colnames
        )

        # Split results by modality
        split_results_data_tables = _split_mdata_results_by_modality(
            mdata, results_data_table
        )
    else:
        # Use modality-level tables
        split_results_data_tables = {}
        for modality in mdata.mod.keys():
            # Load raw table from this modality
            raw_results_table = _load_raw_table(
                mdata.mod[modality], table_type, table_name
            )

            # Convert to DataFrame
            results_data_table = _select_results_attrs(
                mdata.mod[modality],
                raw_results_table,
                table_type,
                results_attrs,
                table_colnames,
            )

            split_results_data_tables[modality] = results_data_table

    # Extract each modality's ontology table and then merge it with
    # the modality's data table
    split_results_tables = {}
    for modality in mdata.mod.keys():
        # Get ontology config for this modality
        modality_ontology_spec = mudata_ontologies[modality]

        # Extract ontologies according to the modality's specification
        ontology_table = _extract_ontologies(
            mdata.mod[modality].var,
            modality_ontology_spec.ontologies,
            modality_ontology_spec.index_which_ontology,
        )

        # Combine ontologies with results
        split_results_tables[modality] = pd.concat(
            [ontology_table, split_results_data_tables[modality]], axis=1
        )

    return split_results_tables


def _load_raw_table(
    adata: Union[anndata.AnnData, mudata.MuData],
    table_type: str,
    table_name: Optional[str] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Load an AnnData table.

    This function loads an AnnData table and returns it as a pd.DataFrame.

    Parameters
    ----------
    adata : anndata.AnnData or mudata.MuData
        The AnnData or MuData object to load the table from.
    table_type : str
        The type of table to load.
    table_name : str, optional
        The name of the table to load.

    Returns
    -------
    pd.DataFrame or np.ndarray
        The loaded table.
    """

    valid_attrs = ADATA_DICTLIKE_ATTRS | ADATA_IDENTITY_ATTRS
    if table_type not in valid_attrs:
        raise ValueError(
            f"table_type {table_type} is not a valid AnnData attribute. Valid attributes are: {valid_attrs}"
        )

    if table_type in ADATA_IDENTITY_ATTRS:
        if table_name is not None:
            logger.debug(
                f"table_name {table_name} is not None, but table_type is in IDENTITY_TABLES. "
                f"table_name will be ignored."
            )
        return getattr(adata, table_type)

    # pull out a dict-like attribute
    return _get_table_from_dict_attr(adata, table_type, table_name)


def _get_table_from_dict_attr(
    adata: Union[anndata.AnnData, mudata.MuData],
    attr_name: str,
    table_name: Optional[str] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Get a table from a dict-like AnnData attribute (varm, layers, etc.)

    Parameters
    ----------
    adata : anndata.AnnData or mudata.MuData
        The AnnData or MuData object to load the table from
    attr_name : str
        Name of the attribute ('varm', 'layers', etc.)
    table_name : str, optional
        Specific table name to retrieve. If None and only one table exists,
        that table will be returned. If None and multiple tables exist,
        raises ValueError

    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        The table data. For array-type attributes (varm, varp, X, layers),
        returns numpy array. For other attributes, returns DataFrame

    Raises
    ------
    ValueError
        If attr_name is not a valid dict-like attribute
        If no tables found in the attribute
        If multiple tables found and table_name not specified
        If specified table_name not found
    """

    if attr_name not in ADATA_DICTLIKE_ATTRS:
        raise ValueError(
            f"attr_name {attr_name} is not a dict-like AnnData attribute. Valid attributes are: {ADATA_DICTLIKE_ATTRS}"
        )

    attr_dict = getattr(adata, attr_name)
    available_tables = list(attr_dict.keys())

    if len(available_tables) == 0:
        raise ValueError(f"No tables found in adata.{attr_name}")
    elif (len(available_tables) > 1) and (table_name is None):
        raise ValueError(
            f"Multiple tables found in adata.{attr_name} and table_name is not specified. "
            f"Available: {available_tables}"
        )
    elif (len(available_tables) == 1) and (table_name is None):
        return attr_dict[available_tables[0]]
    elif table_name not in available_tables:
        raise ValueError(
            f"table_name '{table_name}' not found in adata.{attr_name}. "
            f"Available: {available_tables}"
        )
    else:
        return attr_dict[table_name]


def _select_results_attrs(
    adata: anndata.AnnData,
    raw_results_table: Union[pd.DataFrame, np.ndarray],
    table_type: str,
    results_attrs: Optional[List[str]] = None,
    table_colnames: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Select results attributes from an AnnData object.

    This function selects results attributes from raw_results_table derived
    from an AnnData object and converts them if needed to a pd.DataFrame
    with appropriate indices.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the results to be formatted.
    raw_results_table : pd.DataFrame or np.ndarray
        The raw results table to be formatted.
    table_type: str,
        The type of table `raw_results_table` refers to.
    results_attrs : list of str, optional
        The attributes to extract from the raw_results_table.
    table_colnames: list of str, optional,
        If `table_type` is `varm`, this is the names of all columns (e.g., PC1, PC2, etc.). Ignored otherwise

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the formatted results.
    """
    logger.debug(
        f"_select_results_attrs called with table_type={table_type}, results_attrs={results_attrs}"
    )

    # Validate that array-type tables are not passed as DataFrames
    if table_type in ADATA_ARRAY_ATTRS and isinstance(raw_results_table, pd.DataFrame):
        raise ValueError(
            f"Table type {table_type} must be a numpy array, not a DataFrame. Got {type(raw_results_table)}"
        )

    if isinstance(raw_results_table, pd.DataFrame):
        if results_attrs is not None:
            # Get available columns for better error message
            available_attrs = raw_results_table.columns.tolist()
            missing_attrs = [
                attr for attr in results_attrs if attr not in available_attrs
            ]
            if missing_attrs:
                raise ValueError(
                    f"The following results attributes were not found: {missing_attrs}\n"
                    f"Available attributes are: {available_attrs}"
                )
            results_table_data = raw_results_table.loc[:, results_attrs]
        else:
            results_table_data = raw_results_table
        return results_table_data

    # Convert sparse matrix to dense if needed
    if hasattr(raw_results_table, "toarray"):
        raw_results_table = raw_results_table.toarray()

    valid_attrs = _get_valid_attrs_for_feature_level_array(
        adata, table_type, raw_results_table, table_colnames
    )

    if results_attrs is not None:
        invalid_results_attrs = [x for x in results_attrs if x not in valid_attrs]
        if len(invalid_results_attrs) > 0:
            raise ValueError(
                f"The following results attributes were not found: {invalid_results_attrs}\n"
                f"Available attributes are: {valid_attrs}"
            )

        # Get positions based on table type
        if table_type == ADATA.VARM:
            positions = [table_colnames.index(attr) for attr in results_attrs]
            selected_array = raw_results_table[:, positions]
        elif table_type == ADATA.VARP:
            positions = [adata.var.index.get_loc(attr) for attr in results_attrs]
            selected_array = raw_results_table[:, positions]
        else:  # X or layers
            positions = [adata.obs.index.get_loc(attr) for attr in results_attrs]
            selected_array = raw_results_table[positions, :]

        results_table_data = _create_results_df(
            selected_array, results_attrs, adata.var.index, table_type
        )
    else:
        results_table_data = _create_results_df(
            raw_results_table, valid_attrs, adata.var.index, table_type
        )

    return results_table_data


def _get_valid_attrs_for_feature_level_array(
    adata: anndata.AnnData,
    table_type: str,
    raw_results_table: np.ndarray,
    table_colnames: Optional[List[str]] = None,
) -> list[str]:
    """
    Get valid attributes for a feature-level array.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object
    table_type : str
        The type of table
    raw_results_table : np.ndarray
        The raw results table for dimension validation
    table_colnames : Optional[List[str]]
        Column names for varm tables

    Returns
    -------
    list[str]
        List of valid attributes for this table type

    Raises
    ------
    ValueError
        If table_type is invalid or if table_colnames validation fails for varm tables
    """
    if table_type not in ADATA_ARRAY_ATTRS:
        raise ValueError(
            f"table_type {table_type} is not a valid AnnData array attribute. Valid attributes are: {ADATA_ARRAY_ATTRS}"
        )

    if table_type in [ADATA.X, ADATA.LAYERS]:
        valid_attrs = adata.obs.index.tolist()
    elif table_type == ADATA.VARP:
        valid_attrs = adata.var.index.tolist()
    else:  # varm
        if table_colnames is None:
            raise ValueError("table_colnames is required for varm tables")
        if len(table_colnames) != raw_results_table.shape[1]:
            raise ValueError(
                f"table_colnames must have length {raw_results_table.shape[1]}"
            )
        valid_attrs = table_colnames

    return valid_attrs


def _create_results_df(
    array: np.ndarray, attrs: List[str], var_index: pd.Index, table_type: str
) -> pd.DataFrame:
    """Create a DataFrame with the right orientation based on table type.

    For varm/varp tables:
        - rows are vars (var_index)
        - columns are attrs (features/selected vars)
    For X/layers:
        - rows are attrs (selected observations)
        - columns are vars (var_index)
        - then transpose to get vars as rows
    """
    if table_type in [ADATA.VARM, ADATA.VARP]:
        return pd.DataFrame(array, index=var_index, columns=attrs)
    else:
        return pd.DataFrame(array, index=attrs, columns=var_index).T


def _split_mdata_results_by_modality(
    mdata: mudata.MuData,
    results_data_table: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Split a results table by modality and verify compatibility with var tables.

    Parameters
    ----------
    mdata : mudata.MuData
        MuData object containing multiple modalities
    results_data_table : pd.DataFrame
        Results table with vars as rows, typically from prepare_anndata_results_df()

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with modality names as keys and DataFrames as values.
        Each DataFrame contains just the results for that modality.
        The index of each DataFrame is guaranteed to match the corresponding
        modality's var table for later merging.

    Raises
    ------
    ValueError
        If any modality's vars are not found in the results table
        If any modality's results have different indices than its var table
    """
    # Initialize results dictionary
    results: Dict[str, pd.DataFrame] = {}

    # Process each modality
    for modality in mdata.mod.keys():
        # Get the var_names for this modality
        mod_vars = mdata.mod[modality].var_names

        # Check if all modality vars exist in results
        missing_vars = set(mod_vars) - set(results_data_table.index)
        if missing_vars:
            raise ValueError(
                f"Index mismatch in {modality}: vars {missing_vars} not found in results table"
            )

        # Extract results for this modality
        mod_results = results_data_table.loc[mod_vars]

        # Verify index alignment with var table
        if not mod_results.index.equals(mdata.mod[modality].var.index):
            raise ValueError(
                f"Index mismatch in {modality}: var table and results subset have different indices"
            )

        # Store just the results
        results[modality] = mod_results

    return results


def _extract_ontologies(
    var_table: pd.DataFrame,
    ontologies: Optional[Union[Set[str], Dict[str, str]]] = None,
    index_which_ontology: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract ontology columns from a var table, optionally including the index as an ontology.

    Parameters
    ----------
    var_table : pd.DataFrame
        The var table containing systematic identifiers
    ontologies : Optional[Union[Set[str], Dict[str, str]]], default=None
        Either:
        - Set of columns to treat as ontologies (these should be entries in ONTOLOGIES_LIST)
        - Dict mapping wide column names to ontology names in the ONTOLOGIES_LIST controlled vocabulary
        - None to automatically detect valid ontology columns based on ONTOLOGIES_LIST
    index_which_ontology : Optional[str], default=None
        If provided, extract the index as this ontology. Must not already exist in var table.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the ontology columns, with the same index as var_table

    Raises
    ------
    ValueError
        If index_which_ontology already exists in var table
        If any renamed ontology column already exists in var table
        If any rename values are duplicated
        If any final column names are not in ONTOLOGIES_LIST
    """
    # Make a copy to avoid modifying original
    var_table = var_table.copy()

    # Extract index as ontology if requested
    if index_which_ontology is not None:
        if index_which_ontology in var_table.columns:
            raise ValueError(
                f"Cannot use '{index_which_ontology}' as index_which_ontology - "
                f"column already exists in var table"
            )
        # Add the column with index values
        var_table[index_which_ontology] = var_table.index

    # if ontologies is a dict, validate rename values are unique and don't exist
    if isinstance(ontologies, dict):
        # Check for duplicate rename values
        rename_values = list(ontologies.values())
        if len(rename_values) != len(set(rename_values)):
            duplicates = [val for val in rename_values if rename_values.count(val) > 1]
            raise ValueError(
                f"Duplicate rename values found in ontologies mapping: {duplicates}. "
                "Each ontology must be renamed to a unique value."
            )

        # Check for existing columns with rename values
        existing_rename_cols = set(rename_values) & set(var_table.columns)
        if existing_rename_cols:
            # Filter out cases where we're mapping a column to itself
            actual_conflicts = {
                rename_val
                for src, rename_val in ontologies.items()
                if rename_val in existing_rename_cols and src != rename_val
            }
            if actual_conflicts:
                raise ValueError(
                    f"Cannot rename ontologies - columns already exist in var table: {actual_conflicts}"
                )

    # Validate and get matching ontologies
    matching_ontologies = species._validate_wide_ontologies(var_table, ontologies)
    if isinstance(ontologies, dict):
        var_ontologies = var_table.loc[:, ontologies.keys()]
        # Rename columns according to the mapping
        var_ontologies = var_ontologies.rename(columns=ontologies)
    else:
        var_ontologies = var_table.loc[:, list(matching_ontologies)]

    # Final validation: ensure all column names are in ONTOLOGIES_LIST
    invalid_cols = set(var_ontologies.columns) - set(ONTOLOGIES_LIST)
    if invalid_cols:
        raise ValueError(
            f"The following column names are not in ONTOLOGIES_LIST: {invalid_cols}. "
            f"All column names must be one of: {ONTOLOGIES_LIST}"
        )

    return var_ontologies


class ModalityOntologyConfig(BaseModel):
    """Configuration for ontology handling in a single modality."""

    ontologies: Optional[Union[Set[str], Dict[str, str]]] = Field(
        description="Ontology configuration. Can be either:\n"
        "- None to automatically detect valid ontology columns\n"
        "- Set of columns to treat as ontologies\n"
        "- Dict mapping wide column names to ontology names"
    )
    index_which_ontology: Optional[str] = Field(
        default=None, description="If provided, extract the index as this ontology"
    )


class MultiModalityOntologyConfig(RootModel):
    """Configuration for ontology handling across multiple modalities."""

    root: Dict[str, ModalityOntologyConfig]

    def __getitem__(self, key: str) -> ModalityOntologyConfig:
        return self.root[key]

    def items(self):
        return self.root.items()

    @classmethod
    def from_dict(
        cls,
        data: Dict[
            str,
            Dict[str, Union[Optional[Union[Set[str], Dict[str, str]]], Optional[str]]],
        ],
    ) -> "MultiModalityOntologyConfig":
        """
        Create a MultiModalityOntologyConfig from a dictionary.

        Parameters
        ----------
        data : Dict[str, Dict[str, Union[Optional[Union[Set[str], Dict[str, str]]], Optional[str]]]]
            Dictionary mapping modality names to their ontology configurations.
            Each modality config should have 'ontologies' and optionally 'index_which_ontology'.
            The 'ontologies' field can be:
            - None to automatically detect valid ontology columns
            - Set of columns to treat as ontologies
            - Dict mapping wide column names to ontology names

        Returns
        -------
        MultiModalityOntologyConfig
            Validated ontology configuration
        """
        return cls(
            root={
                modality: ModalityOntologyConfig(**config)
                for modality, config in data.items()
            }
        )

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)
