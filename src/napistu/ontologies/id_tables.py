import logging
from typing import Optional, Set, Union

import pandas as pd

from napistu import utils
from napistu.constants import (
    BQB,
    BQB_DEFINING_ATTRS_LOOSE,
    IDENTIFIERS,
    SBML_DFS_SCHEMA,
    SCHEMA_DEFS,
    VALID_BQB_TERMS,
)

logger = logging.getLogger(__name__)


def filter_id_table(
    id_table: pd.DataFrame,
    identifiers: Optional[Union[str, list, set]] = None,
    ontologies: Optional[Union[str, list, set]] = None,
    bqbs: Optional[Union[str, list, set]] = BQB_DEFINING_ATTRS_LOOSE + [BQB.HAS_PART],
) -> pd.DataFrame:
    """
    Filter an identifier table by identifiers, ontologies, and BQB terms for a given entity type.

    Parameters
    ----------
    id_table : pd.DataFrame
        DataFrame containing identifier mappings to be filtered.
    identifiers : str, list, set, or None, optional
        Identifiers to filter by. If None, no filtering is applied on identifiers.
    ontologies : str, list, set, or None, optional
        Ontologies to filter by. If None, no filtering is applied on ontologies.
    bqbs : str, list, set, or None, optional
        BQB terms to filter by. If None, no filtering is applied on BQB terms. Default is [BQB.IS, BQB.HAS_PART].

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows matching the specified criteria.

    Raises
    ------
    ValueError
        If the id_table or filter values are invalid, or required columns are missing.
    """

    entity_type = utils.infer_entity_type(id_table)
    _validate_id_table(id_table, entity_type)

    # bqbs
    if bqbs is not None:
        bqbs = _sanitize_id_table_bqbs(bqbs, id_table)
        id_table = id_table.query("bqb in @bqbs")

    # ontologies
    if ontologies is not None:
        ontologies = _sanitize_id_table_ontologies(ontologies, id_table)
        id_table = id_table.query("ontology in @ontologies")

    # identifiers
    if identifiers is not None:
        identifiers = _sanitize_id_table_identifiers(identifiers, id_table)
        id_table = id_table.query("identifier in @identifiers")

    # return the filtered id_table
    return id_table


def _validate_id_table(id_table: pd.DataFrame, entity_type: str) -> None:
    """
    Validate that the id_table contains the required columns and matches the schema for the given entity_type.

    Parameters
    ----------
    id_table : pd.DataFrame
        DataFrame containing identifier mappings for a given entity type.
    entity_type : str
        The type of entity (e.g., 'species', 'reactions') to validate against the schema.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If entity_type is not present in the schema, or if required columns are missing in id_table.
    """

    schema = SBML_DFS_SCHEMA.SCHEMA

    if entity_type not in schema.keys():
        raise ValueError(
            f"{entity_type} does not match a table in the SBML_dfs object. The tables "
            f"which are present are {', '.join(schema.keys())}"
        )

    entity_table_attrs = schema[entity_type]

    if SCHEMA_DEFS.ID not in entity_table_attrs.keys():
        raise ValueError(f"{entity_type} does not have an 'id' attribute")

    entity_pk = entity_table_attrs[SCHEMA_DEFS.PK]

    utils.match_pd_vars(
        id_table,
        req_vars={
            entity_pk,
            IDENTIFIERS.ONTOLOGY,
            IDENTIFIERS.IDENTIFIER,
            IDENTIFIERS.URL,
            IDENTIFIERS.BQB,
        },
        allow_series=False,
    ).assert_present()

    return None


def _sanitize_id_table_values(
    values: Union[str, list, set],
    id_table: pd.DataFrame,
    column_name: str,
    valid_values: Optional[Set[str]] = None,
    value_type_name: str = None,
) -> set:
    """
    Generic function to sanitize and validate values against an id_table column.

    Parameters
    ----------
    values : str, list, or set
        Values to sanitize and validate. Can be a single string, list of strings,
        or set of strings.
    id_table : pd.DataFrame
        DataFrame containing the reference data to validate against.
    column_name : str
        Name of the column in id_table to check values against.
    valid_values : set of str, optional
        Optional set of globally valid values for additional validation
        (e.g., VALID_BQB_TERMS). If provided, values must be a subset of this set.
    value_type_name : str, optional
        Human-readable name for the value type used in error messages.
        If None, defaults to column_name.

    Returns
    -------
    set
        Set of sanitized and validated values.

    Raises
    ------
    ValueError
        If values is not a string, list, or set.
        If any values are not in valid_values (when provided).
        If none of the requested values are present in the id_table.

    Warnings
    --------
    Logs a warning if some (but not all) requested values are missing from id_table.
    """
    if value_type_name is None:
        value_type_name = column_name

    # Convert to set
    if isinstance(values, str):
        values = {values}
    elif isinstance(values, list):
        values = set(values)
    elif isinstance(values, set):
        pass
    else:
        raise ValueError(
            f"{value_type_name} must be a string, a set, or list, got {type(values).__name__}"
        )

    # Check against global valid values if provided
    if valid_values is not None:
        invalid_values = values.difference(valid_values)
        if len(invalid_values) > 0:
            raise ValueError(
                f"The following {value_type_name} are not valid: {', '.join(invalid_values)}.\n"
                f"Valid {value_type_name} are {', '.join(valid_values)}"
            )

    # Check against values present in the id_table
    available_values = set(id_table[column_name].unique())
    missing_values = values.difference(available_values)

    if len(missing_values) == len(values):
        raise ValueError(
            f"None of the requested {value_type_name} are present in the id_table: {', '.join(missing_values)}.\n"
            f"The included {value_type_name} are {', '.join(available_values)}"
        )
    elif len(missing_values) > 0:
        logger.warning(
            f"The following {value_type_name} are not present in the id_table: {', '.join(missing_values)}.\n"
            f"The included {value_type_name} are {', '.join(available_values)}"
        )

    return values


def _sanitize_id_table_ontologies(
    ontologies: Union[str, list, set], id_table: pd.DataFrame
) -> set:
    """
    Sanitize and validate ontologies against the id_table.

    Parameters
    ----------
    ontologies : str, list, or set
        Ontology names to validate.
    id_table : pd.DataFrame
        DataFrame containing ontology reference data.

    Returns
    -------
    set
        Set of validated ontology names.
    """
    return _sanitize_id_table_values(
        values=ontologies,
        id_table=id_table,
        column_name=IDENTIFIERS.ONTOLOGY,
        value_type_name="ontologies",
    )


def _sanitize_id_table_bqbs(bqbs: Union[str, list, set], id_table: pd.DataFrame) -> set:
    """
    Sanitize and validate BQBs against the id_table.

    Parameters
    ----------
    bqbs : str, list, or set
        BQB terms to validate.
    id_table : pd.DataFrame
        DataFrame containing BQB reference data.

    Returns
    -------
    set
        Set of validated BQB terms.
    """
    return _sanitize_id_table_values(
        values=bqbs,
        id_table=id_table,
        column_name=IDENTIFIERS.BQB,
        valid_values=VALID_BQB_TERMS,
        value_type_name="bqbs",
    )


def _sanitize_id_table_identifiers(
    identifiers: Union[str, list, set], id_table: pd.DataFrame
) -> set:
    """
    Sanitize and validate identifiers against the id_table.

    Parameters
    ----------
    identifiers : str, list, or set
        Identifier values to validate.
    id_table : pd.DataFrame
        DataFrame containing identifier reference data.

    Returns
    -------
    set
        Set of validated identifiers.
    """
    return _sanitize_id_table_values(
        values=identifiers,
        id_table=id_table,
        column_name=IDENTIFIERS.IDENTIFIER,
        value_type_name="identifiers",
    )
