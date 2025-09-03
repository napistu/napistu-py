import logging
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from napistu.ontologies.constants import MIRBASE_TABLE_SPECS, MIRBASE_TABLES

logger = logging.getLogger(__name__)


def load_mirbase_xrefs() -> pd.DataFrame:
    """
    Load miRBase cross-reference data by combining mature database and links tables.

    This function loads the miRBase mature database and links tables, then merges them
    to create a comprehensive cross-reference dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing merged miRBase cross-reference data with database
        information and links.

    Raises
    ------
    ValueError
        If table loading fails or merge operation fails
    ConnectionError
        If unable to connect to miRBase URLs
    """

    mirbase_databases = load_mirbase_table(MIRBASE_TABLES.MATURE_DATABASE_URL)
    mirbase_xrefs = load_mirbase_table(MIRBASE_TABLES.MATURE_DATABASE_LINKS).merge(
        mirbase_databases[[MIRBASE_TABLES.DATABASE_ENTRY, MIRBASE_TABLES.DATABASE]],
        on=MIRBASE_TABLES.DATABASE_ENTRY,
    )

    return mirbase_xrefs


def load_mirbase_table(
    table_type: str, table_defs: Dict = MIRBASE_TABLE_SPECS, timeout: Optional[int] = 30
) -> pd.DataFrame:
    """
    Read miRBase cross-reference data from an HTML page.

    This function parses HTML content that contains miRBase cross-reference data
    stored in paragraph tags with <br> separators, converting it to a pandas DataFrame.

    Parameters
    ----------
    table_type : str
        The type of miRBase table to load (e.g., 'mature_database_url', 'mature_database_links').
    table_defs : Dict, optional
        Dictionary containing table definitions with URLs and headers, by default MIRBASE_TABLE_SPECS.
    timeout : int, optional
        Timeout in seconds for the HTTP request, by default 30.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed miRBase cross-reference data.
        Each row represents one cross-reference entry.

    Raises
    ------
    ValueError
        If the table_type is invalid or no data is found in the HTML.
    ConnectionError
        If unable to connect to the URL or HTTP request fails.

    Examples
    --------
    >>> df = load_mirbase_table("mature_database_url")
    >>> print(df.shape)
    (1000, 4)
    >>> print(df.head())
    """

    if table_type not in table_defs:
        raise ValueError(
            f"Invalid table type: {table_type}. Valid options are: {list(table_defs.keys())}"
        )

    url = table_defs[table_type][MIRBASE_TABLES.URL]
    header = table_defs[table_type][MIRBASE_TABLES.HEADER]

    logger.info(f"Reading miRBase {table_type} data from {url}")

    try:
        # Read HTML directly from URL
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        html_content = response.text

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the paragraph containing the data
        paragraph = soup.find("p")

        if paragraph is None:
            raise ValueError("No paragraph tag found in the HTML content")

        # Get the raw HTML of the paragraph to preserve <br> tags
        p_html = str(paragraph)

        # Remove <p> and </p> tags
        p_html = p_html.replace("<p>", "").replace("</p>", "")

        # Replace <br> tags with newlines (handle both <br> and <br/>)
        content = p_html.replace("<br>", "\n").replace("<br/>", "\n")

        # Split into lines and filter out empty ones
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        if not lines:
            raise ValueError("No data lines found after parsing HTML content")

        # Split each line by tabs (assuming tab-separated data)
        data = [line.split("\t") for line in lines]

        # Create DataFrame
        df = pd.DataFrame(data, columns=header)

        return df

    except requests.RequestException as e:
        raise ConnectionError(f"Failed to fetch data from URL: {url}. Error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing HTML content: {str(e)}")
