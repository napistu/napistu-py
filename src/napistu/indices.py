from __future__ import annotations

import copy
import datetime
import os
import re
import warnings
from os import PathLike
from typing import Iterable, Optional

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs
import pandas as pd

from napistu.constants import (
    EXPECTED_PW_INDEX_COLUMNS,
    SOURCE_SPEC,
)
from napistu.utils import path_exists


class PWIndex:
    """Pathway Index for organizing metadata and paths of pathway representations.

    The PWIndex class manages a collection of pathway files and their associated
    metadata. It provides functionality to filter, search, and validate pathway
    data across different sources and species.

    Attributes
    ----------
    index : pd.DataFrame
        A table describing the location and contents of pathway files.
        Contains columns for pathway_id, name, source, organismal_species,
        file path, URL, and other metadata.
    base_path : str or None
        Path to directory of indexed files. Set to None if path validation
        is disabled.

    Methods
    -------
    filter(data_sources, organismal_species)
        Filter index based on pathway source and/or organismal species
    search(query)
        Filter index to pathways matching the search query

    Examples
    --------
    >>> # Create a pathway index from a file
    >>> pw_index = PWIndex("path/to/pw_index.tsv")
    >>>
    >>> # Filter for specific sources and species
    >>> pw_index.filter(data_sources=["BiGG", "Reactome"], organismal_species="human")
    >>>
    >>> # Search for pathways containing "metabolism"
    >>> pw_index.search("metabolism")
    >>>
    >>> # Create from DataFrame
    >>> df = pd.DataFrame({
    ...     'pathway_id': ['R-HSA-123456'],
    ...     'name': ['Test Pathway'],
    ...     'source': ['Reactome'],
    ...     'organismal_species': ['human'],
    ...     'file': ['test.sbml'],
    ...     'url': ['https://example.com'],
    ...     'sbml_path': ['/path/to/test.sbml'],
    ...     'date': ['20231201']
    ... })
    >>> pw_index = PWIndex(df)
    """

    def __init__(
        self,
        pw_index: PathLike[str] | str | pd.DataFrame,
        pw_index_base_path=None,
        validate_paths=True,
    ) -> None:
        """Initialize a Pathway Index object.

        Creates a PWIndex instance from a file path, DataFrame, or PathLike object.
        The index contains metadata about pathway files and can optionally validate
        that the referenced files exist.

        Parameters
        ----------
        pw_index : PathLike[str] or str or pd.DataFrame
            Path to index file, or a DataFrame containing pathway index data.
            The DataFrame should contain all required columns defined in
            EXPECTED_PW_INDEX_COLUMNS.
        pw_index_base_path : str or None, optional
            Base path that relative paths in pw_index will reference.
            If None and pw_index is a file path, uses the directory of pw_index.
        validate_paths : bool, optional
            If True, validates that files referenced in the index exist.
            If False, skips file validation and sets base_path to None.
            Default is True.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If pw_index is not a valid type or if required columns are missing.
        FileNotFoundError
            If validate_paths is True and base_path is not a valid directory.
        TypeError
            If pw_index_base_path is not a string or validate_paths is not a boolean.

        Examples
        --------
        >>> # Create from file path
        >>> pw_index = PWIndex("path/to/pw_index.tsv")
        >>>
        >>> # Create from DataFrame
        >>> df = pd.DataFrame({
        ...     'pathway_id': ['R-HSA-123456'],
        ...     'name': ['Test Pathway'],
        ...     'source': ['Reactome'],
        ...     'organismal_species': ['human'],
        ...     'file': ['test.sbml'],
        ...     'url': ['https://example.com'],
        ...     'sbml_path': ['/path/to/test.sbml'],
        ...     'date': ['20231201']
        ... })
        >>> pw_index = PWIndex(df)
        >>>
        >>> # Create with custom base path and no validation
        >>> pw_index = PWIndex(
        ...     "pw_index.tsv",
        ...     pw_index_base_path="/custom/path",
        ...     validate_paths=False
        ... )
        """

        # read index either directly from pandas or from a file
        if isinstance(pw_index, pd.DataFrame):
            self.index = pw_index
        elif isinstance(pw_index, PathLike) or isinstance(pw_index, str):
            base_path = os.path.dirname(pw_index)
            file_name = os.path.basename(pw_index)
            with open_fs(base_path) as base_fs:
                with base_fs.open(file_name) as f:
                    self.index = pd.read_table(f)
        else:
            raise ValueError(
                f"pw_index needs to be of type PathLike[str] | str | pd.DataFrame but was {type(pw_index).__name__}"
            )

        # format option arguments
        if (pw_index_base_path is not None) and (
            not isinstance(pw_index_base_path, str)
        ):
            raise TypeError(
                f"pw_index_base_path was as {type(pw_index_base_path).__name__} and must be a str if provided"
            )

        if not isinstance(validate_paths, bool):
            raise TypeError(
                f"validate_paths was as {type(validate_paths).__name__} and must be a bool"
            )

        # verify that the index is syntactically correct

        observed_columns = set(self.index.columns.to_list())

        if EXPECTED_PW_INDEX_COLUMNS != observed_columns:
            missing = ", ".join(EXPECTED_PW_INDEX_COLUMNS.difference(observed_columns))
            extra = ", ".join(observed_columns.difference(EXPECTED_PW_INDEX_COLUMNS))
            raise ValueError(
                f"Observed pw_index columns did not match expected columns:\n"
                f"Missing columns: {missing}\nExtra columns: {extra}"
            )

        # verify that all pathway_ids are unique
        duplicated_pathway_ids = list(
            self.index[SOURCE_SPEC.PATHWAY_ID][
                self.index[SOURCE_SPEC.PATHWAY_ID].duplicated()
            ]
        )
        if len(duplicated_pathway_ids) != 0:
            path_str = "\n".join(duplicated_pathway_ids)
            raise ValueError(
                f"{len(duplicated_pathway_ids)} pathway_ids were duplicated:\n{path_str}"
            )

        if validate_paths:
            if pw_index_base_path is not None:
                self.base_path = pw_index_base_path
            elif isinstance(pw_index, PathLike) or isinstance(pw_index, str):
                self.base_path = os.path.dirname(pw_index)
            else:
                raise ValueError(
                    "validate_paths was True but neither pw_index_base_path "
                    "nor an index path were provided. Please provide "
                    "pw_index_base_path if you intend to verify that "
                    "the files present in pw_index exist"
                )

            if path_exists(self.base_path) is False:
                raise FileNotFoundError(
                    "base_path at {self.base_path} is not a valid directory"
                )

            # verify that pathway files exist
            self._check_files()

        elif pw_index_base_path is not None:
            print(
                "validate_paths is False so pw_index_base_path will be ignored and paths will not be validated"
            )

    def _check_files(self):
        """Verify that all files referenced in the pathway index exist.

        Checks that all files listed in the index's 'file' column exist
        in the base_path directory. This is used for validation during
        initialization when validate_paths=True.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If any files referenced in the index are missing from the base_path.

        Examples
        --------
        >>> # This method is called internally during initialization
        >>> pw_index = PWIndex("path/to/pw_index.tsv", validate_paths=True)
        >>> # If any files are missing, FileNotFoundError will be raised
        """
        with open_fs(self.base_path) as base_fs:
            # verify that pathway files exist
            files = base_fs.listdir(".")
            missing_pathway_files = set(self.index[SOURCE_SPEC.FILE]) - set(files)
            if len(missing_pathway_files) != 0:
                file_str = "\n".join(missing_pathway_files)
                raise FileNotFoundError(
                    f"{len(missing_pathway_files)} were missing:\n{file_str}"
                )

    def filter(
        self,
        data_sources: str | Iterable[str] | None = None,
        organismal_species: str | Iterable[str] | None = None,
    ):
        """Filter the pathway index by data sources and/or organismal species.

        Modifies the index in-place to include only pathways that match the
        specified criteria. If no filters are provided, the index remains unchanged.

        Parameters
        ----------
        data_sources : str or Iterable[str] or None, optional
            Data sources to filter for (e.g., ["BiGG", "Reactome"]).
            If None, no filtering by data source is applied.
        organismal_species : str or Iterable[str] or None, optional
            Organismal species to filter for (e.g., ["human", "mouse"]).
            If None, no filtering by species is applied.

        Returns
        -------
        None
            Modifies the index in-place.

        Examples
        --------
        >>> # Filter for specific data sources
        >>> pw_index.filter(data_sources=["BiGG", "Reactome"])
        >>>
        >>> # Filter for specific species
        >>> pw_index.filter(organismal_species="human")
        >>>
        >>> # Filter for both sources and species
        >>> pw_index.filter(
        ...     data_sources=["BiGG"],
        ...     organismal_species=["human", "mouse"]
        ... )
        >>>
        >>> # No filtering (index remains unchanged)
        >>> pw_index.filter()
        """
        pw_index = self.index
        if data_sources is not None:
            pw_index = pw_index.query(f"{SOURCE_SPEC.DATA_SOURCE} in @data_sources")

        if organismal_species is not None:
            pw_index = pw_index.query(
                f"{SOURCE_SPEC.ORGANISMAL_SPECIES} in @organismal_species"
            )

        self.index = pw_index

    def search(self, query):
        """Search the pathway index for pathways matching a query string.

        Filters the index in-place to include only pathways whose names
        contain the query string (case-insensitive). Uses regex matching
        for flexible pattern matching.

        Parameters
        ----------
        query : str
            Search query to match against pathway names.
            Case-insensitive regex matching is used.

        Returns
        -------
        None
            Modifies the index in-place.

        Examples
        --------
        >>> # Search for pathways containing "metabolism"
        >>> pw_index.search("metabolism")
        >>>
        >>> # Search for pathways containing "glycolysis"
        >>> pw_index.search("glycolysis")
        >>>
        >>> # Search with regex pattern
        >>> pw_index.search("glucose.*pathway")
        >>>
        >>> # Case-insensitive search
        >>> pw_index.search("METABOLISM")  # Same as "metabolism"
        """

        pw_index = self.index
        # find matches to query
        fil = pw_index[SOURCE_SPEC.NAME].str.contains(
            query, regex=True, flags=re.IGNORECASE
        )
        pw_index = pw_index[fil]
        self.index = pw_index


def adapt_pw_index(
    source: str | PWIndex,
    organismal_species: str | Iterable[str] | None,
    outdir: str | None = None,
    update_index: bool = False,
) -> PWIndex:
    """Adapt a pathway index by filtering for specific organismal species.

    This function is helpful for filtering pathway indices for specific species
    before reconstructing models or performing other operations.

    Parameters
    ----------
    source : str or PWIndex
        URI for pw_index.csv file or PWIndex object to adapt
    organismal_species : str or Iterable[str] or None
        Organismal species to filter for. Should match the nomenclature
        of the pathway index. If None, no filtering is applied.
    outdir : str or None, optional
        Optional directory to write the filtered pw_index to.
        If provided and update_index is True, the filtered index will be
        saved as "pw_index.tsv" in this directory.
    update_index : bool, optional
        Whether to write the filtered pathway index to the output directory.
        Only used if outdir is provided. Default is False.

    Returns
    -------
    PWIndex
        Filtered pathway index containing only entries for the specified
        organismal species.

    Raises
    ------
    ValueError
        If source is neither a string nor a PWIndex object.

    Examples
    --------
    >>> # Filter pathway index for human species
    >>> filtered_index = adapt_pw_index("path/to/pw_index.csv", "human")
    >>>
    >>> # Filter and save to output directory
    >>> filtered_index = adapt_pw_index(
    ...     pw_index_obj,
    ...     ["human", "mouse"],
    ...     outdir="filtered_data",
    ...     update_index=True
    ... )
    """
    if isinstance(source, str):
        pw_index = PWIndex(source)
    elif isinstance(source, PWIndex):
        pw_index = copy.deepcopy(source)
    else:
        raise ValueError("'source' needs to be str or PWIndex")
    pw_index.filter(organismal_species=organismal_species)

    if outdir is not None and update_index:
        with open_fs(outdir, create=True) as fs:
            with fs.open(SOURCE_SPEC.PW_INDEX_FILE, "w") as f:
                pw_index.index.to_csv(f, sep="\t")

    return pw_index


def create_pathway_index_df(
    model_keys: dict[str, str],
    model_urls: dict[str, str],
    model_organismal_species: dict[str, str],
    base_path: str,
    data_source: str,
    model_names: Optional[dict[str, str]] = None,
    file_extension: str = ".sbml",
) -> pd.DataFrame:
    """Create a pathway index DataFrame from model definitions.

    This function creates a standardized pathway index DataFrame that can be used
    across different model sources. It handles file paths and metadata consistently,
    generating all required columns for a valid pathway index.

    Parameters
    ----------
    model_keys : dict[str, str]
        Mapping of species identifiers to model keys/IDs.
        Keys should be species identifiers, values are model keys.
    model_urls : dict[str, str]
        Mapping of species identifiers to model download URLs.
        Keys should match those in model_keys.
    model_organismal_species : dict[str, str]
        Mapping of species identifiers to full organismal species names.
        Keys should match those in model_keys.
    base_path : str
        Base directory path where model files will be stored.
    data_source : str
        Name of the source database (e.g., "BiGG", "Reactome").
    model_names : dict[str, str] or None, optional
        Optional mapping of model keys to display names.
        If None, uses model keys as display names.
    file_extension : str, optional
        File extension for model files. Default is ".sbml".

    Returns
    -------
    pd.DataFrame
        DataFrame containing pathway index information with columns:
        - pathway_id: Unique identifier for the pathway (from model_keys)
        - name: Display name for the pathway
        - source: Source database name
        - organismal_species: Organismal species name
        - file: Basename of the model file
        - url: URL to download the model from
        - sbml_path: Full path where model will be stored
        - date: Current date in YYYYMMDD format

    Raises
    ------
    TypeError
        If model_names is provided but not a dictionary.
    ValueError
        If model_names is provided but contains keys not present in model_keys.

    Examples
    --------
    >>> # Create a basic pathway index
    >>> model_keys = {"human": "HUMAN", "mouse": "MOUSE"}
    >>> model_urls = {
    ...     "human": "https://bigg.ucsd.edu/models/HUMAN",
    ...     "mouse": "https://bigg.ucsd.edu/models/MOUSE"
    ... }
    >>> model_species = {"human": "Homo sapiens", "mouse": "Mus musculus"}
    >>>
    >>> df = create_pathway_index_df(
    ...     model_keys=model_keys,
    ...     model_urls=model_urls,
    ...     model_organismal_species=model_species,
    ...     base_path="/path/to/models",
    ...     data_source="BiGG"
    ... )
    >>>
    >>> # Create with custom display names
    >>> model_names = {"HUMAN": "Human Metabolic Network", "MOUSE": "Mouse Metabolic Network"}
    >>> df = create_pathway_index_df(
    ...     model_keys=model_keys,
    ...     model_urls=model_urls,
    ...     model_organismal_species=model_species,
    ...     base_path="/path/to/models",
    ...     data_source="BiGG",
    ...     model_names=model_names
    ... )
    >>>
    >>> # Create with custom file extension
    >>> df = create_pathway_index_df(
    ...     model_keys=model_keys,
    ...     model_urls=model_urls,
    ...     model_organismal_species=model_species,
    ...     base_path="/path/to/models",
    ...     data_source="BiGG",
    ...     file_extension=".xml"
    ... )
    """
    models = {
        model_keys[species]: {
            SOURCE_SPEC.URL: model_urls[species],
            SOURCE_SPEC.ORGANISMAL_SPECIES: model_organismal_species[species],
        }
        for species in model_keys.keys()
    }

    models_df = pd.DataFrame(models).T
    models_df[SOURCE_SPEC.SBML_PATH] = [
        os.path.join(base_path, k) + file_extension for k in models_df.index.tolist()
    ]
    models_df[SOURCE_SPEC.FILE] = [
        os.path.basename(x) for x in models_df[SOURCE_SPEC.SBML_PATH]
    ]

    # add other attributes which will be used in the pw_index
    models_df[SOURCE_SPEC.DATE] = datetime.date.today().strftime("%Y%m%d")
    models_df.index = models_df.index.rename(SOURCE_SPEC.PATHWAY_ID)
    models_df = models_df.reset_index()

    if model_names is not None:
        if not isinstance(model_names, dict):
            raise TypeError(
                f"If provided, model_names must be a dict but was {type(model_names).__name__}"
            )

        defined_model_names = set(model_names.keys())
        undefined_model_names = (
            set(models_df[SOURCE_SPEC.PATHWAY_ID]) - defined_model_names
        )
        if len(undefined_model_names) != 0:
            raise ValueError(
                f"The following model names were not defined in 'model_names': {', '.join(undefined_model_names)}"
            )

        models_df[SOURCE_SPEC.NAME] = models_df[SOURCE_SPEC.PATHWAY_ID].map(model_names)
    else:
        models_df[SOURCE_SPEC.NAME] = models_df[SOURCE_SPEC.PATHWAY_ID]

    models_df = models_df.assign(**{SOURCE_SPEC.DATA_SOURCE: data_source})

    return models_df
