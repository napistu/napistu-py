from __future__ import annotations

import copy
import os
import re
from os import PathLike
from typing import Iterable

from fs import open_fs
import pandas as pd

from napistu.utils import path_exists
from napistu.constants import EXPECTED_PW_INDEX_COLUMNS
from napistu.constants import SOURCE_SPEC


class PWIndex:
    """
    Pathway Index

    Organizing metadata (and optionally paths) of individual pathway representations

    Attributes
    ----------
    index : pd.DataFrame
        A table describing the location and contents of pathway files.
    base_path: str
        Path to directory of indexed files

    Methods
    -------
    filter(sources, species)
        Filter index based on pathway source an/or category
    search(query)
        Filter index to pathways matching the search query
    """

    def __init__(
        self,
        pw_index: PathLike[str] | str | pd.DataFrame,
        pw_index_base_path=None,
        validate_paths=True,
    ) -> None:
        """
        Tracks pathway file locations and contents.

        Parameters
        ----------
        pw_index : str or None
            Path to index file or a pd.DataFrame containing the contents of PWIndex.index
        pw_index_base_path : str or None
            A Path that relative paths in pw_index will reference
        validate_paths : bool
            If True then paths constructed from base_path + file will be tested for existence.
            If False then paths will not be validated and base_path attribute will be set to None

        Returns
        -------
        None
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
        """Verifies that all files in the pwindex are present

        Raises:
            FileNotFoundError: Error if a file not present
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
        sources: str | Iterable[str] | None = None,
        species: str | Iterable[str] | None = None,
    ):
        """
        Filter Pathway Index

        Args:
            sources (str | Iterable[str] | None, optional): A list of valid sources or None for all
            species (str | Iterable[str] | None, optional): A list of valid species or None all all
        """
        pw_index = self.index
        if sources is not None:
            pw_index = pw_index.query("source in @sources")

        if species is not None:
            pw_index = pw_index.query("species in @species")

        self.index = pw_index

    def search(self, query):
        """
        Search Pathway Index

        Parameters:
        query: str
            Filter to rows of interest based on case-insensitive match to names.

        Returns:
        None
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
    species: str | Iterable[str] | None,
    outdir: str | None = None,
) -> PWIndex:
    """Adapts a pw_index

    Helpful to filter for species before reconstructing.

    Args:
        source (str | PWIndex): uri for pw_index.csv file or PWIndex object
        species (str):
        outdir (str | None, optional): Optional directory to write pw_index to.
            Defaults to None.

    Returns:
        PWIndex: Filtered pw index
    """
    if isinstance(source, str):
        pw_index = PWIndex(source)
    elif isinstance(source, PWIndex):
        pw_index = copy.deepcopy(source)
    else:
        raise ValueError("'source' needs to be str or PWIndex")
    pw_index.filter(species=species)

    if outdir is not None:
        with open_fs(outdir, create=True) as fs:
            with fs.open("pw_index.tsv", "w") as f:
                pw_index.index.to_csv(f, sep="\t")
    return pw_index
