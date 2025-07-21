from __future__ import annotations

import gzip
import io
import json
import logging
import os
import pickle
import re
import requests
import shutil
import urllib.request as request
import zipfile
from contextlib import closing
from itertools import starmap
from textwrap import fill
from typing import Any, Dict, Optional, List, Union
from urllib.parse import urlparse
from pathlib import Path
from requests.adapters import HTTPAdapter
from requests.adapters import Retry

import igraph as ig
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fs import open_fs
from fs.copy import copy_dir
from fs.copy import copy_file
from fs.copy import copy_fs
from fs.errors import CreateFailed
from fs.errors import ResourceNotFound
from fs.tarfs import TarFS
from fs.tempfs import TempFS
from fs.zipfs import ZipFS

from napistu.constants import FILE_EXT_GZ
from napistu.constants import FILE_EXT_ZIP

logger = logging.getLogger(__name__)


def initialize_dir(output_dir_path: str, overwrite: bool):
    """Initializes a filesystem directory

    Args:
        output_dir_path (str): path to new directory
        overwrite (bool): overwrite? if true, directory will be
            deleted and recreated

    Raises:
        FileExistsError
    """
    output_dir_path = str(output_dir_path)
    try:
        with open_fs(output_dir_path) as out_fs:
            if overwrite:
                out_fs.removetree("/")
            else:
                raise FileExistsError(
                    f"{output_dir_path} already exists and overwrite is False"
                )
    except CreateFailed:
        # If gcs bucket did not exist yet, create it
        with open_fs(output_dir_path, create=True):
            pass


def download_and_extract(
    url: str,
    output_dir_path: str = ".",
    download_method: str = "wget",
    overwrite: bool = False,
) -> None:
    """
    Download and Unpack

    Download an archive and then extract to a new folder

    Args:
        url (str): Url of archive.
        output_dir_path (str): Path to output directory.
        overwrite (bool): Overwrite an existing output directory.


    Returns:
        None
    """

    # initialize output directory
    output_dir_path = str(output_dir_path)
    initialize_dir(output_dir_path, overwrite)

    out_fs = open_fs(output_dir_path)
    extn = get_extn_from_url(url)

    # download archive file
    tmp_fs = TempFS()
    tmp_file = os.path.join(tmp_fs.root_path, f"cpr_tmp{extn}")

    if download_method == "wget":
        download_wget(url, tmp_file)
    elif download_method == "ftp":
        download_ftp(url, tmp_file)
    else:
        raise ValueError("undefined download_method, defined methods are wget and ftp")

    if re.search(".tar\\.gz$", extn) or re.search("\\.tgz$", extn):
        # untar .tar.gz into individual files
        with TarFS(tmp_file) as tar_fs:
            copy_fs(tar_fs, out_fs)
            logger.info(f"Archive downloaded and untared to {output_dir_path}")
    elif re.search("\\.zip$", extn):
        with ZipFS(tmp_file) as zip_fs:
            copy_fs(zip_fs, out_fs)
            logger.info(f"Archive downloaded and unzipped to {output_dir_path}")
    elif re.search("\\.gz$", extn):
        outfile = url.split("/")[-1].replace(".gz", "")
        # gunzip file
        with gzip.open(tmp_file, "rb") as f_in:
            with out_fs.open(outfile, "wb") as f_out:
                f_out.write(f_in.read())
    else:
        raise ValueError(f"{extn} is not supported")

    # Close fs
    tmp_fs.close()
    out_fs.close()

    return None


def extract(file: str):
    """
    Download and Unpack

    Untar, unzip and ungzip

    Args:
        file (str): Path to compressed file

    Returns:
        None
    """

    extn = get_extn_from_url(file)
    if re.search(".tar\\.gz$", extn) or re.search("\\.tgz$", extn):
        output_dir_path = os.path.join(
            os.path.join(
                os.path.dirname(file), os.path.basename(file).replace(extn, "")
            )
        )
    else:
        output_dir_path = os.path.dirname(file)

    try:
        initialize_dir(output_dir_path, overwrite=False)
    except FileExistsError:
        pass

    out_fs = open_fs(output_dir_path)

    if re.search(".tar\\.gz$", extn) or re.search("\\.tgz$", extn):
        # untar .tar.gz into individual files
        with TarFS(file) as tar_fs:
            copy_fs(tar_fs, out_fs)
            logger.info(f"Archive downloaded and untared to {output_dir_path}")
    elif re.search("\\.zip$", extn):
        with ZipFS(file) as zip_fs:
            copy_fs(zip_fs, out_fs)
            logger.info(f"Archive downloaded and unzipped to {output_dir_path}")
    elif re.search("\\.gz$", extn):
        outfile = file.split("/")[-1].replace(".gz", "")
        # gunzip file
        with gzip.open(file, "rb") as f_in:
            with out_fs.open(outfile, "wb") as f_out:
                f_out.write(f_in.read())
    else:
        raise ValueError(f"{extn} is not supported")

    # Close fs
    out_fs.close()

    return None


def gunzip(gzipped_path: str, outpath: str | None = None) -> None:
    """Gunzip a file to an output path."""

    if not os.path.exists(gzipped_path):
        raise FileNotFoundError(f"{gzipped_path} not found")

    if not re.search("\\.gz$", gzipped_path):
        logger.warning("{gzipped_path} does not have the .gz extension")

    if outpath is None:
        # determine outfile name automatically if not provided
        outpath = os.path.join(
            os.path.dirname(gzipped_path),
            gzipped_path.split("/")[-1].replace(".gz", ""),
        )
    outfile = os.path.basename(outpath)

    out_fs = open_fs(os.path.dirname(outpath))
    # gunzip file
    with gzip.open(gzipped_path, "rb") as f_in:
        with out_fs.open(outfile, "wb") as f_out:
            f_out.write(f_in.read())
    out_fs.close()

    return None


def get_extn_from_url(url: str) -> str:
    """Retrieves file extension from an URL

    Args:
        url (str): url

    Raises:
        ValueError: Raised when no extension identified

    Returns:
        str: the identified extension

    Examples:
    >>> get_extn_from_url('https://test/test.gz')
    '.gz'
    >>> get_extn_from_url('https://test/test.tar.gz')
    '.tar.gz'
    >>> get_extn_from_url('https://test/test.tar.gz/bla')
    Traceback (most recent call last):
    ...
    ValueError: File extension not identifiable: https://test/test.tar.gz/bla
    """
    match = re.search("\\..+$", os.path.split(url)[1])
    if match is None:
        raise ValueError(f"File extension not identifiable: {url}")
    else:
        extn = match.group(0)
    return extn


def write_file_contents_to_path(path: str, contents) -> None:
    """Helper function to write file contents to the path.

    Args:
        path (str): destination
        contents (Any): file contents

    Returns:
        None
    """
    if hasattr(path, "write") and hasattr(path, "__iter__"):
        path.write(contents)  # type: ignore
    else:
        base, filename = get_target_base_and_path(path)
        with open_fs(base, create=True) as fs:
            with fs.open(filename, "wb") as f:
                f.write(contents)  # type: ignore

    return None


def download_wget(
    url: str,
    path,
    target_filename: str = None,
    verify: bool = True,
    timeout: int = 30,
    max_retries: int = 3,
) -> None:
    """Downloads file / archive with wget

    Args:
        url (str): url
        path (FilePath | WriteBuffer): file path or buffer
        target_filename (str): specific file to extract from ZIP if URL is a ZIP file
        verify (bool): verify argument to pass to requests.get
        timeout (int): timeout in seconds for the request
        max_retries (int): number of times to retry the download if it fails

    Returns:
        None
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        r = session.get(url, allow_redirects=True, verify=verify, timeout=timeout)
        r.raise_for_status()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        logger.error(f"Failed to download {url} after {max_retries} retries: {str(e)}")
        raise

    # check if the content is a ZIP file
    if (
        r.headers.get("Content-Type") == "application/zip"
        or url.endswith(f".{FILE_EXT_ZIP}")
    ) and target_filename:
        # load the ZIP file in memory
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # check if the target file exists in the ZIP archive
            if target_filename in z.namelist():
                with z.open(target_filename) as target_file:
                    # apply the same logic as below to the target file
                    return write_file_contents_to_path(path, target_file.read())
            else:
                raise FileNotFoundError(
                    f"{target_filename} not found in the ZIP archive"
                )
    # check if the content is a GZIP (single-file compression)
    elif url.endswith(f".{FILE_EXT_GZ}"):
        with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
            return write_file_contents_to_path(path, gz.read())
    else:
        # not an archive -> default case -> write file directly
        return write_file_contents_to_path(path, r.content)


def download_ftp(url, path):
    with closing(request.urlopen(url)) as r:
        with open(path, "wb") as f:
            shutil.copyfileobj(r, f)

    return None


def requests_retry_session(
    retries=5,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 503, 504),
    session: requests.Session | None = None,
    **kwargs,
) -> requests.Session:
    """Requests session with retry logic

    This should help to combat flaky apis, eg Brenda.
    From: https://stackoverflow.com/a/58687549

    Args:
        retries (int, optional): Number of retries. Defaults to 5.
        backoff_factor (float, optional): backoff. Defaults to 0.3.
        status_forcelist (tuple, optional): errors to retry. Defaults to (500, 502, 503, 504).
        session (Optional[requests.Session], optional): existing session. Defaults to None.

    Returns:
        requests.Session: new requests session
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        **kwargs,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def pickle_cache(path: str, overwrite: bool = False):
    """A decorator to cache a function call result to pickle

    Attention: this does not care about the function arguments
    All function calls will be served by the same pickle file.

    Args:
        path (str): path to the cache pickle file
        overwrite (bool): should an existing cache be overwritten even
          if it exists?

    Returns:
        A function whos output will be cached to pickle.
    """

    if overwrite:
        if path_exists(path):
            if not os.path.isfile(path):
                logger.warning(
                    f"{path} is a GCS URI and cannot be deleted using overwrite = True"
                )
            else:
                logger.info(
                    f"Deleting {path} because file exists and overwrite is True"
                )
                os.remove(path)

    def decorator(fkt):
        def wrapper(*args, **kwargs):
            if path_exists(path):
                logger.info(
                    "Not running function %s but using cache file '%s' instead.",
                    fkt.__name__,
                    path,
                )
                dat = load_pickle(path)
            else:
                dat = fkt(*args, **kwargs)
                save_pickle(path, dat)
            return dat

        return wrapper

    return decorator


def path_exists(path: str) -> bool:
    """Checks if path or uri exists

    Args:
        path (str): path/uri

    Returns:
        bool: exists?
    """
    dir, file = os.path.split(path)
    try:
        with open_fs(dir) as f:
            return f.exists(file)
    except CreateFailed:
        # If the path is on gcfs,
        # it could be that the parent
        # does not exist, but the path does
        pass

    # If the path is a directory
    # it is enough that it itself
    # exists
    try:
        with open_fs(path) as f:
            return True
    except CreateFailed:
        return False


def save_pickle(path: str, dat: object):
    """Saves object to path as pickle

    Args:
        path (str): target path
        dat (object): object
    """
    dir, file = get_target_base_and_path(path)
    with open_fs(dir, create=True) as f:
        with f.open(file, "wb") as f:
            pickle.dump(dat, f)


def load_pickle(path: str):
    """Loads pickle object to path

    Args:
        path (str): path to pickle

    Returns:
        Any: Object
    """
    dir, file = get_source_base_and_path(path)
    with open_fs(dir) as source_fs:
        try:
            with source_fs.open(file, "rb") as f:
                return pickle.load(f)
        except ResourceNotFound as e:
            if hasattr(source_fs, "fix_storage"):
                logger.info(
                    "File could not be opened. Trying to fix storage for FS-GCFS. "
                    "This is required because of: https://fs-gcsfs.readthedocs.io/en/latest/#limitations "
                    "and will add empty blobs to indicate directories."
                )
                source_fs.fix_storage()
            else:
                raise e


read_pickle = load_pickle
write_pickle = save_pickle


def get_source_base_and_path(uri: str) -> tuple[str, str]:
    """Get the base of a bucket or folder and the path to the file

    Args:
        uri (str): uri

    Returns:
        tuple[str, str]: base: the base folder of the bucket

    Example:
    >>> get_source_base_and_path("gs://bucket/folder/file")
    ('gs://bucket', 'folder/file')
    >>> get_source_base_and_path("/bucket/folder/file")
    ('/bucket/folder', 'file')
    """
    uri = str(uri)
    urlelements = urlparse(uri)
    if len(urlelements.scheme) > 0:
        base = urlelements.scheme + "://" + urlelements.netloc
        path = urlelements.path[1:]
    else:
        base, path = os.path.split(uri)
    return base, path


def get_target_base_and_path(uri):
    """Get the base of a bucket + directory and the file

    Args:
        uri (str): uri

    Returns:
        tuple[str, str]: base: the base folder + path of the bucket
            file: the file

    Example:
    >>> get_target_base_and_path("gs://bucket/folder/file")
    ('gs://bucket/folder', 'file')
    >>> get_target_base_and_path("bucket/folder/file")
    ('bucket/folder', 'file')
    >>> get_target_base_and_path("/bucket/folder/file")
    ('/bucket/folder', 'file')
    """
    base, path = os.path.split(uri)
    return base, path


def copy_uri(input_uri: str, output_uri: str, is_file=True):
    """Copy a file or folder from one uri to another

    Args:
        input_uri (str): input file uri (gcs, http, ...)
        output_uri (str): path to output file (gcs, local)
        is_file (bool, optional): Is this a file or folder?. Defaults to True.
    """
    logger.info("Copy uri from %s to %s", input_uri, output_uri)
    source_base, source_path = get_source_base_and_path(input_uri)
    target_base, target_path = get_target_base_and_path(output_uri)
    if is_file:
        copy_fun = copy_file
    else:
        copy_fun = copy_dir
    with open_fs(source_base) as source_fs:
        with open_fs(target_base, create=True) as target_fs:
            try:
                copy_fun(source_fs, source_path, target_fs, target_path)
            except ResourceNotFound as e:
                if hasattr(source_fs, "fix_storage"):
                    logger.info(
                        "File could not be opened. Trying to fix storage for FS-GCFS. "
                        "This is required because of: https://fs-gcsfs.readthedocs.io/en/latest/#limitations "
                        "and will add empty blobs to indicate directories."
                    )
                    source_fs.fix_storage()
                    copy_fun(source_fs, source_path, target_fs, target_path)
                else:
                    raise (e)


def save_json(uri: str, object: Any) -> None:
    """Write object to json file at uri

    Args:
        object (Any): object to write
        uri (str): path to json file
    """
    target_base, target_path = get_target_base_and_path(uri)
    with open_fs(target_base, create=True) as target_fs:
        target_fs.writetext(target_path, json.dumps(object))


def load_json(uri: str) -> Any:
    """Read json from uri

    Args:
        uri (str): path to json file
    """
    logger.info("Read json from %s", uri)
    source_base, source_path = get_source_base_and_path(uri)
    with open_fs(source_base) as source_fs:
        try:
            txt = source_fs.readtext(source_path)
        except ResourceNotFound as e:
            if hasattr(source_fs, "fix_storage"):
                logger.info(
                    "File could not be opened. Trying to fix storage for FS-GCFS. "
                    "This is required because of: https://fs-gcsfs.readthedocs.io/en/latest/#limitations "
                    "and will add empty blobs to indicate directories."
                )
                source_fs.fix_storage()
                txt = source_fs.readtext(source_path)
            else:
                raise (e)
        return json.loads(txt)


def save_parquet(
    df: pd.DataFrame, uri: Union[str, Path], compression: str = "snappy"
) -> None:
    """
    Write a DataFrame to a single Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save
    uri : Union[str, Path]
        Path where to save the Parquet file. Can be a local path or a GCS URI.
        Recommended extensions: .parquet or .pq
    compression : str, default 'snappy'
        Compression algorithm. Options: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'

    Raises
    ------
    OSError
        If the file cannot be written to (permission issues, etc.)
    """

    uri_str = str(uri)

    # Warn about non-standard extensions
    if not any(uri_str.endswith(ext) for ext in [".parquet", ".pq"]):
        logger.warning(
            f"File '{uri_str}' doesn't have a standard Parquet extension (.parquet or .pq)"
        )

    target_base, target_path = get_target_base_and_path(uri_str)

    with open_fs(target_base, create=True) as target_fs:
        with target_fs.openbin(target_path, "w") as f:
            # Convert to Arrow table and write as single file
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                f,
                compression=compression,
                use_dictionary=True,  # Efficient for repeated values
                write_statistics=True,  # Enables query optimization
            )


def load_parquet(uri: Union[str, Path]) -> pd.DataFrame:
    """
    Read a DataFrame from a Parquet file.

    Parameters
    ----------
    uri : Union[str, Path]
        Path to the Parquet file to load

    Returns
    -------
    pd.DataFrame
        The DataFrame loaded from the Parquet file

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    """
    try:
        target_base, target_path = get_target_base_and_path(str(uri))

        with open_fs(target_base) as target_fs:
            with target_fs.openbin(target_path, "r") as f:
                return pd.read_parquet(f, engine="pyarrow")

    except ResourceNotFound as e:
        raise FileNotFoundError(f"File not found: {uri}") from e


def extract_regex_search(regex: str, query: str, index_value: int = 0) -> str:
    """
    Match an identifier substring and otherwise throw an error

    Args:
        regex (str): regular expression to search
        query (str): string to search against
        index_value (int): entry in index to return

    return:
        match (str): a character string match

    """

    if m := re.search(regex, query):
        match = m[index_value]
    else:
        raise ValueError(
            f"{query} does not match the identifier regular expression: {regex}"
        )

    return match


def extract_regex_match(regex: str, query: str) -> str:
    """
    Args:
        regex (str): regular expression to search
        query (str): string to search against

    return:
        match (str): a character string match
    """

    if m := re.match(regex, query):
        if len(m.groups()) > 0:
            match = m.groups()[0]
        else:
            raise ValueError(
                f"{query} does not match a subgroup in the regular expression: {regex}"
            )
    else:
        raise ValueError(f"{query} does not match the regular expression: {regex}")

    return match


class match_pd_vars:
    """
    Match Pandas Variables.

    Attributes
    ----------
    req_vars:
        A set of variables which should exist in df
    missing_vars:
        Required variables which are not present in df
    extra_vars:
        Non-required variables which are present in df
    are_present:
        Returns True if req_vars are present and False otherwise

    Methods
    -------
    assert_present()
        Raise an exception of req_vars are absent

    """

    def __init__(
        self, df: pd.DataFrame | pd.Series, req_vars: set, allow_series: bool = True
    ) -> None:
        """
        Connects to an SBML file

        Parameters
        ----------
        df
            A pd.DataFrame or pd.Series
        req_vars
            A set of variables which should exist in df
        allow_series:
            Can a pd.Series be provided as df?

        Returns
        -------
        None.
        """

        if isinstance(df, pd.Series):
            if not allow_series:
                raise TypeError("df was a pd.Series and must be a pd.DataFrame")
            vars_present = set(df.index.tolist())
        elif isinstance(df, pd.DataFrame):
            vars_present = set(df.columns.tolist())
        else:
            raise TypeError(
                f"df was a {type(df).__name__} and must be a pd.DataFrame or pd.Series"
            )

        self.req_vars = req_vars
        self.missing_vars = req_vars.difference(vars_present)
        self.extra_vars = vars_present.difference(req_vars)

        if len(self.missing_vars) == 0:
            self.are_present = True
        else:
            self.are_present = False

    def assert_present(self) -> None:
        """
        Raise an error if required variables are missing
        """

        if not self.are_present:
            raise ValueError(
                f"{len(self.missing_vars)} required variables were "
                "missing from the provided pd.DataFrame or pd.Series: "
                f"{', '.join(self.missing_vars)}"
            )

        return None


def ensure_pd_df(pd_df_or_series: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Ensure Pandas DataFrame

    Convert a pd.Series to a DataFrame if needed.

    Args:
        pd_df_or_series (pd.Series | pd.DataFrame):
            a pandas df or series

    Returns:
        pd_df converted to a pd.DataFrame if needed

    """

    if isinstance(pd_df_or_series, pd.DataFrame):
        return pd_df_or_series
    elif isinstance(pd_df_or_series, pd.Series):
        return pd_df_or_series.to_frame().T
    else:
        raise TypeError(
            "ensure_pd_df expects either a pandas DataFrame or Series but received"
            f" a {type(pd_df_or_series)}"
        )


def drop_extra_cols(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    always_include: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Remove columns in df_out that are not in df_in, except those specified in always_include.

    Parameters
    ----------
    df_in : pd.DataFrame
        Reference DataFrame whose columns determine what to keep
    df_out : pd.DataFrame
        DataFrame to filter columns from
    always_include : Optional[List[str]], optional
        List of column names to always include in output, even if not in df_in

    Returns
    -------
    pd.DataFrame
        DataFrame with columns filtered to match df_in plus any always_include columns.
        Column order follows df_in, with always_include columns appended at the end.

    Examples
    --------
    >>> df_in = pd.DataFrame({'a': [1], 'b': [2]})
    >>> df_out = pd.DataFrame({'a': [3], 'c': [4], 'd': [5]})
    >>> _drop_extra_cols(df_in, df_out)
    # Returns DataFrame with just column 'a'

    >>> _drop_extra_cols(df_in, df_out, always_include=['d'])
    # Returns DataFrame with columns ['a', 'd']
    """
    # Handle None case for always_include
    if always_include is None:
        always_include = []

    # Get columns to retain: intersection with df_in plus always_include
    retained_cols = df_in.columns.intersection(df_out.columns).union(always_include)

    # Filter to only columns that exist in df_out
    retained_cols = retained_cols.intersection(df_out.columns)

    # Order columns: first those matching df_in's order, then any remaining always_include
    ordered_cols = []
    # Add columns that are in df_in in their original order
    for col in df_in.columns:
        if col in retained_cols:
            ordered_cols.append(col)
    # Add any remaining always_include columns that weren't in df_in
    for col in always_include:
        if col in retained_cols and col not in ordered_cols:
            ordered_cols.append(col)

    return df_out.loc[:, ordered_cols]


def update_pathological_names(names: pd.Series, prefix: str) -> pd.Series:
    """
    Update pathological names in a pandas Series.

    Add a prefix to the names if they are all numeric.
    """
    if names.apply(lambda x: x.isdigit()).all():
        names = names.apply(lambda x: f"{prefix}{x}")
    return names


def format_identifiers_as_edgelist(
    df: pd.DataFrame, defining_vars: list[str]
) -> pd.DataFrame:
    """
    Format Identifiers as Edgelist

    Collapse a multiindex to an index (if needed), and similarly collapse multiple variables to a single entry.
    This indexed pd.Sereies of index - ids can be treated as an edgelist for greedy clustering.

    Args:
        df (pd.DataFrame):
            Any pd.DataFrame
        defining_vars (list(str)):
            A set of attributes which define a distinct entry in df

    Returns:
        df (pd.DataFrame):
            A pd.DataFrame with an "ind" and "id" variable added indicating rolled up
            values of the index and defining_vars
    """

    # requires a named index by convention
    if None in df.index.names:
        raise ValueError(
            "df did not have a named index. A named index or multindex is expected"
        )

    if not isinstance(defining_vars, list):
        raise TypeError("defining_vars must be a list")

    logger.info(
        f"creating an edgelist linking index levels {', '.join(df.index.names)} and linking it "
        f"to levels defined by {', '.join(defining_vars)}"
    )

    # df is a pd.DataFrame and contains defining_vars
    match_pd_vars(df, req_vars=set(defining_vars), allow_series=False).assert_present()

    # combine all components of a multindex into a single index value
    if df.index.nlevels == 1:
        df.loc[:, "ind"] = ["ind_" + x for x in df.index]
    else:
        # handle a multiindex
        fstr = "ind_" + "_".join(["{}"] * df.index.nlevels)
        df.loc[:, "ind"] = list(starmap(fstr.format, df.index))

    # aggregate defining variables
    df.loc[:, "id"] = df[defining_vars].apply(
        lambda x: "id_" + "_".join(x.dropna().astype(str)), axis=1
    )

    return df


def matrix_to_edgelist(matrix, row_labels=None, col_labels=None):
    rows, cols = np.where(~np.isnan(matrix))

    edgelist = pd.DataFrame(
        {
            "row": rows if row_labels is None else [row_labels[i] for i in rows],
            "column": cols if col_labels is None else [col_labels[i] for i in cols],
            "value": matrix[rows, cols],
        }
    )

    return edgelist


def find_weakly_connected_subgraphs(edgelist: pd.DataFrame) -> pd.DataFrame:
    """Find all cliques of loosly connected components."""

    if edgelist.shape[1] != 2:
        raise ValueError("edgelist must have exactly 2 columns")
    if edgelist.columns.tolist() != ["ind", "id"]:
        raise ValueError("edgelist columns must be ['ind', 'id']")
    if not any(edgelist["ind"].str.startswith("ind")):
        raise ValueError("At least some entries in 'ind' must start with 'ind'")

    id_graph = ig.Graph.TupleList(edgelist.itertuples(index=False))

    id_graph_names = [v.attributes()["name"] for v in id_graph.vs]
    id_graphs_clusters = id_graph.connected_components().membership
    id_graph_df = pd.DataFrame({"name": id_graph_names, "cluster": id_graphs_clusters})
    # clusters based on index or identifiers will be the same when joined to id table
    ind_clusters = id_graph_df[id_graph_df.name.str.startswith("ind")].rename(
        columns={"name": "ind"}
    )

    return ind_clusters


def style_df(
    df: pd.DataFrame,
    headers: Union[str, list[str], None] = "keys",
    hide_index: bool = False,
) -> pd.io.formats.style.Styler:
    """
    Style DataFrame

    Provide some simple options for styling a pd.DataFrame

    Args:
        df: pd.DataFrame
            A table to style
        headers:
            - "keys" to use the current column names
            - None to suppress column names
            - list[str] to overwrite and show column names
        hide_index: bool
            Should rows be displayed?

    Returns:
        styled_df: pd.io.formats.style.Styler
            `df` with styles updated
    """

    if isinstance(headers, list):
        if len(headers) != df.shape[1]:
            raise ValueError(
                f"headers was a list with {len(headers)} entries, but df has {df.shape[1]} "
                "columns. These dimensions should match"
            )

        df.columns = headers  # type: ignore

    styled_df = df.style.format(precision=3).set_table_styles(
        [{"selector": "th", "props": "color: limegreen;"}]
    )

    if hide_index:
        styled_df = styled_df.hide(axis="index")

    if headers is None:
        return styled_df.hide(axis="columns")
    elif isinstance(headers, str):
        if headers == "keys":
            # just plot with the index as headers
            return styled_df
        else:
            raise ValueError(
                f"headers was a string: {headers} but this option is not recognized. "
                'The only defined value is "keys".'
            )
    else:
        assert isinstance(headers, list)
        return styled_df


def safe_series_tolist(x):
    """Convert either a list or str to a list."""

    if isinstance(x, str):
        return [x]
    elif isinstance(x, pd.Series):
        return x.tolist()
    else:
        raise TypeError(f"x was a {type(x)} but only str and pd.Series are supported")


def check_unique_index(df, label=""):
    """Validate that each index value only maps to a single row."""

    if len(df.index) != len(df.index.unique()):
        raise ValueError(f"{label} index entries are not unique")

    return None


def score_nameness(string: str):
    """
    Score Nameness

    This utility assigns a numeric score to a string reflecting how likely it is to be
    a human readable name. This will help to prioritize readable entries when we are
    trying to pick out a single name to display from a set of values which may also
    include entries like systematic ids.

    Args:
        string (str):
            An alphanumeric string

    Returns:
        score (int):
            An integer score indicating how name-like the string is (low is more name-like)
    """

    return (
        # string length
        string.__len__()
        # no-space penalty
        + (sum(c.isspace() for c in string) == 0) * 10
        # penalty for each number
        + sum(c.isdigit() for c in string) * 5
    )


def click_str_to_list(string: str) -> list[str]:
    """Convert a string-based representation of a list inputted from the CLI into a list of strings."""

    var_extract_regex = re.compile("\\'?([a-zA-Z_]+)\\'?")

    re_search = re.search("^\\[(.*)\\]$", string)
    if re_search:
        return var_extract_regex.findall(re_search.group(0))
    else:
        raise ValueError(
            f"The provided string, {string}, could not be reformatted as a list. An example string which can be formatted is: \"['weight', 'upstream_weight']\""
        )


def safe_fill(x: str, fill_width: int = 15) -> str:
    """
    Safely wrap a string to a specified width.

    Parameters
    ----------
    x : str
        The string to wrap.
    fill_width : int, optional
        The width to wrap the string to. Default is 15.

    Returns
    -------
    str
        The wrapped string.
    """

    # TODO - move to non-network utils

    if x == "":
        return ""
    else:
        return fill(x, fill_width)


def match_regex_dict(s: str, regex_dict: Dict[str, any]) -> Optional[any]:
    """
    Apply each regex in regex_dict to the string s. If a regex matches, return its value.
    If no regex matches, return None.

    Parameters
    ----------
    s : str
        The string to test.
    regex_dict : dict
        Dictionary where keys are regex patterns (str), and values are the values to return.

    Returns
    -------
    The value associated with the first matching regex, or None if no match.
    """
    for pattern, value in regex_dict.items():
        if re.search(pattern, s):
            return value
    return None


def _add_nameness_score_wrapper(df, name_var, table_schema):
    """Call _add_nameness_score with default value."""

    if name_var in table_schema.keys():
        return _add_nameness_score(df, table_schema[name_var])
    else:
        logger.debug(
            f"{name_var} is not defined in table_schema; adding a constant (1)"
        )
        return df.assign(nameness_score=1)


def _add_nameness_score(df, name_var):
    """Add a nameness_score variable which reflects how name-like each entry is."""

    df.loc[:, "nameness_score"] = df[name_var].apply(score_nameness)
    return df


def _merge_and_log_overwrites(
    left_df: pd.DataFrame, right_df: pd.DataFrame, merge_context: str, **merge_kwargs
) -> pd.DataFrame:
    """
    Merge two DataFrames and log any column overwrites.

    Parameters
    ----------
    left_df : pd.DataFrame
        Left DataFrame for merge
    right_df : pd.DataFrame
        Right DataFrame for merge
    merge_context : str
        Description of the merge operation for logging
    **merge_kwargs : dict
        Additional keyword arguments passed to pd.merge

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with overwritten columns removed
    """
    # Track original columns
    original_cols = left_df.columns.tolist()

    # Ensure we're using the correct suffixes
    merge_kwargs["suffixes"] = ("_old", "")

    # Perform merge
    merged_df = pd.merge(left_df, right_df, **merge_kwargs)

    # Check for and log any overwritten columns
    new_cols = merged_df.columns.tolist()
    overwritten_cols = [col for col in original_cols if col + "_old" in new_cols]
    if overwritten_cols:
        logger.warning(
            f"The following columns were overwritten during {merge_context} merge and their original values "
            f"have been suffixed with '_old': {', '.join(overwritten_cols)}"
        )
        # Drop the old columns
        cols_to_drop = [col + "_old" for col in overwritten_cols]
        merged_df = merged_df.drop(columns=cols_to_drop)

    return merged_df
