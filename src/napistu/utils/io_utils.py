"""
Utilities for input and output operations.

Public Functions
----------------
download_and_extract(url: str, output_dir_path: str = ".", download_method: str = DOWNLOAD_METHODS.WGET, overwrite: bool = False) -> None:
    Download an archive and then extract to a new folder.
download_ftp(url: str, path: str) -> None:
    Download a file from an FTP server.
download_wget(url: str, path: str, target_filename: str = None, verify: bool = True, timeout: int = 30, max_retries: int = 3) -> None:
    Download a file / archive with wget.
extract(file: str) -> None:
    Untar, unzip and ungzip compressed files.
gunzip(gzipped_path: str, outpath: str | None = None) -> None:
    Gunzip a file to an output path.
load_json(uri: str) -> Any:
    Read JSON from URI.
load_parquet(uri: Union[str, Path]) -> pd.DataFrame:
    Read a DataFrame from a Parquet file.
load_pickle(path: str) -> Any:
    Load pickle object from path.
pickle_cache(path: str, overwrite: bool = False) -> Callable:
    Decorator to cache a function call result to pickle.
read_pickle(path: str) -> Any:
    Alias for load_pickle.
requests_retry_session(retries: int = 5, backoff_factor: float = 0.3, status_forcelist: tuple = (500, 502, 503, 504), session: requests.Session | None = None, **kwargs) -> requests.Session:
    Create a requests session with retry logic.
save_json(uri: str, object: Any) -> None:
    Write object to JSON file at URI.
save_parquet(df: pd.DataFrame, uri: Union[str, Path], compression: str = "snappy") -> None:
    Write a DataFrame to a single Parquet file.
save_pickle(path: str, dat: object) -> None:
    Save object to path as pickle.
write_file_contents_to_path(path: str, contents: Any) -> None:
    Helper function to write file contents to a path.
write_pickle(path: str, dat: object) -> None:
    Alias for save_pickle.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import pickle
import re
import shutil
import urllib.request as request
import warnings
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter, Retry

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs
    from fs.copy import copy_fs
    from fs.errors import ResourceNotFound
    from fs.tarfs import TarFS
    from fs.tempfs import TempFS
    from fs.zipfs import ZipFS

from napistu.constants import FILE_EXT_GZ, FILE_EXT_ZIP
from napistu.utils.constants import DOWNLOAD_METHODS, VALID_DOWNLOAD_METHODS

# Import helper functions from path_utils
from napistu.utils.path_utils import (
    get_extn_from_url,
    get_source_base_and_path,
    get_target_base_and_path,
    initialize_dir,
    path_exists,
)

logger = logging.getLogger(__name__)


def download_and_extract(
    url: str,
    output_dir_path: str = ".",
    download_method: str = DOWNLOAD_METHODS.WGET,
    overwrite: bool = False,
) -> None:
    """
    Download and Unpack

    Download an archive and then extract to a new folder

    Parameters
    ----------
    url : str
        Url of archive.
    output_dir_path : str
        Path to output directory.
    download_method : str
        Method to use to download the archive.
    overwrite : bool
        Overwrite an existing output directory.

    Returns
    -------
    None
        Files are downloaded and extracted to the specified directory
    """

    # initialize output directory
    output_dir_path = str(output_dir_path)
    initialize_dir(output_dir_path, overwrite)

    out_fs = open_fs(output_dir_path)
    extn = get_extn_from_url(url)

    # download archive file
    tmp_fs = TempFS()
    tmp_file = os.path.join(tmp_fs.root_path, f"napistu_tmp{extn}")

    if download_method == DOWNLOAD_METHODS.WGET:
        download_wget(url, tmp_file)
    elif download_method == DOWNLOAD_METHODS.FTP:
        download_ftp(url, tmp_file)
    else:
        raise ValueError(
            f"Undefined download_method, defined methods are {VALID_DOWNLOAD_METHODS}"
        )

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


def download_ftp(url: str, path: str) -> None:
    """
    Download a file from an FTP server.

    Parameters
    ----------
    url : str
        URL of the file to download
    path : str
        Path to the output file

    Returns
    -------
    None
    """
    with closing(request.urlopen(url)) as r:
        with open(path, "wb") as f:
            shutil.copyfileobj(r, f)

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

    Parameters
    ----------
    url : str
        URL of the file to download
    path : FilePath | WriteBuffer
        File path or buffer
    target_filename : str
        Specific file to extract from ZIP if URL is a ZIP file
    verify : bool
        url (str): url
    timeout : int
        Timeout in seconds for the request
    max_retries : int
        Number of times to retry the download if it fails

    Returns
    -------
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


def extract(file: str):
    """
    Untar, unzip and ungzip

    Parameters
    ----------
    file : str
        Path to compressed file

    Returns
    -------
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
    """
    Gunzip a file to an output path.

    Parameters
    ----------
    gzipped_path : str
        Path to the gzipped file
    outpath : str | None
        Path to the output file

    Returns
    -------
    None
    """

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


def load_json(uri: str) -> Any:
    """Read json from uri

    Parameters
    ----------
    uri : str
        Path to the json file

    Returns
    -------
    object : Any
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


def load_pickle(path: str):
    """Loads pickle object to path

    Parameters
    ----------
    path : str
        Path to the pickle file

    Returns
    -------
    Any
        Object
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
                with source_fs.open(file, "rb") as f:
                    return pickle.load(f)
            else:
                raise e


def pickle_cache(path: str, overwrite: bool = False) -> Callable:
    """A decorator to cache a function call result to pickle

    Attention: this does not care about the function arguments
    All function calls will be served by the same pickle file.

    Parameters
    ----------
    path : str
        Path to the cache pickle file
    overwrite : bool
        Should an existing cache be overwritten even if it exists?

    Returns
    -------
    Callable
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


def requests_retry_session(
    retries=5,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 503, 504),
    session: requests.Session | None = None,
    **kwargs,
) -> requests.Session:
    """
    Requests session with retry logic

    This should help to combat flaky apis, eg Brenda.
    From: https://stackoverflow.com/a/58687549

    Parameters
    ----------
    retries : int
        Number of retries. Defaults to 5.
    backoff_factor : float
        Backoff factor. Defaults to 0.3.
    status_forcelist : tuple
        Errors to retry. Defaults to (500, 502, 503, 504).
    session : requests.Session | None
        Existing session. Defaults to None.

    Returns
    -------
    requests.Session
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


def save_json(uri: str, object: Any) -> None:
    """Write object to json file at uri

    Parameters
    ----------
    uri : str
        Path to the json file
    object : Any
        Object to write

    Returns
    -------
    None
    """
    target_base, target_path = get_target_base_and_path(uri)
    with open_fs(target_base, create=True) as target_fs:
        target_fs.writetext(target_path, json.dumps(object))


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


def write_file_contents_to_path(path: str, contents) -> None:
    """
    Helper function to write file contents to the path.

    Parameters
    ----------
    path : str
        Destination
    contents : Any
        File contents

    Returns
    -------
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


read_pickle = load_pickle
write_pickle = save_pickle
