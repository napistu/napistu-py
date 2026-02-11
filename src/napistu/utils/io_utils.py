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
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import pickle
import shutil
import tarfile
import tempfile
import urllib.request as request
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Union

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter, Retry

from napistu.constants import FILE_EXT_ZIP
from napistu.utils.constants import DOWNLOAD_METHODS

# Import helper functions from path_utils
from napistu.utils.path_utils import (
    get_extn_from_url,
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
    """Download archive and extract to directory."""

    initialize_dir(output_dir_path, overwrite)
    extn = get_extn_from_url(url)

    # Download to temp file (AS-IS, no decompression)
    with tempfile.NamedTemporaryFile(suffix=extn, delete=False) as tmp:
        if download_method == DOWNLOAD_METHODS.WGET:
            download_wget(url, tmp.name)
        elif download_method == DOWNLOAD_METHODS.FTP:
            download_ftp(url, tmp.name)
        else:
            raise ValueError(f"Unsupported method: {download_method}")
        tmp_path = tmp.name

    try:
        # Now extract (handles all decompression)
        if extn.endswith((".tar.gz", ".tgz")):
            _extract_tarball(tmp_path, output_dir_path)
        elif extn.endswith(".zip"):
            _extract_zip(tmp_path, output_dir_path)
        elif extn.endswith(".gz"):
            # Single file - extract directly
            outfile = url.split("/")[-1].replace(".gz", "")
            gunzip(tmp_path, f"{output_dir_path}/{outfile}")
        else:
            raise ValueError(f"Unsupported format: {extn}")
    finally:
        os.unlink(tmp_path)


def _extract_tarball(tar_path: str, output_uri: str) -> None:
    """Extract tarball using standard library."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Security: filter to prevent path traversal
            tar.extractall(tmpdir, filter="data")  # Python 3.12+
        _copy_tree(tmpdir, output_uri)


def _extract_zip(zip_path: str, output_uri: str) -> None:
    """Extract zip using standard library."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)
        _copy_tree(tmpdir, output_uri)


def _copy_tree(source_dir: str, dest_uri: str) -> None:
    """Copy directory tree to any fsspec destination."""
    source_path = Path(source_dir)

    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(source_path)
            dest_file_uri = f"{dest_uri}/{rel_path}".replace(
                "\\", "/"
            )  # Windows compat

            with open(file_path, "rb") as src:
                with fsspec.open(dest_file_uri, "wb") as dst:
                    dst.write(src.read())


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
    """
    Downloads file / archive with wget

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

    # Special case: ZIP with target_filename
    if target_filename and (
        r.headers.get("Content-Type") == "application/zip"
        or url.endswith(f".{FILE_EXT_ZIP}")
    ):
        # Extract specific file from ZIP
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            if target_filename in z.namelist():
                with z.open(target_filename) as target_file:
                    return write_file_contents_to_path(path, target_file.read())
            else:
                raise FileNotFoundError(
                    f"{target_filename} not found in the ZIP archive"
                )

    return write_file_contents_to_path(path, r.content)


def extract(file_uri: str) -> None:
    """Extract archive at file_uri to same directory.

    Supports: .tar.gz, .tgz, .zip, .gz
    """
    extn = get_extn_from_url(file_uri)

    # Determine output directory
    if extn.endswith((".tar.gz", ".tgz")):
        output_uri = file_uri.replace(extn, "")
    else:
        output_uri = os.path.dirname(file_uri)

    try:
        initialize_dir(output_uri, overwrite=False)
    except FileExistsError:
        pass  # OK if exists

    # Download to temp if remote
    if file_uri.startswith(("gs://", "s3://", "http://", "https://")):
        with tempfile.NamedTemporaryFile(suffix=extn, delete=False) as tmp:
            with fsspec.open(file_uri, "rb") as src:
                with open(tmp.name, "wb") as dst:
                    dst.write(src.read())
            local_path = tmp.name
        delete_after = True
    else:
        local_path = file_uri
        delete_after = False

    try:
        if extn.endswith((".tar.gz", ".tgz")):
            _extract_tarball(local_path, output_uri)
        elif extn.endswith(".zip"):
            _extract_zip(local_path, output_uri)
        elif extn.endswith(".gz"):
            outfile = file_uri.split("/")[-1].replace(".gz", "")
            gunzip(file_uri, f"{output_uri}/{outfile}")
        else:
            raise ValueError(f"Unsupported format: {extn}")
    finally:
        if delete_after:
            os.unlink(local_path)


def gunzip(gzipped_path: str, outpath: str | None = None) -> None:
    """Gunzip a file to an output path.

    Parameters
    ----------
    gzipped_path : str
        Path or URI to the gzipped file (e.g., '/local/file.gz', 'gs://bucket/file.gz').
    outpath : str | None, optional
        Path or URI to the output file. If None, automatically determined by removing
        the .gz extension from gzipped_path.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If gzipped_path does not exist.

    Examples
    --------
    >>> gunzip('/tmp/data.txt.gz')  # Creates /tmp/data.txt
    >>> gunzip('gs://bucket/data.txt.gz', 'gs://bucket/output.txt')
    """
    # Check if source exists
    fs, path = fsspec.core.url_to_fs(gzipped_path)
    if not fs.exists(path):
        raise FileNotFoundError(f"{gzipped_path} not found")

    # Warn if doesn't have .gz extension
    if not gzipped_path.endswith(".gz"):
        logger.warning(f"{gzipped_path} does not have the .gz extension")

    # Determine output path if not provided
    if outpath is None:
        # Remove .gz extension
        outpath = (
            gzipped_path.rstrip(".gz")
            if gzipped_path.endswith(".gz")
            else gzipped_path + ".uncompressed"
        )

    # Read gzipped file and write uncompressed
    with fsspec.open(gzipped_path, "rb") as f_in:
        with gzip.open(f_in, "rb") as gz:
            with fsspec.open(outpath, "wb") as f_out:
                f_out.write(gz.read())


def load_json(uri: str) -> Any:
    """Read JSON from a URI.

    Parameters
    ----------
    uri : str
        Path or URI to the JSON file (e.g., '/local/path.json', 'gs://bucket/file.json').

    Returns
    -------
    Any
        The parsed JSON object (dict, list, etc.).

    Examples
    --------
    >>> data = load_json('/tmp/config.json')
    >>> data = load_json('gs://bucket/config.json')
    """
    logger.info("Read json from %s", uri)
    with fsspec.open(uri, "r") as f:
        return json.load(f)


def load_parquet(uri: Union[str, Path]) -> pd.DataFrame:
    """
    Read a DataFrame from a Parquet file.

    Parameters
    ----------
    uri : Union[str, Path]
        Path or URI to the Parquet file to load (e.g., '/local/data.parquet', 'gs://bucket/data.parquet').

    Returns
    -------
    pd.DataFrame
        The DataFrame loaded from the Parquet file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    >>> df = load_parquet('/tmp/data.parquet')
    >>> df = load_parquet('gs://bucket/data.parquet')
    """
    try:
        with fsspec.open(str(uri), "rb") as f:
            return pd.read_parquet(f, engine="pyarrow")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {uri}") from e


def load_pickle(path: str) -> Any:
    """Load a pickle object from a path or URI.

    Parameters
    ----------
    path : str
        Path or URI to the pickle file (e.g., '/local/file.pkl', 'gs://bucket/file.pkl').

    Returns
    -------
    Any
        The unpickled object.

    Examples
    --------
    >>> obj = load_pickle('/tmp/data.pkl')
    >>> obj = load_pickle('gs://bucket/data.pkl')
    """
    with fsspec.open(path, "rb") as f:
        return pickle.load(f)


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


def save_json(uri: str, obj: Any) -> None:
    """
    Write object to JSON file at URI.

    Parameters
    ----------
    uri : str
        Path or URI to the JSON file (e.g., '/local/path.json', 'gs://bucket/file.json').
    obj : Any
        Object to serialize to JSON.

    Returns
    -------
    None

    Examples
    --------
    >>> save_json('/tmp/config.json', {'key': 'value'})
    >>> save_json('gs://bucket/config.json', {'key': 'value'})
    """
    with fsspec.open(uri, "w") as f:
        json.dump(obj, f)


def save_parquet(
    df: pd.DataFrame, uri: Union[str, Path], compression: str = "snappy"
) -> None:
    """
    Write a DataFrame to a single Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    uri : Union[str, Path]
        Path or URI where to save the Parquet file (e.g., '/local/data.parquet', 'gs://bucket/data.parquet').
        Recommended extensions: .parquet or .pq
    compression : str, default='snappy'
        Compression algorithm. Options: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'.

    Raises
    ------
    OSError
        If the file cannot be written to (permission issues, etc.).

    Examples
    --------
    >>> save_parquet(df, '/tmp/data.parquet')
    >>> save_parquet(df, 'gs://bucket/data.parquet', compression='gzip')
    """
    uri_str = str(uri)

    # Warn about non-standard extensions
    if not any(uri_str.endswith(ext) for ext in [".parquet", ".pq"]):
        logger.warning(
            f"File '{uri_str}' doesn't have a standard Parquet extension (.parquet or .pq)"
        )

    with fsspec.open(uri_str, "wb") as f:
        # Convert to Arrow table and write as single file
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            f,
            compression=compression,
            use_dictionary=True,  # Efficient for repeated values
            write_statistics=True,  # Enables query optimization
        )


def save_pickle(path: str, dat: Any) -> None:
    """
    Save object to path as pickle.

    Parameters
    ----------
    path : str
        Path or URI where to save the pickle file (e.g., '/local/file.pkl', 'gs://bucket/file.pkl').
    dat : Any
        Object to pickle.

    Returns
    -------
    None

    Examples
    --------
    >>> save_pickle('/tmp/data.pkl', my_object)
    >>> save_pickle('gs://bucket/data.pkl', my_object)
    """
    with fsspec.open(path, "wb") as f:
        pickle.dump(dat, f)


def write_file_contents_to_path(path: str, contents: bytes) -> None:
    """
    Write file contents to a path or URI.

    Handles both file-like objects with write() method and string paths/URIs.

    Parameters
    ----------
    path : str
        Destination path or URI, or a file-like object with write() method.
    contents : bytes
        File contents to write.

    Returns
    -------
    None

    Examples
    --------
    >>> write_file_contents_to_path('/tmp/file.txt', b'Hello')
    >>> write_file_contents_to_path('gs://bucket/file.txt', b'Hello')
    """
    if hasattr(path, "write") and hasattr(path, "__iter__"):
        path.write(contents)  # type: ignore
    else:
        with fsspec.open(path, "wb") as f:
            f.write(contents)
