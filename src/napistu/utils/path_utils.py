"""
Utilities for path and URI operations.

Public Functions
----------------
copy_uri(input_uri: str, output_uri: str, is_file: bool = True) -> None:
    Copy a file or folder from one URI to another.
ensure_path(path: Union[str, Path], expand_user: bool = True) -> Path:
    Convert a string or Path to a Path object, optionally expanding user home directory.
get_extn_from_url(url: str) -> str:
    Retrieve file extension from a URL.
get_source_base_and_path(uri: str) -> tuple[str, str]:
    Get the base of a bucket or folder and the path to the file.
get_target_base_and_path(uri: str) -> tuple[str, str]:
    Get the base of a bucket + directory and the file.
initialize_dir(output_dir_path: str, overwrite: bool) -> None:
    Initialize a filesystem directory.
path_exists(path: str) -> bool:
    Check if a path or URI exists.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import fsspec

logger = logging.getLogger(__name__)


def copy_uri(input_uri: str, output_uri: str, is_file: bool = True) -> None:
    """Copy a file or folder from one URI to another.

    Parameters
    ----------
    input_uri : str
        Input file URI (e.g., 'gs://bucket/file', '/local/path', 'memory://path').
    output_uri : str
        Output file URI (e.g., 'gs://bucket/file', '/local/path', 'memory://path').
    is_file : bool, default=True
        If True, copy a single file. If False, copy directory recursively.

    Examples
    --------
    >>> copy_uri('/local/source.txt', '/local/dest.txt')
    >>> copy_uri('gs://bucket/source/', 'gs://bucket/dest/', is_file=False)
    """
    logger.info("Copy uri from %s to %s", input_uri, output_uri)

    # Parse source and target filesystems
    source_fs, source_path = fsspec.core.url_to_fs(input_uri)
    target_fs, target_path = fsspec.core.url_to_fs(output_uri)

    # Ensure target directory exists
    target_dir = os.path.dirname(target_path)
    if target_dir:
        target_fs.makedirs(target_dir, exist_ok=True)

    if is_file:
        # Copy single file using fsspec's generic copy
        with source_fs.open(source_path, "rb") as src:
            with target_fs.open(target_path, "wb") as dst:
                dst.write(src.read())
    else:
        # Copy directory recursively
        # List all files in source directory
        all_files = source_fs.find(source_path)
        for src_file in all_files:
            # Calculate relative path and target path
            rel_path = os.path.relpath(src_file, source_path)
            dst_file = os.path.join(target_path, rel_path)

            # Ensure target subdirectory exists
            dst_dir = os.path.dirname(dst_file)
            if dst_dir:
                target_fs.makedirs(dst_dir, exist_ok=True)

            # Copy file
            with source_fs.open(src_file, "rb") as src:
                with target_fs.open(dst_file, "wb") as dst:
                    dst.write(src.read())


def ensure_path(path: Union[str, Path], expand_user: bool = True) -> Path:
    """
    Convert a string or Path to a Path object, optionally expanding user home directory.

    Parameters
    ----------
    path : Union[str, Path]
        Path to convert. Can be a string (e.g., "~/data/store") or Path object.
    expand_user : bool, default=True
        If True, expand tildes (~) to the user's home directory.

    Returns
    -------
    Path
        Path object, with user expanded if expand_user=True.

    Raises
    ------
    TypeError
        If path is not a str or Path object.

    Examples
    --------
    >>> ensure_path("~/data/store")
    PosixPath('/home/user/data/store')
    >>> ensure_path(Path("./relative/path"))
    PosixPath('./relative/path')
    >>> ensure_path("~/data", expand_user=False)
    PosixPath('~/data')
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path object, got {type(path).__name__}")
    if isinstance(path, str):
        path = Path(path)
    if expand_user:
        path = path.expanduser()
    return path


def get_extn_from_url(url: str) -> str:
    """Retrieve file extension from a URL.

    Parameters
    ----------
    url : str
        URL to extract extension from.

    Returns
    -------
    str
        File extension including the leading dot (e.g., '.gz', '.tar.gz').

    Raises
    ------
    ValueError
        If no file extension can be identified in the URL.

    Examples
    --------
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


def get_source_base_and_path(uri: str) -> tuple[str, str]:
    """Get the base of a bucket or folder and the path to the file.

    For URIs with a scheme (e.g., 'gs://'), returns the scheme + netloc as base.
    For local paths, returns the directory as base.

    Parameters
    ----------
    uri : str
        URI or path to parse.

    Returns
    -------
    tuple[str, str]
        A tuple of (base, path) where:
        - base : str
            The base URI or directory (e.g., 'gs://bucket' or '/local/dir').
        - path : str
            The relative path to the file (e.g., 'folder/file' or 'file').

    Examples
    --------
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


def get_target_base_and_path(uri: str) -> tuple[str, str]:
    """Get the base directory + parent path and the filename.

    Splits the URI at the last path separator to extract the filename.

    Parameters
    ----------
    uri : str
        URI or path to parse.

    Returns
    -------
    tuple[str, str]
        A tuple of (base, filename) where:
        - base : str
            The directory path (e.g., 'gs://bucket/folder' or '/local/folder').
        - filename : str
            The filename (e.g., 'file').

    Examples
    --------
    >>> get_target_base_and_path("gs://bucket/folder/file")
    ('gs://bucket/folder', 'file')
    >>> get_target_base_and_path("bucket/folder/file")
    ('bucket/folder', 'file')
    >>> get_target_base_and_path("/bucket/folder/file")
    ('/bucket/folder', 'file')
    """
    base, path = os.path.split(uri)
    return base, path


def initialize_dir(output_dir_path: str, overwrite: bool) -> None:
    """Initialize a filesystem directory.

    Creates a new directory or optionally overwrites an existing one.
    Works with any fsspec-supported filesystem (local, GCS, S3, etc.).

    Parameters
    ----------
    output_dir_path : str
        Path or URI to the directory to create (e.g., '/local/path', 'gs://bucket/path').
    overwrite : bool
        If True, delete and recreate the directory if it exists.
        If False, raise FileExistsError if the directory exists.

    Raises
    ------
    FileExistsError
        If directory exists and overwrite is False.

    Examples
    --------
    >>> initialize_dir('/tmp/newdir', overwrite=False)
    >>> initialize_dir('gs://bucket/path', overwrite=True)
    """
    output_dir_path = str(output_dir_path)
    fs, path = fsspec.core.url_to_fs(output_dir_path)

    if fs.exists(path):
        if overwrite:
            fs.rm(path, recursive=True)
            fs.makedirs(path, exist_ok=True)
        else:
            raise FileExistsError(
                f"{output_dir_path} already exists and overwrite is False"
            )
    else:
        fs.makedirs(path, exist_ok=True)


def path_exists(path: str) -> bool:
    """Check if a path or URI exists.

    Works with any fsspec-supported filesystem (local, GCS, S3, memory, etc.).

    Parameters
    ----------
    path : str
        Path or URI to check (e.g., '/local/path', 'gs://bucket/path', 'memory://path').

    Returns
    -------
    bool
        True if the path exists, False otherwise.

    Examples
    --------
    >>> path_exists('/tmp/myfile.txt')
    False
    >>> path_exists('gs://bucket/existing_file.txt')
    True
    >>> path_exists('.')
    True
    """
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)
