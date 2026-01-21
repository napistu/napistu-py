"""
Utilities for path and URI operations.

Public Functions
----------------
copy_uri(input_uri: str, output_uri: str, is_file: bool = True) -> None:
    Copy a file or folder from one URI to another.
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
import warnings
from urllib.parse import urlparse

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs
    from fs.copy import copy_dir, copy_file
    from fs.errors import CreateFailed, ResourceNotFound

logger = logging.getLogger(__name__)


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
