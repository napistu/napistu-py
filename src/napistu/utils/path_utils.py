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

import contextlib
import logging
import os
import posixpath
import re
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import fsspec

logger = logging.getLogger(__name__)


def copy_uri(input_uri: str, output_uri: str, is_file=True):
    """
    Copy a file or folder from one uri to another

    Parameters
    ----------
    input_uri : str
        Input file uri (gcs, http, ...)
    output_uri : str
        Path to output file (gcs, local)
    is_file : bool, optional
        Is this a file or folder?. Defaults to True.

    Returns
    -------
    None
    """
    logger.info(f"Copy uri from {input_uri} to {output_uri}")
    source_base, source_path = get_source_base_and_path(input_uri)
    target_base, target_path = get_target_base_and_path(output_uri)
    copy_fun = _copy_file if is_file else _copy_dir
    with _open_fs(source_base) as source_fs:
        with _open_fs(target_base, create=True) as target_fs:
            copy_fun(source_fs, source_path, target_fs, target_path)


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
    """
    Retrieves file extension from an URL

    Parameters
    ----------
    url : str
        URL to retrieve file extension from

    Returns
    -------
    str
        The identified extension

    Raises
    ------
    ValueError
        Raised when no extension identified

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
    """
    Get the base of a bucket or folder and the path to the file

    Parameters
    ----------
    uri : str
        URI to get the base and path from

    Returns
    -------
    tuple[str, str]
        The base and path of the URI

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


def get_target_base_and_path(uri):
    """
    Get the base of a bucket + directory and the file

    Parameters
    ----------
    uri : str
        URI to get the base and path from

    Returns
    -------
    tuple[str, str]
        The base and path of the URI
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


def initialize_dir(output_dir_path: str, overwrite: bool):
    """
    Initializes a filesystem directory

    Parameters
    ----------
    output_dir_path : str
        Path to new directory
    overwrite : bool
        Overwrite? if true, directory will be deleted and recreated

    Raises
    ------
    FileExistsError
        If directory already exists and overwrite is False
    """
    output_dir_path = str(output_dir_path)
    with _open_fs(output_dir_path) as out_fs:
        exists = out_fs.exists("") or out_fs.exists("/")
        if exists:
            if overwrite:
                out_fs.removetree("/")
            else:
                raise FileExistsError(
                    f"{output_dir_path} already exists and overwrite is False"
                )
    if not exists:
        with _open_fs(output_dir_path, create=True):
            pass


def path_exists(path: str) -> bool:
    """

    Parameters
    ----------
    path : str
        Path/URI to check

    Returns
    -------
    bool
        Exists?
    """
    dir_part, file_part = os.path.split(path)
    try:
        with _open_fs(dir_part) as f:
            if f.exists(file_part):
                return True
    except (OSError, FileNotFoundError, ValueError):
        pass

    try:
        with _open_fs(path) as f:
            return f.exists("") or f.exists("/")
    except (OSError, FileNotFoundError, ValueError):
        return False


class _FsspecFS:
    """Internal adapter wrapping fsspec to match the fs API used by path_utils."""

    def __init__(self, fs: fsspec.AbstractFileSystem, base_path: str, protocol: str):
        self._fs = fs
        self._base = base_path.rstrip("/") if base_path else ""
        self._protocol = protocol

    def _full_path(self, path: str) -> str:
        if not path or path == "/":
            return self._base or "/"
        path = path.lstrip("/")
        return _join(self._protocol, self._base, path)

    def exists(self, path: str) -> bool:
        try:
            return self._fs.exists(self._full_path(path))
        except Exception:
            return False

    def removetree(self, path: str = "/") -> None:
        full = self._full_path(path)
        try:
            if self._fs.exists(full):
                self._fs.rm(full, recursive=True)
        except Exception:
            pass
        if path == "/" and self._base:
            try:
                self._fs.makedirs(self._base, exist_ok=True)
            except Exception:
                pass

    def readbytes(self, path: str) -> bytes:
        full = self._full_path(path)
        return self._fs.cat_file(full)

    def writebytes(self, path: str, data: bytes) -> None:
        full = self._full_path(path)
        self._fs.pipe_file(full, data)

    def listdir(self, path: str = "/") -> list[str]:
        full = self._full_path(path)
        names = self._fs.ls(full)
        return [os.path.basename(p.rstrip("/")) for p in names]

    def isdir(self, path: str) -> bool:
        full = self._full_path(path)
        try:
            info = self._fs.info(full)
            return info.get("type") == "directory"
        except Exception:
            return False

    def makedirs(self, path: str, recreate: bool = False) -> None:
        full = self._full_path(path)
        self._fs.makedirs(full, exist_ok=True)


def _copy_dir(
    source: _FsspecFS, source_path: str, target: _FsspecFS, target_path: str
) -> None:
    target.makedirs(target_path, recreate=True)
    for name in source.listdir(source_path):
        src_child = posixpath.join(source_path, name) if source_path else name
        tgt_child = posixpath.join(target_path, name) if target_path else name
        if source.isdir(src_child):
            _copy_dir(source, src_child, target, tgt_child)
        else:
            _copy_file(source, src_child, target, tgt_child)


def _copy_file(
    source: _FsspecFS, source_path: str, target: _FsspecFS, target_path: str
) -> None:
    data = source.readbytes(source_path)
    dir_part = posixpath.dirname(target_path)
    if dir_part:
        target.makedirs(dir_part, recreate=True)
    target.writebytes(target_path, data)


def _join(protocol: str, base: str, *parts: str) -> str:
    """Join base with path parts; use posixpath for gcs, os.path for file."""
    path = parts[0] if parts else ""
    for p in parts[1:]:
        path = path.rstrip("/") + "/" + p.lstrip("/") if path else p
    if protocol == "file":
        return os.path.normpath(os.path.join(base, path))
    return posixpath.join(base, path) if path else base.rstrip("/")


@contextlib.contextmanager
def _open_fs(url: str, create: bool = False):
    """Context manager yielding _FsspecFS for the given base URL (internal adapter)."""
    protocol, base_path = _parse_uri(url)
    if protocol == "gcs" and not base_path:
        raise ValueError("GCS URI must include bucket")
    fs = fsspec.filesystem(protocol)
    if create and base_path:
        try:
            fs.makedirs(base_path, exist_ok=True)
        except Exception:
            pass
    try:
        yield _FsspecFS(fs, base_path, protocol)
    finally:
        try:
            fs.close()
        except Exception:
            pass


def _parse_uri(uri: str) -> tuple[str, str]:
    """Parse URI into (protocol, path) for fsspec. Returns ('file', path) for local."""
    uri = str(uri).strip()
    if not uri or uri == ".":
        return "file", os.path.abspath(".")
    parsed = urlparse(uri)
    if parsed.scheme in ("gs", "gcs"):
        # gs://bucket or gs://bucket/path -> protocol gcs, path bucket or bucket/path
        netloc = parsed.netloc or parsed.path.split("/", 1)[0]
        path = (parsed.path or "").lstrip("/")
        full = f"{netloc}/{path}" if path else netloc
        return "gcs", full.rstrip("/") or netloc
    if parsed.scheme == "file" or (parsed.scheme == "" and os.path.isabs(uri)):
        path = parsed.path or uri
        if parsed.scheme == "file" and path.startswith("///"):
            path = path[2:]  # file:///abs -> /abs
        return "file", os.path.normpath(path) if path else os.path.abspath(".")
    if parsed.scheme == "":
        # Relative path
        return "file", os.path.normpath(os.path.abspath(uri))
    # Unknown scheme, treat as file path
    return "file", os.path.normpath(uri)
