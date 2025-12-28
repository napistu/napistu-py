"""Module for downloading and loading Napistu public assets from GCS."""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import List, Optional

from napistu import utils
from napistu.gcs.assets import GCSAssets
from napistu.gcs.constants import (
    GCS_ASSET_DEFS,
    GCS_ASSETS,
    INIT_DATA_DIR_MSG,
)
from napistu.gcs.utils import _initialize_data_dir

logger = logging.getLogger(__name__)


def download_public_napistu_asset(
    asset: str, version: str | None, out_path: str, gcs_assets: GCSAssets | None = None
) -> None:
    """
    Download Public Napistu Asset

    Parameters
    -----
    asset: str
        The name of a Napistu public asset stored in Google Cloud Storage (GCS)
    version: str
        The version of the asset to download
    out_path: str
        Local location where the file should be saved.
    gcs_assets: GCSAssets | None
        GCS assets configuration. If None (default), uses constants.GCS_ASSETS via from_dict.
        Can be overridden to use custom asset configurations.

    Returns
    -------
    None

    Examples
    --------
    >>> from napistu.gcs import downloads
    >>> from napistu.gcs.constants import GCS_ASSETS_NAMES
    >>> downloads.download_public_napistu_asset(
    ...     asset=GCS_ASSETS_NAMES.TEST_PATHWAY,
    ...     version=None,
    ...     out_path="/tmp/test_pathway.tar.gz"
    ... )
    """

    # Use default GCS_ASSETS if not provided
    if gcs_assets is None:
        gcs_assets = GCSAssets.from_dict(GCS_ASSETS)

    _validate_gcs_asset(asset, gcs_assets)
    if version is None:
        target_uri = gcs_assets.ASSETS[asset][GCS_ASSET_DEFS.PUBLIC_URL]
    else:
        _validate_gcs_asset_version(asset, version, gcs_assets)
        target_uri = gcs_assets.ASSETS[asset][GCS_ASSET_DEFS.VERSIONS][version]

    logger.info(f"Downloading target URI: {target_uri} to {out_path}")

    utils.download_wget(target_uri, out_path)

    if not os.path.isfile(out_path):
        raise FileNotFoundError(f"Download failed: {out_path} was not created.")

    return None


def load_public_napistu_asset(
    asset: str,
    data_dir: str,
    subasset: str | None = None,
    version: str | None = None,
    init_msg: str = INIT_DATA_DIR_MSG,
    overwrite: bool = False,
    gcs_assets: GCSAssets | None = None,
) -> str:
    """
    Load Public Napistu Asset

    Download the `asset` asset to `data_dir` if it doesn't
    already exist and return a path

    Parameters
    ----------
    asset: str
        The file to download (which will be unpacked if its a .tar.gz)
    data_dir: str
        The local directory where assets should be stored
    subasset: str
        The name of a subasset to load from within the asset bundle
    version: str
        The version of the asset to load (if None, the latest version will be used)
    init_msg: str
        Message to display if data_dir does not exist
    overwrite: bool
        If True, always download the asset and re-extract it, even if it already exists
    gcs_assets: GCSAssets | None
        GCS assets configuration. If None (default), uses constants.GCS_ASSETS via from_dict.
        Can be overridden to use custom asset configurations.

    Returns
    -------
    str
        asset_path: the path to a local file

    Examples
    --------
    >>> from napistu.gcs import downloads
    >>> from napistu.gcs.constants import GCS_ASSETS_NAMES, GCS_SUBASSET_NAMES
    >>> path = downloads.load_public_napistu_asset(
    ...     asset=GCS_ASSETS_NAMES.TEST_PATHWAY,
    ...     data_dir="/tmp/napistu_data",
    ...     subasset=GCS_SUBASSET_NAMES.SBML_DFS
    ... )
    >>> print(path)
    /tmp/napistu_data/test_pathway/sbml_dfs.pkl
    """

    # Use default GCS_ASSETS if not provided
    if gcs_assets is None:
        gcs_assets = GCSAssets.from_dict(GCS_ASSETS)

    # validate data_directory
    _initialize_data_dir(data_dir, init_msg)
    _validate_gcs_asset(asset, gcs_assets)
    _validate_gcs_subasset(asset, subasset, gcs_assets)
    _validate_gcs_asset_version(asset, version, gcs_assets)

    # get the path for the asset (which may have been downloaded in a tar-ball)
    asset_path = os.path.join(
        data_dir, _get_gcs_asset_path(asset, subasset, gcs_assets)
    )
    if os.path.isfile(asset_path) and not overwrite:
        return asset_path

    download_path = os.path.join(
        data_dir, os.path.basename(gcs_assets.ASSETS[asset][GCS_ASSET_DEFS.FILE])
    )
    if overwrite:
        _remove_asset_files_if_needed(asset, data_dir, gcs_assets)
    if not os.path.isfile(download_path):
        download_public_napistu_asset(asset, version, download_path, gcs_assets)

    # gunzip if needed
    extn = utils.get_extn_from_url(download_path)
    if (
        re.search(".tar\\.gz$", extn)
        or re.search("\\.tgz$", extn)
        or re.search("\\.zip$", extn)
        or re.search("\\.gz$", extn)
    ):
        utils.extract(download_path)

    # check that the asset_path exists
    if not os.path.isfile(asset_path):
        raise FileNotFoundError(
            f"Something went wrong and {asset_path} was not created."
        )

    return asset_path


def _get_gcs_asset_path(
    asset: str, subasset: Optional[str], gcs_assets: GCSAssets
) -> str:
    """
    Get the GCS path for a given asset and subasset.

    Parameters
    ----------
    asset : str
        The name of the asset.
    subasset : Optional[str]
        The name of the subasset.
    gcs_assets : GCSAssets
        GCS assets configuration.

    Returns
    -------
    str
        The GCS path for the asset or subasset.
    """
    asset_dict = gcs_assets.ASSETS[asset]
    if asset_dict[GCS_ASSET_DEFS.SUBASSETS] is None:
        out_file = asset_dict[GCS_ASSET_DEFS.FILE]
    else:
        extract_dir = asset_dict[GCS_ASSET_DEFS.FILE].split(".")[0]
        out_file = os.path.join(
            extract_dir, asset_dict[GCS_ASSET_DEFS.SUBASSETS][subasset]
        )
    return out_file


def _remove_asset_files_if_needed(
    asset: str, data_dir: str, gcs_assets: GCSAssets | None = None
) -> List[str]:
    """
    Remove asset archive and any extracted directory from data_dir.

    Parameters
    ----------
    asset: str
        The asset key (e.g., 'test_pathway').
    data_dir: str
        The directory where assets are stored.
    gcs_assets: GCSAssets | None
        GCS assets configuration. If None (default), uses constants.GCS_ASSETS via from_dict.

    Returns
    -------
    List[str]
        A list of the paths of the removed files.
    """

    # Use default GCS_ASSETS if not provided
    if gcs_assets is None:
        gcs_assets = GCSAssets.from_dict(GCS_ASSETS)

    logger = logging.getLogger(__name__)
    removed = []

    # Remove the archive file (any extension)
    archive_filename = os.path.basename(gcs_assets.ASSETS[asset][GCS_ASSET_DEFS.FILE])
    archive_path = os.path.join(data_dir, archive_filename)
    if os.path.exists(archive_path):
        os.remove(archive_path)
        logger.info(f"Removed asset archive: {archive_path}")
        removed.append(archive_path)

    # Remove extracted directory (if any)
    asset_dict = gcs_assets.ASSETS[asset]
    subassets = asset_dict[GCS_ASSET_DEFS.SUBASSETS]
    if subassets is not None or any(
        archive_filename.endswith(ext) for ext in [".tar.gz", ".tgz", ".zip", ".gz"]
    ):
        extract_dir = os.path.join(data_dir, archive_filename.split(".")[0])
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir)
            logger.info(f"Removed extracted asset directory: {extract_dir}")
            removed.append(extract_dir)

    if not removed:
        logger.debug("No asset files found to remove.")

    return removed


def _validate_gcs_asset(asset: str, gcs_assets: GCSAssets) -> None:
    """Validate a GCS asset by name."""

    valid_gcs_assets = gcs_assets.assets.keys()
    if asset not in valid_gcs_assets:
        raise ValueError(
            f"asset was {asset} and must be one of the keys in GCS_ASSETS.ASSETS: {', '.join(valid_gcs_assets)}"
        )

    return None


def _validate_gcs_asset_version(
    asset: str, version: Optional[str], gcs_assets: GCSAssets
) -> None:
    """Validate a GCS asset version if specified."""

    if version is None:
        return None

    asset_dict = gcs_assets.ASSETS[asset]
    versions = asset_dict[GCS_ASSET_DEFS.VERSIONS]
    if versions is None:
        raise ValueError(
            f"Asset '{asset}' does not support versioning. If version is None, the standard 'latest' asset will be used."
        )

    valid_versions = versions.keys()
    if version not in valid_versions:
        raise ValueError(
            f"Version '{version}' is not valid for asset '{asset}'. Valid versions are: {', '.join(valid_versions)}. "
            f"If version is None, the standard 'latest' asset will be used."
        )

    return None


def _validate_gcs_subasset(
    asset: str, subasset: str | None, gcs_assets: GCSAssets
) -> None:
    """Validate a subasset as belonging to a given asset."""

    if gcs_assets.ASSETS[asset][GCS_ASSET_DEFS.SUBASSETS] is None:
        if subasset is not None:
            logger.warning(
                f"subasset was not None but asset {asset} does not have subassets. Ignoring subasset."
            )

        return None

    valid_subassets = gcs_assets.ASSETS[asset][GCS_ASSET_DEFS.SUBASSETS]

    if subasset is None:
        raise ValueError(
            f"subasset was None and must be one of {', '.join(valid_subassets)}"
        )

    if subasset not in valid_subassets:
        raise ValueError(
            f"subasset, {subasset}, was not found in asset {asset}. Valid subassets are {', '.join(valid_subassets)}"
        )

    return None
