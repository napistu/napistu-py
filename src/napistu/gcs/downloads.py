from __future__ import annotations

import logging
import os
import re
import shutil
from pydantic import BaseModel
from typing import Optional, List

from napistu import utils
from napistu.gcs.utils import _initialize_data_dir
from napistu.gcs.constants import (
    GCS_ASSETS,
    GCS_ASSETS_DEFS,
    INIT_DATA_DIR_MSG,
)

logger = logging.getLogger(__name__)


def load_public_napistu_asset(
    asset: str,
    data_dir: str,
    subasset: str | None = None,
    version: str | None = None,
    init_msg: str = INIT_DATA_DIR_MSG,
    overwrite: bool = False,
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

    Returns
    -------
    str
        asset_path: the path to a local file
    """

    # validate data_directory
    _initialize_data_dir(data_dir, init_msg)
    _validate_gcs_asset(asset)
    _validate_gcs_subasset(asset, subasset)
    _validate_gcs_asset_version(asset, version)

    # get the path for the asset (which may have been downloaded in a tar-ball)
    asset_path = os.path.join(data_dir, _get_gcs_asset_path(asset, subasset))
    if os.path.isfile(asset_path) and not overwrite:
        return asset_path

    download_path = os.path.join(
        data_dir, os.path.basename(GCS_ASSETS.ASSETS[asset][GCS_ASSETS_DEFS.FILE])
    )
    if overwrite:
        _remove_asset_files_if_needed(asset, data_dir)
    if not os.path.isfile(download_path):
        download_public_napistu_asset(asset, version, download_path)

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


def download_public_napistu_asset(
    asset: str, version: str | None, out_path: str
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

    Returns
    -------
    None
    """

    _validate_gcs_asset(asset)
    if version is None:
        target_uri = GCS_ASSETS.ASSETS[asset][GCS_ASSETS_DEFS.PUBLIC_URL]
    else:
        _validate_gcs_asset_version(asset, version)
        target_uri = GCS_ASSETS.ASSETS[asset][GCS_ASSETS_DEFS.VERSIONS][version]

    logger.info(f"Downloading target URI: {target_uri} to {out_path}")

    utils.download_wget(target_uri, out_path)

    if not os.path.isfile(out_path):
        raise FileNotFoundError(f"Download failed: {out_path} was not created.")

    return None


def _validate_gcs_asset(asset: str) -> None:
    """Validate a GCS asset by name."""

    assets = _NapistuAssetsValidator(assets=GCS_ASSETS.ASSETS).assets
    valid_gcs_assets = assets.keys()
    if asset not in valid_gcs_assets:
        raise ValueError(
            f"asset was {asset} and must be one of the keys in GCS_ASSETS.ASSETS: {', '.join(valid_gcs_assets)}"
        )

    return None


def _validate_gcs_subasset(asset: str, subasset: str) -> None:
    """Validate a subasset as belonging to a given asset."""

    if GCS_ASSETS.ASSETS[asset][GCS_ASSETS_DEFS.SUBASSETS] is None:
        if subasset is not None:
            logger.warning(
                f"subasset was not None but asset {asset} does not have subassets. Ignoring subasset."
            )

        return None

    valid_subassets = GCS_ASSETS.ASSETS[asset][GCS_ASSETS_DEFS.SUBASSETS]

    if subasset is None:
        raise ValueError(
            f"subasset was None and must be one of {', '.join(valid_subassets)}"
        )

    if subasset not in valid_subassets:
        raise ValueError(
            f"subasset, {subasset}, was not found in asset {asset}. Valid subassets are {', '.join(valid_subassets)}"
        )

    return None


def _validate_gcs_asset_version(asset: str, version: Optional[str]) -> None:
    """Validate a GCS asset version if specified."""

    if version is None:
        return None

    asset_dict = GCS_ASSETS.ASSETS[asset]
    if asset_dict.get(GCS_ASSETS_DEFS.VERSIONS) is None:
        raise ValueError(
            f"Asset '{asset}' does not support versioning. If version is None, the standard 'latest' asset will be used."
        )

    valid_versions = asset_dict[GCS_ASSETS_DEFS.VERSIONS].keys()
    if version not in valid_versions:
        raise ValueError(
            f"Version '{version}' is not valid for asset '{asset}'. Valid versions are: {', '.join(valid_versions)}. "
            f"If version is None, the standard 'latest' asset will be used."
        )

    return None


def _get_gcs_asset_path(asset: str, subasset: Optional[str] = None) -> str:
    """
    Get the GCS path for a given asset and subasset.

    Parameters
    ----------
    asset : str
        The name of the asset.
    subasset : Optional[str]
        The name of the subasset.

    Returns
    -------
    str
        The GCS path for the asset or subasset.
    """
    asset_dict = GCS_ASSETS.ASSETS[asset]
    if asset_dict[GCS_ASSETS_DEFS.SUBASSETS] is None:
        out_file = asset_dict[GCS_ASSETS_DEFS.FILE]
    else:
        extract_dir = asset_dict[GCS_ASSETS_DEFS.FILE].split(".")[0]
        out_file = os.path.join(
            extract_dir, asset_dict[GCS_ASSETS_DEFS.SUBASSETS][subasset]
        )
    return out_file


class _NapistuAssetValidator(BaseModel):
    file: str
    subassets: dict[str, str] | None
    public_url: str
    versions: dict[str, str] | None


class _NapistuAssetsValidator(BaseModel):
    assets: dict[str, _NapistuAssetValidator]


def _remove_asset_files_if_needed(asset: str, data_dir: str) -> List[str]:
    """
    Remove asset archive and any extracted directory from data_dir.

    Parameters
    ----------
    asset: str
        The asset key (e.g., 'test_pathway').
    data_dir: str
        The directory where assets are stored.

    Returns
    -------
    List[str]
        A list of the paths of the removed files.
    """

    logger = logging.getLogger(__name__)
    removed = []

    # Remove the archive file (any extension)
    archive_filename = os.path.basename(GCS_ASSETS.ASSETS[asset][GCS_ASSETS_DEFS.FILE])
    archive_path = os.path.join(data_dir, archive_filename)
    if os.path.exists(archive_path):
        os.remove(archive_path)
        logger.info(f"Removed asset archive: {archive_path}")
        removed.append(archive_path)

    # Remove extracted directory (if any)
    asset_dict = GCS_ASSETS.ASSETS[asset]
    if asset_dict.get(GCS_ASSETS_DEFS.SUBASSETS) is not None or any(
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
