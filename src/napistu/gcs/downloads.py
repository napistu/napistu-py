from __future__ import annotations

import logging
import os
import re
import shutil
from pydantic import BaseModel
from typing import Optional

from napistu import utils
from napistu.gcs.constants import GCS_ASSETS

from napistu.gcs.constants import INIT_DATA_DIR_MSG
from napistu.gcs.utils import _initialize_data_dir

logger = logging.getLogger(__name__)


def load_public_napistu_asset(
    asset: str,
    data_dir: str,
    subasset: str | None = None,
    init_msg: str = INIT_DATA_DIR_MSG,
    overwrite: bool = False,
) -> str:
    """
    Load Public Napistu Asset

    Download the `asset` asset to `data_dir` if it doesn't
    already exist and return a path

    asset: the file to download (which will be unpacked if its a .tar.gz)
    subasset: the name of a subasset to load from within the asset bundle
    data_dir: the local directory where assets should be stored
    init_msg: message to display if data_dir does not exist
    overwrite: if True, always download the asset and re-extract it, even if it already exists

    returns:
        asset_path: the path to a local file
    """

    # validate data_directory
    _initialize_data_dir(data_dir, init_msg)
    _validate_gcs_asset(asset)
    _validate_gcs_subasset(asset, subasset)

    # get the path for the asset (which may have been downloaded in a tar-ball)
    asset_path = os.path.join(data_dir, _get_gcs_asset_path(asset, subasset))
    if os.path.isfile(asset_path) and not overwrite:
        return asset_path

    download_path = os.path.join(
        data_dir, os.path.basename(GCS_ASSETS.ASSETS[asset]["file"])
    )
    if overwrite:
        _remove_asset_files_if_needed(asset, data_dir)
    if not os.path.isfile(download_path):
        download_public_napistu_asset(asset, download_path)

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


def download_public_napistu_asset(asset: str, out_path: str) -> None:
    """
    Download Public Napistu Asset

    Args:
        asset (str): The name of a Napistu public asset stored in Google Cloud Storage (GCS)
        out_path (list): Local location where the file should be saved.

    Returns:
        None
    """

    _validate_gcs_asset(asset)
    selected_file = GCS_ASSETS.ASSETS[asset]["public_url"]

    logger.info(f"Downloading {os.path.basename(selected_file)} to {out_path}")
    logger.info(f"Download URI: {selected_file}")

    utils.download_wget(selected_file, out_path)

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

    if GCS_ASSETS.ASSETS[asset]["subassets"] is None:
        if subasset is not None:
            logger.warning(
                f"subasset was not None but asset {asset} does not have subassets. Ignoring subasset."
            )

        return None

    valid_subassets = GCS_ASSETS.ASSETS[asset]["subassets"]

    if subasset is None:
        raise ValueError(
            f"subasset was None and must be one of {', '.join(valid_subassets)}"
        )

    if subasset not in valid_subassets:
        raise ValueError(
            f"subasset, {subasset}, was not found in asset {asset}. Valid subassets are {', '.join(valid_subassets)}"
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
    if asset_dict["subassets"] is None:
        out_file = asset_dict["file"]
    else:
        extract_dir = asset_dict["file"].split(".")[0]
        out_file = os.path.join(extract_dir, asset_dict["subassets"][subasset])
    return out_file


class _NapistuAssetValidator(BaseModel):
    file: str
    subassets: dict[str, str] | None
    public_url: str


class _NapistuAssetsValidator(BaseModel):
    assets: dict[str, _NapistuAssetValidator]


def _remove_asset_files_if_needed(asset: str, data_dir: str):
    """
    Remove asset archive and any extracted directory from data_dir.

    Args:
        asset (str): The asset key (e.g., 'test_pathway').
        data_dir (str): The directory where assets are stored.
    """
    logger = logging.getLogger(__name__)
    removed = []

    # Remove the archive file (any extension)
    archive_filename = os.path.basename(GCS_ASSETS.ASSETS[asset]["file"])
    archive_path = os.path.join(data_dir, archive_filename)
    if os.path.exists(archive_path):
        os.remove(archive_path)
        logger.info(f"Removed asset archive: {archive_path}")
        removed.append(archive_path)

    # Remove extracted directory (if any)
    asset_dict = GCS_ASSETS.ASSETS[asset]
    if asset_dict.get("subassets") is not None or any(
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
