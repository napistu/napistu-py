from __future__ import annotations

import os

import pytest

from napistu.gcs import downloads
from napistu.gcs.assets import GCSAssets
from napistu.gcs.constants import (
    GCS_ASSETS,
    GCS_ASSETS_NAMES,
    GCS_FILETYPES,
    GCS_SUBASSET_NAMES,
)


@pytest.fixture
def gcs_assets():
    """Fixture that creates a GCSAssets instance from the default GCS_ASSETS."""
    return GCSAssets.from_dict(GCS_ASSETS)


@pytest.mark.skip_on_windows
def test_download_and_load_gcs_asset(tmp_path):

    local_path = downloads.load_public_napistu_asset(
        asset=GCS_ASSETS_NAMES.TEST_PATHWAY,
        subasset=GCS_SUBASSET_NAMES.SBML_DFS,
        data_dir=str(tmp_path),
    )

    expected_path = os.path.join(
        tmp_path, GCS_ASSETS_NAMES.TEST_PATHWAY, GCS_FILETYPES.SBML_DFS
    )
    assert local_path == expected_path
    assert os.path.isfile(local_path)


def test_validate_gcs_asset_version(gcs_assets):
    """Test _validate_gcs_asset_version function with various scenarios."""

    # Test with None version (should pass)
    downloads._validate_gcs_asset_version(
        GCS_ASSETS_NAMES.TEST_PATHWAY, None, gcs_assets
    )
    downloads._validate_gcs_asset_version(
        GCS_ASSETS_NAMES.HUMAN_CONSENSUS, None, gcs_assets
    )

    # Test with valid versions
    downloads._validate_gcs_asset_version(
        GCS_ASSETS_NAMES.TEST_PATHWAY, "20250901", gcs_assets
    )
    downloads._validate_gcs_asset_version(
        GCS_ASSETS_NAMES.HUMAN_CONSENSUS, "20250901", gcs_assets
    )
    downloads._validate_gcs_asset_version(
        GCS_ASSETS_NAMES.HUMAN_CONSENSUS_W_DISTANCES, "20250901", gcs_assets
    )

    # Test with invalid version for versioned asset
    with pytest.raises(
        ValueError,
        match="Version 'invalid_version' is not valid for asset 'test_pathway'. Valid versions are: 20250901",
    ):
        downloads._validate_gcs_asset_version(
            GCS_ASSETS_NAMES.TEST_PATHWAY, "invalid_version", gcs_assets
        )

    # Test with version for non-versioned asset
    with pytest.raises(
        ValueError, match="Asset 'reactome_members' does not support versioning"
    ):
        downloads._validate_gcs_asset_version(
            GCS_ASSETS_NAMES.REACTOME_MEMBERS, "20250901", gcs_assets
        )

    # Test with version for non-versioned asset
    with pytest.raises(
        ValueError, match="Asset 'reactome_xrefs' does not support versioning"
    ):
        downloads._validate_gcs_asset_version(
            GCS_ASSETS_NAMES.REACTOME_XREFS, "20250901", gcs_assets
        )


def test_validate_gcs_asset(gcs_assets):
    """Test _validate_gcs_asset function."""

    # Test with valid assets
    downloads._validate_gcs_asset(GCS_ASSETS_NAMES.TEST_PATHWAY, gcs_assets)
    downloads._validate_gcs_asset(GCS_ASSETS_NAMES.HUMAN_CONSENSUS, gcs_assets)
    downloads._validate_gcs_asset(GCS_ASSETS_NAMES.REACTOME_MEMBERS, gcs_assets)

    # Test with invalid asset
    with pytest.raises(
        ValueError,
        match="asset was invalid_asset and must be one of the keys in GCS_ASSETS.ASSETS",
    ):
        downloads._validate_gcs_asset("invalid_asset", gcs_assets)


def test_validate_gcs_subasset(gcs_assets):
    """Test _validate_gcs_subasset function."""

    # Test with valid subasset
    downloads._validate_gcs_subasset(
        GCS_ASSETS_NAMES.TEST_PATHWAY, GCS_SUBASSET_NAMES.SBML_DFS, gcs_assets
    )
    downloads._validate_gcs_subasset(
        GCS_ASSETS_NAMES.TEST_PATHWAY, GCS_SUBASSET_NAMES.NAPISTU_GRAPH, gcs_assets
    )

    # Test with None subasset for asset without subassets
    downloads._validate_gcs_subasset(
        GCS_ASSETS_NAMES.REACTOME_MEMBERS, None, gcs_assets
    )

    # Test with None subasset for asset with subassets (should fail)
    with pytest.raises(ValueError, match="subasset was None and must be one of"):
        downloads._validate_gcs_subasset(
            GCS_ASSETS_NAMES.TEST_PATHWAY, None, gcs_assets
        )

    # Test with invalid subasset
    with pytest.raises(
        ValueError,
        match="subasset, invalid_subasset, was not found in asset test_pathway",
    ):
        downloads._validate_gcs_subasset(
            GCS_ASSETS_NAMES.TEST_PATHWAY, "invalid_subasset", gcs_assets
        )

    # Test with subasset for asset without subassets (should warn but not fail)
    downloads._validate_gcs_subasset(
        GCS_ASSETS_NAMES.REACTOME_MEMBERS, "some_subasset", gcs_assets
    )
