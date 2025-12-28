"""Tests for GCS assets Pydantic models."""

from __future__ import annotations

from napistu.gcs.assets import GCSAsset, GCSAssets
from napistu.gcs.constants import (
    GCS_ASSET_DEFS,
    GCS_ASSETS,
    GCS_ASSETS_DEFS,
    GCS_ASSETS_NAMES,
    GCS_SUBASSET_NAMES,
)


def test_gcs_assets_from_dict():
    """Test GCSAssets.from_dict with GCS_ASSETS."""

    # Test that from_dict successfully creates a GCSAssets instance
    gcs_assets = GCSAssets.from_dict(GCS_ASSETS)

    # Verify it's a GCSAssets instance
    assert isinstance(gcs_assets, GCSAssets)

    # Verify project and bucket are set correctly using constants
    assert gcs_assets.project == getattr(GCS_ASSETS, GCS_ASSETS_DEFS.PROJECT.upper())
    assert gcs_assets.bucket == getattr(GCS_ASSETS, GCS_ASSETS_DEFS.BUCKET.upper())

    # Verify ASSETS attribute access works (backward compatibility)
    assert hasattr(gcs_assets, GCS_ASSETS_DEFS.ASSETS.upper())
    assert getattr(gcs_assets, GCS_ASSETS_DEFS.ASSETS.upper()) == gcs_assets.assets

    # Verify assets dictionary is not empty
    assert len(gcs_assets.assets) > 0

    # Verify all assets are GCSAsset instances
    for _, asset in gcs_assets.assets.items():
        assert isinstance(asset, GCSAsset)
        # Verify required fields exist using constants
        assert hasattr(asset, GCS_ASSET_DEFS.FILE)
        assert hasattr(asset, GCS_ASSET_DEFS.PUBLIC_URL)
        assert getattr(asset, GCS_ASSET_DEFS.PUBLIC_URL).startswith(
            ("http://", "https://")
        )

    # Verify specific asset exists and has correct structure
    assert GCS_ASSETS_NAMES.TEST_PATHWAY in gcs_assets.assets
    test_pathway_asset = gcs_assets.assets[GCS_ASSETS_NAMES.TEST_PATHWAY]
    assert (
        getattr(test_pathway_asset, GCS_ASSET_DEFS.FILE)
        == f"{GCS_ASSETS_NAMES.TEST_PATHWAY}.tar.gz"
    )
    assert getattr(test_pathway_asset, GCS_ASSET_DEFS.SUBASSETS) is not None
    assert GCS_SUBASSET_NAMES.SBML_DFS in getattr(
        test_pathway_asset, GCS_ASSET_DEFS.SUBASSETS
    )

    # Verify assets without subassets work correctly
    assert GCS_ASSETS_NAMES.REACTOME_MEMBERS in gcs_assets.assets
    reactome_asset = gcs_assets.assets[GCS_ASSETS_NAMES.REACTOME_MEMBERS]
    assert getattr(reactome_asset, GCS_ASSET_DEFS.SUBASSETS) is None
