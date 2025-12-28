"""Pydantic models for GCS assets configuration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from napistu.gcs.constants import GCS_ASSETS_DEFS


class GCSAsset(BaseModel):
    """Pydantic model for a single GCS asset configuration."""

    file: str
    subassets: Optional[dict[str, str]] = None
    public_url: str
    versions: Optional[dict[str, str]] = None

    @field_validator("public_url")
    @classmethod
    def validate_public_url(cls, v: str) -> str:
        """Validate that public_url is a valid URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"public_url must start with http:// or https://, got: {v}"
            )
        return v

    @field_validator("versions")
    @classmethod
    def validate_versions(cls, v: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
        """Validate that all version URLs are valid."""
        if v is not None:
            for version, url in v.items():
                if not url.startswith(("http://", "https://")):
                    raise ValueError(
                        f"Version URL for '{version}' must start with http:// or https://, got: {url}"
                    )
        return v

    def __getitem__(self, key: str):
        """Support dictionary-style access for backward compatibility."""
        # Map GCS_ASSETS_DEFS keys to model attributes
        if key == "file":
            return self.file
        if key == "subassets":
            return self.subassets
        if key == "public_url":
            return self.public_url
        if key == "versions":
            return self.versions
        # Fall back to attribute access
        return getattr(self, key)


class GCSAssets(BaseModel):
    """Pydantic model for GCS assets configuration."""

    project: str
    bucket: str
    assets: dict[str, GCSAsset]

    @field_validator("assets")
    @classmethod
    def validate_assets(cls, v: dict[str, GCSAsset]) -> dict[str, GCSAsset]:
        """Validate that assets dictionary is not empty."""
        if not v:
            raise ValueError("assets dictionary cannot be empty")
        return v

    def __getattr__(self, name: str):
        """Support SimpleNamespace-style access for backward compatibility."""
        # Map uppercase names to lowercase attributes
        if name == "ASSETS":
            return self.assets
        if name == "PROJECT":
            return self.project
        if name == "BUCKET":
            return self.bucket
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    @classmethod
    def from_dict(cls, assets_dict: dict | SimpleNamespace) -> GCSAssets:
        """
        Create a GCSAssets instance from a dictionary or SimpleNamespace.

        Parameters
        ----------
        assets_dict : dict | SimpleNamespace
            Dictionary or SimpleNamespace containing 'PROJECT'/'project', 'BUCKET'/'bucket',
            and 'ASSETS'/'assets' keys/attributes.
            The 'ASSETS' key should map to a dictionary of asset names to asset configurations.

        Returns
        -------
        GCSAssets
            A validated GCSAssets instance.

        Examples
        --------
        >>> from napistu.gcs.constants import GCS_ASSETS
        >>> gcs_assets = GCSAssets.from_dict(GCS_ASSETS)
        """
        # Handle SimpleNamespace input (uses uppercase attribute names)
        if isinstance(assets_dict, SimpleNamespace):
            project = getattr(assets_dict, GCS_ASSETS_DEFS.PROJECT.upper(), None)
            bucket = getattr(assets_dict, GCS_ASSETS_DEFS.BUCKET.upper(), None)
            assets = getattr(assets_dict, GCS_ASSETS_DEFS.ASSETS.upper(), None)
        else:
            # Handle dict input (supports both uppercase and lowercase keys)
            project = assets_dict.get(
                GCS_ASSETS_DEFS.PROJECT.upper()
            ) or assets_dict.get(GCS_ASSETS_DEFS.PROJECT)
            bucket = assets_dict.get(GCS_ASSETS_DEFS.BUCKET.upper()) or assets_dict.get(
                GCS_ASSETS_DEFS.BUCKET
            )
            assets = assets_dict.get(GCS_ASSETS_DEFS.ASSETS.upper()) or assets_dict.get(
                GCS_ASSETS_DEFS.ASSETS
            )

        if project is None or bucket is None or assets is None:
            raise ValueError(
                f"assets_dict must contain '{GCS_ASSETS_DEFS.PROJECT}'/'{GCS_ASSETS_DEFS.PROJECT.upper()}', "
                f"'{GCS_ASSETS_DEFS.BUCKET}'/'{GCS_ASSETS_DEFS.BUCKET.upper()}', and "
                f"'{GCS_ASSETS_DEFS.ASSETS}'/'{GCS_ASSETS_DEFS.ASSETS.upper()}' keys/attributes"
            )

        # Convert asset dicts to GCSAsset instances
        validated_assets = {}
        for asset_name, asset_data in assets.items():
            if isinstance(asset_data, GCSAsset):
                validated_assets[asset_name] = asset_data
            else:
                validated_assets[asset_name] = GCSAsset(**asset_data)

        return cls(project=project, bucket=bucket, assets=validated_assets)

    model_config = ConfigDict(
        # Allow access via attribute names (e.g., gcs_assets.ASSETS)
        # and also support dictionary-style access
        populate_by_name=True,
    )
