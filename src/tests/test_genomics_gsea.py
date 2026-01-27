"""Tests for gene set enrichment analysis (GSEA) functionality."""

import pytest

from napistu.genomics.gsea import GenesetCollection, GmtsConfig
from napistu.utils.optional import import_gseapy

try:
    gp = import_gseapy()
except ImportError:
    pytest.skip("gseapy is not available", allow_module_level=True)


def test_add_gmts_h_all():
    """Test adding the 'h.all' gene set to a GenesetCollection."""
    collection = GenesetCollection(organismal_species="Homo sapiens")

    config = GmtsConfig(
        engine=gp.msigdb.Msigdb,
        categories=["h.all"],
        dbver="2023.2.Hs",
    )

    collection.add_gmts(gmts_config=config)

    assert "h.all" in collection.gmts
    assert isinstance(collection.gmts["h.all"], dict)
    assert len(collection.gmts["h.all"]) > 0
