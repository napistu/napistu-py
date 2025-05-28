from __future__ import annotations

import os
import pytest
import shutil

from napistu.gcs.downloads import load_public_napistu_asset


@pytest.mark.skip_on_windows
def test_download_and_load_gcs_asset():

    local_path = load_public_napistu_asset(
        asset="test_pathway", subasset="sbml_dfs", data_dir="/tmp"
    )

    assert local_path == "/tmp/test_pathway/sbml_dfs.pkl"

    # clean-up
    clean_up_dir = "/tmp/test_pathway"
    shutil.rmtree(clean_up_dir)

    if os.path.exists(clean_up_dir):
        raise Exception(f"Failed to clean up {clean_up_dir}")
