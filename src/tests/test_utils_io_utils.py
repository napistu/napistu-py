from __future__ import annotations

import gzip
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fs.tarfs import TarFS
from fs.zipfs import ZipFS
from google.cloud import storage
from pytest import fixture
from testcontainers.core.container import DockerContainer

from napistu import utils
from napistu.network.constants import DISTANCES
from napistu.utils.io_utils import (
    download_and_extract,
    load_parquet,
    load_pickle,
    pickle_cache,
    save_parquet,
    save_pickle,
)


@fixture(scope="session")
def gcs_storage():
    """A container running a GCS emulator"""
    with (
        DockerContainer("fsouza/fake-gcs-server:1.44")
        .with_bind_ports(4443, 4443)
        .with_command("-scheme http -backend memory")
    ) as gcs:
        os.environ["STORAGE_EMULATOR_HOST"] = "http://0.0.0.0:4443"
        yield gcs


@fixture
def gcs_bucket_name(gcs_storage):
    bucket_name = f"testbucket-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    return bucket_name


@fixture
def gcs_bucket(gcs_bucket_name):
    """A GCS bucket"""
    client = storage.Client()
    client.create_bucket(gcs_bucket_name)
    bucket = client.bucket(gcs_bucket_name)
    yield bucket
    bucket.delete(force=True)


@fixture
def gcs_bucket_uri(gcs_bucket, gcs_bucket_name):
    return f"gs://{gcs_bucket_name}"


@fixture
def gcs_bucket_subdir_uri(gcs_bucket_uri):
    return f"{gcs_bucket_uri}/testdir"


@fixture
def tmp_new_subdir(tmp_path):
    """An empty temporary directory"""
    return tmp_path / "test_dir"


def mock_targ_gz(url, tmp_file):
    with TarFS(tmp_file, write=True) as fol:
        with fol.open("test.txt", "w") as f:
            f.write("test")


def mock_zip(url, tmp_file):
    with ZipFS(tmp_file, write=True) as fol:
        with fol.open("test.txt", "w") as f:
            f.write("test")


def mock_gz(url, tmp_file):
    with gzip.open(tmp_file, mode="wt") as f:
        f.write("test")


@patch("napistu.utils.io_utils.download_wget", side_effect=mock_targ_gz)
def test_download_and_extract_tar_gz(mock_download, tmp_new_subdir):
    download_and_extract(
        url="http://asdf/bla.tar.gz",
        output_dir_path=tmp_new_subdir,
        download_method="wget",
    )
    assert (tmp_new_subdir / "test.txt").exists()


@patch("napistu.utils.io_utils.download_ftp", side_effect=mock_zip)
def test_download_and_extract_zip(mock_download, tmp_new_subdir):
    download_and_extract(
        url="http://asdf/bla.txt.zip",
        output_dir_path=tmp_new_subdir,
        download_method="ftp",
    )
    assert (tmp_new_subdir / "test.txt").exists()


@patch("napistu.utils.io_utils.download_wget", side_effect=mock_gz)
def test_download_and_extract_gz(mock_download, tmp_new_subdir):
    download_and_extract(
        url="http://asdf/bla.txt.gz",
        output_dir_path=tmp_new_subdir,
        download_method="wget",
    )
    assert (tmp_new_subdir / "bla.txt").exists()


def test_download_and_extract_invalid_method(tmp_new_subdir):
    with pytest.raises(ValueError):
        download_and_extract(
            url="http://asdf/bla.txt.zip",
            output_dir_path=tmp_new_subdir,
            download_method="bla",
        )


@patch("napistu.utils.io_utils.download_ftp", side_effect=mock_zip)
def test_download_and_extract_invalid_ext(mock_download, tmp_new_subdir):
    with pytest.raises(ValueError):
        download_and_extract(
            url="http://asdf/bla.txt.zipper",
            output_dir_path=tmp_new_subdir,
            download_method="ftp",
        )


@pytest.mark.unix_only
def test_save_load_pickle_existing_folder(tmp_path):
    fn = tmp_path / "test.pkl"
    payload = "test"
    save_pickle(fn, payload)
    assert fn.exists()
    assert load_pickle(fn) == payload


@pytest.mark.skip_on_windows
def test_save_load_pickle_new_folder(tmp_new_subdir):
    fn = tmp_new_subdir / "test.pkl"
    payload = "test"
    save_pickle(fn, payload)
    assert fn.exists()
    assert load_pickle(fn) == payload


@pytest.mark.unix_only
def test_save_load_pickle_existing_folder_gcs(gcs_bucket_uri):
    fn = f"{gcs_bucket_uri}/test.pkl"
    payload = "test"
    save_pickle(fn, payload)
    assert utils.path_exists(fn)
    assert load_pickle(fn) == payload


@pytest.mark.unix_only
def test_save_load_pickle_new_folder_gcs(gcs_bucket_subdir_uri):
    fn = f"{gcs_bucket_subdir_uri}/test.pkl"
    payload = "test"
    save_pickle(fn, payload)
    assert utils.path_exists(fn)
    assert load_pickle(fn) == payload


@pytest.mark.skip_on_windows
def test_pickle_cache(tmp_path):
    fn = tmp_path / "test.pkl"

    mock = Mock()
    result = "test"

    @pickle_cache(fn)
    def test_func():
        mock()
        return result

    test_func()
    r = test_func()
    assert r == result
    # only called once as second
    # call should be cached
    assert mock.call_count == 1


def test_parquet_save_load():
    """Test that save_parquet and load_parquet work correctly."""
    # Create test DataFrame
    original_df = pd.DataFrame(
        {
            DISTANCES.SC_ID_ORIGIN: ["A", "B", "C"],
            DISTANCES.SC_ID_DEST: ["B", "C", "A"],
            DISTANCES.PATH_LENGTH: [1, 2, 3],
            DISTANCES.PATH_WEIGHT: [0.1, 0.5, 0.8],
            "has_connection": [True, False, True],
        }
    )

    # Write and read using temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.parquet"
        save_parquet(original_df, file_path)
        result_df = load_parquet(file_path)

        # Verify they're identical
        pd.testing.assert_frame_equal(original_df, result_df)
