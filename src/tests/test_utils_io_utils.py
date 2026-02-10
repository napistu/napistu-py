from __future__ import annotations

import gzip
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fs.tarfs import TarFS
from fs.zipfs import ZipFS

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


# Pickle tests - now cross-platform with both local and memory filesystems
def test_save_load_pickle_existing_folder(tmp_path):
    """Test pickle with existing local folder."""
    fn = tmp_path / "test.pkl"
    payload = "test"
    save_pickle(fn, payload)
    assert fn.exists()
    assert load_pickle(fn) == payload


def test_save_load_pickle_new_folder(tmp_new_subdir):
    """Test pickle with new local folder - fsspec creates it automatically."""
    fn = tmp_new_subdir / "test.pkl"
    payload = "test"
    save_pickle(fn, payload)
    assert fn.exists()
    assert load_pickle(fn) == payload


def test_save_load_pickle_existing_folder_mock(mock_bucket_uri):
    """Test pickle with existing memory filesystem folder."""
    fn = f"{mock_bucket_uri}/test.pkl"
    payload = "test"
    save_pickle(fn, payload)
    assert utils.path_exists(fn)
    assert load_pickle(fn) == payload


def test_save_load_pickle_new_folder_mock(mock_bucket_subdir_uri):
    """Test pickle with new memory filesystem folder - created automatically."""
    fn = f"{mock_bucket_subdir_uri}/test.pkl"
    payload = "test"
    save_pickle(fn, payload)
    assert utils.path_exists(fn)
    assert load_pickle(fn) == payload


def test_pickle_cache(tmp_path):
    """Test pickle caching decorator works correctly."""
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


# Parquet tests - now with parametrization for cross-filesystem testing
def test_parquet_save_load_local():
    """Test that save_parquet and load_parquet work with local files."""
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


def test_parquet_save_load_mock(mock_bucket_uri):
    """Test that save_parquet and load_parquet work with memory filesystem."""
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

    # Write and read using memory filesystem
    file_uri = f"{mock_bucket_uri}/test.parquet"
    save_parquet(original_df, file_uri)
    result_df = load_parquet(file_uri)

    # Verify they're identical
    pd.testing.assert_frame_equal(original_df, result_df)


@pytest.mark.parametrize("filesystem_fixture", ["tmp_path", "mock_bucket_uri"])
def test_parquet_roundtrip(filesystem_fixture, request):
    """Test parquet save/load works across different filesystems."""
    filesystem_base = request.getfixturevalue(filesystem_fixture)

    # Create test DataFrame with various types
    original_df = pd.DataFrame(
        {
            "string_col": ["A", "B", "C"],
            "int_col": [1, 2, 3],
            "float_col": [0.1, 0.5, 0.8],
            "bool_col": [True, False, True],
        }
    )

    # Setup file path/URI
    if hasattr(filesystem_base, "exists"):  # Path
        file_path = filesystem_base / "roundtrip.parquet"
    else:  # URI string
        file_path = f"{filesystem_base}/roundtrip.parquet"

    # Save and load
    save_parquet(original_df, file_path)
    result_df = load_parquet(file_path)

    # Verify
    pd.testing.assert_frame_equal(original_df, result_df)


@pytest.mark.parametrize("compression", ["snappy", "gzip", "brotli"])
def test_parquet_compression(tmp_path, compression):
    """Test that different compression algorithms work."""
    original_df = pd.DataFrame(
        {
            "col1": ["A"] * 100,  # Repeated values compress well
            "col2": range(100),
        }
    )

    file_path = tmp_path / f"test_{compression}.parquet"
    save_parquet(original_df, file_path, compression=compression)
    result_df = load_parquet(file_path)

    pd.testing.assert_frame_equal(original_df, result_df)


@pytest.mark.parametrize("filesystem_fixture", ["tmp_path", "mock_bucket_uri"])
def test_pickle_complex_objects(filesystem_fixture, request):
    """Test that complex Python objects pickle correctly across filesystems."""
    filesystem_base = request.getfixturevalue(filesystem_fixture)

    # Complex nested object
    payload = {
        "nested_dict": {"a": 1, "b": [2, 3, 4]},
        "tuple": (1, 2, 3),
        "set": {1, 2, 3},
        "dataframe": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
    }

    # Setup file path/URI
    if hasattr(filesystem_base, "exists"):  # Path
        file_path = filesystem_base / "complex.pkl"
    else:  # URI string
        file_path = f"{filesystem_base}/complex.pkl"

    # Save and load
    save_pickle(file_path, payload)
    result = load_pickle(file_path)

    # Verify
    assert result["nested_dict"] == payload["nested_dict"]
    assert result["tuple"] == payload["tuple"]
    assert result["set"] == payload["set"]
    pd.testing.assert_frame_equal(result["dataframe"], payload["dataframe"])
