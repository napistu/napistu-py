from __future__ import annotations

import pytest

from napistu import utils
from napistu.utils.path_utils import (
    copy_uri,
    get_source_base_and_path,
    initialize_dir,
    path_exists,
)
from tests.conftest import create_blob


def test_get_source_base_and_path_gcs():
    source_base, source_path = get_source_base_and_path(
        "gs://cpr-ml-dev-us-east1/cpr/tests/test_data/pw_index.tsv"
    )
    assert source_base == "gs://cpr-ml-dev-us-east1"
    assert source_path == "cpr/tests/test_data/pw_index.tsv"


def test_get_source_base_and_path_local():
    source_base, source_path = get_source_base_and_path("/test_data/bla/pw_index.tsv")
    assert source_base == "/test_data/bla"
    assert source_path == "pw_index.tsv"


def test_get_source_base_and_path_local_rel():
    source_base, source_path = get_source_base_and_path("./test_data/bla/pw_index.tsv")
    assert source_base == "./test_data/bla"
    assert source_path == "pw_index.tsv"


def test_get_source_base_and_path_local_direct():
    source_base, source_path = get_source_base_and_path("pw_index.tsv")
    assert source_base == ""
    assert source_path == "pw_index.tsv"


def test_initialize_dir_new(tmp_new_subdir):
    initialize_dir(tmp_new_subdir, overwrite=False)
    assert tmp_new_subdir.exists()


def test_initialize_dir_new_mock(mock_bucket_uri):
    test_uri = f"{mock_bucket_uri}/testdir"
    initialize_dir(test_uri, overwrite=False)
    path_exists(test_uri)


def test_initialize_dir_new_2_layers(tmp_new_subdir):
    target_sub_dir = tmp_new_subdir / "test_dir_2"
    initialize_dir(target_sub_dir, overwrite=False)
    assert target_sub_dir.exists()


def test_initialize_dir_new_2_layers_mock(mock_bucket_uri):
    test_uri = f"{mock_bucket_uri}/testdir/testdir2"
    initialize_dir(test_uri, overwrite=False)
    path_exists(test_uri)


def test_initialize_dir_existing(tmp_new_subdir):
    tmp_new_subdir.mkdir()

    test_file = tmp_new_subdir / "test_file"
    test_file.touch()

    with pytest.raises(FileExistsError):
        initialize_dir(tmp_new_subdir, overwrite=False)
    assert test_file.exists()

    initialize_dir(tmp_new_subdir, overwrite=True)
    assert test_file.exists() is False


def test_initialize_dir_existing_mock(mock_fs, mock_bucket_uri):
    # Get the bucket name from URI for filesystem operations
    bucket_name = mock_bucket_uri.replace("memory://", "")

    # create the file
    create_blob(mock_fs, f"{bucket_name}/testdir/file")

    test_uri = f"{mock_bucket_uri}/testdir"
    test_uri_file = f"{test_uri}/file"
    with pytest.raises(FileExistsError):
        initialize_dir(test_uri, overwrite=False)
        assert path_exists(test_uri_file)

    initialize_dir(test_uri, overwrite=True)
    assert path_exists(test_uri_file) is False


def test_path_exists(tmp_path, tmp_new_subdir):
    assert path_exists(tmp_path)
    assert path_exists(tmp_new_subdir) is False
    fn = tmp_path / "test.txt"
    assert path_exists(fn) is False
    fn.touch()
    assert path_exists(fn)
    assert path_exists(".")
    tmp_new_subdir.mkdir()
    assert path_exists(tmp_new_subdir)


def test_path_exists_mock(mock_fs, mock_bucket_uri):
    bucket_name = mock_bucket_uri.replace("memory://", "")

    assert path_exists(mock_bucket_uri)
    test_dir = "testdir"
    mock_test_dir_uri = f"{mock_bucket_uri}/{test_dir}"

    # Directory doesn't exist yet
    assert path_exists(mock_test_dir_uri) is False

    # Create directory by creating a file in it
    test_file = f"{test_dir}/test.txt"
    mock_test_file_uri = f"{mock_bucket_uri}/{test_file}"
    assert path_exists(mock_test_file_uri) is False

    # Create the file (this implicitly creates the directory)
    create_blob(mock_fs, f"{bucket_name}/{test_file}")
    assert path_exists(mock_test_file_uri)

    # Now the directory exists too
    assert path_exists(mock_test_dir_uri)


def test_copy_uri_file(tmp_path, tmp_new_subdir):
    basename = "test.txt"
    fn = tmp_path / basename
    fn.write_text("test")
    fn_out = tmp_new_subdir / "test_out.txt"
    copy_uri(fn, fn_out)
    assert fn_out.read_text() == "test"


def test_copy_uri_fol(tmp_path, tmp_new_subdir):
    tmp_new_subdir.mkdir()
    (tmp_new_subdir / "test").touch()

    out_dir = tmp_path / "out"
    out_file = out_dir / "test"
    copy_uri(tmp_new_subdir, out_dir, is_file=False)
    assert out_file.exists()


def test_copy_uri_file_mock(mock_bucket_uri, mock_bucket_subdir_uri):
    basename = "test.txt"
    content = "test"
    fn = f"{mock_bucket_uri}/{basename}"
    utils.save_pickle(fn, content)
    fn_out = f"{mock_bucket_subdir_uri}/{basename}"
    copy_uri(fn, fn_out)
    assert path_exists(fn_out)
    assert utils.load_pickle(fn_out) == content


def test_copy_uri_fol_mock(mock_bucket_uri, mock_bucket_subdir_uri):
    basename = "test.txt"
    content = "test"
    fn = f"{mock_bucket_subdir_uri}/{basename}"
    utils.save_pickle(fn, content)
    out_dir = f"{mock_bucket_uri}/new_dir"
    out_file = f"{out_dir}/{basename}"
    copy_uri(mock_bucket_subdir_uri, out_dir, is_file=False)
    assert path_exists(out_file)


@pytest.mark.parametrize(
    "source_fixture,dest_fixture",
    [
        ("tmp_path", "tmp_new_subdir"),  # local -> local
        ("mock_bucket_uri", "tmp_path"),  # memory -> local
        ("tmp_path", "mock_bucket_uri"),  # local -> memory
        ("mock_bucket_uri", "mock_bucket_subdir_uri"),  # memory -> memory
    ],
)
def test_copy_uri_file_cross_filesystem(source_fixture, dest_fixture, request):
    """Test copying files between different filesystem types."""
    source_base = request.getfixturevalue(source_fixture)
    dest_base = request.getfixturevalue(dest_fixture)

    # Setup source file - use pickle format for all to be consistent
    basename = "test_cross.pkl"
    content = "cross-filesystem test"

    # Create source file using pickle for consistency across all filesystems
    if hasattr(source_base, "exists"):  # It's a Path
        source_uri = str(source_base / basename)
    else:  # It's a URI string
        source_uri = f"{source_base}/{basename}"

    utils.save_pickle(source_uri, content)

    # Setup destination
    if hasattr(dest_base, "exists"):  # It's a Path
        dest_uri = str(dest_base / basename)
    else:  # It's a URI string
        dest_uri = f"{dest_base}/{basename}"

    # Copy and verify
    copy_uri(source_uri, dest_uri)
    assert path_exists(dest_uri)

    # Verify content using pickle for consistency
    assert utils.load_pickle(dest_uri) == content


@pytest.mark.parametrize(
    "source_fixture,dest_fixture",
    [
        ("tmp_path", "tmp_new_subdir"),  # local -> local
        ("mock_bucket_uri", "tmp_path"),  # memory -> local
        ("tmp_path", "mock_bucket_uri"),  # local -> memory
        (
            "mock_bucket_subdir_uri",
            "mock_bucket_uri",
        ),  # memory -> memory (different paths)
    ],
)
def test_copy_uri_directory_cross_filesystem(source_fixture, dest_fixture, request):
    """Test copying directories between different filesystem types."""
    source_base = request.getfixturevalue(source_fixture)
    dest_base = request.getfixturevalue(dest_fixture)

    # Setup source directory with files - use pickle for consistency
    files = ["file1.pkl", "file2.pkl", "subdir/file3.pkl"]
    content = "directory test"

    # Create source directory structure
    if hasattr(source_base, "exists"):  # It's a Path
        source_uri = str(source_base / "test_dir")
    else:  # It's a URI string
        source_uri = f"{source_base}/test_dir"

    # Create files using pickle for consistency
    for file in files:
        file_uri = f"{source_uri}/{file}"
        utils.save_pickle(file_uri, content)

    # Setup destination
    if hasattr(dest_base, "exists"):  # It's a Path
        dest_uri = str(dest_base / "dest_dir")
    else:  # It's a URI string
        dest_uri = f"{dest_base}/dest_dir"

    # Copy directory
    copy_uri(source_uri, dest_uri, is_file=False)

    # Verify all files were copied
    for file in files:
        file_dest = f"{dest_uri}/{file}"
        assert path_exists(file_dest)
        # Verify content
        assert utils.load_pickle(file_dest) == content
