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


@pytest.mark.unix_only
def test_initialize_dir_new_gcs(gcs_bucket_uri):
    test_uri = f"{gcs_bucket_uri}/testdir"
    initialize_dir(test_uri, overwrite=False)
    path_exists(test_uri)


def test_initialize_dir_new_2_layers(tmp_new_subdir):
    target_sub_dir = tmp_new_subdir / "test_dir_2"
    initialize_dir(target_sub_dir, overwrite=False)
    assert target_sub_dir.exists()


@pytest.mark.unix_only
def test_initialize_dir_new_2_layers_gcs(gcs_bucket_uri):
    test_uri = f"{gcs_bucket_uri}/testdir/testdir2"
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


@pytest.mark.unix_only
def test_initialize_dir_existing_gcs(gcs_bucket, gcs_bucket_uri):
    # create the file
    create_blob(gcs_bucket, "testdir/file")
    # This is a drawback of the current implementation - folders are only
    # recognized if they have a marker file.
    create_blob(gcs_bucket, "testdir/")

    test_uri = f"{gcs_bucket_uri}/testdir"
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


@pytest.mark.unix_only
def test_path_exists_gcs(gcs_bucket, gcs_bucket_uri):
    assert path_exists(gcs_bucket_uri)
    test_dir = "testdir"
    gcs_test_dir_uri = f"{gcs_bucket_uri}/{test_dir}"
    assert path_exists(gcs_test_dir_uri) is False
    # Create the marker file for the directory, such that it 'exists'
    create_blob(gcs_bucket, f"{test_dir}/")
    assert path_exists(gcs_test_dir_uri)

    # Test if files exists
    test_file = f"{test_dir}/test.txt"
    gcs_test_file_uri = f"{gcs_bucket_uri}/{test_file}"
    assert path_exists(gcs_test_file_uri) is False
    # create the file
    create_blob(gcs_bucket, test_file)
    assert path_exists(gcs_test_file_uri)


@pytest.mark.skip_on_windows
def test_copy_uri_file(tmp_path, tmp_new_subdir):
    basename = "test.txt"
    fn = tmp_path / basename
    fn.write_text("test")
    fn_out = tmp_new_subdir / "test_out.txt"
    copy_uri(fn, fn_out)
    assert fn_out.read_text() == "test"


@pytest.mark.skip_on_windows
def test_copy_uri_fol(tmp_path, tmp_new_subdir):
    tmp_new_subdir.mkdir()
    (tmp_new_subdir / "test").touch()

    out_dir = tmp_path / "out"
    out_file = out_dir / "test"
    copy_uri(tmp_new_subdir, out_dir, is_file=False)
    assert out_file.exists()


@pytest.mark.unix_only
def test_copy_uri_file_gcs(gcs_bucket_uri, gcs_bucket_subdir_uri):
    basename = "test.txt"
    content = "test"
    fn = f"{gcs_bucket_uri}/{basename}"
    utils.save_pickle(fn, content)
    fn_out = f"{gcs_bucket_subdir_uri}/{basename}"
    copy_uri(fn, fn_out)
    assert path_exists(fn_out)
    assert utils.load_pickle(fn_out) == content


@pytest.mark.unix_only
def test_copy_uri_fol_gcs(gcs_bucket_uri, gcs_bucket_subdir_uri):
    basename = "test.txt"
    content = "test"
    fn = f"{gcs_bucket_subdir_uri}/{basename}"
    utils.save_pickle(fn, content)
    out_dir = f"{gcs_bucket_uri}/new_dir"
    out_file = f"{out_dir}/{basename}"
    copy_uri(gcs_bucket_subdir_uri, out_dir, is_file=False)
    assert path_exists(out_file)
