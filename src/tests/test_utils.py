from __future__ import annotations

import gzip
import os
from datetime import datetime
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from napistu import utils
from fs.tarfs import TarFS
from fs.zipfs import ZipFS
from google.cloud import storage
from pytest import fixture
from testcontainers.core.container import DockerContainer

from napistu.utils import drop_extra_cols


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


def create_blob(bucket, blob_name, content=b"test"):
    # create the marker file
    bucket.blob(blob_name).upload_from_string(content)


def test_get_source_base_and_path_gcs():
    source_base, source_path = utils.get_source_base_and_path(
        "gs://cpr-ml-dev-us-east1/cpr/tests/test_data/pw_index.tsv"
    )
    assert source_base == "gs://cpr-ml-dev-us-east1"
    assert source_path == "cpr/tests/test_data/pw_index.tsv"


def test_get_source_base_and_path_local():
    source_base, source_path = utils.get_source_base_and_path(
        "/test_data/bla/pw_index.tsv"
    )
    assert source_base == "/test_data/bla"
    assert source_path == "pw_index.tsv"


def test_get_source_base_and_path_local_rel():
    source_base, source_path = utils.get_source_base_and_path(
        "./test_data/bla/pw_index.tsv"
    )
    assert source_base == "./test_data/bla"
    assert source_path == "pw_index.tsv"


def test_get_source_base_and_path_local_direct():
    source_base, source_path = utils.get_source_base_and_path("pw_index.tsv")
    assert source_base == ""
    assert source_path == "pw_index.tsv"


def test_initialize_dir_new(tmp_new_subdir):
    utils.initialize_dir(tmp_new_subdir, overwrite=False)
    assert tmp_new_subdir.exists()


@pytest.mark.unix_only
def test_initialize_dir_new_gcs(gcs_bucket_uri):
    test_uri = f"{gcs_bucket_uri}/testdir"
    utils.initialize_dir(test_uri, overwrite=False)
    utils.path_exists(test_uri)


def test_initialize_dir_new_2_layers(tmp_new_subdir):
    target_sub_dir = tmp_new_subdir / "test_dir_2"
    utils.initialize_dir(target_sub_dir, overwrite=False)
    assert target_sub_dir.exists()


@pytest.mark.unix_only
def test_initialize_dir_new_2_layers_gcs(gcs_bucket_uri):
    test_uri = f"{gcs_bucket_uri}/testdir/testdir2"
    utils.initialize_dir(test_uri, overwrite=False)
    utils.path_exists(test_uri)


def test_initialize_dir_existing(tmp_new_subdir):
    tmp_new_subdir.mkdir()

    test_file = tmp_new_subdir / "test_file"
    test_file.touch()

    with pytest.raises(FileExistsError):
        utils.initialize_dir(tmp_new_subdir, overwrite=False)
    assert test_file.exists()

    utils.initialize_dir(tmp_new_subdir, overwrite=True)
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
        utils.initialize_dir(test_uri, overwrite=False)
        assert utils.path_exists(test_uri_file)

    utils.initialize_dir(test_uri, overwrite=True)
    assert utils.path_exists(test_uri_file) is False


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


@patch("napistu.utils.download_wget", side_effect=mock_targ_gz)
def test_download_and_extract_tar_gz(mock_download, tmp_new_subdir):
    utils.download_and_extract(
        url="http://asdf/bla.tar.gz",
        output_dir_path=tmp_new_subdir,
        download_method="wget",
    )
    assert (tmp_new_subdir / "test.txt").exists()


@patch("napistu.utils.download_ftp", side_effect=mock_zip)
def test_download_and_extract_zip(mock_download, tmp_new_subdir):
    utils.download_and_extract(
        url="http://asdf/bla.txt.zip",
        output_dir_path=tmp_new_subdir,
        download_method="ftp",
    )
    assert (tmp_new_subdir / "test.txt").exists()


@patch("napistu.utils.download_wget", side_effect=mock_gz)
def test_download_and_extract_gz(mock_download, tmp_new_subdir):
    utils.download_and_extract(
        url="http://asdf/bla.txt.gz",
        output_dir_path=tmp_new_subdir,
        download_method="wget",
    )
    assert (tmp_new_subdir / "bla.txt").exists()


def test_download_and_extract_invalid_method(tmp_new_subdir):
    with pytest.raises(ValueError):
        utils.download_and_extract(
            url="http://asdf/bla.txt.zip",
            output_dir_path=tmp_new_subdir,
            download_method="bla",
        )


@patch("napistu.utils.download_ftp", side_effect=mock_zip)
def test_download_and_extract_invalid_ext(mock_download, tmp_new_subdir):
    with pytest.raises(ValueError):
        utils.download_and_extract(
            url="http://asdf/bla.txt.zipper",
            output_dir_path=tmp_new_subdir,
            download_method="ftp",
        )


def test_path_exists(tmp_path, tmp_new_subdir):
    assert utils.path_exists(tmp_path)
    assert utils.path_exists(tmp_new_subdir) is False
    fn = tmp_path / "test.txt"
    assert utils.path_exists(fn) is False
    fn.touch()
    assert utils.path_exists(fn)
    assert utils.path_exists(".")
    tmp_new_subdir.mkdir()
    assert utils.path_exists(tmp_new_subdir)


@pytest.mark.unix_only
def test_path_exists_gcs(gcs_bucket, gcs_bucket_uri):
    assert utils.path_exists(gcs_bucket_uri)
    test_dir = "testdir"
    gcs_test_dir_uri = f"{gcs_bucket_uri}/{test_dir}"
    assert utils.path_exists(gcs_test_dir_uri) is False
    # Create the marker file for the directory, such that it 'exists'
    create_blob(gcs_bucket, f"{test_dir}/")
    assert utils.path_exists(gcs_test_dir_uri)

    # Test if files exists
    test_file = f"{test_dir}/test.txt"
    gcs_test_file_uri = f"{gcs_bucket_uri}/{test_file}"
    assert utils.path_exists(gcs_test_file_uri) is False
    # create the file
    create_blob(gcs_bucket, test_file)
    assert utils.path_exists(gcs_test_file_uri)


@pytest.mark.unix_only
def test_save_load_pickle_existing_folder(tmp_path):
    fn = tmp_path / "test.pkl"
    payload = "test"
    utils.save_pickle(fn, payload)
    assert fn.exists()
    assert utils.load_pickle(fn) == payload


@pytest.mark.skip_on_windows
def test_save_load_pickle_new_folder(tmp_new_subdir):
    fn = tmp_new_subdir / "test.pkl"
    payload = "test"
    utils.save_pickle(fn, payload)
    assert fn.exists()
    assert utils.load_pickle(fn) == payload


@pytest.mark.unix_only
def test_save_load_pickle_existing_folder_gcs(gcs_bucket_uri):
    fn = f"{gcs_bucket_uri}/test.pkl"
    payload = "test"
    utils.save_pickle(fn, payload)
    assert utils.path_exists(fn)
    assert utils.load_pickle(fn) == payload


@pytest.mark.unix_only
def test_save_load_pickle_new_folder_gcs(gcs_bucket_subdir_uri):
    fn = f"{gcs_bucket_subdir_uri}/test.pkl"
    payload = "test"
    utils.save_pickle(fn, payload)
    assert utils.path_exists(fn)
    assert utils.load_pickle(fn) == payload


@pytest.mark.skip_on_windows
def test_copy_uri_file(tmp_path, tmp_new_subdir):
    basename = "test.txt"
    fn = tmp_path / basename
    fn.write_text("test")
    fn_out = tmp_new_subdir / "test_out.txt"
    utils.copy_uri(fn, fn_out)
    assert fn_out.read_text() == "test"


@pytest.mark.skip_on_windows
def test_copy_uri_fol(tmp_path, tmp_new_subdir):
    tmp_new_subdir.mkdir()
    (tmp_new_subdir / "test").touch()

    out_dir = tmp_path / "out"
    out_file = out_dir / "test"
    utils.copy_uri(tmp_new_subdir, out_dir, is_file=False)
    assert out_file.exists()


@pytest.mark.unix_only
def test_copy_uri_file_gcs(gcs_bucket_uri, gcs_bucket_subdir_uri):
    basename = "test.txt"
    content = "test"
    fn = f"{gcs_bucket_uri}/{basename}"
    utils.save_pickle(fn, content)
    fn_out = f"{gcs_bucket_subdir_uri}/{basename}"
    utils.copy_uri(fn, fn_out)
    assert utils.path_exists(fn_out)
    assert utils.load_pickle(fn_out) == content


@pytest.mark.unix_only
def test_copy_uri_fol_gcs(gcs_bucket_uri, gcs_bucket_subdir_uri):
    basename = "test.txt"
    content = "test"
    fn = f"{gcs_bucket_subdir_uri}/{basename}"
    utils.save_pickle(fn, content)
    out_dir = f"{gcs_bucket_uri}/new_dir"
    out_file = f"{out_dir}/{basename}"
    utils.copy_uri(gcs_bucket_subdir_uri, out_dir, is_file=False)
    assert utils.path_exists(out_file)


@pytest.mark.skip_on_windows
def test_pickle_cache(tmp_path):
    fn = tmp_path / "test.pkl"

    mock = Mock()
    result = "test"

    @utils.pickle_cache(fn)
    def test_func():
        mock()
        return result

    test_func()
    r = test_func()
    assert r == result
    # only called once as second
    # call should be cached
    assert mock.call_count == 1


def test_extract_regex():
    assert utils.extract_regex_search("ENS[GT][0-9]+", "ENST0005") == "ENST0005"
    assert utils.extract_regex_search("ENS[GT]([0-9]+)", "ENST0005", 1) == "0005"
    with pytest.raises(ValueError):
        utils.extract_regex_search("ENS[GT][0-9]+", "ENSA0005")

    assert utils.extract_regex_match(".*type=([a-zA-Z]+).*", "Ltype=abcd5") == "abcd"
    # use for formatting identifiers
    assert utils.extract_regex_match("^([a-zA-Z]+)_id$", "sc_id") == "sc"
    with pytest.raises(ValueError):
        utils.extract_regex_match(".*type=[a-zA-Z]+.*", "Ltype=abcd5")


def test_match_pd_vars():
    a_series = pd.Series({"foo": 1, "bar": 2})
    a_dataframe = pd.DataFrame({"foo": ["a", "b"], "bar": [1, 2]})

    assert utils.match_pd_vars(a_series, {"foo", "bar"}).are_present
    assert not utils.match_pd_vars(a_series, {"baz"}).are_present
    assert utils.match_pd_vars(a_dataframe, {"foo", "bar"}).are_present
    assert not utils.match_pd_vars(a_dataframe, {"baz"}).are_present


def test_ensure_pd_df():
    source_df = pd.DataFrame({"a": "b"}, index=[0])
    source_series = pd.Series({"a": "b"}).rename(0)

    converted_series = utils.ensure_pd_df(source_series)

    assert isinstance(utils.ensure_pd_df(source_df), pd.DataFrame)
    assert isinstance(converted_series, pd.DataFrame)
    assert all(converted_series.index == source_df.index)
    assert all(converted_series.columns == source_df.columns)
    assert all(converted_series == source_df)


def test_format_identifiers_as_edgelist():
    DEGEN_EDGELIST_DF_1 = pd.DataFrame(
        {
            "ind1": [0, 0, 1, 1, 1, 1],
            "ind2": ["a", "a", "b", "b", "c", "d"],
            "ont": ["X", "X", "X", "Y", "Y", "Y"],
            "val": ["A", "B", "C", "D", "D", "E"],
        }
    ).set_index(["ind1", "ind2"])

    DEGEN_EDGELIST_DF_2 = pd.DataFrame(
        {
            "ind": ["a", "a", "b", "b", "c", "d"],
            "ont": ["X", "X", "X", "Y", "Y", "Y"],
            "val": ["A", "B", "C", "D", "D", "E"],
        }
    ).set_index("ind")

    edgelist_df = utils.format_identifiers_as_edgelist(
        DEGEN_EDGELIST_DF_1, ["ont", "val"]
    )
    assert edgelist_df["ind"].iloc[0] == "ind_0_a"
    assert edgelist_df["id"].iloc[0] == "id_X_A"

    edgelist_df = utils.format_identifiers_as_edgelist(DEGEN_EDGELIST_DF_1, ["val"])
    assert edgelist_df["ind"].iloc[0] == "ind_0_a"
    assert edgelist_df["id"].iloc[0] == "id_A"

    edgelist_df = utils.format_identifiers_as_edgelist(
        DEGEN_EDGELIST_DF_2, ["ont", "val"]
    )
    assert edgelist_df["ind"].iloc[0] == "ind_a"
    assert edgelist_df["id"].iloc[0] == "id_X_A"

    with pytest.raises(ValueError):
        utils.format_identifiers_as_edgelist(
            DEGEN_EDGELIST_DF_2.reset_index(drop=True), ["ont", "val"]
        )


def test_find_weakly_connected_subgraphs():
    DEGEN_EDGELIST_DF_2 = pd.DataFrame(
        {
            "ind": ["a", "a", "b", "b", "c", "d"],
            "ont": ["X", "X", "X", "Y", "Y", "Y"],
            "val": ["A", "B", "C", "D", "D", "E"],
        }
    ).set_index("ind")

    edgelist_df = utils.format_identifiers_as_edgelist(
        DEGEN_EDGELIST_DF_2, ["ont", "val"]
    )
    edgelist = edgelist_df[["ind", "id"]]

    connected_indices = utils.find_weakly_connected_subgraphs(edgelist[["ind", "id"]])
    assert all(connected_indices["cluster"] == [0, 1, 1, 2])


def test_style_df():
    np.random.seed(0)
    simple_df = pd.DataFrame(np.random.randn(20, 4), columns=["A", "B", "C", "D"])
    simple_df.index.name = "foo"

    multiindexed_df = (
        pd.DataFrame(
            {
                "category": ["foo", "foo", "foo", "bar", "bar", "bar"],
                "severity": ["major", "minor", "minor", "major", "major", "minor"],
            }
        )
        .assign(message="stuff")
        .groupby(["category", "severity"])
        .count()
    )

    # style a few pd.DataFrames
    isinstance(utils.style_df(simple_df), pd.io.formats.style.Styler)
    isinstance(
        utils.style_df(simple_df, headers=None, hide_index=True),
        pd.io.formats.style.Styler,
    )
    isinstance(
        utils.style_df(simple_df, headers=["a", "b", "c", "d"], hide_index=True),
        pd.io.formats.style.Styler,
    )
    isinstance(utils.style_df(multiindexed_df), pd.io.formats.style.Styler)


def test_score_nameness():
    assert utils.score_nameness("p53") == 23
    assert utils.score_nameness("ENSG0000001") == 56
    assert utils.score_nameness("pyruvate kinase") == 15


def test_click_str_to_list():
    assert utils.click_str_to_list("['foo', bar]") == ["foo", "bar"]
    with pytest.raises(ValueError):
        utils.click_str_to_list("foo")


def test_drop_extra_cols():
    """Test the _drop_extra_cols function for removing and reordering columns."""
    # Setup test DataFrames
    df_in = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})

    df_out = pd.DataFrame(
        {
            "col2": [10, 11, 12],
            "col3": [13, 14, 15],
            "col4": [16, 17, 18],  # Extra column that should be dropped
            "col1": [19, 20, 21],  # Different order than df_in
        }
    )

    # Test basic functionality without always_include
    result = drop_extra_cols(df_in, df_out)

    # Check that extra column was dropped
    assert "col4" not in result.columns

    # Check that columns are in the same order as df_in
    assert list(result.columns) == list(df_in.columns)

    # Check that values are preserved
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {"col1": [19, 20, 21], "col2": [10, 11, 12], "col3": [13, 14, 15]}
        )[
            list(df_in.columns)
        ],  # Ensure same column order
    )

    # Test with always_include
    result_with_include = drop_extra_cols(df_in, df_out, always_include=["col4"])

    # Check that col4 is retained and appears at the end
    assert list(result_with_include.columns) == list(df_in.columns) + ["col4"]
    assert result_with_include["col4"].equals(df_out["col4"])

    # Test with always_include containing non-existent column
    result_non_existent = drop_extra_cols(
        df_in, df_out, always_include=["col4", "col5"]
    )
    assert list(result_non_existent.columns) == list(df_in.columns) + ["col4"]

    # Test with always_include containing column from df_in
    result_overlap = drop_extra_cols(df_in, df_out, always_include=["col1", "col4"])
    assert list(result_overlap.columns) == list(df_in.columns) + ["col4"]

    # Test with no overlapping columns but some in always_include
    df_out_no_overlap = pd.DataFrame({"col4": [1, 2, 3], "col5": [4, 5, 6]})
    result_no_overlap = drop_extra_cols(df_in, df_out_no_overlap)
    assert result_no_overlap.empty
    assert list(result_no_overlap.columns) == []

    result_no_overlap_with_include = drop_extra_cols(
        df_in, df_out_no_overlap, always_include=["col4"]
    )
    assert list(result_no_overlap_with_include.columns) == ["col4"]
    assert result_no_overlap_with_include["col4"].equals(df_out_no_overlap["col4"])

    # Test with subset of columns
    df_out_subset = pd.DataFrame(
        {"col1": [1, 2, 3], "col3": [7, 8, 9], "col4": [10, 11, 12]}
    )
    result_subset = drop_extra_cols(df_in, df_out_subset)
    assert list(result_subset.columns) == ["col1", "col3"]
    pd.testing.assert_frame_equal(result_subset, df_out_subset[["col1", "col3"]])

    result_subset_with_include = drop_extra_cols(
        df_in, df_out_subset, always_include=["col4"]
    )
    assert list(result_subset_with_include.columns) == ["col1", "col3", "col4"]
    pd.testing.assert_frame_equal(
        result_subset_with_include, df_out_subset[["col1", "col3", "col4"]]
    )
