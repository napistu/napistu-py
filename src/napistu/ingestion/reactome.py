from __future__ import annotations

import datetime
import logging
import os
import random
from io import StringIO
from typing import Iterable

import pandas as pd
import requests
from napistu import indices
from napistu import sbml_dfs_core
from napistu import utils
from napistu.consensus import construct_consensus_model
from napistu.consensus import construct_sbml_dfs_dict
from napistu.ingestion.constants import REACTOME_PATHWAY_INDEX_COLUMNS
from napistu.ingestion.constants import REACTOME_PATHWAY_LIST_COLUMNS
from napistu.ingestion.constants import REACTOME_PATHWAYS_URL
from napistu.ingestion.constants import REACTOME_SMBL_URL
from fs import open_fs

logger = logging.getLogger(__name__)


def reactome_sbml_download(output_dir_path: str, overwrite: bool = False):
    """
    Reactome SBML Download

    Download Reactome SBML (systems biology markup language) for all reactome species.

    Args:
        output_dir_path (str): Paths to a directory where .sbml files should be saved.
        overwrite (bool): Overwrite an existing output directory. Default: False
    """
    utils.download_and_extract(
        REACTOME_SMBL_URL,
        output_dir_path=output_dir_path,
        overwrite=overwrite,
    )
    # create the pathway index
    pw_index = _build_reactome_pw_index(output_dir_path, file_ext="sbml")

    # save as tsv
    out_fs = open_fs(output_dir_path)
    with out_fs.open("pw_index.tsv", "wb") as index_path:
        pw_index.to_csv(index_path, sep="\t", index=False)


# Functions useful to integrate reactome pathways into a consensus
def construct_reactome_consensus(
    pw_index_inp: str | indices.PWIndex,
    species: str | Iterable[str] | None = None,
    outdir: str | None = None,
    strict: bool = True,
) -> sbml_dfs_core.SBML_dfs:
    """Constructs a basic consensus model by merging all models from a pw_index

    Args:
        pw_index_inp (str | indices.PWIndex): PWIndex or uri pointing to PWIndex
        species (str | Iterable[str] | None): one or more species to filter by. Default: no filtering
        outdir (str | None, optional): output directory used to cache results. Defaults to None.
        strict (bool): should failure of loading any given model throw an exception? If False a warning is thrown.

    Returns:
        sbml_dfs_core.SBML_dfs: A consensus SBML
    """
    if isinstance(pw_index_inp, str):
        pw_index = indices.adapt_pw_index(pw_index_inp, species=species, outdir=outdir)
    elif isinstance(pw_index_inp, indices.PWIndex):
        pw_index = pw_index_inp
    else:
        raise ValueError("pw_index_inp needs to be a PWIndex or a str to a location.")
    if outdir is not None:
        construct_sbml_dfs_dict_fkt = utils.pickle_cache(
            os.path.join(outdir, "model_pool.pkl")
        )(construct_sbml_dfs_dict)
        construct_consensus_model_fkt = utils.pickle_cache(
            os.path.join(outdir, "consensus.pkl")
        )(construct_consensus_model)
    else:
        construct_sbml_dfs_dict_fkt = construct_sbml_dfs_dict
        construct_consensus_model_fkt = construct_consensus_model

    sbml_dfs_dict = construct_sbml_dfs_dict_fkt(pw_index, strict)
    consensus_model = construct_consensus_model_fkt(sbml_dfs_dict, pw_index)
    return consensus_model


def _build_reactome_pw_index(
    output_dir: str,
    file_ext: str,
    species_filter: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a reactome pathway index

    Builds the index based on available files and cross-checkes it with the
    expected reactome pathway list.

    Args:
        output_dir (str): File directory
        file_ext (str): File extension
        species_filter (Optional[Iterable[str]], optional): Filter the expected
            pathway list based on a list of species. Eg in cases only one species available. Defaults to None.

    Returns:
        pd.DataFrame: pathway index
    """
    # create the pathway index
    out_fs = open_fs(output_dir)
    all_files = [os.path.basename(f.path) for f in out_fs.glob(f"**/*.{file_ext}")]

    if len(all_files) == 0:
        raise ValueError(f"Zero files in {output_dir} have the {file_ext} extension")

    pw_index = pd.DataFrame({"file": all_files}).assign(source="Reactome")
    pw_index["pathway_id"] = [os.path.splitext(x)[0] for x in pw_index["file"]]

    # test before merging
    pathway_list = _get_reactome_pathway_list()
    if species_filter is not None:
        pathway_list = pathway_list.loc[pathway_list["species"].isin(species_filter)]

    _check_reactome_pw_index(pw_index, pathway_list)
    pw_index = pw_index.merge(pathway_list)
    pw_index = pw_index[REACTOME_PATHWAY_INDEX_COLUMNS]
    pw_index["date"] = datetime.date.today().strftime("%Y%m%d")

    return pw_index


def _check_reactome_pw_index(pw_index: indices.PWIndex, reactome_pathway_list: list):
    """Compare local files defined in the pathway index to a list of Reactome's pathways."""

    # check extension in pw_index
    extn = set([os.path.splitext(x)[1] for x in pw_index["file"]])
    if len(extn) != 1:
        raise ValueError(
            f"Expected all files to have the same extension, but found extensions: {extn}"
        )
    if len(extn.intersection({".sbml"})) != 1:
        raise ValueError(
            f"Expected all files to have the .sbml extension, but found: {extn}"
        )
    extn_string = extn.pop()

    local_reactome_pws = set(pw_index["pathway_id"])
    remote_reactome_pws = set(reactome_pathway_list["pathway_id"])

    extra_local = local_reactome_pws.difference(remote_reactome_pws)
    if len(extra_local) != 0:
        n_samples = min(5, len(extra_local))
        local_str = ", ".join(random.sample(list(extra_local), n_samples))

        logger.warning(
            f"{len(extra_local)} Reactome {extn_string} files were detected "
            "which are not found in reactome.get_reactome_pathway_list(). "
            f"The include {local_str}. "
            "These files will be excluded from the pathway index"
        )

    extra_remote = remote_reactome_pws.difference(local_reactome_pws)

    if len(extra_remote) != 0:
        n_samples = min(5, len(extra_remote))
        remote_str = ", ".join(random.sample(list(extra_remote), n_samples))

        logger.warning(
            f"{len(extra_remote)} Reactome {extn_string} files were missing "
            "which should be present based on reactome.get_reactome_pathway_list(). "
            f"These include {remote_str}."
        )
    return None


def _get_reactome_pathway_list():
    """Reactome Pathway List
    Produce a pd.DataFrame listing all pathways in reactome and their internal ids

    Parameters:
        None

    Returns:
        pd.DataFrame containing pathway_id, name and species
    """
    page = requests.get(REACTOME_PATHWAYS_URL)
    if page.status_code != 200:
        raise ValueError(
            f"Reactome data could not be accessed at {REACTOME_PATHWAYS_URL}"
        )
    StringData = StringIO(page.content.decode())
    df = pd.read_csv(StringData, sep="\t", names=REACTOME_PATHWAY_LIST_COLUMNS)

    return df
