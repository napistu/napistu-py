from __future__ import annotations

import logging
import os
from typing import Iterable

from napistu import indices
from napistu import sbml_dfs_core
from napistu import utils
from napistu.consensus import construct_sbml_dfs_dict
from napistu.ontologies.renaming import rename_species_ontologies
from napistu.ingestion.constants import BIGG_MODEL_KEYS
from napistu.ingestion.constants import BIGG_MODEL_URLS
from napistu.ingestion.constants import SPECIES_FULL_NAME_HUMAN
from napistu.ingestion.constants import SPECIES_FULL_NAME_MOUSE
from napistu.ingestion.constants import SPECIES_FULL_NAME_YEAST
from fs import open_fs

logger = logging.getLogger(__name__)


def bigg_sbml_download(bg_pathway_root: str, overwrite: bool = False) -> None:
    """
    BiGG SBML Download

    Download SBML models from BiGG. Currently just the human Recon3D model

    Parameters:
    bg_pathway_root (str): Paths to a directory where a \"sbml\" directory should be created.
    overwrite (bool): Overwrite an existing output directory.

    Returns:
    None

    """
    utils.initialize_dir(bg_pathway_root, overwrite)

    bigg_models_df = indices.create_pathway_index_df(
        model_keys=BIGG_MODEL_KEYS,
        model_urls=BIGG_MODEL_URLS,
        model_species={
            SPECIES_FULL_NAME_HUMAN: SPECIES_FULL_NAME_HUMAN,
            SPECIES_FULL_NAME_MOUSE: SPECIES_FULL_NAME_MOUSE,
            SPECIES_FULL_NAME_YEAST: SPECIES_FULL_NAME_YEAST,
        },
        base_path=bg_pathway_root,
        source_name="BiGG",
    )

    with open_fs(bg_pathway_root, create=True) as bg_fs:
        for _, row in bigg_models_df.iterrows():
            with bg_fs.open(row["file"], "wb") as f:
                utils.download_wget(row["url"], f)  # type: ignore

        pw_index = bigg_models_df[
            ["file", "source", "species", "pathway_id", "name", "date"]
        ]

        # save index to sbml dir
        with bg_fs.open("pw_index.tsv", "wb") as f:
            pw_index.to_csv(f, sep="\t", index=False)

    return None


def construct_bigg_consensus(
    pw_index_inp: str | indices.PWIndex,
    species: str | Iterable[str] | None = None,
    outdir: str | None = None,
) -> sbml_dfs_core.SBML_dfs:
    """Construct a BiGG SBML DFs pathway representation.

    Parameters
    ----------
    pw_index_inp : str or indices.PWIndex
        PWIndex object or URI pointing to PWIndex
    species : str or Iterable[str] or None, optional
        One or more species to filter by, by default None (no filtering)
    outdir : str or None, optional
        Output directory used to cache results, by default None

    Returns
    -------
    sbml_dfs_core.SBML_dfs
        A consensus SBML representation

    Notes
    -----
    Currently this only works for a single model. Integration of multiple
    models is not yet supported in BiGG.

    The function:
    1. Loads/validates the pathway index
    2. Constructs SBML DFs dictionary
    3. Processes the single model:
        - Infers compartmentalization for species without location
        - Names compartmentalized species
        - Validates the final model

    Raises
    ------
    ValueError
        If pw_index_inp is neither a PWIndex nor a string
    NotImplementedError
        If attempting to merge multiple models
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
    else:
        construct_sbml_dfs_dict_fkt = construct_sbml_dfs_dict

    sbml_dfs_dict = construct_sbml_dfs_dict_fkt(pw_index)
    if len(sbml_dfs_dict) > 1:
        raise NotImplementedError("Merging of models not implemented yet for BiGG")

    # In Bigg there should be only one model
    sbml_dfs = list(sbml_dfs_dict.values())[0]
    # fix missing compartimentalization
    sbml_dfs.infer_uncompartmentalized_species_location()
    sbml_dfs.name_compartmentalized_species()
    rename_species_ontologies(sbml_dfs)
    sbml_dfs.validate()
    return sbml_dfs
