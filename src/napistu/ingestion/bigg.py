from __future__ import annotations

import datetime
import logging
import os
from typing import Iterable

import pandas as pd
from napistu import indices
from napistu import sbml_dfs_core
from napistu import utils
from napistu.consensus import construct_sbml_dfs_dict
from napistu.ingestion import sbml
from napistu.ingestion.constants import BIGG_MODEL_FIELD_SPECIES
from napistu.ingestion.constants import BIGG_MODEL_FIELD_URL
from napistu.ingestion.constants import BIGG_MODEL_KEYS
from napistu.ingestion.constants import BIGG_MODEL_URLS
from napistu.ingestion.constants import BIGG_RECON3D_FIELD_ANNOTATION
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

    bigg_models = {
        BIGG_MODEL_KEYS[SPECIES_FULL_NAME_HUMAN]: {
            BIGG_MODEL_FIELD_URL: BIGG_MODEL_URLS[SPECIES_FULL_NAME_HUMAN],
            BIGG_MODEL_FIELD_SPECIES: SPECIES_FULL_NAME_HUMAN,
        },
        BIGG_MODEL_KEYS[SPECIES_FULL_NAME_MOUSE]: {
            BIGG_MODEL_FIELD_URL: BIGG_MODEL_URLS[SPECIES_FULL_NAME_MOUSE],
            BIGG_MODEL_FIELD_SPECIES: SPECIES_FULL_NAME_MOUSE,
        },
        BIGG_MODEL_KEYS[SPECIES_FULL_NAME_YEAST]: {
            BIGG_MODEL_FIELD_URL: BIGG_MODEL_URLS[SPECIES_FULL_NAME_YEAST],
            BIGG_MODEL_FIELD_SPECIES: SPECIES_FULL_NAME_YEAST,
        },
    }
    bigg_models_df = pd.DataFrame(bigg_models).T
    bigg_models_df["sbml_path"] = [
        os.path.join(bg_pathway_root, k) + ".sbml"
        for k in bigg_models_df.index.tolist()
    ]
    bigg_models_df["file"] = [os.path.basename(x) for x in bigg_models_df["sbml_path"]]

    # add other attributes which will be used in the pw_index
    bigg_models_df["date"] = datetime.date.today().strftime("%Y%m%d")
    bigg_models_df.index = bigg_models_df.index.rename("pathway_id")
    bigg_models_df = bigg_models_df.reset_index()
    bigg_models_df["name"] = bigg_models_df["pathway_id"]
    bigg_models_df = bigg_models_df.assign(source="BiGG")

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


def annotate_recon(raw_model_path: str, annotated_model_path: str) -> None:
    """Annotate Recon3D
    Add compartment annotations to Recon3D so it can be merged with other pathways
    """
    logger.warning(
        "add_sbml_annotations is deprecated and maybe removed in a future version of rcpr; "
        "we are now adding these annotation during ingestion by sbml.sbml_df_from_sbml() rather "
        "than directly appending them to the raw .sbml"
    )
    recon_3d_annotations = pd.DataFrame(BIGG_RECON3D_FIELD_ANNOTATION)
    sbml_model = sbml.SBML(raw_model_path)
    sbml.add_sbml_annotations(
        sbml_model, recon_3d_annotations, save_path=annotated_model_path
    )

    return None


def construct_bigg_consensus(
    pw_index_inp: str | indices.PWIndex,
    species: str | Iterable[str] | None = None,
    outdir: str | None = None,
) -> sbml_dfs_core.SBML_dfs:
    """Constructs a BiGG SBML DFs Pathway Representation

    Attention: curently this does work only for a singly model. Integraiton of multiple
    models is not supported yet in BiGG.

    Args:
        pw_index_inp (str | indices.PWIndex): PWIndex or uri pointing to PWIndex
        species (str | Iterable[str] | None): one or more species to filter by. Default: no filtering
        outdir (str | None, optional): output directory used to cache results. Defaults to None.

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
    else:
        construct_sbml_dfs_dict_fkt = construct_sbml_dfs_dict

    sbml_dfs_dict = construct_sbml_dfs_dict_fkt(pw_index)
    if len(sbml_dfs_dict) > 1:
        raise NotImplementedError("Merging of models not implemented yet for BiGG")

    # In Bigg there should be only one model
    model = list(sbml_dfs_dict.values())[0]
    # fix missing compartimentalization
    model = sbml_dfs_core.infer_uncompartmentalized_species_location(model)
    model = sbml_dfs_core.name_compartmentalized_species(model)
    model.validate()
    return model
