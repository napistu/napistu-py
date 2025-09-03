from __future__ import annotations

import logging
import os
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    from fs import open_fs

from napistu import indices, utils
from napistu.constants import (
    EXPECTED_PW_INDEX_COLUMNS,
    SOURCE_SPEC,
)
from napistu.ingestion.constants import (
    BIGG_MODEL_KEYS,
    BIGG_MODEL_URLS,
    DATA_SOURCES,
    LATIN_SPECIES_NAMES,
    MODEL_SOURCE_DESCRIPTIONS,
)
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator
from napistu.ingestion.sbml import SBML
from napistu.ontologies.renaming import rename_species_ontologies
from napistu.sbml_dfs_core import SBML_dfs
from napistu.source import Source

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
        model_organismal_species={
            LATIN_SPECIES_NAMES.HOMO_SAPIENS: LATIN_SPECIES_NAMES.HOMO_SAPIENS,
            LATIN_SPECIES_NAMES.MUS_MUSCULUS: LATIN_SPECIES_NAMES.MUS_MUSCULUS,
            LATIN_SPECIES_NAMES.SACCHAROMYCES_CEREVISIAE: LATIN_SPECIES_NAMES.SACCHAROMYCES_CEREVISIAE,
        },
        base_path=bg_pathway_root,
        data_source=DATA_SOURCES.BIGG,
        model_names=MODEL_SOURCE_DESCRIPTIONS,
    )

    with open_fs(bg_pathway_root, create=True) as bg_fs:
        for _, row in bigg_models_df.iterrows():
            with bg_fs.open(row[SOURCE_SPEC.FILE], "wb") as f:
                utils.download_wget(row[SOURCE_SPEC.URL], f)  # type: ignore

        pw_index = bigg_models_df[[c for c in EXPECTED_PW_INDEX_COLUMNS]]

        # save index to sbml dir
        with bg_fs.open(SOURCE_SPEC.PW_INDEX_FILE, "wb") as f:
            pw_index.to_csv(f, sep="\t", index=False)

    return None


def construct_bigg_consensus(
    pw_index_inp: str | indices.PWIndex,
    organismal_species: str | None = None,
) -> SBML_dfs:
    """Construct a BiGG SBML DFs pathway representation.

    Parameters
    ----------
    pw_index_inp : str or indices.PWIndex
        PWIndex object or URI pointing to PWIndex
    organismal_species : str or None, optional
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

    # select the model assocaited with the organismal species
    if isinstance(pw_index_inp, str):

        # sannitize organismal species name and validate against supported species
        organismal_species_validator = OrganismalSpeciesValidator(organismal_species)
        organismal_species_validator.assert_supported(
            supported_species=list(BIGG_MODEL_URLS.keys())
        )

        pw_index = indices.adapt_pw_index(
            pw_index_inp, organismal_species=organismal_species_validator.latin_name
        )

        if SOURCE_SPEC.MODEL not in pw_index.index.columns:
            pw_index.index = pw_index.index.assign(
                model=pw_index.index[SOURCE_SPEC.PATHWAY_ID]
            )

    elif isinstance(pw_index_inp, indices.PWIndex):
        pw_index = pw_index_inp
    else:
        raise ValueError("pw_index_inp needs to be a PWIndex or a str to a location.")

    if pw_index.index.shape[0] != 1:
        raise ValueError(
            f"The filtered pw_index contained {pw_index.index.shape[0]} rows, expected 1"
        )

    model_file = pw_index.index.iloc[0][SOURCE_SPEC.FILE]
    if isinstance(pw_index_inp, str):
        sbml_dfs_path = os.path.join(os.path.dirname(pw_index_inp), model_file)
    else:
        # if a PWIndex is provided, assume that the file field is the full path to the model file
        sbml_dfs_path = model_file

    model_source = Source(pw_index.index)

    sbml = SBML(sbml_dfs_path)
    sbml_dfs = SBML_dfs(sbml, model_source)
    rename_species_ontologies(sbml_dfs)
    sbml_dfs.validate()

    return sbml_dfs
