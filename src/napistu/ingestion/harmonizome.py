"""Ingestion utilities for Harmonizome datasets like Achilles and the Replogle et al. Perturb-seq datasets, Cell 2022."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from napistu.constants import ONTOLOGIES
from napistu.ingestion.constants import (
    HARMONIZOME_DATASET_DEFS,
    HARMONIZOME_DATASET_FILES,
    HARMONIZOME_DATASET_SHORTNAMES,
    HARMONIZOME_DATASETS,
    HARMONIZOME_DEFS,
    HARMONIZOME_URL_TEMPLATES,
)
from napistu.utils import download_wget, initialize_dir

logger = logging.getLogger(__name__)


class HarmonizomeDataset(BaseModel):
    """Configuration and methods for a Harmonizome dataset."""

    name: str = Field(..., description="Full display name of the dataset")
    short_name: str = Field(
        ..., description="Slug used in URLs (e.g., 'reploglek562essential')"
    )
    description: str = Field(..., description="Brief description of the dataset")
    download_files: List[str] = Field(
        default_factory=lambda: [
            HARMONIZOME_DATASET_FILES.INTERACTIONS,
            HARMONIZOME_DATASET_FILES.GENES,
            HARMONIZOME_DATASET_FILES.ATTRIBUTES,
        ],
        description="List of file types to download",
    )
    custom_urls: Dict[str, str] = Field(
        default_factory=dict,
        description="Override specific file URLs if they don't follow the pattern",
    )
    custom_loaders: Dict[str, Callable[[Path], pd.DataFrame]] = Field(
        default_factory=dict,
        description="Custom loader functions for specific file types",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @field_validator(HARMONIZOME_DATASET_DEFS.DOWNLOAD_FILES)
    @classmethod
    def validate_download_files(cls, v: List[str]) -> List[str]:
        """Ensure download_files contains valid file types."""
        valid_types = {
            HARMONIZOME_DATASET_FILES.INTERACTIONS,
            HARMONIZOME_DATASET_FILES.GENES,
            HARMONIZOME_DATASET_FILES.ATTRIBUTES,
        }
        invalid = set(v) - valid_types
        if invalid:
            raise ValueError(
                f"Invalid file types: {invalid}. Must be one of {valid_types}"
            )
        return v

    @classmethod
    def from_dict(cls, short_name: str, config: Dict) -> "HarmonizomeDataset":
        """
        Create a HarmonizomeDataset from a configuration dictionary.

        Parameters
        ----------
        short_name : str
            The short name/key for the dataset
        config : Dict
            Configuration dictionary with keys: name, description, download_files

        Returns
        -------
        HarmonizomeDataset
            A new dataset instance

        Examples
        --------
        >>> config = {
        ...     'name': 'My Dataset',
        ...     'description': 'A test dataset',
        ...     'download_files': ['interactions', 'genes']
        ... }
        >>> dataset = HarmonizomeDataset.from_dict('mydataset', config)
        """
        # Add short_name to config and let Pydantic handle validation
        return cls(short_name=short_name, **config)

    @classmethod
    def ensure_harmonizome_dataset(
        cls, dataset: "HarmonizomeDataset | Dict"
    ) -> "HarmonizomeDataset":
        """
        Ensure input is a HarmonizomeDataset instance.

        If a dict is provided, it must have 'short_name' key.

        Parameters
        ----------
        dataset : HarmonizomeDataset | Dict
            Either a HarmonizomeDataset instance or a config dict

        Returns
        -------
        HarmonizomeDataset
            A HarmonizomeDataset instance

        Examples
        --------
        >>> # Pass through existing instance
        >>> ds = HarmonizomeDataset(name='test', short_name='test', description='test')
        >>> result = HarmonizomeDataset.ensure_harmonizome_dataset(ds)
        >>> assert result is ds

        >>> # Convert from dict
        >>> config = {'short_name': 'test', 'name': 'Test Dataset', 'description': 'A test'}
        >>> result = HarmonizomeDataset.ensure_harmonizome_dataset(config)
        >>> isinstance(result, HarmonizomeDataset)
        True
        """
        if isinstance(dataset, cls):
            return dataset
        elif isinstance(dataset, dict):
            if HARMONIZOME_DATASET_DEFS.SHORT_NAME not in dataset:
                raise ValueError(
                    f"Dictionary must contain {HARMONIZOME_DATASET_DEFS.SHORT_NAME} key"
                )
            # Pydantic will validate all fields
            return cls(**dataset)
        else:
            raise TypeError(
                f"Expected {cls.__name__} or dict, got {type(dataset).__name__}"
            )

    def get_download_urls(self) -> Dict[str, str]:
        """Generate download URLs, using custom URLs when provided."""
        base_url = HARMONIZOME_URL_TEMPLATES.BASE_URL.format(short_name=self.short_name)

        # Standard patterns
        standard_urls = {
            HARMONIZOME_DATASET_FILES.INTERACTIONS: HARMONIZOME_URL_TEMPLATES.INTERACTIONS.format(
                base_url=base_url
            ),
            HARMONIZOME_DATASET_FILES.GENES: HARMONIZOME_URL_TEMPLATES.GENES.format(
                base_url=base_url
            ),
            HARMONIZOME_DATASET_FILES.ATTRIBUTES: HARMONIZOME_URL_TEMPLATES.ATTRIBUTES.format(
                base_url=base_url
            ),
        }

        # Override with custom URLs where provided
        urls = {k: v for k, v in standard_urls.items() if k in self.download_files}
        urls.update(self.custom_urls)

        return urls

    def download(self, output_dir: Path, overwrite: bool = False) -> Dict[str, Path]:
        """Download all files for this dataset using napistu's download_wget utility."""

        output_dir = Path(output_dir).expanduser()
        dataset_dir = output_dir / self.short_name
        urls = self.get_download_urls()
        downloaded_files = {}

        try:
            initialize_dir(str(dataset_dir), overwrite=overwrite)
        except FileExistsError:
            if not overwrite:
                logger.debug(
                    f"Dataset directory {dataset_dir} already exists. Set overwrite=True to replace."
                )

        dataset_dir.mkdir(parents=True, exist_ok=True)

        for file_type, url in urls.items():
            # Save as .tsv since download_wget auto-decompresses .gz files
            output_path = dataset_dir / f"{file_type}.tsv"
            downloaded_files[file_type] = output_path

            if output_path.is_file() and not overwrite:
                logger.debug(
                    f"File {output_path} already exists. Set overwrite=True to replace."
                )
                continue

            try:
                logger.info(f"Downloading {file_type} from {url}...")
                # download_wget automatically decompresses .gz files
                download_wget(
                    url=url,
                    path=str(output_path),
                    verify=True,
                    timeout=30,
                    max_retries=3,
                )

                logger.info(f"Saved {file_type} to {output_path}")

            except Exception as e:
                logger.error(f"Failed to download {file_type}: {e}")

        return downloaded_files

    def load(self, output_dir: Path, file_type: str) -> pd.DataFrame:
        """
        Load a dataset file as a DataFrame.

        Parameters
        ----------
        output_dir : Path
            Directory containing the dataset files
        file_type : str
            Type of file to load (e.g., 'interactions', 'genes', 'attributes')

        Returns
        -------
        pd.DataFrame
            The loaded data
        """
        output_dir = Path(output_dir).expanduser()
        dataset_dir = output_dir / self.short_name
        filepath = dataset_dir / f"{file_type}.tsv"

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Use custom loader if defined for this file type
        if file_type in self.custom_loaders:
            logger.debug(f"Using custom loader for {file_type}")
            return self.custom_loaders[file_type](filepath)

        # Default loader
        return self._default_loader(filepath)

    def process_edge_list(self, dataset_dir: Path) -> dict:
        """Parse gene-attribute edge list and return statistics."""
        df = self.load(dataset_dir, "interactions")

        return {
            "n_edges": len(df),
            "n_genes": df.iloc[:, 0].nunique() if len(df) > 0 else 0,
            "n_attributes": df.iloc[:, 1].nunique() if len(df) > 0 else 0,
            "dataframe": df,
        }

    def download_and_process(
        self, output_dir: Path, overwrite: bool = False
    ) -> Dict[str, any]:
        """Download all files and process them."""
        logger.debug("=" * 60)
        logger.debug(f"Processing: {self.name}")
        logger.debug("=" * 60)

        # Download files
        files = self.download(output_dir, overwrite=overwrite)

        # Process interactions if available
        results = {"files": files}

        if "interactions" in files:
            stats = self.process_edge_list(output_dir)
            results["stats"] = stats

            logger.debug("Dataset stats:")
            logger.debug(f"  Edges: {stats['n_edges']:,}")
            logger.debug(f"  Genes: {stats['n_genes']:,}")
            logger.debug(f"  Attributes: {stats['n_attributes']:,}")

        return results

    @staticmethod
    def _default_loader(filepath: Path) -> pd.DataFrame:
        """Default file loading logic."""
        return pd.read_csv(filepath, sep="\t", index_col=0)


# public functions


def load_harmonizome_datasets(
    dataset_short_names: List[str],
    output_dir: Path,
    datasets_dict: Dict[str, Union[HarmonizomeDataset, Dict]] = None,
    file_types: Optional[List[str]] = None,
    custom_loaders_registry: Optional[Dict[str, Dict[str, Callable]]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load multiple datasets into nested dictionary of DataFrames.

    Parameters
    ----------
    dataset_short_names : List[str]
        List of dataset short names from datasets_dict to load
    output_dir : Path
        Directory containing the downloaded files
    datasets_dict : Optional[Dict[str, Union[HarmonizomeDataset, Dict]]],
        Dictionary of available datasets.
        Defaults to HARMONIZOME_DATASETS if None.
    file_types : Optional[List[str]], optional
        List of file types to load. If None, loads all available files.
        Options: 'interactions', 'genes', 'attributes'
    custom_loaders_registry : Optional[Dict[str, Dict[str, Callable]]], optional
        Registry of custom loader functions. If None, uses CUSTOM_LOADERS.
        Maps dataset short_name -> file_type -> loader function.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary: {dataset_short_name: {file_type: DataFrame}}

    Examples
    --------
    >>> data = load_harmonizome_datasets(['reploglek562essential', 'achilles'], Path('data'))
    >>> interactions = data['reploglek562essential']['interactions']
    >>> genes = data['achilles']['genes']

    >>> # Load only specific file types
    >>> data = load_harmonizome_datasets(
    ...     ['reploglek562essential'],
    ...     Path('data'),
    ...     file_types=['interactions']
    ... )

    >>> # Provide custom loaders
    >>> my_loaders = {
    ...     'achilles': {'interactions': load_achilles_interactions}
    ... }
    >>> data = load_harmonizome_datasets(
    ...     ['achilles'],
    ...     Path('data'),
    ...     custom_loaders_registry=my_loaders
    ... )
    """
    if datasets_dict is None:
        from napistu.ingestion.constants import HARMONIZOME_DATASETS

        datasets_dict = HARMONIZOME_DATASETS

    if custom_loaders_registry is None:
        custom_loaders_registry = CUSTOM_LOADERS

    output_dir = Path(output_dir).expanduser()
    all_data = {}

    for name in dataset_short_names:
        if name not in datasets_dict:
            logger.error(f"Dataset key '{name}' not found in datasets dictionary")
            continue

        # Ensure we have a HarmonizomeDataset instance
        try:
            dataset = HarmonizomeDataset.ensure_harmonizome_dataset(datasets_dict[name])
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert dataset {name}: {e}")
            continue

        # Apply custom loaders if available for this dataset
        dataset = _apply_custom_loaders(dataset, custom_loaders_registry)

        # Determine which file types to load for THIS dataset
        # Use dataset.download_files to respect which files should exist for this dataset
        # Use a local variable to avoid modifying the parameter across iterations
        files_to_load = dataset.download_files if file_types is None else file_types

        logger.debug(f"Loading dataset: {name}")

        all_data[name] = {}

        for file_type in files_to_load:
            try:
                df = dataset.load(output_dir, file_type)
                all_data[name][file_type] = df
                logger.debug(f"  Loaded {file_type}: {df.shape}")
            except FileNotFoundError:
                logger.warning(
                    f"  File not found: {file_type} for {name}. "
                    "Use process_harmonizome_datasets to download and process the dataset or "
                    "use the HarmonizomeDataset instance to call the download method directly."
                )
            except Exception as e:
                logger.error(f"  Failed to load {file_type} for {name}: {e}")

    return all_data


def process_harmonizome_datasets(
    dataset_short_names: List[str],
    output_dir: Path,
    datasets_dict: Dict[str, Union[HarmonizomeDataset, Dict]] = HARMONIZOME_DATASETS,
    overwrite: bool = False,
    custom_loaders_registry: Optional[Dict[str, Dict[str, Callable]]] = None,
) -> Dict[str, Dict]:
    """
    Download and process multiple datasets with statistics.

    This is the main function for downloading Harmonizome datasets.

    Parameters
    ----------
    dataset_short_names : List[str]
        List of dataset short names from datasets_dict to process
    output_dir : Path
        Directory to save downloaded files
    datasets_dict : Optional[Dict]
        Dictionary of available datasets.
        Defaults to HARMONIZOME_DATASETS if None.
    overwrite : bool, optional
        Whether to overwrite existing files
    custom_loaders_registry : Optional[Dict[str, Dict[str, Callable]]], optional
        Registry of custom loader functions. If None, uses CUSTOM_LOADERS.
        Maps dataset short_name -> file_type -> loader function.

    Returns
    -------
    Dict[str, Dict]
        Nested dictionary with files and stats for each dataset:
        {
            'dataset_short_name': {
                'files': {file_type: filepath},
                'stats': {
                    'n_edges': int,
                    'n_genes': int,
                    'n_attributes': int,
                    'dataframe': pd.DataFrame
                }
            }
        }

    Examples
    --------
    >>> results = process_harmonizome_datasets(['reploglek562essential'], Path('data'))
    >>> results['reploglek562essential']['stats']['n_edges']
    12345
    >>> results['reploglek562essential']['files']['interactions']
    PosixPath('data/reploglek562essential/interactions.tsv')

    >>> # Provide custom loaders
    >>> my_loaders = {
    ...     'achilles': {'interactions': load_achilles_interactions}
    ... }
    >>> results = process_harmonizome_datasets(
    ...     ['achilles'],
    ...     Path('data'),
    ...     custom_loaders_registry=my_loaders
    ... )
    """

    if custom_loaders_registry is None:
        custom_loaders_registry = CUSTOM_LOADERS

    output_dir = Path(output_dir).expanduser()
    all_results = {}

    for name in dataset_short_names:
        if name not in datasets_dict:
            logger.error(f"Dataset key '{name}' not found in datasets dictionary")
            continue

        try:
            dataset = HarmonizomeDataset.ensure_harmonizome_dataset(datasets_dict[name])
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert dataset {name}: {e}")
            all_results[name] = {"error": str(e)}
            continue

        # Apply custom loaders if available for this dataset
        dataset = _apply_custom_loaders(dataset, custom_loaders_registry)

        try:
            results = dataset.download_and_process(output_dir, overwrite=overwrite)
            all_results[name] = results
        except Exception as e:
            logger.error(f"Failed to process dataset {name}: {e}")
            all_results[name] = {"error": str(e)}

    return all_results


# private functions


def _apply_custom_loaders(
    dataset: "HarmonizomeDataset",
    custom_loaders_registry: Dict[str, Dict[str, Callable[[Path], pd.DataFrame]]],
) -> "HarmonizomeDataset":
    """
    Apply custom loaders to a HarmonizomeDataset instance if available.

    This creates a modified copy of the dataset with custom loaders added.

    Parameters
    ----------
    dataset : HarmonizomeDataset
        The dataset instance to potentially add custom loaders to
    custom_loaders_registry : Dict[str, Dict[str, Callable]]
        Registry mapping dataset short_name to file_type loaders

    Returns
    -------
    HarmonizomeDataset
        Dataset instance with custom loaders applied (if any exist for this dataset)
    """
    # Check if this dataset has custom loaders
    if dataset.short_name in custom_loaders_registry:
        logger.debug(f"Applying custom loaders for dataset: {dataset.short_name}")

        # Create a new instance with custom loaders added
        # Pydantic models are immutable by default, so we use model_copy with update
        return dataset.model_copy(
            update={
                HARMONIZOME_DATASET_DEFS.CUSTOM_LOADERS: custom_loaders_registry[
                    dataset.short_name
                ]
            }
        )

    return dataset


def _format_perturbatalas_perturbations(interactions: pd.DataFrame) -> pd.DataFrame:

    # use the perturbed gene's symbol<->entrez as an adapter since the perutbation only has a symbol
    symbol_to_entrez = interactions[
        [ONTOLOGIES.SYMBOL, ONTOLOGIES.NCBI_ENTREZ_GENE]
    ].drop_duplicates()
    # Convert entrez ID to string
    symbol_to_entrez[ONTOLOGIES.NCBI_ENTREZ_GENE] = symbol_to_entrez[
        ONTOLOGIES.NCBI_ENTREZ_GENE
    ].astype(str)
    # convert to upper case symbols
    symbol_to_entrez[ONTOLOGIES.SYMBOL] = symbol_to_entrez[
        ONTOLOGIES.SYMBOL
    ].str.upper()

    perturbations = (
        interactions[[HARMONIZOME_DEFS.PERTURBATION]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Split the Perturbation column by underscore into 5 columns
    decoded_perturbations = perturbations[HARMONIZOME_DEFS.PERTURBATION].str.split(
        "_", n=4, expand=True
    )
    decoded_perturbations.columns = [
        "prefix",
        "ID",
        HARMONIZOME_DEFS.PERTURBATION_TYPE,
        ONTOLOGIES.SYMBOL,
        HARMONIZOME_DEFS.PERTURBATION_STUDY,
    ]
    # convert to upper case symbols
    decoded_perturbations[ONTOLOGIES.SYMBOL] = decoded_perturbations[
        ONTOLOGIES.SYMBOL
    ].str.upper()
    # add the entrez ID of the perturbation
    decoded_perturbations = decoded_perturbations.merge(
        symbol_to_entrez, on=ONTOLOGIES.SYMBOL, how="left"
    )

    # Convert entrez ID to string, converting NaN to None
    # After merge, NaN values cause pandas to convert the column to float
    # Convert numeric values to string (removing decimal) and NaN to None
    decoded_perturbations[ONTOLOGIES.NCBI_ENTREZ_GENE] = decoded_perturbations[
        ONTOLOGIES.NCBI_ENTREZ_GENE
    ].apply(lambda x: str(int(float(x))) if pd.notna(x) else None)

    # Add these columns to your original dataframe
    perturbations[
        [
            HARMONIZOME_DEFS.PERTURBED_SYMBOL,
            HARMONIZOME_DEFS.PERTURBED_NCBI_ENTREZ_GENE,
            HARMONIZOME_DEFS.PERTURBATION_TYPE,
            HARMONIZOME_DEFS.PERTURBATION_STUDY,
        ]
    ] = decoded_perturbations[
        [
            ONTOLOGIES.SYMBOL,
            ONTOLOGIES.NCBI_ENTREZ_GENE,
            HARMONIZOME_DEFS.PERTURBATION_TYPE,
            HARMONIZOME_DEFS.PERTURBATION_STUDY,
        ]
    ]
    return perturbations


# custom loaders


def _load_achilles_interactions(filepath: Path) -> pd.DataFrame:
    return (
        pd.read_csv(
            filepath,
            sep="\t",
            skiprows=1,  # Skip the first metadata header
        )
        .drop(columns=["NA", "NA.1"])
        .rename(
            columns={
                HARMONIZOME_DEFS.GENE_SYM: ONTOLOGIES.SYMBOL,
                HARMONIZOME_DEFS.GENE_ID: ONTOLOGIES.NCBI_ENTREZ_GENE,
            }
        )
        .assign(
            **{
                ONTOLOGIES.NCBI_ENTREZ_GENE: lambda x: x[
                    ONTOLOGIES.NCBI_ENTREZ_GENE
                ].astype(str)
            }
        )
    )


def _load_clinvar_interactions(filepath: Path) -> pd.DataFrame:
    return (
        pd.read_csv(filepath, sep="\t")
        .rename(
            columns={
                HARMONIZOME_DEFS.GENE: ONTOLOGIES.SYMBOL,
                HARMONIZOME_DEFS.GENE_ID_SPACED: ONTOLOGIES.NCBI_ENTREZ_GENE,
            }
        )
        .drop(columns=["id", "Phenotype ID", "Threshold Value"])
        .assign(
            **{
                ONTOLOGIES.NCBI_ENTREZ_GENE: lambda x: x[
                    ONTOLOGIES.NCBI_ENTREZ_GENE
                ].astype(str)
            }
        )
    )


def _load_dbgap_interactions(filepath: Path) -> pd.DataFrame:

    return (
        pd.read_csv(filepath, sep="\t", skiprows=1)
        .rename(
            columns={
                HARMONIZOME_DEFS.GENE_SYM: ONTOLOGIES.SYMBOL,
                HARMONIZOME_DEFS.GENE_ID: ONTOLOGIES.NCBI_ENTREZ_GENE,
            }
        )
        .assign(
            **{
                ONTOLOGIES.NCBI_ENTREZ_GENE: lambda x: x[
                    ONTOLOGIES.NCBI_ENTREZ_GENE
                ].astype(str)
            }
        )[[ONTOLOGIES.SYMBOL, ONTOLOGIES.NCBI_ENTREZ_GENE, HARMONIZOME_DEFS.TRAIT]]
    )


def _load_drugbank_interactions(filepath: Path) -> pd.DataFrame:
    return (
        pd.read_csv(filepath, sep="\t", encoding="latin-1", skiprows=1)
        .rename(
            columns={
                HARMONIZOME_DEFS.GENE_SYM: ONTOLOGIES.SYMBOL,
                HARMONIZOME_DEFS.UNIPROT_ACC: ONTOLOGIES.UNIPROT,
                HARMONIZOME_DEFS.GENE_ID: ONTOLOGIES.NCBI_ENTREZ_GENE,
                HARMONIZOME_DEFS.DRUGBANK_ID: ONTOLOGIES.DRUGBANK,
            }
        )
        .drop(columns=["NA", "weight"])
        .assign(
            **{
                ONTOLOGIES.NCBI_ENTREZ_GENE: lambda x: x[
                    ONTOLOGIES.NCBI_ENTREZ_GENE
                ].astype(str)
            }
        )
    )


def _load_drugbank_attributes(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath, sep="\t", encoding="latin-1").rename(
        columns={HARMONIZOME_DEFS.DRUGBANK_ID: ONTOLOGIES.DRUGBANK}
    )


def _load_gad_interactions(filepath: Path) -> pd.DataFrame:
    return (
        pd.read_csv(filepath, sep="\t", skiprows=1)
        .rename(
            columns={
                HARMONIZOME_DEFS.GENE_SYM: ONTOLOGIES.SYMBOL,
                HARMONIZOME_DEFS.GENE_ID: ONTOLOGIES.NCBI_ENTREZ_GENE,
            }
        )
        .assign(
            **{
                ONTOLOGIES.NCBI_ENTREZ_GENE: lambda x: x[
                    ONTOLOGIES.NCBI_ENTREZ_GENE
                ].astype(str)
            }
        )[
            [
                ONTOLOGIES.SYMBOL,
                ONTOLOGIES.NCBI_ENTREZ_GENE,
                HARMONIZOME_DEFS.DISEASE,
                HARMONIZOME_DEFS.DISEASE_CLASS,
            ]
        ]
    )


def _load_replogle_interactions(filepath: Path) -> pd.DataFrame:
    return (
        pd.read_csv(filepath, sep="\t", index_col=0)
        .rename(
            columns={
                HARMONIZOME_DEFS.GENE: ONTOLOGIES.SYMBOL,
                HARMONIZOME_DEFS.GENE_ID_SPACED: ONTOLOGIES.NCBI_ENTREZ_GENE,
            }
        )
        .assign(
            **{
                ONTOLOGIES.NCBI_ENTREZ_GENE: lambda x: x[
                    ONTOLOGIES.NCBI_ENTREZ_GENE
                ].astype(str)
            }
        )
        .assign(
            **{
                HARMONIZOME_DEFS.PERTURBED_ENSEMBL_GENE: lambda x: x[
                    HARMONIZOME_DEFS.GENE_PERTURBATION_ID
                ].str.extract(r"(ENS[A-Z]*G[0-9]+)$", expand=False)
            }
        )
    )


def _load_perturbatlas_interactions(filepath: Path) -> pd.DataFrame:
    interactions = (
        pd.read_csv(filepath, sep="\t", index_col=0)
        .rename(
            columns={
                HARMONIZOME_DEFS.GENE: ONTOLOGIES.SYMBOL,
                HARMONIZOME_DEFS.GENE_ID_SPACED: ONTOLOGIES.NCBI_ENTREZ_GENE,
            }
        )
        .assign(
            **{
                ONTOLOGIES.NCBI_ENTREZ_GENE: lambda x: x[
                    ONTOLOGIES.NCBI_ENTREZ_GENE
                ].astype(str)
            }
        )
    )

    formatted_perturbations = _format_perturbatalas_perturbations(interactions)
    return interactions.merge(
        formatted_perturbations, on=HARMONIZOME_DEFS.PERTURBATION, how="inner"
    )


# Custom loaders registry - maps dataset short_name to file_type loaders
CUSTOM_LOADERS: Dict[str, Dict[str, Callable[[Path], pd.DataFrame]]] = {
    HARMONIZOME_DATASET_SHORTNAMES.ACHILLES: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_achilles_interactions,
    },
    HARMONIZOME_DATASET_SHORTNAMES.CLINVAR: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_clinvar_interactions,
    },
    HARMONIZOME_DATASET_SHORTNAMES.DBGAP: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_dbgap_interactions,
    },
    HARMONIZOME_DATASET_SHORTNAMES.DRUG_BANK: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_drugbank_interactions,
        HARMONIZOME_DATASET_FILES.ATTRIBUTES: _load_drugbank_attributes,
    },
    HARMONIZOME_DATASET_SHORTNAMES.GENETIC_ASSOCIATION_DATABASE: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_gad_interactions,
    },
    HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_K562_GENOME_WIDE: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_replogle_interactions,
    },
    HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_K562_ESSENTIAL: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_replogle_interactions,
    },
    HARMONIZOME_DATASET_SHORTNAMES.REPLOGLE_RPE1_ESSENTIAL: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_replogle_interactions,
    },
    HARMONIZOME_DATASET_SHORTNAMES.PERTURB_ATLAS: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_perturbatlas_interactions,
    },
    HARMONIZOME_DATASET_SHORTNAMES.PERTURB_ATLAS_MOUSE: {
        HARMONIZOME_DATASET_FILES.INTERACTIONS: _load_perturbatlas_interactions,
    },
}
