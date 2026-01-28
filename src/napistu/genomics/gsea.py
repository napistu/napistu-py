"""
Functions for organizing gene sets and for applying gene set enrichment analysis (GSEA) to vertices or edges.

Classes
-------
GenesetCollection:
    A collection of gene sets for a given organismal species.

Public Functions
----------------
get_default_collection_config:
    Get the default collection configuration for a given organismal species.

"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from igraph import Graph
from pydantic import BaseModel

from napistu.constants import ONTOLOGIES, SBML_DFS
from napistu.genomics.constants import GENESET_COLLECTION_DEFS, GMTS_CONFIG_FIELDS
from napistu.identifiers import _check_species_identifiers_table
from napistu.ingestion.constants import LATIN_SPECIES_NAMES
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator
from napistu.matching.species import features_to_pathway_species
from napistu.network.constants import IGRAPH_DEFS
from napistu.utils.optional import import_gseapy, require_gseapy

logger = logging.getLogger(__name__)


@require_gseapy
def get_default_collection_config(
    organismal_species: Union[str, OrganismalSpeciesValidator],
) -> GmtsConfig:

    gp = import_gseapy()

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)

    GENESET_COLLECTION_DEFAULTS = {
        LATIN_SPECIES_NAMES.HOMO_SAPIENS: {
            GMTS_CONFIG_FIELDS.ENGINE: gp.msigdb.Msigdb,
            GMTS_CONFIG_FIELDS.CATEGORIES: ["h.all", "c2.cp.kegg_legacy", "c5.go.bp"],
            GMTS_CONFIG_FIELDS.DBVER: "2023.2.Hs",
        }
    }

    organismal_species_str = organismal_species.latin_name
    if organismal_species_str not in GENESET_COLLECTION_DEFAULTS:
        raise ValueError(
            f"The organismal species {organismal_species_str} does not have a default collection config available through `get_default_collection_config`. Please create a config manually."
        )

    return GmtsConfig(**GENESET_COLLECTION_DEFAULTS[organismal_species_str])


class GenesetCollection:
    """
    A collection of gene sets for a given organismal species

    Parameters
    ----------
    organismal_species: Union[str, OrganismalSpeciesValidator]
        The organismal species to create a gene set collection for.

    Attributes
    ----------
    organismal_species: OrganismalSpeciesValidator
        The organismal species to create a gene set collection for.
    gmt: Dict[str, List[str]]
        A dictionary of gene set categories to their gene sets.
    gmts: Dict[str, Dict[str, List[str]]]
        A nested dictionary of gene set categories to their gene sets for each ontology.

    Public Methods
    --------------
    add_gmts:
        Add gene sets to the gene set collection.
    get_gmt_as_df:
        Convert the GMT dictionary to a DataFrame format suitable for matching.

    Examples
    --------
    >>> geneset_collection = GenesetCollection(organismal_species="Homo sapiens")
    >>> # Add the default gene set collection
    >>> geneset_collection.add_gmts()
    >>> # Add a custom gene set collection
    >>> geneset_collection.add_gmts(gmts_config=GmtsConfig(engine=gp.MsigDB, categories=["c5.go.bp", "c5.go.cc", "c5.go.mf"], dbver="2023.2.Hs"))
    """

    def __init__(self, organismal_species: Union[str, OrganismalSpeciesValidator]):
        self.organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)
        self.gmt: Dict[str, List[str]] = {}
        self.gmts: Dict[str, Dict[str, List[str]]] = {}
        self.deep_to_shallow_lookup: pd.DataFrame = None

    def add_gmts(
        self,
        gmts_config: Union[Dict[str, Any], GmtsConfig, None] = None,
        entrez: bool = True,
    ):
        """
        Add gene sets to the gene set collection.

        Parameters
        ----------
        gmts_config: Union[Dict[str, Any], GmtsConfig, None]
            The configuration for the gene set collection.
        entrez: bool
            Whether to use Entrez gene IDs (True) or gene symbols (False).
        """

        gmts_config = self._format_gmts_config(gmts_config)

        caller = gmts_config.engine
        ontology_names = gmts_config.categories
        dbver = gmts_config.dbver

        if not hasattr(caller, "list_category"):
            raise ValueError(f"Caller {caller} does not have a list_category method")
        if not hasattr(caller, "get_gmt"):
            raise ValueError(f"Caller {caller} does not have a get_gmt method")

        ontologies = caller.list_category(dbver=dbver)
        for ontology_name in ontology_names:
            if ontology_name not in ontologies:
                raise ValueError(
                    f"Ontology {ontology_name} not found in {caller} database version {dbver}. Available ontologies: {ontologies}"
                )
            if ontology_name in self.gmts:
                logger.warning(
                    f"Ontology {ontology_name} already exists in the gene set collection. Overwriting."
                )

            self.gmts[ontology_name] = caller.get_gmt(
                category=ontology_name, dbver=dbver, entrez=entrez
            )

        # map from multiple gmt ontologies to unambiguous names
        self.deep_to_shallow_lookup = self._create_deep_to_shallow_lookup()

        # create the shallow gmt
        self.gmt = self._create_gmt()

    def get_gmt_as_df(self) -> pd.DataFrame:
        """
        Convert the GMT dictionary to a DataFrame format suitable for matching.

        Returns
        -------
        pd.DataFrame
            A DataFrame with two columns:
            - "gene_set": The gene set name
            - "identifier": The identifier (e.g., Entrez ID) for each gene in the set

        Examples
        --------
        >>> collection = GenesetCollection(organismal_species="Homo sapiens")
        >>> collection.add_gmts()
        >>> gmt_df = collection.get_gmt_as_df()
        """
        if len(self.gmt) == 0:
            raise ValueError(
                "No gene sets found in the gene set collection. "
                "Please add gene sets using the `add_gmts` method."
            )

        rows = []
        for gene_set_name, identifiers in self.gmt.items():
            for identifier in identifiers:
                rows.append(
                    {
                        GENESET_COLLECTION_DEFS.GENESET: gene_set_name,
                        GENESET_COLLECTION_DEFS.IDENTIFIER: identifier,
                    }
                )

        return pd.DataFrame(rows)

    def get_gmt_w_napistu_ids(
        self,
        species_identifiers: pd.DataFrame,
        id_type: str = SBML_DFS.S_ID,
    ) -> pd.DataFrame:
        """
        Get the gene set collection with Napistu molecular species IDs.

        Parameters
        ----------
        species_identifiers: pd.DataFrame
            A DataFrame with the species identifiers. Either updated with sbml_dfs.get_characteristic_species_ids()
            or loaded from a tsv distributed as part of a Napistu GCS tar-balls. To map to compartmentalized species IDs
            use identifiers.construct_cspecies_identifiers() to add the sc_id column.
        id_type: str
            The type of identifier to use. Must be one of {SBML_DFS.S_ID, SBML_DFS.SC_ID}. If using sc_id, then
            the species_identifiers table must be update to add the sc_id column.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary of gene set names to their Napistu molecular species IDs.
        """

        _check_species_identifiers_table(species_identifiers)
        if id_type not in [SBML_DFS.S_ID, SBML_DFS.SC_ID]:
            raise ValueError(
                f"Invalid id_type: {id_type}. Must be one of {SBML_DFS.S_ID, SBML_DFS.SC_ID}"
            )
        if id_type not in species_identifiers.columns:
            raise ValueError(
                f"id_type {id_type} not found in species_identifiers columns: {species_identifiers.columns}"
            )

        gmt_df = self.get_gmt_as_df()

        gmt_df_w_napistu_ids = features_to_pathway_species(
            gmt_df.assign(feature_id=lambda x: x.identifier.astype(str)),
            species_identifiers,
            ontologies={ONTOLOGIES.NCBI_ENTREZ_GENE},
        )

        return (
            gmt_df_w_napistu_ids.groupby("geneset")[id_type]
            .unique()
            .apply(list)
            .to_dict()
        )

    def _create_deep_to_shallow_lookup(self):
        """
        Create a lookup from deep gene set categories to shallow gene set categories.

        If there is only one ontology, the lookup is simply the gene set names.
        If there are multiple ontologies, the lookup is a concatenation of the ontology name and the gene set name.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the deep gene set names and the shallow gene set names.
        """

        if len(self.gmts.keys()) == 0:
            raise ValueError(
                "No gene sets found in the gene set collection. Please add gene sets using the `add_gmts` method."
            )

        if len(self.gmts.keys()) == 1:
            ontology_name = next(iter(self.gmts.keys()))
            df = (
                pd.DataFrame(
                    {
                        GENESET_COLLECTION_DEFS.DEEP_NAME: list(
                            self.gmts[ontology_name].keys()
                        )
                    }
                )
                .assign(**{GENESET_COLLECTION_DEFS.ONTOLOGY_NAME: ontology_name})
                .assign(
                    **{
                        GENESET_COLLECTION_DEFS.SHALLOW_NAME: lambda x: x[
                            GENESET_COLLECTION_DEFS.DEEP_NAME
                        ]
                    }
                )
            )
            return df

        tables = list()
        for ontology_name in self.gmts:
            df = pd.DataFrame(
                {
                    GENESET_COLLECTION_DEFS.ONTOLOGY_NAME: ontology_name,
                    GENESET_COLLECTION_DEFS.DEEP_NAME: list(
                        self.gmts[ontology_name].keys()
                    ),
                }
            ).assign(
                **{
                    GENESET_COLLECTION_DEFS.SHALLOW_NAME: lambda x: ontology_name
                    + "_"
                    + x[GENESET_COLLECTION_DEFS.DEEP_NAME]
                }
            )
            tables.append(df)

        return pd.concat(tables).reset_index(drop=True)

    def _create_gmt(self):
        """
        Create a GMT dictionary from the gmts dictionary.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary of shallow gene set names to their gene sets.
        """

        if len(self.gmts.keys()) == 0:
            raise ValueError(
                "No gene sets found in the gene set collection. Please add gene sets using the `add_gmts` method."
            )

        if len(self.gmts.keys()) == 1:
            return self.gmts[next(iter(self.gmts.keys()))]

        if self.deep_to_shallow_lookup is None:
            raise ValueError(
                "Deep to shallow lookup not found. Please create the deep to shallow lookup using the `_create_deep_to_shallow_lookup` method."
            )

        return {
            row[GENESET_COLLECTION_DEFS.SHALLOW_NAME]: self.gmts[
                row[GENESET_COLLECTION_DEFS.ONTOLOGY_NAME]
            ][row[GENESET_COLLECTION_DEFS.DEEP_NAME]]
            for _, row in self.deep_to_shallow_lookup.iterrows()
        }

    def _format_gmts_config(
        self, gmts_config: Optional[Union[Dict[str, Any], GmtsConfig]] = None
    ):
        """
        Format a gmts config into a GmtsConfig object.

        Parameters
        ----------
        gmts_config: Optional[Union[Dict[str, Any], GmtsConfig]]
            The gmts config to format.

        Returns
        -------
        GmtsConfig
            The formatted gmts config.
        """

        if gmts_config is None:
            gmts_config = get_default_collection_config(self.organismal_species)

        if isinstance(gmts_config, dict):
            gmts_config = GmtsConfig(**gmts_config)

        if not isinstance(gmts_config, GmtsConfig):
            raise ValueError(
                f"gmts_config must be a GmtsConfig object, got {type(gmts_config)}"
            )

        return gmts_config


def _calculate_geneset_edge_counts(
    edgelist: pd.DataFrame,
    genesets: Dict[str, List[str]],
    universe: Graph,
    min_set_size: int = 5,
    max_set_size: Optional[int] = None,
    directed: bool = False,
) -> pd.DataFrame:
    """
    Calculate edge counts between all geneset pairs.

    Parameters
    ----------
    edgelist : pd.DataFrame
        Edgelist with 'source' and 'target' columns containing vertex names.
        These are the actual edges to count.
    genesets : Dict[str, List[str]]
        Dictionary mapping geneset names to lists of vertex names
    universe : igraph.Graph
        Universe graph defining the possible edges (used for filtering genesets to valid vertices)
    min_set_size : int
        Minimum number of genes in universe for a geneset to be included
    max_set_size : int, optional
        Maximum number of genes in universe for a geneset to be included
    directed : bool
        Whether edges are directed

    Returns
    -------
    pd.DataFrame
        Columns: source_geneset, target_geneset, observed_edges, n_genes_source, n_genes_target
        One row per geneset pair (upper triangle only if undirected)
    """
    # Step 1: Filter genesets to universe vertices and create membership dataframe
    filtered_genesets, geneset_df = _filter_genesets_to_universe(
        universe, genesets, min_set_size, max_set_size
    )

    if len(filtered_genesets) == 0:
        raise ValueError(
            "No genesets found in universe after filtering to minimum size"
        )

    # Step 2: Join observed edges to source genesets
    edges_with_source = edgelist.merge(
        geneset_df, left_on=IGRAPH_DEFS.SOURCE, right_on="vertex_name", how="inner"
    ).rename(columns={"geneset": "source_geneset"})

    # Step 3: Join to target genesets
    edges_with_both = edges_with_source.merge(
        geneset_df,
        left_on=IGRAPH_DEFS.TARGET,
        right_on="vertex_name",
        how="inner",
        suffixes=("_src", "_tgt"),
    ).rename(columns={"geneset": "target_geneset"})

    # Step 4: Count edges per geneset pair
    edge_counts = (
        edges_with_both.groupby(["source_geneset", "target_geneset"])
        .size()
        .reset_index(name="observed_edges")
    )

    # Step 5: Create all possible pairs (including those with 0 edges)
    geneset_sizes = {name: len(genes) for name, genes in filtered_genesets.items()}
    pathway_names = list(filtered_genesets.keys())

    if directed:
        all_pairs = [
            {"source_geneset": a, "target_geneset": b}
            for a in pathway_names
            for b in pathway_names
        ]
    else:
        all_pairs = [
            {"source_geneset": pathway_names[i], "target_geneset": pathway_names[j]}
            for i in range(len(pathway_names))
            for j in range(i, len(pathway_names))
        ]

    all_pairs_df = pd.DataFrame(all_pairs)
    all_pairs_df["n_genes_source"] = all_pairs_df["source_geneset"].map(geneset_sizes)
    all_pairs_df["n_genes_target"] = all_pairs_df["target_geneset"].map(geneset_sizes)

    # Step 6: Merge to include pairs with 0 edges
    result = all_pairs_df.merge(
        edge_counts[["source_geneset", "target_geneset", "observed_edges"]],
        on=["source_geneset", "target_geneset"],
        how="left",
    )
    result["observed_edges"] = result["observed_edges"].fillna(0).astype(int)

    return result


def _filter_genesets_to_universe(
    universe: Graph,
    genesets: Dict[str, List[str]],
    min_set_size: int = 5,
    max_set_size: Optional[int] = None,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Filter genesets to universe vertices and create membership dataframe.

    Parameters
    ----------
    universe : igraph.Graph
        Universe graph with 'name' attribute on vertices
    genesets : Dict[str, List[str]]
        Dictionary mapping geneset names to lists of vertex names
    min_set_size : int
        Minimum number of genes in universe for inclusion
    max_set_size : int, optional
        Maximum number of genes in universe for inclusion

    Returns
    -------
    filtered_genesets : Dict[str, List[str]]
        Geneset name -> list of vertex names in universe
    geneset_df : pd.DataFrame
        Long format with columns: geneset, vertex_name
        Each row is one gene in one geneset
    """
    # Get valid vertex names in universe
    universe_vertex_names = set(universe.vs[IGRAPH_DEFS.NAME])

    # Filter genesets to universe and by size
    filtered_genesets = {}
    geneset_members = []

    for geneset_name, gene_names in genesets.items():
        # Filter to genes that exist in universe
        valid_genes = [g for g in gene_names if g in universe_vertex_names]

        # Filter by size
        if len(valid_genes) >= min_set_size:
            if max_set_size is None or len(valid_genes) <= max_set_size:
                filtered_genesets[geneset_name] = valid_genes

                # Add to membership list
                for gene in valid_genes:
                    geneset_members.append(
                        {
                            "geneset": geneset_name,
                            "vertex_name": gene,
                        }
                    )

    geneset_df = pd.DataFrame(geneset_members)

    return filtered_genesets, geneset_df


class GmtsConfig(BaseModel):
    """Pydantic model for GMT (Gene Matrix Transposed) configuration.

    This class validates the configuration used for gene set collections,
    including the engine, categories, and database version.

    Parameters
    ----------
    engine : Any
        The gene set engine class (e.g., MsigDB from gseapy).
    categories : List[str]
        List of gene set categories to use (e.g., ["h.all", "c2.cp.kegg"]).
    dbver : str
        Database version string (e.g., "2023.2.Hs").

    Examples
    --------
    >>> config = GmtsConfig(
    ...     engine=gp.MsigDB,
    ...     categories=["h.all", "c2.cp.kegg", "c5.go.bp"],
    ...     dbver="2023.2.Hs"
    ... )
    """

    engine: Any
    categories: List[str]
    dbver: str
