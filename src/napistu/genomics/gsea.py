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
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from igraph import Graph
from pydantic import BaseModel

from napistu.constants import ONTOLOGIES, SBML_DFS
from napistu.genomics.constants import (
    GENESET_COLLECTION_DEFS,
    GENESET_COLLECTIONS,
    GENESET_DEFAULT_CONFIG_NAMES,
    GENESET_SOURCE_VERSIONS,
    GENESET_SOURCES,
    GMTS_CONFIG_FIELDS,
    VALID_GENESET_DEFAULT_CONFIG_NAMES,
)
from napistu.identifiers import _check_species_identifiers_table
from napistu.ingestion.constants import LATIN_SPECIES_NAMES
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator
from napistu.matching.species import features_to_pathway_species
from napistu.network.constants import IGRAPH_DEFS, NAPISTU_GRAPH
from napistu.network.edgelist import Edgelist
from napistu.network.ig_utils import define_graph_universe
from napistu.statistics.constants import ENRICHMENT_TESTS, VALID_ENRICHMENT_TESTS
from napistu.statistics.hypothesis_testing import (
    binomial_test_vectorized,
    fisher_exact_vectorized,
    proportion_test_vectorized,
)
from napistu.utils.optional import (
    import_gseapy,
    import_statsmodels_multitest,
    require_gseapy,
    require_statsmodels,
)

logger = logging.getLogger(__name__)


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
    >>> # Add a custom gene set collection using string engine name
    >>> geneset_collection.add_gmts(gmts_config=GmtsConfig(engine="msigdb", categories=["c5.go.bp", "c5.go.cc", "c5.go.mf"], dbver="2023.2.Hs"))
    >>> # Or using a dict with string engine name (dbver is optional)
    >>> geneset_collection.add_gmts(gmts_config={"engine": "msigdb", "categories": ["c5.go.bp"]})
    """

    def __init__(self, organismal_species: Union[str, OrganismalSpeciesValidator]):
        self.organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)
        self.gmt: Dict[str, List[str]] = {}
        self.gmts: Dict[str, Dict[str, List[str]]] = {}
        self.deep_to_shallow_lookup: pd.DataFrame = None

    def add_gmts(
        self,
        gmts_config: Union[Dict[str, Any], GmtsConfig, str, None] = None,
        entrez: bool = True,
    ):
        """
        Add gene sets to the gene set collection.

        Parameters
        ----------
        gmts_config: Union[Dict[str, Any], GmtsConfig, str, None]
            The configuration for the gene set collection. Can be:
            - A string name from GENESET_DEFAULT_CONFIG_NAMES (e.g., "hallmarks", "bp_kegg_hallmarks", "wikipathways")
            - A dict with engine, categories, and optionally dbver
            - A GmtsConfig object
            - None: uses the default collection config for the organismal species from GENESET_DEFAULT_BY_SPECIES
            The engine can be specified as a string (e.g., "msigdb") or as a callable class.
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

        # Call list_category with dbver if provided, otherwise let it use default
        if dbver is not None:
            ontologies = caller.list_category(dbver=dbver)
        else:
            ontologies = caller.list_category()

        for ontology_name in ontology_names:
            if ontology_name not in ontologies:
                dbver_msg = f" database version {dbver}" if dbver is not None else ""
                raise ValueError(
                    f"Ontology {ontology_name} not found in {caller}{dbver_msg}. Available ontologies: {ontologies}"
                )
            if ontology_name in self.gmts:
                logger.warning(
                    f"Ontology {ontology_name} already exists in the gene set collection. Overwriting."
                )

            # Call get_gmt with dbver if provided, otherwise let it use default
            if dbver is not None:
                self.gmts[ontology_name] = caller.get_gmt(
                    category=ontology_name, dbver=dbver, entrez=entrez
                )
            else:
                self.gmts[ontology_name] = caller.get_gmt(
                    category=ontology_name, entrez=entrez
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
        self, gmts_config: Optional[Union[Dict[str, Any], GmtsConfig, str]] = None
    ) -> GmtsConfig:
        """
        Format a gmts config into a GmtsConfig object.

        Parameters
        ----------
        gmts_config: Optional[Union[Dict[str, Any], GmtsConfig, str]]
            The gmts config to format. Can be:
            - A string name from GENESET_DEFAULT_CONFIG_NAMES (e.g., "hallmarks", "bp_kegg_hallmarks", "wikipathways")
            - A dict with engine, categories, and optionally dbver
            - A GmtsConfig object
            - None: uses the default collection config for the organismal species from GENESET_DEFAULT_BY_SPECIES
            If a dict is provided, the engine can be specified as a string (e.g., "msigdb") or as a callable class.

        Returns
        -------
        GmtsConfig
            The formatted gmts config.
        """

        if gmts_config is None:
            gmts_config = get_default_collection_config(self.organismal_species)
        elif isinstance(gmts_config, str):
            if gmts_config not in VALID_GENESET_DEFAULT_CONFIG_NAMES:
                raise ValueError(
                    f"Invalid config name: {gmts_config}. Must be one of {VALID_GENESET_DEFAULT_CONFIG_NAMES}"
                )
            gmts_config = GENESET_DEFAULT_CONFIGS[gmts_config]

        if isinstance(gmts_config, dict):
            # Convert string engine names to callables if needed
            if GMTS_CONFIG_FIELDS.ENGINE in gmts_config:
                engine = gmts_config[GMTS_CONFIG_FIELDS.ENGINE]
                if isinstance(engine, str):
                    gmts_config[GMTS_CONFIG_FIELDS.ENGINE] = _get_engine_from_string(
                        engine
                    )
            gmts_config = GmtsConfig(**gmts_config)
        elif isinstance(gmts_config, GmtsConfig):
            # Convert string engine to callable if GmtsConfig was created directly with string
            if isinstance(gmts_config.engine, str):
                # Create a new GmtsConfig with the converted engine
                engine_dict = gmts_config.model_dump()
                engine_dict[GMTS_CONFIG_FIELDS.ENGINE] = _get_engine_from_string(
                    gmts_config.engine
                )
                gmts_config = GmtsConfig(**engine_dict)
        else:
            raise ValueError(
                f"gmts_config must be a GmtsConfig instance a string, a dict or None; got {type(gmts_config)}"
            )

        return gmts_config


@require_statsmodels
def edgelist_gsea(
    edgelist: Union[pd.DataFrame, Edgelist],
    genesets: Union[GenesetCollection, Dict[str, List[str]]],
    graph: Graph,
    enrichment_test: str = ENRICHMENT_TESTS.FISHER_EXACT,
    universe_vertex_names: Optional[Union[List[str], pd.Series]] = None,
    universe_edgelist: Optional[pd.DataFrame] = None,
    universe_observed_only: bool = False,
    universe_edge_filter_logic: str = "and",
    include_self_edges: bool = False,
    min_set_size: int = 5,
    max_set_size: Optional[int] = None,
    min_x_geneset_edges_possible: int = 5,
    chunk_size: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test pathway edge enrichment using NEAT degree-corrected method.

    Performs gene set edge enrichment analysis to identify pairs of pathways
    with more edges between them than expected by chance based on a Fisher's exact test.

    Parameters
    ----------
    edgelist : Union[pd.DataFrame, Edgelist]
        Edgelist with 'source' and 'target' columns containing vertex names.
        These are the edges to test for enrichment.
    genesets : GenesetCollection or Dict[str, List[str]]
        Gene sets to test. Either a GenesetCollection object or a dictionary
        mapping geneset names to lists of gene names.
    graph : ig.Graph
        Source network graph
    enrichment_test : str
        The enrichment test to use. Must be one of "fisher_exact", "proportion", or "binomial". Default is "fisher_exact".
        - "fisher_exact": Uses a Fisher's exact test to test for enrichment.
        - "proportion": Uses a proportion test to test for enrichment.
        - "binomial": Uses a binomial test to test for enrichment.
    universe_vertex_names : list of str or pd.Series, optional
        Vertex names to include in universe. If None, filter to the vertices present in at least one geneset.
    universe_edgelist : pd.DataFrame, optional
        Edgelist defining possible edges in universe. If None and
        universe_observed_only=False, creates complete graph.
    universe_observed_only : bool
        If True, universe includes only observed edges from graph.
    universe_edge_filter_logic : str
        How to combine universe_edgelist and universe_observed_only: 'and' or 'or'
    include_self_edges : bool
        Whether to include self-edges in universe
    min_set_size : int
        Minimum geneset size (after filtering to universe)
    max_set_size : int, optional
        Maximum geneset size (after filtering to universe)
    min_x_geneset_edges_possible : int
        Minimum number of possible edges in universe between geneset pairs to include in results.
        If there are only a small number of possible edges, seeing 1 would be statistically surprising
        but not meaningful. Default is 5.
    chunk_size : int
        Number of target genesets to process at once. Only used when counting edges between genesets in the universe.
    verbose : bool
        If True, print progress information

    Returns
    -------
    pd.DataFrame
        Enrichment results with columns:
        - source_geneset, target_geneset: Pathway names
        - n_genes_source, n_genes_target: Pathway sizes in universe
        - observed_edges: Number of observed edges between pathways
        - p_value: One-tailed p-value (upper tail)
        - q_value: FDR-corrected p-value (Benjamini-Hochberg)

    Examples
    --------
    >>> # Test enrichment in full network
    >>> observed = pd.DataFrame({
    ...     'source': ['A', 'B', 'C'],
    ...     'target': ['B', 'C', 'D']
    ... })
    >>> results = edgelist_gsea(
    ...     observed, genesets, graph
    ... )

    >>> # Test with gene-only universe
    >>> gene_names = [v['name'] for v in graph.vs if v.get('biotype') == 'gene']
    >>> results = edgelist_gsea(
    ...     observed, genesets, graph,
    ...     universe_vertex_names=gene_names
    ... )

    >>> # Test with observed edges only in universe
    >>> results = edgelist_gsea(
    ...     observed, genesets, graph,
    ...     universe_observed_only=True
    ... )
    """

    multipletests_module = import_statsmodels_multitest()

    edgelist = Edgelist.ensure(edgelist)
    if universe_observed_only:
        edgelist.validate_subset(graph)
    else:
        edgelist.validate_subset(graph, validate=NAPISTU_GRAPH.EDGES)

    # resolve edgelist to check for duplicates and reciprocal edges
    edgelist = _resolve_edgelist(graph, edgelist)

    # Extract genesets dict if GenesetCollection provided
    if isinstance(genesets, GenesetCollection):
        genesets_dict = genesets.gmt
    else:
        genesets_dict = genesets

    if enrichment_test not in VALID_ENRICHMENT_TESTS:
        raise ValueError(
            f"Invalid enrichment test: {enrichment_test}. Must be one of {VALID_ENRICHMENT_TESTS}"
        )

    _log_edgelist_gsea_input(verbose, graph, edgelist, genesets_dict)

    # Step 2: Create universe graph

    if not include_self_edges:
        # override if there are self edges in the graph
        logger.warning(
            "Setting include_self_edges to True because there are self edges in the graph"
        )
        include_self_edges = any(graph.is_loop())

    if universe_vertex_names is None:
        universe_vertex_names = list(set(chain.from_iterable(genesets_dict.values())))

    universe = define_graph_universe(
        graph=graph,
        vertex_names=universe_vertex_names,
        edgelist=universe_edgelist,
        observed_only=universe_observed_only,
        edge_filter_logic=universe_edge_filter_logic,
        include_self_edges=include_self_edges,
    )

    # verify that the edgelist is a subset of the universe
    _validate_edgelist_universe(edgelist, universe)

    _log_edgelist_gsea_universe(verbose, universe)

    # Step 3: Calculate edges in the observed edgelist and universe between all geneset pairs

    edge_counts_df = _calculate_geneset_edge_counts(
        edgelist=edgelist,
        genesets=genesets_dict,
        universe=universe,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
        chunk_size=chunk_size,
    )

    # Filter to minimum universe edges threshold
    edge_counts_df = edge_counts_df[
        edge_counts_df["universe_edges"] >= min_x_geneset_edges_possible
    ].copy()

    if len(edge_counts_df) == 0:
        raise ValueError(
            f"No geneset pairs found with at least {min_x_geneset_edges_possible} possible edges in universe"
        )

    # Step 4: Calculate expected edge counts and enrichment statistics
    odds_ratios, p_values = _calculate_enrichment_statistics(
        enrichment_test=enrichment_test,
        edge_counts_df=edge_counts_df,
        edgelist_size=len(edgelist),
        universe_size=universe.ecount(),
    )

    edge_counts_df["odds_ratio"] = odds_ratios
    edge_counts_df["p_value"] = p_values

    # Step 5: Multiple testing correction (FDR)
    if verbose:
        logger.info("Applying FDR correction...")

    _, q_values, _, _ = multipletests_module.multipletests(
        edge_counts_df["p_value"], method="fdr_bh"
    )
    edge_counts_df["q_value"] = q_values

    # Step 6: Sort by significance
    edge_counts_df = edge_counts_df.sort_values("p_value").reset_index(drop=True)

    _log_edgelist_gsea_paired_results(verbose, edge_counts_df)

    return edge_counts_df


@require_gseapy
def get_default_collection_config(
    organismal_species: Union[str, OrganismalSpeciesValidator],
) -> GmtsConfig:

    organismal_species = OrganismalSpeciesValidator.ensure(organismal_species)

    organismal_species_str = organismal_species.latin_name
    if organismal_species_str not in GENESET_DEFAULT_BY_SPECIES:
        raise ValueError(
            f"The organismal species {organismal_species_str} does not have a default collection config available through `get_default_collection_config`. Please create a config manually."
        )

    return GENESET_DEFAULT_BY_SPECIES[organismal_species_str]


def _calculate_geneset_edge_counts(
    edgelist: pd.DataFrame,
    genesets: Dict[str, List[str]],
    universe: Graph,
    min_set_size: int = 5,
    max_set_size: Optional[int] = None,
    chunk_size: int = 10,
) -> pd.DataFrame:
    """
    Calculate edge counts between all geneset pairs in both observed edgelist and universe.

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
    chunk_size : int
        Number of target genesets to process at once. Set to np.inf to process all at once.

    Returns
    -------
    pd.DataFrame
        Columns: source_geneset, target_geneset, observed_edges, universe_edges, n_genes_source, n_genes_target
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

    # Step 2: Convert edgelist to DataFrame if needed
    if isinstance(edgelist, Edgelist):
        edgelist_df = edgelist.to_dataframe()
    elif isinstance(edgelist, pd.DataFrame):
        edgelist_df = edgelist
    else:
        raise ValueError(f"Invalid edgelist type: {type(edgelist)}")

    # Step 3: Create universe edgelist from the graph
    universe_edgelist_df = pd.DataFrame(
        [
            {
                IGRAPH_DEFS.SOURCE: universe.vs[e.source][IGRAPH_DEFS.NAME],
                IGRAPH_DEFS.TARGET: universe.vs[e.target][IGRAPH_DEFS.NAME],
            }
            for e in universe.es
        ]
    )

    # Step 4: Count observed edges (no chunking needed for small edgelists)
    observed_counts = _count_edges_by_geneset_pair_chunked(
        edgelist_df, geneset_df, "observed_edges", chunk_size=np.inf
    )

    # Step 5: Count universe edges (with chunking for large universes)
    universe_counts = _count_edges_by_geneset_pair_chunked(
        universe_edgelist_df, geneset_df, "universe_edges", chunk_size=chunk_size
    )

    # Step 6: Create all possible pairs
    geneset_sizes = {name: len(genes) for name, genes in filtered_genesets.items()}
    pathway_names = list(filtered_genesets.keys())

    if universe.is_directed():
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

    # Step 7: Merge both edge counts
    result = all_pairs_df.merge(
        observed_counts, on=["source_geneset", "target_geneset"], how="left"
    )
    result["observed_edges"] = result["observed_edges"].fillna(0).astype(int)

    result = result.merge(
        universe_counts, on=["source_geneset", "target_geneset"], how="left"
    )
    result["universe_edges"] = result["universe_edges"].fillna(0).astype(int)

    return result


def _count_edges_by_geneset_pair_chunked(
    edgelist_df: pd.DataFrame,
    geneset_df: pd.DataFrame,
    count_column_name: str,
    chunk_size: int = 10,
) -> pd.DataFrame:
    """
    Count edges between geneset pairs, processing target genesets in chunks to limit memory.

    Parameters
    ----------
    edgelist_df : pd.DataFrame
        Edgelist with 'source' and 'target' columns
    geneset_df : pd.DataFrame
        Long format with columns: geneset, vertex_name
    count_column_name : str
        Name for the count column in output
    chunk_size : int
        Number of target genesets to process at once

    Returns
    -------
    pd.DataFrame
        Columns: source_geneset, target_geneset, {count_column_name}
    """
    # Join edges to source genesets once (reuse for all chunks)
    edges_with_source = edgelist_df.merge(
        geneset_df, left_on=IGRAPH_DEFS.SOURCE, right_on="vertex_name", how="inner"
    ).rename(columns={"geneset": "source_geneset"})

    # Get unique genesets for chunking
    unique_genesets = geneset_df["geneset"].unique()
    all_results = []

    # Convert chunk_size to int, handling np.inf
    if chunk_size == np.inf or chunk_size >= len(unique_genesets):
        chunk_size_int = len(unique_genesets)
    else:
        chunk_size_int = int(chunk_size)

    # Process target genesets in chunks
    for i in range(0, len(unique_genesets), chunk_size_int):
        chunk_genesets = unique_genesets[i : i + chunk_size_int]
        geneset_df_chunk = geneset_df[geneset_df["geneset"].isin(chunk_genesets)]

        # Join to target genesets (only for this chunk)
        edges_with_both = edges_with_source.merge(
            geneset_df_chunk,
            left_on=IGRAPH_DEFS.TARGET,
            right_on="vertex_name",
            how="inner",
            suffixes=("_src", "_tgt"),
        ).rename(columns={"geneset": "target_geneset"})

        # Count edges for this chunk
        edge_counts = (
            edges_with_both.groupby(["source_geneset", "target_geneset"])
            .size()
            .reset_index(name=count_column_name)
        )

        all_results.append(edge_counts)

    # Combine all chunks
    return pd.concat(all_results, ignore_index=True)[
        ["source_geneset", "target_geneset", count_column_name]
    ]


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


def _log_edgelist_gsea_input(
    verbose: bool, graph: Graph, edgelist: Edgelist, genesets_dict: Dict[str, List[str]]
):

    if verbose:
        logger.info("Starting pathway edge enrichment analysis")
        logger.info(f"  Input graph: {graph.vcount()} vertices, {graph.ecount()} edges")
        logger.info(f"  Observed edgelist: {len(edgelist)} edges")
        logger.info(f"  Input genesets: {len(genesets_dict)} pathways")
        logger.info("Creating enrichment universe...")


def _log_edgelist_gsea_universe(verbose: bool, universe: Graph):
    if verbose:
        logger.info(
            f"  Universe: {universe.vcount()} vertices, {universe.ecount()} edges"
        )
        logger.info(f"  Directed: {universe.is_directed()}")
        logger.info("Calculating observed edge counts between geneset pairs...")


def _log_edgelist_gsea_paired_counts(
    verbose: bool,
    edge_counts_df: pd.DataFrame,
    min_set_size: int,
    max_set_size: Optional[int],
):

    if verbose:
        n_genesets = len(edge_counts_df["source_geneset"].unique())
        logger.info(
            f"  Filtered to {n_genesets} genesets (size {min_set_size}-{max_set_size or 'inf'})"
        )
        logger.info(f"  Testing {len(edge_counts_df)} geneset pairs")

        # Summary statistics
        n_pairs_with_edges = (edge_counts_df["observed_edges"] > 0).sum()
        logger.info(
            f"  Pairs with edges: {n_pairs_with_edges} ({100*n_pairs_with_edges/len(edge_counts_df):.1f}%)"
        )
        logger.info(
            f"  Median edges per pair: {edge_counts_df['observed_edges'].median():.0f}"
        )
        logger.info(f"  Max edges per pair: {edge_counts_df['observed_edges'].max()}")
        logger.info("Computing NEAT enrichment statistics...")


def _log_edgelist_gsea_paired_results(verbose: bool, results_df: pd.DataFrame):

    if verbose:
        # Summary of results
        n_sig_05 = (results_df["q_value"] < 0.05).sum()
        n_sig_01 = (results_df["q_value"] < 0.01).sum()
        logger.info(
            f"  Significant pairs (q < 0.05): {n_sig_05} ({100*n_sig_05/len(results_df):.1f}%)"
        )
        logger.info(
            f"  Significant pairs (q < 0.01): {n_sig_01} ({100*n_sig_01/len(results_df):.1f}%)"
        )

        if n_sig_05 > 0:
            top_result = results_df.iloc[0]
            logger.info(
                f"  Top enrichment: {top_result['source_geneset']} <-> {top_result['target_geneset']}"
            )


def _resolve_edgelist(graph: Graph, edgelist: Edgelist) -> Edgelist:

    if graph.is_directed():
        if edgelist.has_duplicated_edges:
            logger.warning("Edgelist contains duplicate edges. Removing duplicates.")
            edgelist = edgelist.remove_duplicated_edges()
    else:
        has_duplicates = edgelist.has_duplicated_edges
        if edgelist.has_reciprocal_edges:
            if has_duplicates:
                raise ValueError(
                    "The provided edgelist has both duplicate edges and reciprocal edges (). Please remove duplicates from the edgelist to allow for automatic resolution of reciprocal edges."
                )
            else:
                logger.warning(
                    "The provided graph is undirected but some edges are present in both directions (A-B and B-A). Only retaining a single example of each pair."
                )
                edgelist = edgelist.remove_reciprocal_edges()
        else:
            logger.warning("Edgelist contains duplicate edges. Removing duplicates.")
            edgelist = edgelist.remove_duplicated_edges()

        # add back B-A reciprocal edges so that A-B and B-A are present for all provided edges
        reciprocal_edges = edgelist.to_dataframe().copy()
        reciprocal_edges[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET] = reciprocal_edges[
            IGRAPH_DEFS.TARGET, IGRAPH_DEFS.SOURCE
        ]
        reciprocal_edges[IGRAPH_DEFS.TARGET, IGRAPH_DEFS.SOURCE] = reciprocal_edges[
            IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET
        ]
        edgelist = Edgelist(pd.concat([edgelist.to_dataframe(), reciprocal_edges]))

    return edgelist


def _calculate_enrichment_statistics(
    enrichment_test: str,
    edge_counts_df: pd.DataFrame,
    edgelist_size: int,
    universe_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate enrichment statistics using the specified test method.

    Parameters
    ----------
    enrichment_test : str
        Test method: "proportion", "fisher_exact", or "binomial"
    edge_counts_df : pd.DataFrame
        DataFrame with observed_edges and universe_edges columns
    edgelist_size : int
        Total number of edges in the observed edgelist
    universe_size : int
        Total number of edges in the universe

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (odds_ratios, p_values) arrays
    """
    if enrichment_test == ENRICHMENT_TESTS.PROPORTION:
        _, odds_ratios, p_values = proportion_test_vectorized(
            sample_successes=edge_counts_df["observed_edges"].values,
            sample_total=edgelist_size,
            population_successes=edge_counts_df["universe_edges"].values,
            population_total=universe_size,
        )
    elif enrichment_test == ENRICHMENT_TESTS.FISHER_EXACT:
        observed_members = edge_counts_df["observed_edges"].values
        missing_members = (
            edge_counts_df["universe_edges"].values
            - edge_counts_df["observed_edges"].values
        )
        observed_nonmembers = edgelist_size - observed_members
        nonobserved_nonmembers = (
            universe_size - observed_members - missing_members - observed_nonmembers
        )

        odds_ratios, p_values = fisher_exact_vectorized(
            observed_members=observed_members,
            missing_members=missing_members,
            observed_nonmembers=observed_nonmembers,
            nonobserved_nonmembers=nonobserved_nonmembers,
        )
    elif enrichment_test == ENRICHMENT_TESTS.BINOMIAL:
        _, p_values = binomial_test_vectorized(
            sample_successes=edge_counts_df["observed_edges"].values,
            sample_total=edgelist_size,
            population_successes=edge_counts_df["universe_edges"].values,
            population_total=universe_size,
        )
        # Odds ratio doesn't make sense with binomial test, return NaN
        odds_ratios = np.full(len(edge_counts_df), np.nan, dtype=float)
    else:
        raise ValueError(
            f"Invalid enrichment test: {enrichment_test}. Must be one of {VALID_ENRICHMENT_TESTS}"
        )

    return odds_ratios, p_values


def _validate_edgelist_universe(edgelist, universe):

    try:
        edgelist.validate_subset(graph=universe, graph_name="universe")
    except Exception as e:
        logger.warning(
            "The observed edgelist is not a subset of the universe of possible edges in the universe.\n"
            "This could be because:\n"
            "  1. The edgelist contains vertices which are not in universe_vertex_names\n"
            "  2. The edgelist contains edges which are not in universe_edgelist\n"
            "  3. The universe_observed_only flag is True and the edgelist contains edges which are not observed in the graph"
        )
        raise e


@require_gseapy
def _get_engine_from_string(engine_name: str) -> Any:
    """
    Convert a string engine name to the corresponding gseapy engine class.

    Parameters
    ----------
    engine_name : str
        The engine name (e.g., "msigdb").

    Returns
    -------
    Any
        The engine class (e.g., gp.msigdb.Msigdb).

    Raises
    ------
    ValueError
        If the engine name is not recognized.

    Examples
    --------
    >>> engine = _get_engine_from_string("msigdb")
    >>> engine
    <class 'gseapy.msigdb.Msigdb'>
    """
    gp = import_gseapy()

    engine_map = {
        "msigdb": gp.msigdb.Msigdb,
    }

    engine_name_lower = engine_name.lower()
    if engine_name_lower not in engine_map:
        available = ", ".join(engine_map.keys())
        raise ValueError(
            f"Unknown engine name: '{engine_name}'. "
            f"Available engine names: {available}. "
            f"Alternatively, you can pass the engine class directly."
        )

    return engine_map[engine_name_lower]


class GmtsConfig(BaseModel):
    """Pydantic model for GMT (Gene Matrix Transposed) configuration.

    This class validates the configuration used for gene set collections,
    including the engine, categories, and database version.

    Parameters
    ----------
    engine : Union[str, Any]
        The gene set engine class (e.g., MsigDB from gseapy) or a string name
        (e.g., "msigdb"). Supported string names: "msigdb".
    categories : List[str]
        List of gene set categories to use (e.g., ["h.all", "c2.cp.kegg"]).
    dbver : Optional[str]
        Database version string (e.g., "2023.2.Hs"). If None, the engine's default
        version will be used.

    Examples
    --------
    >>> # Using string engine name (recommended)
    >>> config = GmtsConfig(
    ...     engine="msigdb",
    ...     categories=["h.all", "c2.cp.kegg", "c5.go.bp"],
    ...     dbver="2023.2.Hs"
    ... )
    >>> # Using callable engine class (also supported)
    >>> config = GmtsConfig(
    ...     engine=gp.msigdb.Msigdb,
    ...     categories=["h.all", "c2.cp.kegg", "c5.go.bp"],
    ...     dbver="2023.2.Hs"
    ... )
    >>> # dbver is optional
    >>> config = GmtsConfig(
    ...     engine="msigdb",
    ...     categories=["h.all", "c2.cp.kegg", "c5.go.bp"]
    ... )
    """

    engine: Union[str, Callable]
    categories: List[str]
    dbver: Optional[str] = None


GENESET_DEFAULT_CONFIGS = {
    GENESET_DEFAULT_CONFIG_NAMES.HALLMARKS: GmtsConfig(
        **{
            GMTS_CONFIG_FIELDS.ENGINE: GENESET_SOURCES.MSIGDB,
            GMTS_CONFIG_FIELDS.CATEGORIES: [GENESET_COLLECTIONS.H_ALL],
            GMTS_CONFIG_FIELDS.DBVER: GENESET_SOURCE_VERSIONS.HS_2023_2,
        }
    ),
    GENESET_DEFAULT_CONFIG_NAMES.BP_KEGG_HALLMARKS: GmtsConfig(
        **{
            GMTS_CONFIG_FIELDS.ENGINE: GENESET_SOURCES.MSIGDB,
            GMTS_CONFIG_FIELDS.CATEGORIES: [
                GENESET_COLLECTIONS.H_ALL,
                GENESET_COLLECTIONS.C2_CP_KEGG_LEGACY,
                GENESET_COLLECTIONS.C5_GO_BP,
            ],
            GMTS_CONFIG_FIELDS.DBVER: GENESET_SOURCE_VERSIONS.HS_2023_2,
        }
    ),
    GENESET_DEFAULT_CONFIG_NAMES.WIKIPATHWAYS: GmtsConfig(
        **{
            GMTS_CONFIG_FIELDS.ENGINE: GENESET_SOURCES.MSIGDB,
            GMTS_CONFIG_FIELDS.CATEGORIES: [GENESET_COLLECTIONS.C2_CP_WIKIPATHWAYS],
            GMTS_CONFIG_FIELDS.DBVER: GENESET_SOURCE_VERSIONS.HS_2023_2,
        }
    ),
}

GENESET_DEFAULT_BY_SPECIES = {
    LATIN_SPECIES_NAMES.HOMO_SAPIENS: GENESET_DEFAULT_CONFIGS[
        GENESET_DEFAULT_CONFIG_NAMES.HALLMARKS
    ],
}
