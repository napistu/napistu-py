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
from napistu.genomics.constants import GENESET_COLLECTION_DEFS, GMTS_CONFIG_FIELDS
from napistu.identifiers import _check_species_identifiers_table
from napistu.ingestion.constants import LATIN_SPECIES_NAMES
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator
from napistu.matching.species import features_to_pathway_species
from napistu.network.constants import IGRAPH_DEFS, NAPISTU_GRAPH
from napistu.network.edgelist import Edgelist
from napistu.network.ig_utils import _get_universe_degrees, define_graph_universe
from napistu.statistics.hypothesis_testing import neat_edge_enrichment_test
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
        gmts_config: Union[Dict[str, Any], GmtsConfig, None] = None,
        entrez: bool = True,
    ):
        """
        Add gene sets to the gene set collection.

        Parameters
        ----------
        gmts_config: Union[Dict[str, Any], GmtsConfig, None]
            The configuration for the gene set collection. The engine can be specified
            as a string (e.g., "msigdb") or as a callable class. If None, uses the
            default collection config for the organismal species.
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
        self, gmts_config: Optional[Union[Dict[str, Any], GmtsConfig]] = None
    ):
        """
        Format a gmts config into a GmtsConfig object.

        Parameters
        ----------
        gmts_config: Optional[Union[Dict[str, Any], GmtsConfig]]
            The gmts config to format. If a dict is provided, the engine can be
            specified as a string (e.g., "msigdb") or as a callable class.

        Returns
        -------
        GmtsConfig
            The formatted gmts config.
        """

        if gmts_config is None:
            gmts_config = get_default_collection_config(self.organismal_species)

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

        if not isinstance(gmts_config, GmtsConfig):
            raise ValueError(
                f"gmts_config must be a GmtsConfig object, got {type(gmts_config)}"
            )

        return gmts_config


@require_statsmodels
def edgelist_gsea(
    edgelist: Union[pd.DataFrame, Edgelist],
    genesets: Union[GenesetCollection, Dict[str, List[str]]],
    graph: Graph,
    universe_vertex_names: Optional[Union[List[str], pd.Series]] = None,
    universe_edgelist: Optional[pd.DataFrame] = None,
    universe_observed_only: bool = False,
    universe_edge_filter_logic: str = "and",
    include_self_edges: bool = False,
    min_set_size: int = 5,
    max_set_size: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test pathway edge enrichment using NEAT degree-corrected method.

    Performs gene set edge enrichment analysis to identify pairs of pathways
    with more edges between them than expected by chance, accounting for
    degree heterogeneity using the configuration model.

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
    verbose : bool
        If True, print progress information

    Returns
    -------
    pd.DataFrame
        Enrichment results with columns:
        - source_geneset, target_geneset: Pathway names
        - n_genes_source, n_genes_target: Pathway sizes in universe
        - observed_edges: Number of observed edges between pathways
        - expected_edges: Expected number under configuration model
        - variance: Variance of expected edges
        - z_score: Standardized enrichment score
        - p_value: One-tailed p-value (upper tail)
        - q_value: FDR-corrected p-value (Benjamini-Hochberg)

    Examples
    --------
    >>> # Test enrichment in full network
    >>> observed = pd.DataFrame({
    ...     'source': ['A', 'B', 'C'],
    ...     'target': ['B', 'C', 'D']
    ... })
    >>> results = pathway_edge_enrichment(
    ...     observed, genesets, graph
    ... )

    >>> # Test with gene-only universe
    >>> gene_names = [v['name'] for v in graph.vs if v.get('biotype') == 'gene']
    >>> results = pathway_edge_enrichment(
    ...     observed, genesets, graph,
    ...     universe_vertex_names=gene_names
    ... )

    >>> # Test with observed edges only in universe
    >>> results = pathway_edge_enrichment(
    ...     observed, genesets, graph,
    ...     universe_observed_only=True
    ... )

    References
    ----------
    Signorelli et al. (2016) NEAT: an efficient network enrichment analysis test.
    BMC Bioinformatics 17:558.
    """

    multipletests_module = import_statsmodels_multitest()

    edgelist = Edgelist.ensure(edgelist)
    if universe_observed_only:
        edgelist.validate_subset(graph)
    else:
        edgelist.validate_subset(graph, validate=NAPISTU_GRAPH.EDGES)

    # Extract genesets dict if GenesetCollection provided
    if isinstance(genesets, GenesetCollection):
        genesets_dict = genesets.gmt
    else:
        genesets_dict = genesets

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

    # Step 3: Calculate observed edge counts between all geneset pairs

    edge_counts_df = _calculate_geneset_edge_counts(
        edgelist=edgelist,
        genesets=genesets_dict,
        universe=universe,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
        directed=graph.is_directed(),
    )

    _log_edgelist_gsea_paired_counts(
        verbose, edge_counts_df, min_set_size, max_set_size
    )

    # Step 4: Get universe properties for NEAT test

    out_degrees, in_degrees = _get_universe_degrees(
        universe, directed=graph.is_directed()
    )

    # Create name to index mapping for degree lookup
    name_to_idx = {v[IGRAPH_DEFS.NAME]: v.index for v in universe.vs}

    # Get filtered genesets with universe indices
    filtered_genesets, _ = _filter_genesets_to_universe(
        universe, genesets_dict, min_set_size, max_set_size
    )
    geneset_to_indices = {
        name: np.array([name_to_idx[g] for g in genes])
        for name, genes in filtered_genesets.items()
    }

    # Step 5: Run NEAT test for each geneset pair
    total_edges_universe = universe.ecount()
    total_edges_observed = len(edgelist)
    enrichment_stats = edge_counts_df.apply(
        _test_geneset_pair,
        axis=1,
        geneset_to_indices=geneset_to_indices,
        out_degrees=out_degrees,
        in_degrees=in_degrees,
        total_edges_universe=total_edges_universe,
        total_edges_observed=total_edges_observed,
    )

    # Step 6: Combine results
    results_df = pd.concat([edge_counts_df, enrichment_stats], axis=1)

    # Step 7: Multiple testing correction (FDR)
    if verbose:
        logger.info("Applying FDR correction...")

    _, q_values, _, _ = multipletests_module.multipletests(
        results_df["p_value"], method="fdr_bh"
    )
    results_df["q_value"] = q_values

    # Step 8: Sort by significance
    results_df = results_df.sort_values("p_value").reset_index(drop=True)

    _log_edgelist_gsea_paired_results(verbose, results_df)

    return results_df


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
    if isinstance(edgelist, Edgelist):
        edgelist_df = edgelist.to_dataframe()
    elif isinstance(edgelist, pd.DataFrame):
        edgelist_df = edgelist
    else:
        raise ValueError(f"Invalid edgelist type: {type(edgelist)}")

    edges_with_source = edgelist_df.merge(
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


def _test_geneset_pair(
    row: pd.Series,
    geneset_to_indices: Dict[str, np.ndarray],
    out_degrees: np.ndarray,
    in_degrees: np.ndarray,
    total_edges_universe: int,
    total_edges_observed: int,
) -> pd.Series:
    """Calculate the NEAT enrichment statistic for a single geneset pair in a pd.DataFrame."""

    if row["observed_edges"] == 0:
        return pd.Series(
            {
                "observed_edges": 0,
                "expected_edges": None,
                "variance": None,
                "z_score": None,
                "p_value": 1,
                "n_genes_a": None,
                "n_genes_b": None,
                "sum_out_deg_a": None,
                "sum_in_deg_b": None,
                "total_edges_universe": total_edges_universe,
                "total_edges_observed": total_edges_observed,
            }
        )

    source_name = row["source_geneset"]
    target_name = row["target_geneset"]

    indices_source = geneset_to_indices[source_name]
    indices_target = geneset_to_indices[target_name]

    result = neat_edge_enrichment_test(
        observed_edges=row["observed_edges"],
        out_degrees_a=out_degrees[indices_source],
        in_degrees_b=in_degrees[indices_target],
        total_edges_universe=total_edges_universe,
        total_edges_observed=total_edges_observed,
    )

    return pd.Series(result)


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
