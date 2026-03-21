"""
Functions for clustering an igraph.Graph's vertices.

Public Functions
----------------
run_infomap(graph, source_attr, target_attr, weight_attr, directed, two_level, num_trials, seed, teleportation_probability, use_subprocess)
    Run Infomap community detection on an igraph graph and return module assignments.
"""

import json
import subprocess
from typing import Optional

import igraph as ig
import pandas as pd

from napistu.network.constants import (
    CLUSTERING_DEFS,
    INFOMAP_ARGS,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_VERTICES,
)


def run_infomap(
    graph: ig.Graph,
    weight_attr: str = NAPISTU_GRAPH_EDGES.WEIGHT,
    directed: Optional[bool] = None,
    two_level: bool = True,
    num_trials: int = 5,
    seed: int = 123,
    teleportation_probability: float = 0.15,
    markov_time: float = 1.0,
    use_subprocess: bool = True,
) -> pd.Series:
    """
    Run Infomap community detection on an igraph graph and return module assignments.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    weight_attr : str
        Edge attribute name for edge weights. Default "weight".
        Larger weights indicate stronger associations and will be traversed
        more frequently by the random walker, biasing nodes toward the same module.
    directed : bool, optional
        Whether to treat the graph as directed. Defaults to graph.is_directed().
    two_level : bool
        If True, optimize a flat two-level partition rather than a deep hierarchy.
        Recommended for pathway-level interpretability. Default True.
    num_trials : int
        Number of independent optimization runs. Best solution is kept. Default 5.
    seed : int
        Random seed for reproducibility. Default 123.
    teleportation_probability : float
        Probability of teleporting to a random node at each step (analogous to
        1 - damping in PageRank). Default 0.15.
    markov_time : float
        Markov time for the random walk. Default 1.0.
        Larger values produce coarser modules; smaller values produce finer modules.
    use_subprocess : bool
        If True, run Infomap in an isolated subprocess to avoid OpenMP conflicts
        with PyTorch and other libraries that ship their own OpenMP runtime
        (libiomp5 vs libomp). Default True.

    Returns
    -------
    pd.Series
        Series indexed by vertex name (or integer id if no name attribute),
        containing integer module assignments. Isolated vertices (no edges)
        will have None as their module assignment.

    Dependencies
    ------------
    Requires the `infomap` package and, on macOS, libomp:
        brew install libomp
        pip install infomap

    Note: a missing libomp will cause a kernel crash rather than a Python
    exception. If your notebook crashes on import or instantiation, install
    libomp and restart the kernel before reimporting.
    """
    if directed is None:
        directed = graph.is_directed()

    if weight_attr not in graph.es.attributes():
        raise ValueError(
            f"Edge attribute '{weight_attr}' not found in graph. "
            f"Available attributes: {graph.es.attributes()}."
        )

    edgelist = graph.get_edgelist()
    weights = graph.es[weight_attr]
    edges = [(s, t, float(w)) for (s, t), w in zip(edgelist, weights)]

    infomap_kwargs = dict(
        directed=directed,
        two_level=two_level,
        num_trials=num_trials,
        seed=seed,
        teleportation_probability=teleportation_probability,
        markov_time=markov_time,
    )

    if use_subprocess:
        modules = _run_infomap_subprocess(
            edges=edges,
            n_vertices=graph.vcount(),
            **infomap_kwargs,
        )
    else:
        modules = _run_infomap_local(
            edges=edges,
            **infomap_kwargs,
        )

    vertex_indices = [v.index for v in graph.vs]

    if NAPISTU_GRAPH_VERTICES.NAME in graph.vs.attributes():
        index = graph.vs[NAPISTU_GRAPH_VERTICES.NAME]
    else:
        index = vertex_indices

    return pd.Series(
        [modules.get(i, None) for i in vertex_indices],
        index=index,
        name=CLUSTERING_DEFS.MODULE_ID,
        dtype="Int64",  # use Int64 to allow for missing values
    )


def _run_infomap_local(
    edges: list,
    directed: bool,
    two_level: bool,
    num_trials: int,
    seed: int,
    teleportation_probability: float,
    markov_time: float,
) -> dict:
    """
    Run Infomap in the current process.

    Will conflict with PyTorch and other libraries that ship their own OpenMP
    runtime. Use _run_infomap_subprocess if this is a concern.

    Parameters
    ----------
    edges : list
        List of (source, target, weight) tuples.
    directed : bool
        Whether to treat the graph as directed.
    two_level : bool
        Whether to optimize a flat two-level partition.
    num_trials : int
        Number of independent optimization runs.
    seed : int
        Random seed.
    teleportation_probability : float
        Probability of teleporting to a random node at each step.

    Returns
    -------
    dict
        Dictionary mapping integer node_id to integer module_id.
    """
    try:
        import infomap as im_module
    except ImportError:
        raise ImportError(
            "infomap is not installed. Install with: pip install infomap\n"
            "On macOS also run: brew install libomp"
        )

    im = im_module.Infomap(
        directed=directed,
        two_level=two_level,
        num_trials=num_trials,
        seed=seed,
        teleportation_probability=teleportation_probability,
        markov_time=markov_time,
        silent=True,
        no_file_output=True,
    )
    im.add_links(edges)
    im.run()
    return {node.node_id: node.module_id for node in im.nodes}


def _run_infomap_subprocess(
    edges: list,
    n_vertices: int,
    directed: bool,
    two_level: bool,
    num_trials: int,
    seed: int,
    teleportation_probability: float,
    markov_time: float,
) -> dict:
    """
    Run Infomap in an isolated subprocess to avoid OpenMP conflicts with
    PyTorch and other libraries that ship their own OpenMP runtime.

    Parameters
    ----------
    edges : list
        List of (source, target, weight) tuples.
    n_vertices : int
        Number of vertices in the graph.
    directed : bool
        Whether to treat the graph as directed.
    two_level : bool
        Whether to optimize a flat two-level partition.
    num_trials : int
        Number of independent optimization runs.
    seed : int
        Random seed.
    teleportation_probability : float
        Probability of teleporting to a random node at each step.
    markov_time : float
        Markov time for the random walk.

    Returns
    -------
    dict
        Dictionary mapping integer node_id to integer module_id.
    """
    payload = {
        INFOMAP_ARGS.EDGES: edges,
        INFOMAP_ARGS.N_VERTICES: n_vertices,
        INFOMAP_ARGS.DIRECTED: directed,
        INFOMAP_ARGS.TWO_LEVEL: two_level,
        INFOMAP_ARGS.NUM_TRIALS: num_trials,
        INFOMAP_ARGS.SEED: seed,
        INFOMAP_ARGS.TELEPORTATION_PROBABILITY: teleportation_probability,
        INFOMAP_ARGS.MARKOV_TIME: markov_time,
    }

    script = """
import json
import sys
import infomap as im_module

payload = json.loads(sys.stdin.read())

im = im_module.Infomap(
    directed=payload["directed"],
    two_level=payload["two_level"],
    num_trials=payload["num_trials"],
    seed=payload["seed"],
    teleportation_probability=payload["teleportation_probability"],
    markov_time=payload["markov_time"],
    silent=True,
    no_file_output=True,
)
im.add_links(payload["edges"])
im.run()

modules = {str(node.node_id): node.module_id for node in im.nodes}
print(json.dumps(modules))
"""

    result = subprocess.run(
        ["python", "-c", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Infomap subprocess failed with return code {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )

    return {int(k): v for k, v in json.loads(result.stdout).items()}
