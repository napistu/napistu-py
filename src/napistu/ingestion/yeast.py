from __future__ import annotations

import pandas as pd
from napistu import identifiers
from napistu import sbml_dfs_core
from napistu import sbml_dfs_utils
from napistu import source
from napistu import utils
from napistu.constants import BQB
from napistu.constants import MINI_SBO_FROM_NAME
from napistu.ingestion.constants import YEAST_IDEA_KINETICS_URL
from napistu.ingestion.constants import YEAST_IDEA_PUBMED_ID
from napistu.ingestion.constants import YEAST_IDEA_SOURCE
from napistu.ingestion.constants import YEAST_IDEA_TARGET


def download_idea(output_dir: str) -> None:
    # save to
    utils.download_and_extract(YEAST_IDEA_KINETICS_URL, output_dir)

    # TODO: since only a single file is outputted, it makes sense to download and extract the data, then copy it to a target URI
    # TODO: add GCS support
    pass


def convert_idea_kinetics_to_sbml_dfs(
    idea_path: str,
) -> sbml_dfs_core.SBML_dfs:
    """
    Convert IDEA Kinetics to SBML DFs

    Format yeast induction regulator->target relationships as a directed graph.

    Args:
        idea_path: Path to the IDEA Kinetics file.

    Returns:
        SBML_dfs: an SBML_dfs object containing molecular species and their interactions.
        Kinetic attributes are included as reactions_data.

    """

    # TO DO - replace with GCS support (currently this just reads a local .tsv)
    idea_kinetics_df = pd.read_csv(idea_path, sep="\t")

    # separate based on whether the change is probably direct or indirect
    idea_kinetics_df["directness"] = [
        "direct" if t_rise < 15 else "indirect" for t_rise in idea_kinetics_df["t_rise"]
    ]

    # reduce cases of multiple TF-target pairs to a single entry
    distinct_edges = (
        idea_kinetics_df.groupby([YEAST_IDEA_SOURCE, YEAST_IDEA_TARGET], as_index=True)
        .apply(_summarize_idea_pairs)
        .reset_index()
    )

    # add some more fields are reformat
    formatted_distinct_edges = distinct_edges.rename(
        {YEAST_IDEA_SOURCE: "upstream_name", YEAST_IDEA_TARGET: "downstream_name"},
        axis=1,
    ).assign(
        upstream_compartment="cellular_component",
        downstream_compartment="cellular_component",
        # tag reactions with the IDEA publication
        r_Identifiers=identifiers._format_Identifiers_pubmed(YEAST_IDEA_PUBMED_ID),
        r_isreversible=False,
    )

    # create some nice interaction names before we rename the roles as SBO terms
    formatted_distinct_edges["r_name"] = [
        f"{u} {d} {r} of {t}"
        for u, d, r, t in zip(
            formatted_distinct_edges["upstream_name"],
            formatted_distinct_edges["directness"],
            formatted_distinct_edges["role"],
            formatted_distinct_edges["downstream_name"],
        )
    ]

    # final interaction output
    # replace readable roles with entries in the SBO ontology
    interaction_edgelist = formatted_distinct_edges.replace(
        {"role": MINI_SBO_FROM_NAME}
    ).rename({"role": "sbo_term"}, axis=1)

    species_df = pd.DataFrame(
        {
            "s_name": list(
                {
                    *idea_kinetics_df[YEAST_IDEA_SOURCE],
                    *idea_kinetics_df[YEAST_IDEA_TARGET],
                }
            )
        }
    )

    # create Identifiers objects for each species
    species_df["s_Identifiers"] = [
        identifiers.Identifiers(
            [{"ontology": "gene_name", "identifier": x, "bqb": BQB.IS}]
        )
        for x in species_df["s_name"]
    ]

    # Constant fields (for this data source)

    # setup compartments (just treat this as uncompartmentalized for now)
    compartments_df = sbml_dfs_utils.stub_compartments()

    # Per convention unaggregated models receive an empty source
    interaction_source = source.Source(init=True)

    sbml_dfs = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist=interaction_edgelist,
        species_df=species_df,
        compartments_df=compartments_df,
        interaction_source=interaction_source,
        # additional attributes (directness) are added to reactions_data
        keep_reactions_data="idea",
    )
    sbml_dfs.validate()

    return sbml_dfs


def _summarize_idea_pairs(pairs_data: pd.DataFrame) -> pd.Series:
    """Rollup multiple records of a TF->target pair into a single summary."""

    # specify how to aggregate results if there are more than one entry for a TF-target pair
    # pull most attributes from the earliest change
    # this will favor direct over indirect naturally
    earliest_change = pairs_data.sort_values("t_rise").iloc[0].to_dict()

    KEYS_SUMMARIZED = ["v_inter", "v_final", "t_rise", "t_fall", "rate", "directness"]
    kinetic_timing_dict = {k: earliest_change[k] for k in KEYS_SUMMARIZED}

    # map v_inter (log2 fold-change change following perturbation) onto SBO terms for interactions
    if (any(pairs_data["v_inter"] > 0)) and (any(pairs_data["v_inter"] < 0)):
        kinetic_timing_dict["role"] = "modifier"
    elif all(pairs_data["v_inter"] > 0):
        kinetic_timing_dict["role"] = "stimulator"
    elif all(pairs_data["v_inter"] < 0):
        kinetic_timing_dict["role"] = "inhibitor"
    else:
        ValueError("Unexpected v_inter values")

    return pd.Series(kinetic_timing_dict)
