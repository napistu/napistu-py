from __future__ import annotations

import os
import re

import pandas as pd
import requests

from napistu.ingestion.constants import (
    IDENTIFIERS_ETL_SBO_URL,
    IDENTIFIERS_ETL_YEAST_FIELDS,
    IDENTIFIERS_ETL_YEAST_HEADER_REGEX,
    IDENTIFIERS_ETL_YEAST_URL,
)


def read_yeast_identifiers(url: str = IDENTIFIERS_ETL_YEAST_URL):
    """Read Yeast Identifiers
    Generate a pd.DataFrame which maps between yeast identifiers including
    common and systematic (OLN) names, as well as Swiss-Prot and SGD identifiers.

    Params:
        url (str): url to the identifier file
    Returns:
        pd.DataFrame with one row per gene
    """
    response = requests.get(url).text

    yeast_id_list = list()
    break_line_hit = 0
    for line in response.splitlines():
        if re.match(IDENTIFIERS_ETL_YEAST_HEADER_REGEX, line):
            # find start and end of header indicated by a line of underscores
            break_line_hit += 1
            continue

        if break_line_hit >= 2:
            if line == "":
                # reached the end
                break

            # split each line into a list of fields, the only optional field is 3d
            # all white spaces are space
            line = re.sub(" +", " ", line)
            line = re.sub("; ", ";", line)
            # remove pol and gag designations from transposons since they are an unnecessary extra field
            line = re.sub("(-[0-9]) (GAG)|(POL)", "\\1", line)

            line = line.split()

            if line[6] != "(3)":
                # if no 3D field is present then create one
                line.insert(6, "none")

            # split common fields into a separate list
            common_list = line[0].split(";")
            line[0] = common_list[0]
            line.insert(1, common_list)

            if len(line) != 9:
                raise ValueError(
                    "the yeast id file could not be read; all entries should have 8 fields"
                )

            yeast_id_list.append(dict(zip(IDENTIFIERS_ETL_YEAST_FIELDS, line)))

    return pd.DataFrame(yeast_id_list)


def read_sbo_ontology(
    url: str = IDENTIFIERS_ETL_SBO_URL, verbose: bool = False
) -> pd.DataFrame:
    """Read SBO Ontology
    Read the Systems Biology Ontology (SBO) identifiers and reformat the obo results into a pd.DataFrame.

    Params:
        url (str): url to the obo specification file
        verbose (bool): throw warnings when attributes are overwritten
    Returns:
        pd.DataFrame
    """

    # save the obo file locally
    tmp_file = os.path.join("/tmp", "sbo.obo")
    r = requests.get(url, allow_redirects=True)
    open(tmp_file, "wb").write(r.content)

    with open(tmp_file) as sbo:
        sbo_dict = dict()
        current_id = None
        in_header = True
        for line in sbo:
            # skip the header
            if line == "[Term]\n":
                in_header = False
                continue
            if in_header:
                continue

            line_entries = line.split(":", 1)

            if len(line_entries) == 2:
                entry_type = line_entries[0]
                entry_value = line_entries[1].strip()

                # drop type defs
                if (
                    (current_id is not None)
                    and (entry_type != "id")
                    and (re.match("SBO", current_id) is None)
                ):
                    continue

                # clean-up definitions
                if entry_type == "is_a":
                    entry_value = re.match("SBO:[0-9]+", entry_value)[0]

                # if a new id has been reached then initilize a new dict and
                # update current id

                if entry_type == "id":
                    current_id = entry_value
                    if re.match("SBO", current_id) is not None:
                        sbo_dict[current_id] = {"is_a": []}
                    continue

                if entry_type == "is_a":
                    sbo_dict[current_id]["is_a"].append(entry_value)
                else:
                    # add a new entry
                    if (entry_type in sbo_dict[current_id].keys()) and verbose:
                        print(
                            f"2+ {entry_type} entries were found for {current_id}, only one value should be present "
                        )
                    sbo_dict[current_id][entry_type] = entry_value

    sbo_df = pd.DataFrame(sbo_dict).T

    obsolete_terms = set(
        sbo_df["name"][sbo_df["name"].str.match("obsolete")].index.tolist()
    )
    sbo_df["is_obsolete"] = [
        (x in obsolete_terms) | (len(set(y).intersection(obsolete_terms)) > 0)
        for x, y in zip(sbo_df.index, sbo_df["is_a"])
    ]

    sbo_df = sbo_df[["name", "comment", "is_a", "is_obsolete"]]
    sbo_df.index.name = "sbo_term"
    sbo_df = sbo_df.reset_index()

    return sbo_df
