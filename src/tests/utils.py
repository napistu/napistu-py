from __future__ import annotations

import os
import shutil

from napistu import indices


def test_setup(test_data_path="/home/sean/cpr/lib/python/cpr/tests/test_data"):
    cpr_assets_root = "/group/cpr"

    # setup sbmls

    pw_index = indices.PWIndex(
        os.path.join(cpr_assets_root, "reactome/sbml/pw_index.tsv")
    )
    pw_index.filter(species="Homo sapiens")
    pw_index.search("carbon")

    # add pw_index
    pw_index.index.to_csv(os.path.join(test_data_path, "pw_index.tsv"), sep="\t")

    # move all sbmls in pw_index
    [
        shutil.copyfile(
            os.path.join(cpr_assets_root, "reactome", "sbml", f),
            os.path.join(test_data_path, f),
        )
        for f in pw_index.index["file"].tolist()
    ]
