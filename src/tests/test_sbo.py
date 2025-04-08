from __future__ import annotations

from napistu.constants import MINI_SBO_FROM_NAME
from napistu.constants import MINI_SBO_TO_NAME


def test_sbo_to_from_name():
    assert len(MINI_SBO_TO_NAME) == len(MINI_SBO_FROM_NAME)

    for k, v in MINI_SBO_FROM_NAME.items():
        assert MINI_SBO_TO_NAME[v] == k
