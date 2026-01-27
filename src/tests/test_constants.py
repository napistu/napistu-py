from __future__ import annotations

from napistu.constants import (
    BQB_PRIORITIES,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    MINI_SBO_TO_NAME,
    SBO_NAME_TO_ROLE,
    VALID_BQB_TERMS,
    VALID_SBO_ROLES,
)


def test_sbo_constants():
    # all SBO terms in "MINI_SBO" set have a role
    assert set(SBO_NAME_TO_ROLE.keys()) == set(MINI_SBO_FROM_NAME.keys())
    # all roles are valid
    assert [x in VALID_SBO_ROLES for x in SBO_NAME_TO_ROLE.values()]


def test_bqb_priorities():

    assert not any(BQB_PRIORITIES[IDENTIFIERS.BQB].duplicated())
    assert set(BQB_PRIORITIES[IDENTIFIERS.BQB]) == set(VALID_BQB_TERMS)


def test_sbo_to_from_name():
    assert len(MINI_SBO_TO_NAME) == len(MINI_SBO_FROM_NAME)

    for k, v in MINI_SBO_FROM_NAME.items():
        assert MINI_SBO_TO_NAME[v] == k
