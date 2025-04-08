from __future__ import annotations

from napistu import constants


def test_sbo_constants():
    # all SBO terms in "MINI_SBO" set have a role
    assert set(constants.SBO_NAME_TO_ROLE.keys()) == set(
        constants.MINI_SBO_FROM_NAME.keys()
    )
    # all roles are valid
    assert [x in constants.VALID_SBO_ROLES for x in constants.SBO_NAME_TO_ROLE.values()]


################################################
# __main__
################################################

if __name__ == "__main__":
    test_sbo_constants()
