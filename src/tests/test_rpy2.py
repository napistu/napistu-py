from __future__ import annotations

import sys
from importlib import reload
from unittest.mock import Mock
from unittest.mock import patch

import pytest


# Patch rpy2_arrow.arrow to avoid ImportError
# if the R env is not properly set up during testing
sys.modules["rpy2_arrow.arrow"] = Mock()
import cpr.rpy2.callr  # noqa: E402
import cpr.rpy2.rids  # noqa: E402


def test_rpy2_has_rpy2_false():
    with patch.dict("sys.modules", {"rpy2": None}):
        reload(cpr.rpy2)
        assert cpr.rpy2.has_rpy2 is False
        # Test if other cpr.rpy2 modules can be
        # loaded without rpy2 installed
        reload(cpr.rpy2.callr)
        reload(cpr.rpy2.rids)


# def test_rpy2_has_rpy2_true():
#    with patch.dict("sys.modules", {"rpy2": "pytest"}):
#        reload(cpr.rpy2)
#        assert cpr.rpy2.has_rpy2 is True


@patch("cpr.rpy2.has_rpy2", False)
def test_warn_if_no_rpy2_false():
    @cpr.rpy2.warn_if_no_rpy2
    def test_func():
        pass

    with pytest.raises(ImportError):
        test_func()


@patch("cpr.rpy2.has_rpy2", True)
def test_warn_if_no_rpy2_true():
    @cpr.rpy2.warn_if_no_rpy2
    def test_func():
        pass

    test_func()


################################################
# __main__
################################################

if __name__ == "__main__":
    test_rpy2_has_rpy2_false()
    # test_rpy2_has_rpy2_true()
    test_warn_if_no_rpy2_false()
    test_warn_if_no_rpy2_true()
