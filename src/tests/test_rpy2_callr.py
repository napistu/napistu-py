from __future__ import annotations

import sys
from unittest.mock import Mock, patch
import pandas as pd
import pytest


# Mock all rpy2 dependencies before any imports to prevent ImportErrors
sys.modules["rpy2"] = Mock()
sys.modules["rpy2.robjects"] = Mock()
sys.modules["rpy2.robjects.conversion"] = Mock()
sys.modules["rpy2.robjects.default_converter"] = Mock()
sys.modules["rpy2.robjects.packages"] = Mock()
sys.modules["rpy2.robjects.pandas2ri"] = Mock()
sys.modules["rpy2.rinterface"] = Mock()
sys.modules["rpy2_arrow"] = Mock()
sys.modules["rpy2_arrow.arrow"] = Mock()

import napistu.rpy2.callr  # noqa: E402


def test_get_napistu_r_without_rpy2():
    """Test get_napistu_r when rpy2 not available."""
    with patch("napistu.rpy2.get_rpy2_availability", return_value=False):
        with pytest.raises(ImportError, match="requires `rpy2`"):
            napistu.rpy2.callr.get_napistu_r()


def test_bioconductor_org_r_function_without_rpy2():
    """Test bioconductor function when rpy2 not available."""
    with patch("napistu.rpy2.get_rpy2_availability", return_value=False):
        with pytest.raises(ImportError):
            napistu.rpy2.callr.bioconductor_org_r_function("test", "Homo sapiens")


def test_get_rbase_without_rpy2():
    """Test get_rbase when rpy2 not available."""
    with patch("napistu.rpy2.get_rpy2_availability", return_value=False):
        with pytest.raises(ImportError):
            napistu.rpy2.callr.get_rbase()


def test_pandas_conversion_functions_without_rpy2():
    """Test pandas conversion functions when rpy2 not available."""
    with patch("napistu.rpy2.get_rpy2_availability", return_value=False):
        # Test pandas to R conversion
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ImportError):
            napistu.rpy2.callr.pandas_to_r_dataframe(df)

        # Test R to pandas conversion
        mock_rdf = Mock()
        with pytest.raises(ImportError):
            napistu.rpy2.callr.r_dataframe_to_pandas(mock_rdf)

        # Test internal converter function
        with pytest.raises(ImportError):
            napistu.rpy2.callr._get_py2rpy_pandas_conv()


def test_callr_module_import_safety():
    """Test that callr module can be imported safely without triggering R initialization."""
    import importlib

    # Test module can be imported without rpy2
    with patch.dict("sys.modules", {"rpy2": None}):
        # This should not raise ImportError during import
        importlib.reload(napistu.rpy2.callr)

    # Test that function calls fail appropriately but imports don't
    with patch("napistu.rpy2.get_rpy2_availability", return_value=False):
        # Functions should fail with ImportError, not other errors
        with pytest.raises(ImportError):
            napistu.rpy2.callr.get_napistu_r()

        with pytest.raises(ImportError):
            napistu.rpy2.callr.get_rbase()
