from __future__ import annotations

import sys
from unittest.mock import Mock, patch

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

import napistu.rpy2  # noqa: E402


def test_rpy2_availability_detection():
    """Test rpy2 availability detection in various scenarios."""
    # Test ImportError case
    with patch.dict("sys.modules", {"rpy2": None}):
        if hasattr(napistu.rpy2.get_rpy2_availability, "cache_clear"):
            napistu.rpy2.get_rpy2_availability.cache_clear()
        assert napistu.rpy2.get_rpy2_availability() is False

    # Test other exception case during import
    with patch("builtins.__import__") as mock_import:
        mock_import.side_effect = RuntimeError("R installation broken")
        if hasattr(napistu.rpy2.get_rpy2_availability, "cache_clear"):
            napistu.rpy2.get_rpy2_availability.cache_clear()
        assert napistu.rpy2.get_rpy2_availability() is False

    # Test success case
    mock_rpy2 = Mock()
    with patch.dict("sys.modules", {"rpy2": mock_rpy2}):
        if hasattr(napistu.rpy2.get_rpy2_availability, "cache_clear"):
            napistu.rpy2.get_rpy2_availability.cache_clear()
        assert napistu.rpy2.get_rpy2_availability() is True


def test_caching_behavior():
    """Test that lazy loading functions are properly cached."""
    # Test availability caching
    with patch("builtins.__import__") as mock_import:
        mock_import.return_value = Mock()
        if hasattr(napistu.rpy2.get_rpy2_availability, "cache_clear"):
            napistu.rpy2.get_rpy2_availability.cache_clear()

        result1 = napistu.rpy2.get_rpy2_availability()
        result2 = napistu.rpy2.get_rpy2_availability()

        assert result1 == result2
        # Only test call count if caching is enabled
        if hasattr(napistu.rpy2.get_rpy2_availability, "cache_clear"):
            assert mock_import.call_count == 1  # Should only be called once

    # Test core modules caching
    with patch("napistu.rpy2.get_rpy2_availability", return_value=True):
        with (
            patch("rpy2.robjects.conversion"),
            patch("rpy2.robjects.default_converter"),
            patch("rpy2.robjects.packages.importr"),
        ):
            if hasattr(napistu.rpy2.get_rpy2_core_modules, "cache_clear"):
                napistu.rpy2.get_rpy2_core_modules.cache_clear()

            result1 = napistu.rpy2.get_rpy2_core_modules()
            result2 = napistu.rpy2.get_rpy2_core_modules()

            # Only test object identity if caching is enabled
            if hasattr(napistu.rpy2.get_rpy2_core_modules, "cache_clear"):
                assert result1 is result2  # Same object due to caching


def test_lazy_import_functions_without_rpy2():
    """Test that lazy import functions fail appropriately when rpy2 unavailable."""
    with patch("napistu.rpy2.get_rpy2_availability", return_value=False):
        # Clear all caches if they exist
        if hasattr(napistu.rpy2.get_rpy2_core_modules, "cache_clear"):
            napistu.rpy2.get_rpy2_core_modules.cache_clear()
        if hasattr(napistu.rpy2.get_rpy2_extended_modules, "cache_clear"):
            napistu.rpy2.get_rpy2_extended_modules.cache_clear()
        if hasattr(napistu.rpy2.get_napistu_r_package, "cache_clear"):
            napistu.rpy2.get_napistu_r_package.cache_clear()

        # All should raise ImportError
        with pytest.raises(ImportError, match="requires `rpy2`"):
            napistu.rpy2.get_rpy2_core_modules()

        with pytest.raises(ImportError, match="requires `rpy2`"):
            napistu.rpy2.get_rpy2_extended_modules()

        with pytest.raises(ImportError, match="requires `rpy2`"):
            napistu.rpy2.get_napistu_r_package()


def test_decorators():
    """Test require_rpy2 and report_r_exceptions decorators."""
    # Test require_rpy2 with rpy2 available
    with patch("napistu.rpy2.get_rpy2_availability", return_value=True):

        @napistu.rpy2.require_rpy2
        def test_func_success():
            return "success"

        assert test_func_success() == "success"

    # Test require_rpy2 without rpy2
    with patch("napistu.rpy2.get_rpy2_availability", return_value=False):

        @napistu.rpy2.require_rpy2
        def test_func_fail():
            return "success"

        with pytest.raises(ImportError, match="test_func_fail.*requires `rpy2`"):
            test_func_fail()

    # Test report_r_exceptions with success
    with patch("napistu.rpy2.get_rpy2_availability", return_value=True):

        @napistu.rpy2.report_r_exceptions
        def test_func_report_success():
            return "success"

        assert test_func_report_success() == "success"

    # Test report_r_exceptions with failure
    with patch("napistu.rpy2.get_rpy2_availability", return_value=True):
        with patch("napistu.rpy2.rsession_info") as mock_rsession:

            @napistu.rpy2.report_r_exceptions
            def test_func_report_fail():
                raise ValueError("R function failed")

            with pytest.raises(ValueError, match="R function failed"):
                test_func_report_fail()

            mock_rsession.assert_called_once()


def test_module_import_safety():
    """Test that modules can be imported safely without triggering R initialization."""
    import importlib

    # Test modules can be imported without rpy2
    with patch.dict("sys.modules", {"rpy2": None}):
        # These should not raise ImportError during import
        importlib.reload(napistu.rpy2)
