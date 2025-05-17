"""
Tests for the Napistu MCP server functionality.
"""

import pytest
import importlib.util

# Check if MCP extras are installed
mcp_available = importlib.util.find_spec("mcp") is not None

@pytest.mark.skipif(not mcp_available, reason="MCP extras not installed")
class TestMCPFunctionality:
    
    def test_mcp_availability(self):
        """Test that MCP functionality is correctly marked as available."""
        import napistu
        assert hasattr(napistu, "mcp_available")
        assert napistu.mcp_available == True
    
    def test_server_profiles(self):
        """Test server profiles."""
        from napistu.mcp.profiles import get_profile
        
        # Test local profile
        local_profile = get_profile("local")
        config = local_profile.get_config()
        assert config["server_name"] == "napistu-local"
        assert config["enable_execution"] == True
        
        # Test remote profile
        remote_profile = get_profile("remote")
        config = remote_profile.get_config()
        assert config["server_name"] == "napistu-docs"
        assert config["enable_documentation"] == True
        assert config["enable_codebase"] == True
        assert config["enable_tutorials"] == True
        
        # Test full profile
        full_profile = get_profile("full")
        config = full_profile.get_config()
        assert config["enable_documentation"] == True
        assert config["enable_codebase"] == True
        assert config["enable_execution"] == True
        assert config["enable_tutorials"] == True
        
        # Test overrides
        custom_profile = get_profile("local", enable_documentation=True, server_name="custom")
        config = custom_profile.get_config()
        assert config["server_name"] == "custom"
        assert config["enable_execution"] == True
        assert config["enable_documentation"] == True
    
    def test_start_server(self):
        """Test that servers can be started with different profiles."""
        from napistu.mcp import start_server
        
        # Start local server
        local = start_server("local")
        assert local["status"] == "running"
        assert local["profile"] == "local"
        assert callable(local["register_object"])
        
        # Start remote server
        remote = start_server("remote")
        assert remote["status"] == "running"
        assert remote["profile"] == "remote"
        assert remote["register_object"] is None  # Execution not enabled
        
        # Stop the servers
        local["stop"]()
        remote["stop"]()
    
    def test_register_object(self):
        """Test that objects can be registered with the server."""
        from napistu.mcp import start_server, register_object
        
        # Start the server
        server = start_server("local")
        
        # Create a test object
        test_obj = {"name": "test", "value": 42}
        
        # Register the object
        register_object("test_obj", test_obj)
        
        # Stop the server
        server["stop"]()


class TestMCPFallback:
    
    @pytest.mark.skipif(mcp_available, reason="MCP extras are installed")
    def test_import_error_if_not_installed(self):
        """Test that ImportError is raised if MCP is not installed."""
        import napistu
        assert hasattr(napistu, "mcp_available")
        assert napistu.mcp_available == False
        
        with pytest.raises(ImportError):
            napistu.start_server()
        
        with pytest.raises(ImportError):
            napistu.start_local_server()
        
        with pytest.raises(ImportError):
            napistu.start_remote_server()
        
        with pytest.raises(ImportError):
            napistu.register_object("test", {})