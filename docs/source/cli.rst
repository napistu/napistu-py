Command Line Interface
======================

Napistu CLI
-----------

The main Napistu CLI provides commands for data ingestion, integration, consensus building, and more.

Main Command
~~~~~~~~~~~~

.. click:: napistu.__main__:cli
   :prog: napistu
   :nested: full

This will automatically generate documentation for all commands, subcommands, arguments, and options.


MCP Server CLI
--------------

The MCP server CLI provides commands for starting and managing MCP servers.

.. click:: napistu.mcp.__main__:cli
   :prog: napistu-mcp
   :nested: full