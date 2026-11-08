"""Compatibility entry point for the research MCP server."""

from .forecasting.research_mcp import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
