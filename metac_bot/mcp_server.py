"""Compatibility entry point for the forecasting MCP server."""

from .forecasting.implementation.mcp_server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
