"""Metaculus forecasting capabilities exposed through MCP stdio."""

import contextlib
import sys

# Some forecasting-tools imports print diagnostics; stdout belongs to MCP.
with contextlib.redirect_stdout(sys.stderr):
    from dotenv import load_dotenv
    from .jobs import ROOT, get_job, save_forecast
    from .bayesian_jobs import get_stage, record_stage
    from mcp.server.fastmcp import FastMCP

load_dotenv(ROOT / ".env")
mcp = FastMCP(
    "metaculus",
    instructions=(
        "For a normal forecast job, use get_forecast_job and record_forecast with "
        "the metaculus-forecast skill. For an assigned Bayesian stage, use "
        "get_bayesian_stage and record_bayesian_stage with the bayesian-ensemble skill. "
        "These tools save validated artifacts; the Python runner handles publication."
    ),
)


@mcp.tool()
def get_forecast_job(job_id: str) -> dict:
    """Read a runner-created question snapshot and its exact forecast JSON schema."""
    return get_job(job_id)


@mcp.tool()
def record_forecast(
    job_id: str, prediction: dict, research: str, rationale: str
) -> dict:
    """Validate and save a forecast. Research should cite source URLs and dates.

    prediction must match get_forecast_job's schema. Probabilities are decimals,
    not percentages. Fix any validation errors and call again. Does not publish.
    """
    return save_forecast(job_id, prediction, research, rationale)


@mcp.tool()
def get_bayesian_stage(stage_id: str) -> dict:
    """Read an assigned Bayesian ensemble stage, context, and result JSON schema."""
    return get_stage(stage_id)


@mcp.tool()
def record_bayesian_stage(stage_id: str, result: dict) -> dict:
    """Validate and save the assigned proposal, review, or decision. Never publishes.

    Use the schema from get_bayesian_stage. Fix validation errors and retry.
    Decision groups must account for every candidate exactly once; the same
    normalized source URL must not occur in separate groups.
    """
    return record_stage(stage_id, result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
