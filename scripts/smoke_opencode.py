"""Opt-in live OpenCode/MCP smoke test; uses a synthetic question and never publishes."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting_tools import BinaryQuestion
from metac_bot.jobs import ROOT, create_job, read_report
from metac_bot.opencode import OpenCode, discover_service


async def main():
    job_id = create_job(
        BinaryQuestion(
            question_text="Will a fair coin's next independent toss land heads? (synthetic smoke test)",
            resolution_criteria="Heads resolves YES, tails resolves NO. This is an unperformed future toss.",
            background_info="A hypothetical fair coin, with no edge outcome, has equal probability of heads and tails.",
        )
    )
    url, auth = discover_service()
    async with OpenCode(url, auth) as client:
        session_id = await client.create_session(ROOT, "Forecast MCP smoke test")
        print(f"Session: {session_id}; job: {job_id}", flush=True)
        await client.prompt(
            session_id,
            (
                f"This is a synthetic integration test. Load the metaculus-forecast skill, "
                f"read job {job_id} using the Metaculus MCP tool, and record a forecast using "
                "record_forecast. Use the supplied fair-coin definition as your evidence; "
                "no web research is needed for this synthetic fixture. Research should state "
                "that it is a synthetic test with no external sources. Do not publish anything "
                "or edit any repository files."
            ),
        )
        await client.wait(session_id, 180)
        report = read_report(job_id)
        if report.prediction != 0.5:
            raise AssertionError(
                f"Expected a fair-coin prediction of 0.5, got {report.prediction}"
            )
        print(
            "PASS: OpenCode read the job and recorded a validated 0.5 forecast through MCP."
        )


if __name__ == "__main__":
    asyncio.run(main())
