"""Instructions for the single-agent and Bayesian forecasting workflows."""

from typing import Literal

from .roles import ROLES

Stage = Literal["proposal", "review", "decision"]


def forecast_prompt(job_id: str, *, one_off: bool) -> str:
    return (
        f"Load the metaculus-forecast skill and complete forecasting job {job_id}. "
        "Use the Metaculus MCP tools to read the question and record your validated "
        "forecast. Research with your web tools. The runner handles output and publication settings. "
        + (
            "This is a standalone user-supplied question: do not look it up on Metaculus, "
            "ask for a Metaculus token, or publish it. "
            if one_off
            else ""
        )
        + "Do not change repository files. Do not ask follow-up questions; "
        "finish immediately after recording the result."
    )


def stage_prompt(stage: Stage, label: str, stage_id: str, *, one_off: bool) -> str:
    if stage == "proposal":
        instructions = ROLES.get(label, "")
    elif stage == "review":
        instructions = (
            f"Review the supplied work as the reviewer for the {label} proposal; "
            f"revise the posterior for the {label} perspective when appropriate; "
            "do not redo the initial research or rebuild the model."
        )
    else:
        instructions = ""
    standalone = (
        "This is a standalone user-supplied question: do not look it up on Metaculus "
        "or ask for a Metaculus token."
        if one_off
        else ""
    )
    return (
        f"Load the bayesian-ensemble skill and complete your {stage} stage {stage_id}. "
        f"{instructions} "
        "Use metaculus.get_bayesian_stage and metaculus.record_bayesian_stage. "
        "Follow the provided JSON schema, correct validation errors, and record your result. "
        "The Python runner pools the agents' distributions and stores the forecast; do not publish or edit files. "
        "Do not ask follow-up questions; finish immediately after recording the result. "
        f"{standalone}"
    )
