# Forecasting strategy

This package describes the bot's forecasting behavior; the neighboring
`implementation` package supplies execution, persistence, and validation.

- `workflow.py` selects the single-agent or ensemble path. Binary and
  multiple-choice questions use the ensemble by default; other question types
  and explicit single-agent runs use the single-agent path.
- `roles.py`, `principles.py`, and `prompts.py` define the independent
  perspectives and instructions. The agents and skills in `opencode/` contain
  the detailed forecasting, research, and modeling guidance.
- The ensemble gathers independent proposals, requires the configured quorum
  and both modeling roles, adjudicates an initial evidence ledger, then runs
  review and adjudication rounds. It fails closed unless all reviewers accept
  and the posterior change is within the configured tolerance.
- `aggregation.py` selects the binary or multiple-choice Bayesian update and
  prepares the research and rationale. The implementation validates evidence
  groups, checks source overlap, and calculates each posterior from the prior.
- `research.py` and the `forecasting-research` skill describe available tools
  and source-selection guidance. Tool selection remains an agent judgment;
  network clients and credentials are handled in infrastructure.

Settings such as quorum, review rounds, model selection, and timeouts remain
configurable through the existing CLI and environment settings.
