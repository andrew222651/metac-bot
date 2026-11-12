# Forecasting strategy

This package describes the bot's forecasting behavior; the neighboring
`implementation` package supplies execution, persistence, and validation.

- `workflow.py` selects the single-agent or ensemble path. All supported
  question types use the ensemble by default; single-agent runs are explicit.
- `roles.py`, `principles.py`, and `prompts.py` define the independent
  perspectives and instructions. The agents and skills in `opencode/` contain
  the detailed forecasting, research, and modeling guidance.
- The ensemble gathers all successful posterior proposals and runs independent review
  rounds. It pools current distributions with equal weights, without treating shared sources as
  independent. With completed reviews it fails closed unless all completed reviewers
  accept and the consensus change is within the configured tolerance. If a round
  produces no reviews, it returns the latest available pool with a warning.
- `pooling.py` handles all question types and coherent conditional joints;
  `distributions.py` provides numeric and date CDF interpolation, explicit tail
  assumptions, pooling, and bounded convergence comparison. Platform output
  approximates continuous pools with quantiles.
- `research.py` and the `forecasting-research` skill describe available tools
  and source-selection guidance. Tool selection remains an agent judgment;
  network clients and credentials are handled in infrastructure.

Settings such as review rounds, model selection, and timeouts remain
configurable through the existing CLI and environment settings.
