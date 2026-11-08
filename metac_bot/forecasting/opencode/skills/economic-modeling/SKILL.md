---
name: Economic Modeling
description: Construct and evaluate a transparent economic model, including a DSGE-style model when justified, for an ensemble forecasting question.
---

# Economic modeling for forecasts

1. Read the assigned question and resolution criteria. Decide whether an
   economic mechanism is relevant. If not, mark the model inapplicable, explain
   why, and fill the remaining required artifact fields with "not applicable".
2. Choose the simplest defensible model. For macroeconomic questions a DSGE-style
   model may be appropriate; otherwise a partial-equilibrium, accounting,
   time-series, or scenario model may be better. Explain the choice and horizon.
3. Specify variables, units, timing, behavioral relationships, shocks, boundary
   conditions, and the link from the model outcome to the YES event or each
   multiple-choice option.
   For a DSGE model, specify agents' objectives and constraints, equilibrium and
   market-clearing conditions, expectations, policy rules, shock processes, and
   steady state where applicable. Do not claim a model is DSGE merely because it
   has equations. Explicitly label stylized or partial models.
4. Cite sources for data, calibration, parameters, and base rates. State priors,
   parameter uncertainty, identification assumptions, and alternatives. Distinguish
   historical estimates, assumptions, and hypothetical inputs; never fabricate data.
5. Load the `modeling-python` skill. Give equations and a reproducible numerical procedure or code when feasible.
   Use available computation tools to actually calculate/check results when possible.
   State clearly whether code was executed and which inputs were used; if data,
   permissions or tools are unavailable, give a qualitative model and do not invent
   a numerical forecast. Discuss sensitivity and model misspecification.
6. Fill the `model` object in your Bayesian proposal: `kind` is `economic`;
   describe applicability, structure, equations, input sources, inference and
   implementation, result and uncertainty, and limitations. If you propose
   evidence, cite underlying observations and explain the likelihood ratio. The
   model output is an interpretation of those observations, not an additional
   independent observation to multiply into the evidence ledger.
