---
name: Bayesian Causal Modeling
description: Construct a transparent Bayesian causal model with a causal graph, assumptions, and uncertainty for an ensemble forecasting question.
---

# Bayesian causal modeling for forecasts

1. Read the assigned question and resolution criteria. Determine whether a
   causal model can inform it. If not, mark it inapplicable, explain why, and
   fill the remaining required artifact fields with "not applicable".
2. Define the target outcome, exposure or intervention, time horizon, confounders,
   mediators, measurement processes, and selection mechanisms. Specify a directed
   acyclic graph or structural causal model in text, distinguishing observed and
   latent variables. Explain why the assumed arrows are defensible.
3. Specify the causal estimand and distinguish observing an exposure from
   intervening on it. State assumptions required for identification, such as
   consistency, exchangeability, positivity and measurement assumptions when
   relevant. Avoid conditioning on colliders or post-treatment variables without
   justification. If identification is not possible, say so.
4. Specify a joint probability model or structural equations, parameter priors,
   likelihood, data sources and the relationship to the YES event or each
   multiple-choice option. Treat related
   observations as dependent where appropriate. Do not present an observational
   correlation as a causal effect without supporting assumptions.
5. Load the `modeling-python` skill and describe posterior inference and uncertainty.
   PyMC is available for explicit probabilistic models and posterior sampling. When data and computation tools
   are available, provide reproducible code and perform a numerical check; clearly
   state whether it was executed and cite inputs. Otherwise give a qualitative model
   and do not invent numerical estimates. Evaluate sensitivity to priors, omitted
   confounding, measurement and model misspecification.
6. Fill the `model` object in your Bayesian proposal: `kind` is `causal`;
   describe applicability, graph/structure, equations, input sources, inference and
   implementation, result and uncertainty, and limitations. Cite observations
   used as evidence; a posterior derived from them is not a separate independent
   observation to multiply into the ensemble ledger.
