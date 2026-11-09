# Forecasting strategy

This document describes the forecasting strategy at the level of its agents,
evidence, mathematical aggregation, and review. It is a conceptual specification
of how a forecast is formed, not a description of the software that runs it.

## Purpose and approach

The goal is a calibrated probability distribution for a precisely defined future
outcome, accompanied by an auditable explanation. A coding agent conducts
research, reasons about evidence, and, when useful and feasible, runs
computations or builds explicit models. Different agent roles provide distinct
perspectives. Each agent forms a reasoned Bayesian forecast; the ensemble pools
those forecasts and lets the agents review and revise their judgments.

All supported question types use the ensemble below by default. A single agent
or another configuration may be selected explicitly. Numerical and date
forecasts express uncertainty as distributions; conditional forecasts include
the condition and the outcome distributions given each condition.

## Agents and research

The ensemble begins with independent proposals from seven perspectives:

- **Generalist:** assess the whole question and balance competing scenarios.
- **Base-rate analyst:** find reference classes, historical frequencies, and a
  defensible prior.
- **Current-evidence analyst:** investigate recent developments and authoritative
  sources.
- **Skeptic:** challenge assumptions and seek credible counterevidence.
- **Quantitative analyst:** examine measurable trends, likelihoods, and uncertainty.
- **Economic modeler:** develop an explicit economic model when relevant.
- **Causal modeler:** develop an explicit Bayesian causal model when relevant.

The modeling agents explain their assumptions, data, uncertainty, and limitations,
including whether computations were actually performed. They may declare their
model inapplicable, with reasons. A model's output is an interpretation of its
inputs, not an additional independent observation.

Agents have web search and source retrieval for general research and primary
source verification. Depending on availability, they may also use source-text
search, news research and summaries, and research services that synthesize
sources. They select tools according to the question rather than calling every
service. Provider-generated summaries are leads, not authority: important claims
should be checked against dated sources. Agents search both for recent evidence
and for historical or reference-class information, seek contrary evidence, and
report unavailable tools or missing information without inventing sources.
Computational tools support numerical, statistical, symbolic, and probabilistic
modeling when appropriate and available.

## Shared superforecasting guidance

Every forecasting agent is prompted to clarify resolution criteria and break the
question into tractable parts; start from an outside-view base rate; and then
consider the specifics, time horizon, and alternative scenarios. Agents seek
diverse evidence and counterarguments, distinguish facts from assumptions, and
assess source quality, dates, and dependence. They express uncertainty
quantitatively without false precision, update in light of how expected evidence
is under competing outcomes, and remain open to criticism and revision. They
should learn from calibration and past errors rather than defer blindly to other
agents.

## Ensemble aggregation and review

Each proposing agent supplies its prior and posterior distributions, reasons for
the update, sourced observations, research, assumptions, and limitations. An
observation is not a prior, prediction, model output, or merely an absence of
information. Agents identify overlapping sources and dependence and do not
update again merely because another agent repeats information already considered.

For completed agents with posterior probability measures $P_i$, the initial
consensus is the linear opinion pool

$$
P = \sum_i w_i P_i, \qquad w_i \ge 0, \qquad \sum_i w_i = 1.
$$

Weights are equal by default and remain fixed during review. This pools judgments
without assuming that agents or their evidence are independent. It is not, in
general, a verified Bayesian posterior for all information. For binary questions
it pools event probabilities; for multiple choice it pools complete option
vectors. For numeric and date questions it pools cumulative distribution
functions (CDFs), not quantiles or fixed outcome bins.

Numeric, discrete-numeric, and date agents express distributions using ordered interior quantiles.
The CDF is linear between quantiles and, for a closed tail, linear to the
question's bound. For an open tail it decays exponentially beyond the nearest
quantile, with scale equal to the adjacent quantile spacing. These interpolation
and tail rules are explicit modeling assumptions, not facts determined by the
quantiles. Open tails are not truncated in the underlying pooled distribution.
Representing the pool for external consumption uses a finite set of quantiles
and can introduce additional approximation. For discrete numeric questions,
this is a latent continuous CDF: the platform maps it to its discrete buckets,
rather than the ensemble claiming a continuous outcome is possible. Convergence
uses the latent CDF difference, a conservative bound at any common bucket cut.

For conditional questions, each agent supplies a probability for the condition
and outcome distributions assuming that it is true and false. The ensemble
pools these joint distributions, then derives the condition, conditional outcome,
and unconditional outcome distributions. The weights within each conditional
branch depend on each agent's condition probability. A zero-probability branch
cannot define a conditional distribution and causes the ensemble to fail closed.

## Review, convergence, and output

The proposing perspectives independently review the consensus, other agents'
priors, posteriors, evidence, dependence assumptions, and models. They flag
errors and may revise their own posteriors with reasons; a critique is not itself
a new evidence factor. Each round recomputes the pool from the agents' current
posteriors and fixed weights, rather than repeatedly pooling an earlier consensus.
A round converges only when every completed reviewer accepts the forecast they
reviewed and the new consensus changes by no more than the configured tolerance.
Change is the absolute probability difference for binary questions, the largest
option-probability difference for multiple choice, or the largest CDF difference
for numeric and date questions. For conditional questions it is the largest
change across the condition and all derived outcome distributions. Numerical
comparison includes tails and uses a bounded approximation.

Both proposal and review stages require a configured quorum and the economic and
causal modeling roles. If a required stage fails or the ensemble does not
converge within the allowed rounds, it fails without publishing a forecast.
Agreement and numerical stability do not establish that the assumptions are
correct.

The final result includes the forecast, research, and rationale: agent priors and
posteriors, evidence and sources, dependence assumptions, review and revision
history, weights, model artifacts, and remaining limitations. Probability bounds
and platform conversion may be applied to the reported prediction; the
underlying pooled judgments are retained separately. The pool, its stability, and
reviewer agreement do not guarantee calibration or correctness.
