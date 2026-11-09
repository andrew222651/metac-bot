# Forecasting strategy

This document describes the forecasting strategy at the level of its agents,
evidence, mathematical aggregation, and review. It is a conceptual specification
of how a forecast is formed.


## Purpose and approach

The goal is an accurate probability distribution for a precisely defined future
outcome, accompanied by an auditable explanation. Coding agents conduct
research, reason about evidence, and, when useful and feasible, run
computations or build explicit models. Different agent roles provide distinct
perspectives. Each agent forms a reasoned Bayesian forecast; the ensemble pools
those forecasts and lets the agents review and revise their judgments.

Numerical and date
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

Agents can use these research tools, subject to configuration and availability:

- **OpenCode web search and fetch:** general research and primary-source retrieval.
- **Exa** (`research.exa_search`): source-text search, optionally limited to
  specified domains.
- **AskNews** (`research.asknews_research`): news summaries and deep research.
- **Perplexity** (`research.perplexity_research`): search-grounded research and
  synthesis with sources.

Agents select tools according to the question rather than calling every service.
Provider-generated summaries are leads, not authority: important claims
should be checked against dated sources. Agents search both for recent evidence
and for historical or reference-class information, seek contrary evidence, and
report unavailable tools or missing information without inventing sources.

All agents are informed that their Python environment includes NumPy,
SciPy, SymPy, pandas, statsmodels, NetworkX, PyMC, and ArviZ. They are encouraged
to use basic statistical modeling when useful and feasible, while explaining
assumptions, uncertainty, limitations, and whether computations actually ran.


## Shared superforecasting guidance

Every forecasting agent's prompt includes a summary of superforecasting principles.


## Ensemble aggregation and review

Each proposing agent supplies its prior and posterior distributions, reasons for
the update, sourced observations, research, assumptions, and limitations. An
observation is not a prior, prediction, model output, or merely an absence of
information. Agents identify overlapping sources and dependence and do not
update again merely because another agent repeats information already considered.

Agents may also discuss auxiliary probabilities for related events, such as
possible causal factors, as events in a larger probability space containing the
forecast outcome. They explain how these probabilities relate to the target and
identify relevant assumptions and dependencies. These auxiliary judgments can
inform the reasoning and review, but are not automatically treated as separate
observations or pooled as part of the requested forecast.

For completed agents with posterior probability measures $P_i$, the initial
consensus is the linear opinion pool

$$
P = \sum_i w_i P_i, \qquad w_i \ge 0, \qquad \sum_i w_i = 1.
$$

Weights are equal by default and remain fixed during review. For binary questions
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

In each round, a reviewer is an agent in a new session assigned the perspective
of one completed proposal. Reviewers independently assess the consensus and the
agents' priors, posteriors, evidence, dependence assumptions, and models. Each
reviewer flags errors and may revise the posterior for its assigned perspective
with reasons; a critique is not itself a new evidence factor. Each round recomputes
the pool from the agents' current posteriors and fixed weights.
A round converges only when every completed reviewer accepts the forecast they
reviewed and the new consensus changes by no more than the configured tolerance.
Change is the absolute probability difference for binary questions, the largest
option-probability difference for multiple choice, or the largest CDF difference
for numeric and date questions. For conditional questions it is the largest
change across the condition and all derived outcome distributions. Numerical
comparison includes tails and uses a bounded approximation.

The final result includes the forecast, research, and rationale: agent priors and
posteriors, evidence and sources, dependence assumptions, review and revision
history, weights, model artifacts, and remaining limitations. Probability bounds
and platform conversion may be applied to the reported prediction; the
underlying pooled judgments are retained separately.
