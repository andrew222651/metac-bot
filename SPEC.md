# Forecasting strategy

This document describes the forecasting strategy at the level of its agents,
evidence, mathematical aggregation, and review. It is a conceptual specification
of how a forecast is formed, not a description of the software that runs it.

## Purpose and approach

The goal is a calibrated probability distribution for a precisely defined future
outcome, accompanied by an auditable explanation. A coding agent conducts
research, reasons about evidence, and, when useful and feasible, runs
computations or builds explicit models. Different agent roles provide distinct
perspectives; their proposals are combined through an evidence-based Bayesian
procedure rather than by averaging their forecasts.

Binary and mutually exclusive multiple-choice questions normally use the
ensemble below. Other question types, and explicitly requested single-agent
forecasts, use a single forecasting agent. For numerical and date questions this
agent expresses uncertainty as a distribution; conditional questions require
forecasts conditional on the specified outcomes.

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

## Ensemble and Bayesian aggregation

Each proposing agent supplies a reasoned prior and sourced observations, with
judgments about their likelihood under the possible outcomes. An observation is
not a prior, a prediction, or merely an absence of information. The adjudicating
agent selects a common prior, considers the proposals, and assigns every
candidate observation either to an evidence group or to an exclusion with a
reason. Related reports and overlapping sources belong together; repeated
coverage of the same event must not be counted as independent evidence.

For ordered evidence groups $E_1,\ldots,E_m$ and a binary event $H$, the
adjudicator estimates the conditional likelihood ratio

$$
L_j = \frac{P(E_j\mid H,E_1,\ldots,E_{j-1})}
           {P(E_j\mid \neg H,E_1,\ldots,E_{j-1})}.
$$

The posterior is calculated from the common prior $p_0$ by

$$
\frac{p}{1-p} = \frac{p_0}{1-p_0}\prod_{j=1}^m L_j.
$$

For mutually exclusive outcomes $H_i$, the adjudicator instead supplies a
positive conditional likelihood weight $w_{ij}$ for each outcome and group,
proportional to $P(E_j\mid H_i,E_1,\ldots,E_{j-1})$. The resulting distribution is

$$
P(H_i\mid E_1,\ldots,E_m) =
\frac{P(H_i)\prod_j w_{ij}}{\sum_k P(H_k)\prod_j w_{kj}}.
$$

These likelihoods and dependence assumptions are elicited judgments, not
objectively verified measurements. The adjudicator explains them and handles
unknown dependence conservatively. Source-overlap checks cannot detect every
duplicate or dependence. Each revision recomputes from its selected prior and
complete evidence ledger, not from the previous posterior, avoiding repeated
updates on the same evidence.

## Review, convergence, and output

The proposing perspectives review the common prior, evidence ledger, dependence
assumptions, and posterior independently. They flag errors and propose changes;
their critiques are not themselves new evidence factors. The adjudicator
considers the reviews and may revise the prior, groups, or likelihoods. A round
converges only when every completed reviewer accepts the forecast they reviewed
and the newly calculated posterior changes by no more than the configured
tolerance. For multiple-choice questions, change is the largest absolute change
among the option probabilities.

Both proposal and review stages require a configured quorum and the economic and
causal modeling roles. If a required stage fails or the ensemble does not
converge within the allowed rounds, it fails without publishing a forecast.
Agreement and numerical stability do not establish that the assumptions are
correct.

The final result includes the forecast, research, and rationale: the prior,
posterior, evidence and sources, exclusions, dependence assumptions, model
artifacts, and remaining limitations. Probability bounds may be applied to the
reported prediction; the underlying posterior is retained separately.
