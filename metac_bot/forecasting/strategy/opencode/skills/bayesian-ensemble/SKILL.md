---
name: Bayesian Forecast Ensemble
description: Independently propose evidence, review, or adjudicate a binary or multiple-choice forecast using an auditable shared Bayesian evidence ledger.
---

# Bayesian ensemble stage

Call `metaculus.get_bayesian_stage` with your assigned stage ID. Complete the
specified stage and call `metaculus.record_bayesian_stage` with a `result` object
matching the returned JSON schema. Correct validation errors and retry. The
runner computes the posterior; do not publish or write files.

## Proposal

Work independently from the assigned role. Load `forecasting-research` and
research the question; read primary sources where possible. For binary questions,
offer a prior in (0,1) and evidence with a likelihood ratio
`P(evidence | YES) / P(evidence | NO)`. Greater than 1 favors YES; below 1
favors NO. For multiple-choice questions, give a prior distribution with a
positive probability for every exact option, summing to 1. For each evidence
item provide `likelihoods`: positive relative weights for every exact option,
proportional to `P(evidence | that option)`. The weights need not sum to 1;
their ratios are what matter. Explain the reference class, prior assumptions,
sources, dates, and limitations. These are uncertain elicited judgments, not measurements. Do not
count a belief, prior, forecast, or absence of information as observed evidence.
Include zero evidence items if no verifiable evidence is found; explicitly state
the research limitation. Never invent sources, outcomes or numerical precision.
Limit each claim to a coherent observation, with relevant citations.

If your role is `economic-modeler` or `causal-modeler`, load the corresponding
modeling skill named in your role instructions and fill the `model` object in your
proposal. Explicitly mark the model inapplicable if the question cannot support
one; explain why and use "not applicable" for fields that do not apply. Separate
models and their assumptions from observed evidence. Do not submit a model's
prediction as an independent evidence item on top of its underlying observations.

## Decision

You receive all proposals and candidate IDs. Select an explicit common prior,
explaining the reference class and considering the independently proposed priors.
Assign each candidate exactly once to an evidence group or exclude it with a
reason. Group overlapping, duplicate, and dependent observations, including
different articles derived from the same study, announcement, or data. The
same URL cannot occur in different groups. Do not treat repeated reports as
independent evidence. For each ordered group, estimate its JOINT likelihood
conditional on all earlier groups, with a substantive explanation of the assumed
dependencies and how the estimate was obtained. For binary questions, give a
single conditional likelihood ratio. For multiple choice, provide
`conditional_likelihoods` with a positive relative weight for every exact
option, proportional to the evidence likelihood conditional on earlier groups.
Give a complete prior distribution and use the question's exact option names.
Do not submit separate YES/NO binary forecasts for mutually exclusive options.
Never multiply candidate
likelihood ratios blindly. If dependence is unknown, say so and make a
conservative judgment or exclude the candidate. Source overlap checks are only
partial; semantically duplicated URLs must also be grouped. Describe uncertainty,
model risk, and potential missing evidence in `limitations`.

If reviews are present, consider each critique and use it to revise the prior,
groups or likelihoods as appropriate. Keep the final ledger complete: all
candidate IDs still appear exactly once. The runner recalculates from the final
prior, not from the previous posterior; do not treat previous posterior as a new
prior. Do not invent evidence or new candidate IDs.

## Review

Examine the supplied common prior, candidate evidence, ordered groups, likelihoods,
dependence assumptions, posterior, and original proposals. Identify double
counting, mistaken question interpretation, prior miscalibration, and implausible
likelihoods. Give actionable proposed changes and express uncertainty. The review
is a critique, not a new likelihood factor. If no issues are found, say so.
Set `accepts_current` true only if you accept the current prior, evidence ledger,
dependence assumptions, and posterior as a reasonable forecast despite remaining
uncertainties. Otherwise set it false and describe what must change. Reviewers do
not have access to one another's critiques until the adjudicator collects them.
A stable posterior does not prove the assumptions are correct.
