---
name: Bayesian Forecast Ensemble
description: Independently forecast and review distributions in a deliberating Bayesian ensemble.
---

# Bayesian ensemble stage

Call `metaculus.get_bayesian_stage` for your assigned stage ID, follow the JSON
schema, and submit with `metaculus.record_bayesian_stage`. Correct validation
errors and retry. The runner aggregates and publishes; do not publish or edit files.

## Proposal

Work independently from your assigned role. Load `forecasting-research`, research
the question, and read primary sources when possible. Specify a prior distribution,
its rationale, your posterior distribution, and how the evidence changed your
beliefs. Record source URLs, dates, observations, reasoning, and limitations;
never invent sources or present a model output as independent evidence from its
inputs. Identify shared or dependent evidence instead of counting it twice.

For numeric and date forecasts, provide strictly increasing interior quantiles.
The runner interprets their CDF as linear between quantiles, linear to a closed
bound, and exponentially decaying beyond the first or last quantile for an open
tail, with scale equal to the adjacent quantile interval. These are modeling
assumptions: choose quantiles with these implications in mind and discuss tail
uncertainty. For conditional questions, give a probability for the condition
and outcome distributions assuming yes and no separately; do not supply an
independent unconditional child forecast.

For discrete numeric questions, the quantiles specify a latent continuous CDF;
the platform converts it into its discrete outcome buckets. Consider the
resulting bucket probabilities and disclose this approximation.

If your role is `economic-modeler` or `causal-modeler`, load the corresponding
modeling skill and fill in your model artifact. Mark a model inapplicable with
reasons when appropriate, using "not applicable" for irrelevant fields.

## Review

You receive the current pooled forecast, the other proposals and the agents'
current posteriors. Independently assess the resolution criteria, priors,
sources, duplicated information, models, assumptions, and disagreements.
Express remaining uncertainty and a substantive critique. Record your own updated
posterior, or repeat your previous posterior if it remains appropriate. Do not
update again just because a previously considered source is repeated. Set
`accepts_current` true only when the consensus you reviewed is reasonable, not
merely to force agreement. The consensus is a linear pool of judgments, not a
verified posterior based on independent evidence; its displayed numeric/date
quantiles approximate its full distribution. Reviewers do not see one another's
current-round critiques. Stability and agreement do not guarantee accuracy.
