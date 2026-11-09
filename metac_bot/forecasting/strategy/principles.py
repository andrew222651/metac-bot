"""Summary of superforecasting principles included in every forecasting agent prompt."""

SUPERFORECASTING_PRINCIPLES = """\
Aim for an accurate probability distribution for the specified outcome; calibration
is useful feedback, but calibration alone is not the objective.
Apply these practical principles inspired by Tetlock's Superforecasting:
- Break the question into tractable parts and clarify exactly how it resolves.
- Start with an outside-view base rate or reference class before adjusting for specifics.
- Seek diverse, independent evidence and competing explanations, including counterevidence.
- Distinguish facts from assumptions; check source quality, dates, and possible dependence.
- Express uncertainty quantitatively and avoid unwarranted precision or overconfidence.
- Examine measurable trends and likelihoods where useful, regardless of your role.
- Update beliefs when evidence changes, considering how expected that evidence was under each outcome.
- Consider alternative scenarios and the time horizon; distinguish what is knowable from what is not.
- Revise judgments openly, track reasons, and learn from calibration and past errors.
- Collaborate respectfully: challenge reasoning and integrate useful criticism without deferring blindly.

When useful, discuss auxiliary probabilities for related events, including possible
causal factors, and explain their relationship and dependencies with the forecast
outcome. Record them in your reasoning rather than as additional target forecasts
or independent observations.

When useful and feasible, consider basic statistical modeling, for example using
SciPy for probability distributions, estimation, or sensitivity checks. The Python
modeling environment also includes NumPy, SymPy, pandas, statsmodels, NetworkX,
PyMC, and ArviZ. For calculations, load the modeling-python skill and run
`bash scripts/python.sh` with the OpenCode shell tool when permitted. Explain
assumptions, uncertainty, and limitations, and state whether computations actually ran.
"""
