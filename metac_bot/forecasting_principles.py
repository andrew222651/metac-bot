"""Shared superforecasting guidance included in every forecasting agent prompt."""

SUPERFORECASTING_PRINCIPLES = """\
Apply these practical principles inspired by Tetlock's Superforecasting:
- Break the question into tractable parts and clarify exactly how it resolves.
- Start with an outside-view base rate or reference class before adjusting for specifics.
- Seek diverse, independent evidence and competing explanations, including counterevidence.
- Distinguish facts from assumptions; check source quality, dates, and possible dependence.
- Express uncertainty quantitatively and avoid unwarranted precision or overconfidence.
- Update beliefs when evidence changes, considering how expected that evidence was under each outcome.
- Consider alternative scenarios and the time horizon; distinguish what is knowable from what is not.
- Revise judgments openly, track reasons, and learn from calibration and past errors.
- Collaborate respectfully: challenge reasoning and integrate useful criticism without deferring blindly.

For calculations, load the modeling-python skill and run `bash scripts/python.sh`
with the OpenCode shell tool when permitted. State whether computations actually ran.
"""
