"""Independent forecasting perspectives and required model artifacts."""

ROLES = {
    "generalist": "Evaluate the full question and balanced scenarios.",
    "base-rate": "Focus on reference classes, historical frequencies and prior selection.",
    "current-evidence": "Focus on recent developments and authoritative primary sources.",
    "skeptic": "Challenge key assumptions and seek credible counterevidence.",
    "economic-modeler": (
        "Load the economic-modeling and modeling-python skills and build an explicit economic model, "
        "such as a DSGE-style model when appropriate. Record a model artifact only when an "
        "applicable model exists; if none exists, report no model and the proposal will be excluded."
    ),
    "market-analyst": (
        "Load the market-inference and modeling-python skills and infer the forecast from market variables, "
        "such as stock price movements, interest rates, and public company financials. Record a model "
        "artifact only when an applicable model exists; if none exists, report no model and the "
        "proposal will be excluded."
    ),
}

MODEL_ROLES = {"economic-modeler", "market-analyst"}
