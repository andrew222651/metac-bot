"""Independent forecasting perspectives and required model artifacts."""

ROLES = {
    "generalist": "Evaluate the full question and balanced scenarios.",
    "base-rate": "Focus on reference classes, historical frequencies and prior selection.",
    "current-evidence": "Focus on recent developments and authoritative primary sources.",
    "skeptic": "Challenge key assumptions and seek credible counterevidence.",
    "quantitative": "Focus on measurable trends, explicit likelihoods and uncertainty.",
    "economic-modeler": (
        "Load the economic-modeling and modeling-python skills and build an explicit economic model, "
        "such as a DSGE-style model when appropriate. Record the required model artifact."
    ),
    "causal-modeler": (
        "Load the bayesian-causal-modeling and modeling-python skills and build an explicit Bayesian causal model. "
        "Record the required model artifact."
    ),
}

MODEL_ROLES = {"economic-modeler": "economic", "causal-modeler": "causal"}
