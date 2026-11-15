---
name: Market Inference
description: Infer a forecast from market variables such as stock prices, interest rates, and public company financials.
---

# Market inference for forecasts

1. Read the assigned question and resolution criteria. Decide whether market
   variables can inform it. If not, report no model artifact: omit the `model`
   field and explain in your reasoning why no applicable model exists. The
   runner will exclude your proposal from the pooled forecast.
2. Select relevant market variables: stock price movements and volatility,
   interest rates and yield curves, credit spreads, corporate earnings and
   public company fundamentals (revenue, margins, leverage, guidance), and
   other observable financial series. Confirm tickers and series IDs at the
   source. Use `research.yahoo_financial_series` for dated daily prices for a
   known ticker and `research.fred_financial_series` for recent FRED
   observations for a known series ID when `FRED_API_KEY` is configured.
   For company financials, use filings, investor relations, and other primary
   sources found by web search. State units, timing, and which market
   variables map to which outcome and why.
3. Distinguish a market price as a cause, a reflection of information, and a
   noisy signal. Explain the channel linking the market variable to the
   outcome, address timing and look-ahead risk, and note data limits: the
   latest bar may be incomplete, delayed or revised; FRED observation dates
   are not release dates and values may be revised. Do not treat every price
   move as independent evidence or double-count the same information.
4. Specify a transparent mapping from market variables to the forecast
   distribution: a prior, likelihood, regression or other explicit model,
   and how the market signal updates beliefs. Include parameter uncertainty,
   sensitivity checks, and at least one alternative explanation or contrary
   signal considered. Cite dated sources with URLs; never fabricate prices,
   rates, or fundamentals.
5. Load the `modeling-python` skill. Provide reproducible code and actually
   calculate or check results when feasible. State clearly whether code was
   executed and which inputs were used; otherwise give a qualitative model
   and do not invent numerical estimates. Compare the market-implied signal
   to other evidence and to prediction-market prices where available, and
   explain any divergence. A prediction-market price is not automatically a
   calibrated probability for this question.
6. Fill the `model` object in your Bayesian proposal in any form that suits
   the question; describe its structure, equations, input sources,
   inference and implementation, result and uncertainty, and limitations.
   Cite observations
   used as evidence; a posterior derived from them is not a separate independent
   observation to multiply into the ensemble ledger.
