---
name: Forecasting Research
description: Research forecasting questions using native web tools, AskNews, Perplexity, and Exa; recreate the SmartSearcher search-and-synthesis workflow inside OpenCode.
---

# Research for a forecast

1. Break the question into factual uncertainties: resolution rules, current status,
   base rates, trends, and relevant expert/market estimates. Design at least two
   complementary searches when research is needed: current developments and a
   historical/reference-class view.
2. Call `research.available_providers` to see which optional services have
   credentials. This checks configuration, not account validity or service
   availability. Native OpenCode web search and web fetch do not depend on these
   optional research-service credentials, but also depend on their configuration
   and availability.
3. Choose tools by the evidence needed; do not call every provider by default:
   - Native web search/fetch: general research and primary-source verification.
   - `research.exa_search`: relevant source text, optionally restricted to primary
     source domains. Start with five results; use up to ten for broader coverage.
   - `research.asknews_research`, depth `news`: recent and historical news summaries.
     Depth `low`, `medium`, or `high`: AskNews' original deep-research presets.
    - `research.perplexity_research`: synthesized research with sources. `fast` is
      the starting point; use higher effort when a complex evidence gap warrants it.
    - `research.yahoo_financial_series`: dated daily prices for a known Yahoo ticker;
      `research.fred_financial_series`: recent FRED observations for a known series ID
      when `FRED_API_KEY` is configured. Confirm identifiers and units at the source.
    - `research.prediction_market_search`: candidate markets from Manifold, Polymarket,
      PredictIt, and a bounded partial scan of Kalshi. `research.kalshi_market`
      retrieves a known Kalshi ticker, including one found elsewhere. Compare each
      contract's resolution rules and dates before
      using prices. Bid, ask and last trade are not interchangeable, and a market
      price is not necessarily a calibrated probability.
4. Read the most relevant sources, check dates and units, deduplicate repeated
   reporting, and actively search for contrary evidence. Follow up on gaps rather
   than treating a provider-generated answer as authoritative. AskNews and
   Perplexity perform their own model work outside OpenCode; label their answers
   as provider-generated research and verify important claims at source URLs.
5. Synthesize the findings yourself in OpenCode. This is the replacement for
   `SmartSearcher`: query planning → retrieval → source evaluation → cited summary,
   without a separate Python LLM call. Keep facts distinct from inference and
   record URLs, publication dates, assumptions, and unresolved uncertainty.

If a provider lacks credentials, fails, runs out of credits, or is rate limited,
including a native web tool, use another available source and state the limitation.
An optional MCP result with `status: unavailable` is a failure, not evidence or a
successful search. Do not repeatedly retry a failing paid call, request credentials,
or claim searches were performed when they were not. If no alternative is available,
explain the missing evidence and finish with the best supported forecast only if it
can be made responsibly. Research services only supply evidence; the forecasting
skill determines the final prediction.
