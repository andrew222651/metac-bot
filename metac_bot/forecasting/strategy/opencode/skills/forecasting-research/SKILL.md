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
4. Read the most relevant sources, check dates and units, deduplicate repeated
   reporting, and actively search for contrary evidence. Follow up on gaps rather
   than treating a provider-generated answer as authoritative. AskNews and
   Perplexity perform their own model work outside OpenCode; label their answers
   as provider-generated research and verify important claims at source URLs.
5. Synthesize the findings yourself in OpenCode. This is the replacement for
   `SmartSearcher`: query planning → retrieval → source evaluation → cited summary,
   without a separate Python LLM call. Keep facts distinct from inference and
   record URLs, publication dates, assumptions, and unresolved uncertainty.

If a provider lacks credentials, fails, or is rate limited, use another available
source and state the limitation. Do not repeatedly retry a failing paid call or
claim searches were performed when they were not. Research services only supply
evidence; the forecasting skill determines the final prediction.
