# OpenCode Metaculus forecasting bot

An OpenCode-first bot for [Metaculus FutureEval](https://www.metaculus.com/futureeval/participate/). OpenCode researches and forecasts; Python validates and publishes reports using `forecasting-tools`.

## Install

Install [OpenCode V2](https://opencode.ai/v2/docs/), then run from the repository:

```bash
bash scripts/install.sh
```

On **Termux**, the installer installs missing tools and creates a Debian Trixie container if needed, then installs Python and the locked dependencies there. It reuses existing containers and does not overwrite an incompatible Python installation. Set `METAC_BOT_VENV` to override the default `/root/.venvs/metac-bot` environment.

## Run locally

1. Copy `.env.template` to `.env` and set `METACULUS_TOKEN`.
2. Connect a model and search provider in OpenCode, then start the service:
   ```bash
   opencode service start
   opencode mcp list
   ```
3. Test one question **without publishing**:
   ```bash
   bash scripts/python.sh -m metac_bot --mode test_questions --limit 1 --dry-run
   ```

CLI options override the matching environment variables; use `--help` to list them. With no explicit provider overrides, the bot uses your existing OpenCode service. An explicit `OPENROUTER_API_KEY` starts a private service with that key, without changing your saved OpenCode account. Other saved provider accounts are unavailable to that private service and need environment keys if used. Its session history is temporary, but forecast artifacts remain in `forecast_runs/`. An explicit `OPENCODE_SEARCH_PROVIDER` also uses a private service without changing your shared configuration. These overrides cannot be combined with `OPENCODE_SERVER_URL`.

If OpenCode was already running before the project agents were added, run `opencode reload` so it discovers them. Reloading cancels pending permission requests and forms; resolve those first.

### One-off forecast

Forecast a question you supply directly, without a Metaculus token or publication:

```bash
bash scripts/python.sh -m metac_bot forecast \
  "Which party will win the most votes in Spain's November 29, 2026 general election?" \
  --option PP --option PSOE --option Vox --option other \
  --model 'openrouter/openai/gpt-6.1-terra#low'
```

Omit `--option` for a binary yes/no question. Add `--criteria` or `--background` when needed. Binary and multiple-choice questions use the full ensemble by default; `--single-agent` is an explicit cheaper alternative. This command always saves locally and never publishes.

Both the one-off and tournament commands accept `--model`, per-role `--ensemble-model ROLE=PROVIDER/MODEL`, and `--ensemble-adjudicator-model`. Use `--timeout` (or `--session-timeout`) for the default session deadline, `--api-timeout` for OpenCode API requests, and `--ensemble-proposal-timeout`, `--ensemble-review-timeout`, or `--ensemble-decision-timeout` to override individual stages. Corresponding `METAC_BOT_*` environment variables work on both Termux and GitHub Actions. See `--help` for their names and defaults. MCP and research-provider timeouts are configured separately in OpenCode.

### Forecast modes

```bash
# AI tournament and MiniBench; skips previously forecast questions
bash scripts/python.sh -m metac_bot --mode tournament

# Metaculus Cup
bash scripts/python.sh -m metac_bot --mode metaculus_cup

# Testing area
bash scripts/python.sh -m metac_bot --mode test_questions
```

**These commands publish unless `--dry-run` is included.** Use `--limit N` to restrict questions and `--help` for model, timeout, and ensemble options. Permission requests are not auto-approved; a blocked session is interrupted and reported. A failed publication may already have submitted a prediction before its comment failed, so check Metaculus before retrying.

## GitHub Actions

Add repository secrets for `METACULUS_TOKEN` and at least one of `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`. Add a search key such as `EXA_API_KEY` or `TAVILY_API_KEY`; optional AskNews and Perplexity credentials are listed in `.env.template`.

Optionally set repository variables `OPENCODE_MODEL`, `OPENCODE_SEARCH_PROVIDER`, `METAC_BOT_SESSION_TIMEOUT`, and `METAC_BOT_ENSEMBLE_CONCURRENCY`. Manual workflow inputs override those variables. Without a model variable, CI chooses a default from the available provider keys; search defaults to `random`. The CI-only OpenCode template reads its search setting from the environment and does not change your local OpenCode configuration.

Run **Test Bot** manually first. It publishes to the testing area. The AI tournament workflow runs every 20 minutes, and the Cup workflow every two days. Workflows install OpenCode and dependencies, then upload `forecast_runs/` as artifacts.

## How it works

- **Binary and multiple-choice questions:** seven independent sessions propose priors and evidence. An adjudicator groups evidence and estimates conditional likelihoods; Python computes a shared posterior across all outcomes. Agents review it, and the bot publishes only if the convergence and quorum checks pass. The economic and causal modelers must both participate.
- **Other question types:** a single agent forecasts numeric, date, and conditional questions.
- **Research:** OpenCode's web tools and optional AskNews, Perplexity, and Exa MCP tools provide sources. AskNews and Perplexity perform their own provider-side generation; OpenCode produces the final forecast.

The Bayesian likelihoods and dependence assumptions are model judgments, not guaranteed calibration. Repeated reports and correlated evidence can still be difficult to identify. An ensemble question can require **16 or more model sessions**, with corresponding time and cost.

The modeling agents can run Python with NumPy, SciPy, SymPy, pandas, statsmodels, NetworkX, and PyMC. Shell execution requires permission and is **not sandboxed**. On Termux/PRoot, PyTensor defaults to a slower non-C-compiled mode; set `PYTENSOR_FLAGS` to override it if a suitable Linux compiler and headers are available.

Reports, model artifacts, and session IDs are stored in `forecast_runs/` (ignored by Git). Agent definitions and skills live in `.opencode/`; MCP servers are configured in `opencode.jsonc` and implemented in `metac_bot/`.

Python forecasting behavior lives in `metac_bot/forecasting/` (roles, prompts, ensemble, validation, and MCP tools). Runtime interfaces and adapters live in `metac_bot/infrastructure/` (CLI, configuration, OpenCode, and research services). `metac_bot/runner.py` connects the two and handles question selection and publication. The `python -m metac_bot` and existing MCP entry points remain unchanged.

## Development

```bash
# Include optional integrations and development tools
bash scripts/install.sh --all-groups

bash scripts/python.sh -m unittest discover -s tests -v
bash scripts/python.sh -m ruff format --check .
bash scripts/python.sh -m ruff check .
bash scripts/python.sh -m ty check
```

Ruff, ty, and the unit tests run on push. See [integrations/README.md](integrations/README.md) for LightningRod and bot-review.
