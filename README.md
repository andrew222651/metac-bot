A bot for [Metaculus FutureEval](https://www.metaculus.com/futureeval/participate/).

Other open-source bots are listed [here](https://www.metaculus.com/notebooks/38928/futureeval-resources-page/#open-source-bots).


## Install in Termux

Install [OpenCode V2](https://opencode.ai/v2/docs/) in Termux and clone your copy of this repository. From the repository directory, run:

```bash
bash scripts/install.sh
```

The installer installs missing tools and creates a Debian Trixie container if needed, then installs Python and the locked dependencies there.
Set `METAC_BOT_VENV` to override the default `/root/.venvs/metac-bot` environment.

Each forecasting run starts its own private OpenCode server, loads the bot's agents, skills, and `metaculus` and `research` tools, and stops the server afterward. You do not need to start or configure a shared server. The private server normally uses your saved OpenCode configuration and provider accounts; alternatively, provide model and search credentials through environment variables or the `.env` file described below. Select a model with `--model` or `OPENCODE_MODEL`, and optionally set `OPENCODE_SEARCH_PROVIDER` to choose a search provider.


## Run a test or one-off forecast in Termux

For a standalone question, use `forecast`. This saves locally, never publishes, and does not require a Metaculus token:

```bash
bash scripts/python.sh -m metac_bot forecast \
  "Which party will win the most votes in Spain's November 29, 2026 general election?" \
  --option PP --option PSOE --option Vox --option other \
  --model 'openrouter/openai/gpt-6.1-terra#low'
```

Omit `--option` for a binary yes/no question. Add `--criteria` or `--background` when needed. Binary and multiple-choice questions use the full ensemble by default; `--single-agent` is an explicit cheaper alternative.

To test against one question from the Metaculus testing area **without publishing**, generate a configuration file if you do not already have one:

```bash
bash scripts/python.sh -m metac_bot generate-env > .env
```

The command prints a template generated from the settings; the redirection creates `.env` and **overwrites it if it already exists**. Uncomment and set `METACULUS_TOKEN` in `.env`, then run:

```bash
bash scripts/python.sh -m metac_bot --mode test_questions --limit 1 --dry-run
```

Remove `--dry-run` only if you want to publish to the testing area. Reports and session artifacts are saved in `forecast_runs/`. Forecasting can incur model and research costs; an ensemble forecast can require 16 or more model sessions. Shell execution requires permission and is **not sandboxed**.

CLI options override matching environment variables. The private server normally uses your saved OpenCode provider accounts. An explicit `OPENROUTER_API_KEY` uses an isolated account database without changing your saved account; other saved provider accounts are unavailable to it and need environment keys if used. In that case, session history is temporary, but forecast artifacts remain in `forecast_runs/`. An explicit `OPENCODE_SEARCH_PROVIDER` applies only to the private server.

## Configure secrets and variables in GitHub

In your repository, open **Settings → Secrets and variables → Actions → Secrets** and add:

- `METACULUS_TOKEN` for the Metaculus bot account.
- At least one model provider key: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`.
- A key for the search provider you use, such as `EXA_API_KEY` or `TAVILY_API_KEY`. Optional AskNews and Perplexity credentials and other search keys are listed in the generated environment template and CLI help.

For manually started workflows, the model, search provider, session timeout, and ensemble concurrency inputs override the corresponding repository variables. A blank input, or `configured` for search, falls back to the variable. Scheduled runs have no inputs and use the variables directly. If a variable is not set, the workflow uses the fallback below. Credentials always come from repository secrets, not dispatch inputs.

Set optional variables under **Settings → Secrets and variables → Actions → Variables**:

| Setting | Repository variable | Fallback |
| --- | --- | --- |
| Model | `OPENCODE_MODEL` | Chosen from available provider keys, in OpenRouter, OpenAI, then Anthropic order |
| Search provider | `OPENCODE_SEARCH_PROVIDER` | `random` |
| Session timeout | `METAC_BOT_SESSION_TIMEOUT` | `900` seconds |
| Ensemble concurrency | `METAC_BOT_ENSEMBLE_CONCURRENCY` | `2` |

The model fallbacks are `openrouter/openai/gpt-4.1-mini`, `openai/gpt-4.1-mini`, and `anthropic/claude-sonnet-4-5`, respectively. Without a model input or variable and without any of those provider keys, the workflow fails.

## Run a test forecast in GitHub Actions

After configuring secrets, open **Actions → Run bot → Run workflow**, choose a branch, and select `test_questions` (the default). Optionally set the model, search provider, session timeout, or ensemble concurrency, and start the run. Download the `forecasts-test_questions` artifact from the completed run for its reports and session artifacts.

**The test mode publishes forecasts and comments to the Metaculus testing area.** The workflow does not accept a custom one-off question or offer a dry-run input. For a private, user-supplied forecast, use the Termux one-off command above.

## Participate in a Metaculus competition in GitHub Actions

After configuring secrets, open **Actions → Run bot**. Scheduled runs forecast on the AI tournament and MiniBench every 20 minutes, skipping questions already forecast. To start a competition run manually, choose **Run workflow** and select `tournament`; the manual default is the testing area.
