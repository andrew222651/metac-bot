"""Command-line configuration and application entry point."""

import asyncio
from dataclasses import dataclass
import json
import logging
import sys
from typing import Annotated, Any, Literal

from cyclopts import App, Parameter
from dotenv import load_dotenv

from .config import ROLES, OneOffConfig, RunConfig
from .environment import EnvironmentSettings, environment_help, environment_template
from .paths import ROOT

app = App(
    name="metac-bot",
    help="Run Metaculus forecasts using OpenCode.",
    help_epilogue=environment_help(),
    version_flags=[],
)


@dataclass(frozen=True)
class CommonOptions:
    """Options shared by tournament and standalone forecasts."""

    model: Annotated[
        str | None,
        Parameter(
            help="provider/model; defaults to OPENCODE_MODEL or OpenCode's default."
        ),
    ] = None
    timeout: Annotated[
        float | None,
        Parameter(
            name=("--timeout", "--session-timeout"),
            help="Default agent session deadline in seconds.",
        ),
    ] = None
    api_timeout: Annotated[
        float | None,
        Parameter(
            help="OpenCode API request timeout in seconds, excluding session wait."
        ),
    ] = None
    service_start_timeout: Annotated[
        float | None, Parameter(help="Private OpenCode startup deadline in seconds.")
    ] = None
    single_agent: Annotated[
        bool | None,
        Parameter(help="Use a single agent for binary and multiple-choice questions."),
    ] = None
    ensemble_concurrency: Annotated[
        int | None, Parameter(help="Maximum simultaneous ensemble sessions.")
    ] = None
    ensemble_min_members: Annotated[
        int | None, Parameter(help="Minimum successful proposals and reviews (1–7).")
    ] = None
    ensemble_review_rounds: Annotated[
        int | None, Parameter(help="Maximum review rounds (1–5).")
    ] = None
    ensemble_tolerance: Annotated[
        float | None, Parameter(help="Posterior-change convergence threshold.")
    ] = None
    ensemble_proposal_timeout: Annotated[
        float | None,
        Parameter(help="Proposal session deadline; defaults to --timeout."),
    ] = None
    ensemble_review_timeout: Annotated[
        float | None, Parameter(help="Review session deadline; defaults to --timeout.")
    ] = None
    ensemble_decision_timeout: Annotated[
        float | None,
        Parameter(help="Adjudication session deadline; defaults to --timeout."),
    ] = None
    ensemble_model: Annotated[
        list[str] | None, Parameter(help="ROLE=PROVIDER/MODEL; repeat for other roles.")
    ] = None
    ensemble_adjudicator_model: Annotated[
        str | None, Parameter(help="Model for initial and review adjudication.")
    ] = None

    def run_config(
        self,
        *,
        mode: Literal["tournament", "test_questions"] = "tournament",
        dry_run: bool = False,
        limit: int | None = None,
    ) -> RunConfig:
        overrides: dict[str, Any] = {
            key: value
            for key, value in {
                "OPENCODE_MODEL": self.model,
                "METAC_BOT_SESSION_TIMEOUT": self.timeout,
                "METAC_BOT_API_TIMEOUT": self.api_timeout,
                "METAC_BOT_SERVICE_START_TIMEOUT": self.service_start_timeout,
                "METAC_BOT_SINGLE_AGENT": self.single_agent,
                "METAC_BOT_ENSEMBLE_CONCURRENCY": self.ensemble_concurrency,
                "METAC_BOT_ENSEMBLE_MIN_MEMBERS": self.ensemble_min_members,
                "METAC_BOT_ENSEMBLE_REVIEW_ROUNDS": self.ensemble_review_rounds,
                "METAC_BOT_ENSEMBLE_TOLERANCE": self.ensemble_tolerance,
                "METAC_BOT_ENSEMBLE_PROPOSAL_TIMEOUT": self.ensemble_proposal_timeout,
                "METAC_BOT_ENSEMBLE_REVIEW_TIMEOUT": self.ensemble_review_timeout,
                "METAC_BOT_ENSEMBLE_DECISION_TIMEOUT": self.ensemble_decision_timeout,
                "METAC_BOT_ENSEMBLE_ADJUDICATOR_MODEL": self.ensemble_adjudicator_model,
            }.items()
            if value is not None
        }
        settings = EnvironmentSettings(**overrides)
        models = {}
        if settings.METAC_BOT_ENSEMBLE_MODELS and self.ensemble_model != []:
            try:
                models = json.loads(settings.METAC_BOT_ENSEMBLE_MODELS)
            except ValueError as exc:
                raise ValueError(
                    "METAC_BOT_ENSEMBLE_MODELS must be a JSON object"
                ) from exc
            if not isinstance(models, dict) or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in models.items()
            ):
                raise ValueError(
                    "METAC_BOT_ENSEMBLE_MODELS must map role names to model IDs"
                )
        if self.ensemble_model == []:
            models.clear()
        seen = set()
        for item in self.ensemble_model or []:
            role, separator, role_model = item.partition("=")
            if not separator or role not in ROLES or role in seen:
                raise ValueError(
                    f"Invalid --ensemble-model {item!r}; roles: {', '.join(ROLES)}"
                )
            seen.add(role)
            models[role] = role_model

        def choose(value, default):
            return default if value is None else value

        return RunConfig(
            mode=mode,
            dry_run=dry_run,
            limit=limit,
            model=choose(self.model, settings.OPENCODE_MODEL),
            timeout=choose(self.timeout, settings.METAC_BOT_SESSION_TIMEOUT),
            api_timeout=choose(self.api_timeout, settings.METAC_BOT_API_TIMEOUT),
            service_start_timeout=choose(
                self.service_start_timeout, settings.METAC_BOT_SERVICE_START_TIMEOUT
            ),
            single_agent=choose(self.single_agent, settings.METAC_BOT_SINGLE_AGENT),
            ensemble_concurrency=choose(
                self.ensemble_concurrency, settings.METAC_BOT_ENSEMBLE_CONCURRENCY
            ),
            ensemble_min_members=choose(
                self.ensemble_min_members, settings.METAC_BOT_ENSEMBLE_MIN_MEMBERS
            ),
            ensemble_review_rounds=choose(
                self.ensemble_review_rounds, settings.METAC_BOT_ENSEMBLE_REVIEW_ROUNDS
            ),
            ensemble_tolerance=choose(
                self.ensemble_tolerance, settings.METAC_BOT_ENSEMBLE_TOLERANCE
            ),
            ensemble_proposal_timeout=choose(
                self.ensemble_proposal_timeout,
                settings.METAC_BOT_ENSEMBLE_PROPOSAL_TIMEOUT,
            ),
            ensemble_review_timeout=choose(
                self.ensemble_review_timeout, settings.METAC_BOT_ENSEMBLE_REVIEW_TIMEOUT
            ),
            ensemble_decision_timeout=choose(
                self.ensemble_decision_timeout,
                settings.METAC_BOT_ENSEMBLE_DECISION_TIMEOUT,
            ),
            ensemble_models=models,
            ensemble_adjudicator_model=choose(
                self.ensemble_adjudicator_model,
                settings.METAC_BOT_ENSEMBLE_ADJUDICATOR_MODEL,
            ),
        )


@app.default
def build_config(
    *,
    mode: Annotated[
        Literal["tournament", "test_questions"],
        Parameter(help="Tournament to forecast on."),
    ] = "tournament",
    dry_run: Annotated[
        bool, Parameter(help="Save reports without publishing to Metaculus.")
    ] = False,
    limit: Annotated[int | None, Parameter(help="Maximum number of questions.")] = None,
    options: Annotated[CommonOptions, Parameter(name="*")] = CommonOptions(),
) -> RunConfig:
    """Configure one tournament forecasting run."""
    return options.run_config(mode=mode, dry_run=dry_run, limit=limit)


@app.command
def generate_env() -> None:
    """Print a commented .env template; redirect output to a new .env file."""
    print(environment_template(), end="")


@app.command
def forecast(
    question: Annotated[
        str, Parameter(help="Question to forecast, supplied directly by you.")
    ],
    *,
    option: Annotated[
        list[str] | None,
        Parameter(help="Choice; repeat for each option. Omit for a yes/no question."),
    ] = None,
    criteria: Annotated[
        str | None, Parameter(help="Optional resolution criteria.")
    ] = None,
    background: Annotated[
        str | None, Parameter(help="Optional background information.")
    ] = None,
    options: Annotated[CommonOptions, Parameter(name="*")] = CommonOptions(),
) -> OneOffConfig:
    """Forecast a user-supplied question locally; never publish it."""
    base = options.run_config(dry_run=True)
    return OneOffConfig(
        **vars(base),
        question=question.strip(),
        options=tuple(item.strip() for item in option or ()),
        criteria=criteria,
        background=background,
    )


def parse_args(argv: list[str] | None = None) -> RunConfig:
    command, arguments, _ = app.parse_args(argv)
    try:
        result = command(*arguments.args, **arguments.kwargs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    # Cyclopts handles --help/--version as commands which return no RunConfig.
    if not isinstance(result, RunConfig):
        raise SystemExit(0)
    return result


def main() -> None:
    load_dotenv(ROOT / ".env")
    config = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    from .runtime import silence_noisy_dependencies

    silence_noisy_dependencies()
    if isinstance(config, OneOffConfig):
        # The client library warns at import time when no Metaculus token is set;
        # one-off forecasts deliberately do not need one.
        logging.getLogger("forecasting_tools.helpers.metaculus_client").setLevel(
            logging.ERROR
        )
    from ..runner import run, run_one_off

    result = run_one_off(config) if isinstance(config, OneOffConfig) else run(config)
    raise SystemExit(asyncio.run(result))
