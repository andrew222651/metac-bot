"""Command-line configuration and application entry point."""

import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import Annotated, Literal

from cyclopts import App, Parameter
from dotenv import load_dotenv

from .config import ROLES, OneOffConfig, RunConfig

app = App(
    name="metac-bot", help="Run Metaculus forecasts using OpenCode.", version_flags=[]
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
        float,
        Parameter(
            name=("--timeout", "--session-timeout"),
            help="Default agent session deadline in seconds.",
        ),
    ] = 900
    api_timeout: Annotated[
        float,
        Parameter(
            help="OpenCode API request timeout in seconds, excluding session wait."
        ),
    ] = 30
    single_agent: Annotated[
        bool,
        Parameter(help="Use a single agent for binary and multiple-choice questions."),
    ] = False
    ensemble_concurrency: Annotated[
        int, Parameter(help="Maximum simultaneous ensemble sessions.")
    ] = 2
    ensemble_min_members: Annotated[
        int, Parameter(help="Minimum successful proposals and reviews (1–7).")
    ] = 3
    ensemble_review_rounds: Annotated[
        int, Parameter(help="Maximum review rounds (1–5).")
    ] = 3
    ensemble_tolerance: Annotated[
        float, Parameter(help="Posterior-change convergence threshold.")
    ] = 0.01
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
        mode: Literal["tournament", "metaculus_cup", "test_questions"] = "tournament",
        dry_run: bool = False,
        limit: int | None = None,
    ) -> RunConfig:
        models = {}
        for item in self.ensemble_model or []:
            role, separator, role_model = item.partition("=")
            if not separator or role not in ROLES or role in models:
                raise ValueError(
                    f"Invalid --ensemble-model {item!r}; roles: {', '.join(ROLES)}"
                )
            models[role] = role_model
        return RunConfig(
            mode=mode,
            dry_run=dry_run,
            limit=limit,
            model=self.model
            if self.model is not None
            else os.getenv("OPENCODE_MODEL") or None,
            timeout=self.timeout,
            api_timeout=self.api_timeout,
            single_agent=self.single_agent,
            ensemble_concurrency=self.ensemble_concurrency,
            ensemble_min_members=self.ensemble_min_members,
            ensemble_review_rounds=self.ensemble_review_rounds,
            ensemble_tolerance=self.ensemble_tolerance,
            ensemble_proposal_timeout=self.ensemble_proposal_timeout,
            ensemble_review_timeout=self.ensemble_review_timeout,
            ensemble_decision_timeout=self.ensemble_decision_timeout,
            ensemble_models=models,
            ensemble_adjudicator_model=self.ensemble_adjudicator_model,
        )


@app.default
def build_config(
    *,
    mode: Annotated[
        Literal["tournament", "metaculus_cup", "test_questions"],
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
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
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
    from .runner import run, run_one_off

    result = run_one_off(config) if isinstance(config, OneOffConfig) else run(config)
    raise SystemExit(asyncio.run(result))
