"""Close-aware budgets for live forecasting runs."""

from datetime import datetime, timezone
import math

from forecasting_tools import MetaculusQuestion

from ..forecasting.strategy.roles import ROLES
from .config import RunConfig

PUBLISH_RESERVE = 30.0
MIN_FORECAST_BUDGET = 60.0


def is_closed(question: MetaculusQuestion) -> bool:
    """Honor the cached state as well as a possibly stale close timestamp."""
    return getattr(question.state, "value", question.state) in ("closed", "resolved")


def remaining_seconds(question: MetaculusQuestion) -> float | None:
    """Return time to the cached close, or None when no close is known."""
    close = question.close_time
    if close is None:
        return None
    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    return (close - datetime.now(timezone.utc)).total_seconds()


def forecast_budget(question: MetaculusQuestion, args: RunConfig) -> float | None:
    """Budget the forecast phase, reserving time for publication on live runs."""
    if args.dry_run:
        return None
    remaining = remaining_seconds(question)
    if remaining is None:
        return None
    return remaining - PUBLISH_RESERVE


def stage_timeout_cap(budget: float, args: RunConfig) -> float:
    """Allocate time for at least one proposal and review wave per role."""
    waves = (
        1
        if args.single_agent
        else 2 * math.ceil(len(ROLES) / args.ensemble_concurrency)
    )
    return max(1.0, budget / waves)
