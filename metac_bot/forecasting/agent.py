"""Common OpenCode session lifecycle for all forecasting agents."""

from datetime import datetime, timezone
from typing import Callable

from ..infrastructure.opencode import OpenCode
from ..infrastructure.paths import ROOT
from .principles import SUPERFORECASTING_PRINCIPLES


async def run_agent(
    opencode: OpenCode,
    *,
    title: str,
    prompt: str,
    model: str | None,
    timeout: float,
    on_session: Callable[[str], None],
    agent: str | None = None,
) -> str:
    """Start and wait for one session, persisting its ID before the prompt."""
    if agent:
        session_id = await opencode.create_session(ROOT, title, model, agent=agent)
    else:
        session_id = await opencode.create_session(ROOT, title, model)
    on_session(session_id)
    await opencode.prompt(
        session_id,
        (
            f"{prompt}\nToday is {datetime.now(timezone.utc).isoformat()}.\n\n"
            + SUPERFORECASTING_PRINCIPLES
        ),
    )
    await opencode.wait(session_id, timeout)
    return session_id
