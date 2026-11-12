"""Environment-backed settings and the CLI's environment-variable reference.

Values used by OpenCode, MCP servers, or shell launchers are documented here as
well as values read directly by the Python runner. Secret values are never used
to generate help text.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConnectionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True, env_ignore_empty=True, extra="ignore"
    )

    OPENROUTER_API_KEY: SecretStr | None = Field(
        None,
        description="Explicit OpenRouter key; uses a private server so it overrides saved accounts.",
    )
    OPENAI_API_KEY: SecretStr | None = Field(
        None, description="OpenAI API key for the OpenCode server."
    )
    ANTHROPIC_API_KEY: SecretStr | None = Field(
        None, description="Anthropic API key for the OpenCode server."
    )
    EXA_API_KEY: SecretStr | None = Field(
        None, description="Exa search key for OpenCode and the research MCP server."
    )
    TAVILY_API_KEY: SecretStr | None = Field(
        None, description="Tavily search key for the OpenCode server."
    )
    FIRECRAWL_API_KEY: SecretStr | None = Field(
        None, description="Firecrawl search key for the OpenCode server."
    )
    PARALLEL_API_KEY: SecretStr | None = Field(
        None, description="Parallel search key for the OpenCode server."
    )
    TINYFISH_API_KEY: SecretStr | None = Field(
        None, description="Optional TinyFish search key for the OpenCode server."
    )
    ASKNEWS_API_KEY: SecretStr | None = Field(
        None, description="AskNews API key for the research MCP server."
    )
    ASKNEWS_CLIENT_ID: SecretStr | None = Field(
        None, description="AskNews OAuth client ID; use with ASKNEWS_SECRET."
    )
    ASKNEWS_SECRET: SecretStr | None = Field(
        None, description="AskNews OAuth client secret; use with ASKNEWS_CLIENT_ID."
    )
    ASKNEWS_CACHE_MODE: str | None = Field(
        None,
        description="AskNews cache mode: no_cache, use_cache, or use_cache_with_fallback.",
    )
    PERPLEXITY_API_KEY: SecretStr | None = Field(
        None, description="Perplexity API key for the research MCP server."
    )
    FRED_API_KEY: SecretStr | None = Field(
        None, description="FRED API key for economic time-series research."
    )
    OPENCODE_SEARCH_PROVIDER: str | None = Field(
        None,
        description="OpenCode search-provider override, e.g. random, exa, firecrawl, parallel, tavily, or tinyfish.",
    )
    XDG_CONFIG_HOME: str | None = Field(
        None,
        description="Base directory for OpenCode configuration.",
    )
    XDG_DATA_HOME: str | None = Field(
        None,
        description="Base directory for OpenCode data and saved provider credentials.",
    )
    XDG_STATE_HOME: str | None = Field(
        None,
        description="OpenCode state directory; private servers use temporary state.",
    )
    XDG_CACHE_HOME: str | None = Field(
        None, description="Base directory for OpenCode cache files."
    )
    OPENCODE_DB: str | None = Field(
        None,
        description="Optional OpenCode database path; an explicit OpenRouter key uses an isolated in-memory database.",
    )
    METAC_BOT_HOST_CONFIG_HOME: str | None = Field(
        None,
        description="Termux launcher-provided host OpenCode configuration directory.",
    )
    METAC_BOT_HOST_DATA_HOME: str | None = Field(
        None, description="Termux launcher-provided host OpenCode data directory."
    )


class EnvironmentSettings(ConnectionSettings):
    OPENCODE_MODEL: str | None = Field(
        None,
        description="Default provider/model for bot sessions; --model takes precedence.",
    )
    METAC_BOT_SESSION_TIMEOUT: float = Field(
        900, description="Default agent session deadline, in seconds."
    )
    METAC_BOT_API_TIMEOUT: float = Field(
        30, description="OpenCode API request timeout, in seconds."
    )
    METAC_BOT_SERVICE_START_TIMEOUT: float = Field(
        60,
        description="Wait limit in seconds for private OpenCode startup and, separately, MCP connections.",
    )
    METAC_BOT_SINGLE_AGENT: bool = Field(
        False, description="Use a single agent instead of the ensemble."
    )
    METAC_BOT_ENSEMBLE_CONCURRENCY: int = Field(
        2, description="Maximum simultaneous ensemble sessions."
    )
    METAC_BOT_ENSEMBLE_REVIEW_ROUNDS: int = Field(
        3, description="Maximum ensemble review rounds."
    )
    METAC_BOT_ENSEMBLE_TOLERANCE: float = Field(
        0.01, description="Posterior-change convergence threshold."
    )
    METAC_BOT_ENSEMBLE_PROPOSAL_TIMEOUT: float | None = Field(
        None, description="Override the proposal session deadline, in seconds."
    )
    METAC_BOT_ENSEMBLE_REVIEW_TIMEOUT: float | None = Field(
        None, description="Override the review session deadline, in seconds."
    )
    METAC_BOT_ENSEMBLE_DECISION_TIMEOUT: float | None = Field(
        None,
        description="Legacy decision timeout; unused by the distributional ensemble.",
    )
    METAC_BOT_ENSEMBLE_MODELS: str | None = Field(
        None, description="JSON object mapping ensemble roles to provider/model IDs."
    )
    METAC_BOT_ENSEMBLE_ADJUDICATOR_MODEL: str | None = Field(
        None,
        description="Legacy adjudicator model; unused by the distributional ensemble.",
    )

    METACULUS_TOKEN: SecretStr | None = Field(
        None,
        description="Metaculus bot token; required for tournament runs, not one-off forecasts.",
    )
    METAC_BOT_VENV: str | None = Field(
        None,
        description="Termux Debian virtualenv path; defaults to /root/.venvs/metac-bot.",
    )
    PYTENSOR_FLAGS: str | None = Field(
        None, description="PyTensor flags; the Termux/PRoot launcher defaults to cxx=."
    )
    OPENBLAS_NUM_THREADS: str | None = Field(
        None, description="Optional BLAS thread limit for Python model calculations."
    )
    OMP_NUM_THREADS: str | None = Field(
        None, description="Optional OpenMP thread limit for Python model calculations."
    )
    UV_PROJECT_ENVIRONMENT: str | None = Field(
        None,
        description="uv environment path on standard Linux/macOS; Termux uses METAC_BOT_VENV.",
    )
    UV_NO_SYNC: str | None = Field(
        None,
        description="Skip uv's automatic environment sync when explicitly enabled.",
    )
    TMPDIR: str | None = Field(
        None,
        description="Temporary directory used for private service files and Python tools.",
    )
    UV_LINK_MODE: str | None = Field(
        None, description="uv link mode; the Termux launcher forces copy mode."
    )
    TERMUX_VERSION: str | None = Field(
        None, description="Set by Termux; used to select the Debian launcher."
    )
    PD_CONTAINER: str | None = Field(
        None, description="Set by proot-distro; used to detect a PRoot container."
    )
    ANDROID_ROOT: str | None = Field(
        None,
        description="Set by Android; used to detect the PRoot modeling environment.",
    )
    FORECAST_MODE: str | None = Field(
        None,
        description="GitHub composite-action mode; the action passes it to --mode.",
    )


def environment_help() -> str:
    """Render metadata, not values, so credentials cannot leak through --help."""
    lines = ["Environment variables (names and descriptions only):", ""]
    for name, field in EnvironmentSettings.model_fields.items():
        default = field.default
        suffix = (
            f" (default: {default})" if isinstance(default, (int, float, bool)) else ""
        )
        lines.append(f"- `{name}`: {field.description}{suffix}")
    lines.extend(
        [
            "",
            "CLI options override matching environment variables. Provider keys must be available "
            "to the OpenCode service or relevant MCP server; external tools may support additional variables.",
        ]
    )
    return "\n".join(lines)


def environment_template() -> str:
    """Render a dotenv template from field metadata, never from live settings."""
    lines = [
        "# Uncomment and edit the settings you need. Never commit credentials.",
        "# CLI options override matching environment variables.",
        "",
    ]
    for name, field in EnvironmentSettings.model_fields.items():
        for line in (field.description or "").splitlines():
            lines.append(f"# {line}")
        default = field.default
        if isinstance(default, bool):
            value = str(default).lower()
        elif isinstance(default, (int, float)):
            value = str(default)
        else:
            value = ""
        lines.extend((f"# {name}={value}", ""))
    return "\n".join(lines)
