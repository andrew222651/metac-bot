#!/usr/bin/env bash
# Use the same Python environment for the runner and OpenCode's MCP subprocess.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# PRoot may expose Android's compiler without Linux Python headers. Disable
# PyTensor C compilation there; callers can explicitly override PYTENSOR_FLAGS.
if [[ ( -n "${TERMUX_VERSION:-}" && -z "${PD_CONTAINER:-}" ) || ( -n "${PD_CONTAINER:-}" && -n "${ANDROID_ROOT:-}" ) ]]; then
  export PYTENSOR_FLAGS="${PYTENSOR_FLAGS:-cxx=}"
fi
if [[ -n "${TERMUX_VERSION:-}" && -z "${PD_CONTAINER:-}" ]]; then
  if [[ -z "${OPENCODE_SERVICE_FILE:-}" && -z "${OPENCODE_SERVER_URL:-}" ]]; then
    # The top-level CLI may start the service, but MCP/modeling subprocesses
    # must not attempt to start the service that launched them.
    if [[ ( "${1:-}" == "-m" && "${2:-}" == "metac_bot" ) || "${1:-}" == "main.py" ]]; then
      opencode service start >/dev/null
    fi
    export OPENCODE_SERVICE_FILE="$(opencode debug paths state)/service.json"
  fi
  export METAC_BOT_HOST_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
  export METAC_BOT_HOST_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
  export UV_LINK_MODE=copy
  export UV_PROJECT_ENVIRONMENT="${METAC_BOT_VENV:-/root/.venvs/metac-bot}"
  # PRoot sanitizes the host environment. Forward supported variables over an
  # inherited pipe, rather than exposing credentials in command-line arguments
  # or writing them to a temporary file. Preserve stdin for Python scripts.
  exec 3< <(
    while IFS= read -r name; do
      case "$name" in
        METAC_BOT_*|OPENCODE_DB|OPENCODE_MODEL|OPENCODE_SEARCH_PROVIDER|OPENCODE_SERVICE_FILE|OPENCODE_SERVER_*|METACULUS_TOKEN|OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|EXA_API_KEY|TAVILY_API_KEY|FIRECRAWL_API_KEY|PARALLEL_API_KEY|TINYFISH_API_KEY|ASKNEWS_API_KEY|ASKNEWS_CLIENT_ID|ASKNEWS_SECRET|ASKNEWS_CACHE_MODE|PERPLEXITY_API_KEY|LIGHTNINGROD_API_KEY|PYTENSOR_FLAGS|OPENBLAS_NUM_THREADS|OMP_NUM_THREADS|XDG_CONFIG_HOME|XDG_DATA_HOME|XDG_STATE_HOME|XDG_CACHE_HOME|TMPDIR|UV_LINK_MODE|UV_NO_SYNC|UV_PROJECT_ENVIRONMENT)
          printf '%s=%q\n' "$name" "${!name}"
          ;;
      esac
    done < <(compgen -e)
  )
  exec proot-distro login --work-dir "$ROOT" debian -- /bin/bash -c '
    set -a
    source /dev/fd/3
    set +a
    exec 3<&-
    project=$1
    shift
    exec uv run --locked --project "$project" python "$@"
  ' metac-bot "$ROOT" "$@"
fi
exec uv run --locked --project "$ROOT" python "$@"
