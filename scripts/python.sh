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
  exec proot-distro login --work-dir "$ROOT" debian -- env \
    OPENCODE_SERVICE_FILE="${OPENCODE_SERVICE_FILE:-}" \
    PYTENSOR_FLAGS="$PYTENSOR_FLAGS" \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT="${METAC_BOT_VENV:-/root/.venvs/metac-bot}" \
    uv run --locked --project "$ROOT" python "$@"
fi
exec uv run --locked --project "$ROOT" python "$@"
