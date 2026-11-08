#!/usr/bin/env bash
# Install locked Python dependencies on Linux or in Termux's Debian container.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

sync_options=()
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 1 || "$1" != "--all-groups" ]]; then
    printf 'Usage: bash scripts/install.sh [--all-groups]\n' >&2
    exit 2
  fi
  sync_options+=(--all-groups)
fi

if [[ -n "${TERMUX_VERSION:-}" && -z "${PD_CONTAINER:-}" ]]; then
  missing=()
  command -v uv >/dev/null 2>&1 || missing+=(uv)
  command -v proot-distro >/dev/null 2>&1 || missing+=(proot-distro)
  if [[ ${#missing[@]} -gt 0 ]]; then
    if ! command -v pkg >/dev/null 2>&1; then
      printf 'Termux package manager (pkg) is required to install missing tools.\n' >&2
      exit 1
    fi
    printf 'Installing missing Termux packages: %s\n' "${missing[*]}"
    pkg install -y "${missing[@]}"
    for tool in "${missing[@]}"; do
      if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Installed package %s did not provide an executable on PATH.\n' "$tool" >&2
        exit 1
      fi
    done
  fi

  containers="$(proot-distro list --quiet)"
  if ! grep -Fxq 'debian' <<< "$containers"; then
    printf 'Creating Debian container...\n'
    proot-distro install debian:trixie
  fi

  if ! proot-distro login debian -- test -x /usr/bin/python3; then
    printf 'Installing Python in Debian...\n'
    proot-distro login debian -- apt-get update
    proot-distro login debian -- env DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv ca-certificates
  fi
  if ! proot-distro login debian -- /usr/bin/python3 -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] < (3, 14))'; then
    printf 'The existing Debian container needs Python 3.11–3.13 at /usr/bin/python3. It was not modified.\n' >&2
    exit 1
  fi

  printf 'Installing dependencies in Debian...\n'
  exec proot-distro login --work-dir "$ROOT" debian -- env \
    UV_PROJECT_ENVIRONMENT="${METAC_BOT_VENV:-/root/.venvs/metac-bot}" \
    UV_LINK_MODE=copy \
    uv sync --locked --inexact --python /usr/bin/python3 --project "$ROOT" "${sync_options[@]}"
fi

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required on Linux/macOS; install it first: https://docs.astral.sh/uv/\n' >&2
  exit 1
fi
exec uv sync --locked --inexact --project "$ROOT" "${sync_options[@]}"
