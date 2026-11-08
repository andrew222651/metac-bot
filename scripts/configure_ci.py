"""Configure the ephemeral GitHub runner before starting its OpenCode service."""

import json
import os
from pathlib import Path


def main():
    if os.getenv("GITHUB_ACTIONS") != "true":
        raise SystemExit("This script configures GitHub Actions only")
    model = os.getenv("OPENCODE_MODEL")
    if not model:
        for key, candidate in (
            ("OPENROUTER_API_KEY", "openrouter/openai/gpt-4.1-mini"),
            ("OPENAI_API_KEY", "openai/gpt-4.1-mini"),
            ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-5"),
        ):
            if os.getenv(key):
                model = candidate
                break
    if not model:
        raise SystemExit(
            "Set a provider API key and optionally the OPENCODE_MODEL repository variable"
        )
    directory = (
        Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "opencode"
    )
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "opencode.json"
    if config.exists():
        raise SystemExit(f"Refusing to replace existing CI config: {config}")
    config.write_text(
        json.dumps(
            {
                "model": model,
                "websearch": {
                    "provider": os.getenv("OPENCODE_SEARCH_PROVIDER") or "random"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
