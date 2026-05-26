import os
from pathlib import Path


DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_local_env(env_path: str | Path = DEFAULT_ENV_PATH, override: bool = False) -> None:
    """Loads simple KEY=VALUE pairs from a local .env file into the process."""

    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if not key:
            continue

        current_value = os.environ.get(key)
        should_set = (
            override
            or key not in os.environ
            or current_value is None
            or not current_value.strip()
        )

        if should_set:
            os.environ[key] = value
