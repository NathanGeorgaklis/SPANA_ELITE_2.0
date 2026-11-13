"""API keys: read from the environment or the gitignored .env file. Never logged, printed or written to output."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parent.parent


class SecretError(RuntimeError):
    pass


def load_secret(env_var: str, env_file: Optional[Path] = None) -> str:
    """The value of `env_var`: the process environment first, then .env at the repository root. Fails closed."""
    value = os.environ.get(env_var)
    if not value:
        f = REPO / ".env" if env_file is None else Path(env_file)
        if f.is_file():
            value = dotenv_values(f).get(env_var)
    if not value or not value.strip():
        raise SecretError(f"{env_var} is not set: export it or add it to .env (see .env.example)")
    return value.strip()


def redact(text: str, secret: str) -> str:
    """Remove a secret from text that may be shown or stored (error messages from HTTP clients)."""
    return text.replace(secret, "***") if secret else text
