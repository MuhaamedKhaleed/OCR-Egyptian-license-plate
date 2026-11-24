"""Explicit configuration for the optional dataset download experiment."""

import os
from pathlib import Path

from dotenv import load_dotenv


def roboflow_key(env_file: Path | None = None) -> str:
    """Read a required download key; never override an existing environment value.

    An env file is loaded only when explicitly supplied. Local image inference
    requires no API key and does not call this function.
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise ValueError("Set ROBOFLOW_API_KEY before downloading the dataset.")
    return key
