"""Local settings: data path and adapter, never stored in tracked files.

Order: command-line argument, then the environment variables NEURALFUEL_DATA and
NEURALFUEL_ADAPTER, then `local.toml` in the working directory (or the file named by
NEURALFUEL_CONFIG):

    [data]
    path = "/path/to/flights.parquet"
    adapter = "example_a320"

A relative path of an adapter file (`local/my_adapter.py`) in local.toml starts from
the folder of that local.toml, not from the working directory.
"""

import os
import tomllib
from pathlib import Path


def local_settings() -> dict:
    path = Path(os.environ.get("NEURALFUEL_CONFIG", "local.toml"))
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        d = dict(tomllib.load(fh).get("data", {}))
    a = d.get("adapter")
    if isinstance(a, str) and (a.endswith(".py") or ".py:" in a):
        file, _, fn = a.rpartition(".py")
        f = Path(file + ".py")
        if not f.is_absolute():
            d["adapter"] = str(path.resolve().parent / f) + fn
    return d


def resolve(data=None, adapter=None):
    """(data path, adapter name) from arguments, environment and local.toml."""
    loc = local_settings()
    data = data or os.environ.get("NEURALFUEL_DATA") or loc.get("path")
    adapter = (
        adapter
        or os.environ.get("NEURALFUEL_ADAPTER")
        or loc.get("adapter")
        or "canonical"
    )
    return data, adapter
