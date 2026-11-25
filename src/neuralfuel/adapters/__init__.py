"""Adapters: functions that read a raw file and return a DataFrame in the schema.

An adapter is any function `load(path) -> pandas.DataFrame` whose result passes
`neuralfuel.schema.validate`. Its module can list the raw measurement columns in
MEASUREMENTS (a list of column names); `profile --raw` gives value ranges only for
these columns and the numeric columns of the schema (`measurements`). Register a new
one in ADAPTERS, pass it on the command line as `package.module:function`, or give the
path of a Python file (`my_adapter.py` for its function `load`, `my_adapter.py:function`
for another one).
"""

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from . import example_a320, synthetic


def read_table(path, columns=None) -> pd.DataFrame:
    """Read a parquet or CSV file."""
    path = Path(path)
    if path.suffix.lower() in (".csv", ".gz"):
        return pd.read_csv(path, usecols=columns)
    return pd.read_parquet(path, columns=columns)


def canonical(path) -> pd.DataFrame:
    """A file that is already in the schema (parquet or CSV)."""
    df = read_table(path)
    if "time" in df and not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


ADAPTERS = {
    "canonical": canonical,
    "example_a320": example_a320.load,
    "synthetic": synthetic.load,
}


def load_file(path):
    """The module in a Python file (relative paths from the working directory). The
    module key holds a hash of the full path, so two files with the same name stay
    apart."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ImportError(f"adapter file not found: {path}")
    tag = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    name = f"neuralfuel_adapter_{path.stem}_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measurements(fn) -> tuple:
    """The MEASUREMENTS of the module of the adapter function fn (empty if none)."""
    mod = sys.modules.get(getattr(fn, "__module__", ""), None)
    return tuple(str(c) for c in getattr(mod, "MEASUREMENTS", ()))


def get_adapter(name: str):
    """Adapter by registered name, by 'module:function' or by 'file.py[:function]'."""
    if name in ADAPTERS:
        return ADAPTERS[name]
    if name.endswith(".py") or ".py:" in name:
        file, _, fn = name.rpartition(".py")
        return getattr(load_file(file + ".py"), fn.lstrip(":") or "load")
    if ":" in name:
        mod, fn = name.split(":", 1)
        return getattr(importlib.import_module(mod), fn)
    raise KeyError(
        f"unknown adapter '{name}'; known: {', '.join(ADAPTERS)}, module:function "
        "or file.py[:function]"
    )
