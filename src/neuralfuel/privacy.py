"""Shareable JSON: finite values, allowed keys and a scan for identifying fields.

`clean_json` and `dumps` give standard JSON (no NaN or Infinity; a missing value is
null). `allowed` keeps only the keys of a record format; the formats of the run
record and the validation report are defined next to the code that writes them
(neuralfuel.train.RUN_RECORD, neuralfuel.validate.REPORT). A format (spec) is

  NUM        a number, a bool, null, or a list of these (lists can be nested)
  STR        a string, null, or a list of strings
  {key: spec, ...}
             a dict with these keys (or null); the key NUM_KEY stands for any key
             that is a number written as text (for example "270.9")
  [spec]     a list of entries of this format (or null)
"""

import json
import math
import re

import numpy as np

FORBIDDEN_TOKENS = {
    "flight",
    "tail",
    "registration",
    "reg",
    "callsign",
    "icao",
    "lat",
    "latitude",
    "lon",
    "longitude",
    "position",
    "timestamp",
    "date",
    "utc",
}
DATE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}")
NUM, STR, NUM_KEY = "num", "str", "<number>"
_NUMBER = re.compile(r"^-?\d+(\.\d+)?(e[+-]?\d+)?$")


def clean_json(obj):
    """NaN and inf to None, numpy scalars to Python numbers, tuples to lists."""
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def dumps(obj, indent=1) -> str:
    """Standard JSON text of obj (clean_json first; NaN and Infinity are refused)."""
    return json.dumps(clean_json(obj), indent=indent, allow_nan=False, default=str)


def _leaf_ok(v, kind):
    if isinstance(v, list):
        return all(_leaf_ok(x, kind) for x in v)
    if v is None:
        return True
    if kind == NUM:
        return isinstance(v, (bool, int, float))
    return isinstance(v, str)


def allowed(obj, spec, path="$"):
    """(copy of obj with only the entries of the format spec, paths of the entries
    that were dropped: unknown keys and values of the wrong kind)."""
    if isinstance(spec, dict):
        if obj is None:
            return None, []
        if not isinstance(obj, dict):
            return None, [path]
        out, dropped = {}, []
        for k, v in obj.items():
            sub = spec.get(k)
            if sub is None and NUM_KEY in spec and _NUMBER.match(str(k)):
                sub = spec[NUM_KEY]
            if sub is None:
                dropped.append(f"{path}.{k}")
                continue
            val, d = allowed(v, sub, f"{path}.{k}")
            dropped += d
            if d != [f"{path}.{k}"]:
                out[k] = val
        return out, dropped
    if isinstance(spec, list):
        if obj is None:
            return None, []
        if not isinstance(obj, list):
            return None, [path]
        out, dropped = [], []
        for i, v in enumerate(obj):
            val, d = allowed(v, spec[0], f"{path}[{i}]")
            dropped += d
            out.append(val)
        return out, dropped
    if _leaf_ok(obj, spec):
        return obj, []
    return None, [path]


def find_identifiers(obj, extra_values=(), path="$"):
    """Paths of keys whose name suggests an identifier, date or position, and of string
    values that look like dates or equal one of extra_values (for example the local
    flight identifiers)."""
    extra = {str(v) for v in extra_values}
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            tokens = set(re.split(r"[^a-z0-9]+", str(k).lower()))
            if tokens & FORBIDDEN_TOKENS:
                hits.append(f"{path}.{k}")
            hits += find_identifiers(v, extra, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_identifiers(v, extra, f"{path}[{i}]")
    elif isinstance(obj, str) and (DATE.search(obj) or obj in extra):
        hits.append(path)
    return list(dict.fromkeys(hits))  # a key and its value can give the same path
