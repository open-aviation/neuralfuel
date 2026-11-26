"""Shareable JSON: finite values, allowed keys and a scan for identifying fields.

`clean_json` and `dumps` give standard JSON (no NaN or Infinity; a missing value is
null). `allowed` keeps only the keys of a record format whose values have the declared
shape; the formats of the run record and the validation report are defined next to
the code that writes them (neuralfuel.train.RUN_RECORD, neuralfuel.validate.REPORT). A
format (spec) is

  NUM        a number, a bool or null (not a list)
  Nums(n)    a list of at most n numbers (exactly n with exact=True), or null; an
             entry can be null (a missing value)
  Text(re)   a string that matches the regular expression re as a whole (at most
             MAX_TEXT characters), or null; `enum` gives the Text of fixed strings
  {key: spec, ...}
             a dict with these keys (or null); the key NUM_KEY stands for any key
             that is a number written as text (for example "270.9")
  [spec]     a list of entries of this format (or null)

A value of another shape is dropped. No format holds free text: every string is a
version, a hash, a code, a name of a fixed set or a text that the package writes from
numbers.
"""

import dataclasses
import json
import math
import re
from dataclasses import dataclass

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
NUM, NUM_KEY = "num", "<number>"
_NUMBER = re.compile(r"^-?\d+(\.\d+)?(e[+-]?\d+)?$")
MAX_TEXT = 400  # characters of a Text value
MAX_SETTING_LIST = 32  # numbers in a list field of a settings dataclass


@dataclass(frozen=True)
class Nums:
    """A list of at most n numbers (exactly n with exact=True), or null."""

    n: int
    exact: bool = False

    def ok(self, v):
        if v is None:
            return True
        if not isinstance(v, list) or len(v) > self.n:
            return False
        if self.exact and len(v) != self.n:
            return False
        return all(x is None or _number(x) for x in v)


@dataclass(frozen=True)
class Text:
    """A string that matches the regular expression `pattern` as a whole, or null."""

    pattern: str

    def ok(self, v):
        if v is None:
            return True
        return (
            isinstance(v, str)
            and len(v) <= MAX_TEXT
            and re.fullmatch(self.pattern, v) is not None
        )


def enum(*values) -> Text:
    """The Text of a fixed set of strings."""
    return Text("|".join(re.escape(v) for v in values))


VERSION = Text(
    r"\d{1,4}(\.\d{1,4}){0,3}((a|b|rc|\.post|\.dev)\d{1,4})?(\+[a-z0-9.]{1,16})?"
)
SHA256 = Text(r"[0-9a-f]{64}")
CODE = Text(r"[a-z0-9_]{1,16}")  # an aircraft type code or an engine family
CODE_OR_EMPTY = Text(r"[a-z0-9_]{0,16}")
_G = r"-?[0-9.]{1,12}(e[+-]?\d{1,3})?"  # a number written with the format 'g'


def number_text(template: str) -> Text:
    """Text of a fixed sentence with numbers: each '{}' in template is a number."""
    return Text(_G.join(re.escape(part) for part in template.split("{}")))


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def settings_spec(cls, texts=None):
    """Format of the fields of a settings dataclass: a number, a bool or None is NUM,
    a tuple Nums(MAX_SETTING_LIST), a dataclass its own format. A str field takes
    its Text from texts (field name -> Text); a str field without one is an error."""
    texts = texts or {}
    out = {}
    for f in dataclasses.fields(cls):
        t = f.type if isinstance(f.type, type) else None
        if t is not None and dataclasses.is_dataclass(t):
            out[f.name] = settings_spec(t, texts)
        elif f.type in (str, "str"):
            out[f.name] = texts[f.name]
        elif f.type in (tuple, "tuple"):
            out[f.name] = Nums(MAX_SETTING_LIST)
        else:
            out[f.name] = NUM
    return out


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
    if kind == NUM:
        return v is None or isinstance(v, (bool, int, float))
    return kind.ok(v)


def allowed(obj, spec, path="$"):
    """(copy of obj with only the entries of the format spec, paths of the entries
    that were dropped: unknown keys and values of another shape). The paths hold the
    keys of the dropped entries, which can be private: show them only to the user,
    never write them into a shared file."""
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
