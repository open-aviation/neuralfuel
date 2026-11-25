import numpy as np
import pandas as pd

from neuralfuel.checks import check
from neuralfuel.schema import validate


def errors(df):
    return {p.column for p in validate(df) if p.level == "error"}


def test_valid(syn_samples):
    assert errors(syn_samples) == set()


def test_missing_columns(syn_samples):
    assert errors(syn_samples.drop(columns=["mass_kg"])) == {"mass_kg"}
    assert errors(syn_samples.drop(columns=["mach"])) == {"cas_kt/tas_kt/mach"}
    assert errors(syn_samples.drop(columns=["sat_k"])) == {"tat_k/sat_k"}


def test_unit_problems(syn_samples):
    d = syn_samples.copy()
    d["sat_k"] = d.sat_k - 273.15  # degC
    d["mass_kg"] = d.mass_kg / 1000  # t
    assert errors(d) == {"sat_k", "mass_kg"}
    d = syn_samples.copy()
    d["clean"] = d.clean.astype(int)
    d["time"] = d.time.astype(str)
    assert errors(d) == {"clean", "time"}


def test_check_names_suspect_column(syn_samples):
    assert check(syn_samples)["ok"]
    d = syn_samples.copy()
    d["ff_kgh"] = d.ff_kgh * 2.20462  # total fuel flow in lb/h
    r = check(d)
    failed = {t["column"] for t in r["tests"] if t["status"] == "fail"}
    assert not r["ok"] and failed == {"ff_kgh/mass_kg", "ff1_kgh/ff2_kgh"}
    d = syn_samples.copy()
    d["sat_k"] = d.sat_k + 28.0  # wrong offset in the temperature conversion
    failed = {t["column"] for t in check(d)["tests"] if t["status"] == "fail"}
    assert failed == {"sat_k"}
    d = syn_samples.copy()
    d["gs_kt"] = d.gs_kt * 0.514444  # m/s
    failed = {t["column"] for t in check(d)["tests"] if t["status"] == "fail"}
    assert failed == {"gs_kt"}
    d = syn_samples.copy()
    d["mach"] = np.round(d.mach * 0.6, 4)
    failed = {t["column"] for t in check(d)["tests"] if t["status"] == "fail"}
    assert "mach" in failed


def test_missing_flag_values(syn_samples):
    d = syn_samples.copy()
    d["clean"] = d.clean.astype("boolean")
    d.loc[d.index[50], "clean"] = pd.NA
    assert errors(d) == {"clean"}
    p = [p for p in validate(d) if p.level == "error"]
    assert "missing" in p[0].message


def test_mass_against_fuel_in_continuous_segments(syn_samples):
    name = "mass decrease against integrated fuel flow"

    def result(d):
        return next(t for t in check(d)["tests"] if t["test"] == name)

    assert result(syn_samples)["status"] == "pass"
    # the first and last 20 minutes of each flight: the gap holds burned fuel that the
    # integral does not see, so the test reports too little coverage, not a unit error
    n = syn_samples.groupby("flight_id").cumcount()
    m = syn_samples.groupby("flight_id").cumcount(ascending=False)
    r = result(syn_samples[(n < 1200) | (m < 1200)])
    assert r["status"] == "warn" and "insufficient coverage" in r["message"]
    # the first and last 40 minutes: each continuous segment is tested alone
    r = result(syn_samples[(n < 2400) | (m < 2400)])
    assert r["status"] == "pass" and 0.8 <= r["value"] <= 1.25
