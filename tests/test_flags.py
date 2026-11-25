"""Optional configuration flags: prep exclusion, carry-through and the check report."""

import numpy as np
import pytest

from neuralfuel.checks import check
from neuralfuel.prep import PrepConfig, prepare
from neuralfuel.schema import validate


@pytest.fixture(scope="module")
def flagged(syn_samples):
    """Synthetic samples with flags on in known stretches of each flight."""
    d = syn_samples.copy()
    sec = (d.time - d.groupby("flight_id").time.transform("min")).dt.total_seconds()
    d["speedbrake"] = (sec >= 1000) & (sec < 1100)
    left = d.groupby("flight_id").time.transform("max") - d.time
    d["gear_down"] = left.dt.total_seconds() < 120  # last 2 min, below 10,000 ft
    d["anti_ice"] = (sec >= 2000) & (sec < 2600)
    d["tail"] = np.where(d.flight_id.str[-1].isin(list("02468")), "A", "B")
    d["sec"] = sec
    return d


def block_secs(blocks):
    """Start second of each block in its flight."""
    return blocks.time_s.to_numpy() - 4.5


def test_schema_accepts_flags(flagged):
    assert [p for p in validate(flagged) if p.level == "error"] == []
    bad = flagged.assign(speedbrake=flagged.speedbrake.astype(int))
    assert {p.column for p in validate(bad) if p.level == "error"} == {"speedbrake"}


def test_prep_excludes_and_carries(flagged):
    blocks, windows, summary = prepare(flagged.drop(columns="sec"))
    s = block_secs(blocks)
    assert not ((s > 990) & (s < 1100)).any()  # speed brake
    end = blocks.flight_id.map(flagged.groupby("flight_id").sec.max()).to_numpy()
    assert (s + 9 < end - 119).all()  # gear down in the last 2 min
    assert ((s > 2000) & (s < 2590)).any()  # anti-ice is kept by default
    ai = blocks.anti_ice.to_numpy()
    assert ai[(s >= 2000) & (s <= 2590)].all() and not ai[s < 1990].any()
    assert set(blocks["tail"]) == {"A", "B"}
    assert "anti_ice" in windows and "tail" in windows
    counts = summary["filter_counts"]
    assert counts["after speed brake retracted"] < counts["after clean configuration"]
    assert counts["after gear up"] < counts["after speed brake retracted"]
    assert "after anti-ice off" not in counts
    # with the setting, anti-ice samples go as well, in blocks and windows
    b2, w2, sum2 = prepare(
        flagged.drop(columns="sec"), PrepConfig(exclude_anti_ice=True)
    )
    s2 = block_secs(b2)
    assert not ((s2 > 1990) & (s2 < 2600)).any() and not b2.anti_ice.any()
    assert not w2.anti_ice.any() and len(w2) < len(windows)
    assert "after anti-ice off" in sum2["filter_counts"]


def test_prep_without_flags_unchanged(syn_samples, syn_prepared):
    blocks, _, _ = syn_prepared
    assert "anti_ice" not in blocks
    b2, _, _ = prepare(
        syn_samples.assign(speedbrake=False, gear_down=False, anti_ice=False)
    )
    np.testing.assert_array_equal(b2.ln_ff, blocks.ln_ff)


def test_check_reports_flag_shares(flagged):
    r = check(flagged.drop(columns="sec"))
    assert r["ok"]
    fl = r["flags"]
    assert fl["samples"] == len(flagged)  # all synthetic samples fly above 100 kt
    assert fl["all"]["speedbrake"] == pytest.approx(flagged.speedbrake.mean())
    assert fl["all"]["anti_ice"] == pytest.approx(flagged.anti_ice.mean())
    assert set(fl["bands"]) == {"below 10000 ft", "10000 to 25000 ft", "above 25000 ft"}
    # an inverted gear flag fails the check and names the column
    r = check(flagged.drop(columns="sec").assign(gear_down=~flagged.gear_down))
    failed = {t["column"] for t in r["tests"] if t["status"] == "fail"}
    assert not r["ok"] and failed == {"gear_down"}
    # without the optional flags the shares are None
    r = check(flagged.drop(columns=["sec", "speedbrake", "gear_down", "anti_ice"]))
    assert r["flags"]["all"]["speedbrake"] is None and r["flags"]["all"]["clean"] > 0.9
