import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import run_cli

from neuralfuel.adapters import example_a320, get_adapter
from neuralfuel.cli import main
from neuralfuel.schema import validate


def test_example_a320_scale_factors(tmp_path):
    n = 5
    raw = pd.DataFrame(
        {
            "flight_id": ["F1"] * n,
            "timestamp": pd.date_range("2030-01-01", periods=n, freq="1s"),
            "altitude": np.full(n, 35000.0),
            "CAS": np.full(n, 270.0, dtype=np.float32),
            "groundspeed": np.full(n, 460.0),
            "gross_weight": np.full(n, 62000.0, dtype=np.float32),
            "fuel_flow": np.full(n, 2400.0, dtype=np.float32),
            "ff1": np.full(n, 1200.0, dtype=np.float32),
            "ff2": np.full(n, 1200.0, dtype=np.float32),
            "total_air_temperature": np.array(
                [-100.0, -127.0, 0.0, 4.0, -40.0], dtype=np.float32
            ),
            "flap_raw": np.array([0.0, 1.4, -1.0, 2.0, 145.0], dtype=np.float32),
            "roll": np.zeros(n, dtype=np.float32),
            "unused": np.zeros(n),
        }
    )
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path)
    d = example_a320.load(path)
    np.testing.assert_allclose(
        d.tat_k.to_numpy()[[0, 2, 3, 4]], [248.15, 273.15, 274.15, 263.15]
    )
    assert np.isnan(d.tat_k.iloc[1])  # raw floor is invalid
    assert d.clean.tolist() == [True, True, True, False, False]
    assert str(d.time.dt.tz) == "UTC"
    assert [p.column for p in validate(d) if p.level == "error"] == []


def test_get_adapter():
    assert get_adapter("example_a320") is example_a320.load
    assert get_adapter("neuralfuel.adapters.example_a320:load") is example_a320.load
    with pytest.raises(KeyError):
        get_adapter("nope")


def test_adapter_from_file(tmp_path, syn_csv):
    path = tmp_path / "my_adapter.py"
    path.write_text(
        "from neuralfuel.adapters import synthetic\n\n"
        "def load(path):\n    return synthetic.load(path)\n\n"
        "def other(path):\n    return synthetic.load(path).head(3)\n"
    )
    d = get_adapter(str(path))(syn_csv)
    assert [p.column for p in validate(d) if p.level == "error"] == []
    assert len(get_adapter(f"{path}:other")(syn_csv)) == 3
    with pytest.raises(ImportError):
        get_adapter(str(tmp_path / "missing.py"))


def test_cli_adapter_from_file(tmp_path, syn_csv, capsys):
    path = tmp_path / "my_adapter.py"
    path.write_text("from neuralfuel.adapters.synthetic import load\n")
    code, out = run_cli(capsys, "profile", str(syn_csv), "--adapter", str(path))
    assert code == 0 and out["adapter"] == str(path)
    code, out = run_cli(
        capsys, "profile", str(syn_csv), "--adapter", str(tmp_path / "no.py")
    )
    assert code == 2


def test_local_toml_adapter_path_and_module_keys(tmp_path, syn_csv, monkeypatch):
    from neuralfuel.config import resolve

    cfg = tmp_path / "cfg"
    (cfg / "local").mkdir(parents=True)
    (cfg / "local" / "my_adapter.py").write_text(
        "from neuralfuel.adapters.synthetic import load\nTAG = 1\n"
    )
    (cfg / "local.toml").write_text(
        f'[data]\npath = "{syn_csv}"\nadapter = "local/my_adapter.py"\n'
    )
    other = tmp_path / "other"
    other.mkdir()
    (other / "my_adapter.py").write_text("TAG = 2\n\ndef load(path):\n    return 2\n")
    monkeypatch.setenv("NEURALFUEL_CONFIG", str(cfg / "local.toml"))
    monkeypatch.delenv("NEURALFUEL_ADAPTER", raising=False)
    monkeypatch.chdir(other)  # a relative path from the working directory is wrong
    _, adapter = resolve()
    assert adapter == str(cfg / "local" / "my_adapter.py")
    assert len(get_adapter(adapter)(syn_csv)) > 0
    # two files with the same name do not share a module
    assert get_adapter("my_adapter.py")(syn_csv) == 2
    assert len(get_adapter(adapter)(syn_csv)) > 0


def test_unknown_adapter_message(capsys, syn_csv):
    code, out = run_cli(capsys, "check", str(syn_csv), "--adapter", "nope")
    assert code == 2 and out["error"].startswith("unknown adapter 'nope'")


def test_example_adapter_copy_works(tmp_path, syn_csv):
    import neuralfuel.adapters.synthetic as mod

    copy = tmp_path / "copied.py"
    copy.write_text(Path(mod.__file__).read_text())
    assert len(get_adapter(str(copy))(syn_csv)) > 0


def test_adapter_errors_hold_no_cell_values(capsys, tmp_path):
    """An exception of the adapter is shown with its class, the operation and the
    place in the adapter, never with its message (it can hold a cell value); a
    missing column is named."""
    mark = "PRIVATE-CELL-4711"
    raw = pd.DataFrame({"stamp": ["2030-01-01T00:00:00", mark], "x": [1.0, 2.0]})
    raw.to_csv(tmp_path / "raw.csv", index=False)
    adapter = tmp_path / "bad_adapter.py"
    adapter.write_text(
        "import pandas as pd\n"
        "from neuralfuel.adapters import read_table\n\n\n"
        "def load(path):\n"
        "    r = read_table(path)\n"
        "    return pd.DataFrame({'time': pd.to_datetime(r.stamp, format='ISO8601')})\n"
        "\n\n"
        "def by_value(path):\n"
        "    r = read_table(path)\n"
        "    return {'x': 1}[r.stamp.iloc[1]]\n"
        "\n\n"
        "def no_column(path):\n"
        "    return read_table(path)['altitude_raw']\n"
    )
    data = str(tmp_path / "raw.csv")
    for cmd in ("profile", "check", "prep"):
        out_dir = () if cmd != "prep" else ("--out", str(tmp_path / "p"))
        code, out = run_cli(capsys, cmd, data, "--adapter", str(adapter), *out_dir)
        assert code == 2 and mark not in json.dumps(out), cmd
        assert "ValueError in load, line 7 of the adapter" in out["error"]
        assert main([cmd, data, "--adapter", str(adapter), *out_dir]) == 2
        text = capsys.readouterr()
        assert mark not in text.out + text.err and "ValueError in load" in text.err
    code, out = run_cli(capsys, "profile", data, "--adapter", f"{adapter}:by_value")
    assert code == 2 and mark not in json.dumps(out) and out["column"] is None
    assert "KeyError in by_value, line 12 of the adapter" in out["error"]
    code, out = run_cli(capsys, "check", data, "--adapter", f"{adapter}:no_column")
    assert code == 2 and out["column"] == "altitude_raw"
    assert "column 'altitude_raw' not found" in out["error"]


def test_unexpected_data_errors_hold_no_message(capsys, syn_csv, monkeypatch):
    """An unexpected error (exit code 3) of a command that reads the samples does not
    show its message either."""
    mark = "PRIVATE-CELL-4712"

    def fail(*a, **k):
        raise ValueError(mark)

    monkeypatch.setattr("neuralfuel.checks.profile", fail)
    code, out = run_cli(capsys, "profile", str(syn_csv), "--adapter", "synthetic")
    assert code == 3 and mark not in json.dumps(out)
    assert out["error"].startswith("'profile' failed: ValueError in fail, line ")
