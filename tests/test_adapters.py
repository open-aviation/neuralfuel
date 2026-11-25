from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import run_cli

from neuralfuel.adapters import example_a320, get_adapter
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
