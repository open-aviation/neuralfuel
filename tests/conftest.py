import json
from dataclasses import asdict
from pathlib import Path

import pytest

from neuralfuel import synthetic
from neuralfuel.cli import main
from neuralfuel.prep import PrepConfig


def run_cli(capsys, *argv):
    """Run a CLI command in-process with --json; return (exit code, parsed output).
    The whole stdout must be one JSON object."""
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def write_prep(folder, blocks, windows, cfg=None):
    """A prepared-data folder as 'prep' writes it: blocks, windows and prep.json."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    blocks.to_parquet(folder / "blocks.parquet")
    windows.to_parquet(folder / "windows.parquet")
    rec = {"config": asdict(cfg or PrepConfig())}
    (folder / "prep.json").write_text(json.dumps(rec))
    return folder


@pytest.fixture(scope="session")
def syn_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("syn") / "syn.csv"
    synthetic.generate(n_flights=12, seed=1).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def syn_samples(syn_csv):
    from neuralfuel.adapters import synthetic as adapter

    return adapter.load(syn_csv)


@pytest.fixture(scope="session")
def syn_prepared(syn_samples):
    from neuralfuel.prep import prepare

    return prepare(syn_samples)
