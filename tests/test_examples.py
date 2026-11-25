"""Every script in examples/ runs from the repository root and exits with code 0.

An example that needs an optional dependency group is skipped when the group is not
installed. The training examples run with short step counts here.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))

# example: (module of its optional dependency group, extra arguments)
SETTINGS = {
    "05_casadi_optimize": ("casadi", ["--states", "20"]),
    "06_train_synthetic": (None, ["--steps", "60", "--flights", "12"]),
    "07_federated_round": (None, ["--base-steps", "60", "--steps", "30"]),
    "09_compare_openap": ("openap", []),
}


def run(args, **kw):
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        **kw,
    )


def test_examples_exist():
    assert len(EXAMPLES) == 9
    assert set(SETTINGS) <= {p.stem for p in EXAMPLES}


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(path):
    module, extra = SETTINGS.get(path.stem, (None, []))
    if module and importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} is not installed")
    r = run([str(path.relative_to(ROOT)), *extra])
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip()


def test_cruise_example_plots(tmp_path):
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not installed")
    r = run(["examples/02_cruise_performance.py", "--plot", "--out-dir", str(tmp_path)])
    assert r.returncode == 0, r.stderr[-2000:]
    assert {p.name for p in tmp_path.glob("*.png")} == {
        "sar_mach.png",
        "best_altitude.png",
    }


def test_onnx_example_without_torch():
    code = (
        "import runpy, sys; "
        "runpy.run_path('examples/04_onnx_without_torch.py', run_name='__main__'); "
        "print('torch' in sys.modules, 'neuralfuel' in sys.modules)"
    )
    r = run(["-c", code])
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "False False"
