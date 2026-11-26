"""Symbolic import of the exported ONNX file into CasADi (optional dependency).

Run with the dependency group: uv run --group casadi pytest tests/test_casadi.py
"""

import pytest

pytest.importorskip("casadi")

from neuralfuel.inference import casadi_check, export_onnx
from neuralfuel.support import random_inputs
from neuralfuel.train import TrainConfig, train


@pytest.mark.parametrize("casadi_ops", [False, True])
def test_casadi_matches_pytorch(syn_prepared, tmp_path, casadi_ops):
    blocks, windows, _ = syn_prepared
    cfg = TrainConfig(max_steps=4, eval_every=2, device="cpu", threads=2)
    cfg.loss.n_colloc = 256
    model, _ = train(blocks, windows, cfg, log=lambda s: None)
    path = tmp_path / "m.onnx"
    export_onnx(model, path, casadi=casadi_ops)
    x = random_inputs(model.envelope, 200, seed=0)
    for expand in (False, True):
        r = casadi_check(model, path, x, expand=expand, repeat=1)
        assert r["states"] == 200
        assert r["max_abs_diff_ln_ff"] < 1e-9
        assert r["max_abs_diff_gradient"] < 1e-7 * max(1.0, r["max_abs_gradient"])
        assert r["max_abs_diff_hessian"] < 1e-7 * max(1.0, r["max_abs_hessian"])


def test_casadi_export_refuses_overflow(tmp_path):
    """Softplus as log(1 + exp(x)) overflows in float32 for a large pre-activation."""
    import numpy as np
    import torch

    from neuralfuel.model import FuelFlowNet

    net = FuelFlowNet(np.zeros(5), np.ones(5), 7.0, 0.01, hidden=1, layers=1)
    with torch.no_grad():
        net.net[0].weight.zero_()
        net.net[0].bias.fill_(100.0)
        net.net[2].weight.fill_(1.0)
        net.net[2].bias.zero_()
    with pytest.raises(ValueError, match="overflow in float32"):
        export_onnx(net, tmp_path / "m.onnx", casadi=True, n_check=16)
    assert not (tmp_path / "m.onnx").exists()
    info = export_onnx(net, tmp_path / "m.onnx", n_check=16)  # plain Softplus
    assert info["max_rel_diff_ff"] < 1e-4


def test_example_optimizer_checks_the_fixed_inputs():
    import runpy
    from pathlib import Path

    from neuralfuel.inference import casadi_function, onnx_metadata
    from neuralfuel.reference import model_inputs
    from neuralfuel.support import in_envelope

    root = Path(__file__).parent.parent
    ex = runpy.run_path(str(root / "examples" / "05_casadi_optimize.py"))
    onnx_path = root / "models" / "a320" / "nf-a320-cfm56_5b.onnx"
    env = onnx_metadata(onnx_path)["envelope"]
    f = casadi_function(onnx_path)
    assert ex["best_range"](f, env, 72000.0)[3] == "outside the envelope"
    m, h, _, status = ex["best_range"](f, env, 62000.0)
    assert status == "Solve_Succeeded"
    x = model_inputs(m, h, 62000.0, ex["DT_ISA"], 0.0)
    assert in_envelope(env, x).all()


def test_casadi_function_refuses_external_data(tmp_path):
    """An ONNX file with tensors in an external file is refused before CasADi reads
    it (casadi_function loads the graph without external data)."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from neuralfuel.inference import casadi_function, external_tensors

    w = numpy_helper.from_array(np.ones((6, 1), dtype=np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["ln_fuel_flow"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 6])],
        [helper.make_tensor_value_info("ln_fuel_flow", TensorProto.FLOAT, ["n", 1])],
        [w],
    )
    m = helper.make_model(graph)
    path = tmp_path / "ext.onnx"
    onnx.save_model(
        m,
        str(path),
        save_as_external_data=True,
        location="weights.bin",
        size_threshold=0,
    )
    assert (tmp_path / "weights.bin").exists()
    assert external_tensors(onnx.load(str(path), load_external_data=False).graph)
    with pytest.raises(ValueError, match="external data"):
        casadi_function(path)


def test_example_bounds_without_cruise_limit():
    """examples/05: an envelope without steady windows (no W/delta limit of cruise)
    still gives the level-flight bounds."""
    import copy
    import importlib.util
    from pathlib import Path

    from neuralfuel.inference import casadi_function, onnx_metadata

    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        "example05", root / "examples/05_casadi_optimize.py"
    )
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    onnx_path = root / "models/a320/nf-a320-cfm56_5b-xs.onnx"
    env = copy.deepcopy(onnx_metadata(onnx_path)["envelope"])
    env["cruise"].update(max_wd_t=None, max_cl=None)
    m, h, sar, status = ex.best_range(casadi_function(onnx_path), env, 62000.0)
    lv = env["phases"]["level"]
    assert (
        lv["mach"][0] <= m <= lv["mach"][1] and lv["alt_ft"][0] <= h <= lv["alt_ft"][1]
    )
    assert sar > 0 and status
