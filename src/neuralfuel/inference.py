"""Prediction, inputs from ADS-B / Mode S quality data, and ONNX export."""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

from . import atmos
from .atmos import FT, KT, T0
from .model import FuelFlowNet, state_hash
from .prep import FEATURES, energy_inputs
from .privacy import dumps
from .support import in_envelope, random_inputs

# float32 exp overflows above about 88.7; the export with casadi=True (Softplus as
# log(1 + exp(x))) is refused when a hidden pre-activation inside the envelope exceeds
# this value
MAX_PREACTIVATION = 80.0
# largest relative fuel-flow difference ONNX / PyTorch of an export that is accepted
ONNX_TOLERANCE = 1e-4


def onnx_agrees(max_rel_diff_ff) -> bool:
    """True if the largest relative fuel-flow difference between the ONNX file and
    PyTorch (export_onnx, 'max_rel_diff_ff') is finite and below ONNX_TOLERANCE."""
    return math.isfinite(max_rel_diff_ff) and max_rel_diff_ff < ONNX_TOLERANCE


def predict(model: FuelFlowNet, x, outside=False, offsets=None):
    """Fuel flow (kg/h) from model inputs: an (n, 6) array in the order of FEATURES,
    or a DataFrame with these columns.

    outside=True returns (fuel flow, flag) with the flag True for the states outside
    the valid envelope of the model (neuralfuel.support.in_envelope); a model without
    an envelope raises ValueError.
    offsets: the private local offsets of one aircraft (neuralfuel.local.offsets_for),
    added to ln fuel flow by phase. They make a step at the phase borders: use them for
    fuel estimation, not inside an optimizer. A warning is given when they were fitted
    against another model (model_weights_sha256).
    The model stays on its device.
    """
    if isinstance(x, pd.DataFrame):
        x = x[FEATURES].to_numpy(dtype=float)
    x = np.asarray(x, dtype=float)
    dev = next(model.parameters()).device
    with torch.no_grad():
        y = model(torch.as_tensor(x.astype(np.float32), device=dev))
        ff = np.exp(y.cpu().numpy().astype(float))
    if offsets is not None:
        from .local import offset_values

        h = offsets.get("model_weights_sha256")
        if h is not None and h != state_hash(model):
            warnings.warn(
                "the local offsets were fitted against another model "
                "(model_weights_sha256 differs)",
                stacklevel=2,
            )

        ff = ff * np.exp(
            offset_values(x.reshape(-1, len(FEATURES)), offsets).reshape(ff.shape)
        )
    if not outside:
        return ff
    return ff, ~in_envelope(model.envelope, x.reshape(-1, len(FEATURES))).reshape(
        ff.shape
    )


def fuel_flow(model, mach, alt_ft, mass_kg, sat_k, sin_gamma=0.0, acc_in=0.0):
    """Fuel flow (kg/h) at flight conditions; arguments broadcast."""
    m, a, w, t, sg, ac = np.broadcast_arrays(
        *[
            np.asarray(v, dtype=float)
            for v in (mach, alt_ft, mass_kg, sat_k, sin_gamma, acc_in)
        ]
    )
    x = np.stack(
        [m, np.log(atmos.isa_delta(a)), np.log(w), sg, ac, np.log(t / T0)], axis=-1
    )
    return predict(model, x.reshape(-1, 6)).reshape(m.shape)


def track_segments(time_s, track_id=None, rtol=1e-3):
    """(segment number, time step) per sample of time series of one or more tracks.

    A segment is a run of samples of one track at a constant time step: a new segment
    starts at a new track and at a step that differs from the step of the current
    segment by more than rtol (a gap, a repeated time or another cadence). The step
    of a segment is its first step; a sample alone in its segment has the step NaN.
    """
    t = np.asarray(time_s, dtype=float)
    tid = (
        np.zeros(len(t), int)
        if track_id is None
        else pd.factorize(np.asarray(track_id))[0]
    )
    seg = np.zeros(len(t), int)
    steps = []  # step of each segment
    cur = None
    for i in range(len(t)):
        d = t[i] - t[i - 1] if i else np.nan
        if (
            i == 0
            or tid[i] != tid[i - 1]
            or (cur is not None and abs(d - cur) > rtol * cur)
        ):
            steps.append(np.nan)
            cur = None
        elif cur is None:
            if d > 0:
                cur = steps[-1] = d
            else:  # a repeated or earlier time
                steps.append(np.nan)
        seg[i] = len(steps) - 1
    return seg, np.asarray(steps, dtype=float)[seg]


def _sg_by_segment(x, seg, step, window_s):
    """Savitzky-Golay first derivative (order 2) in each segment at its own step, with
    a window of about window_s seconds (odd, at least 3 samples); NaN in segments
    shorter than the window."""
    out = np.full(len(x), np.nan)
    if not len(x):
        return out
    start = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    end = np.r_[start[1:], len(x)]
    for a, b in zip(start, end, strict=True):
        dt = step[a]
        if not np.isfinite(dt):
            continue
        n = max(round(window_s / dt), 3)
        n = n if n % 2 else n - 1
        n = max(n, 3)
        if b - a >= n:
            out[a:b] = savgol_filter(x[a:b], n, 2, deriv=1, delta=dt)
    return out


def surveillance_inputs(
    time_s,
    alt_ft,
    gs_kt,
    mass_kg,
    sat_k,
    cas_kt=None,
    tas_kt=None,
    mach=None,
    track_id=None,
    window_s=21.0,
) -> pd.DataFrame:
    """Model inputs from ADS-B / Mode S style time series of one or more tracks.

    time_s: sample time (s), sorted within each track (for example at 1 s or 4 s
    steps). alt_ft: barometric (pressure) altitude. gs_kt: groundspeed. sat_k: static
    air temperature, for example from a weather model. One airspeed is needed: cas_kt,
    tas_kt or mach. track_id: optional track label.

    The time steps come from the differences inside each track (track_segments): a
    new segment starts at a new track, at a gap and at a change of the step, so tracks
    with different steps are each used at their own step. The derivatives of altitude
    and groundspeed use a Savitzky-Golay filter (order 2) of about window_s seconds
    inside each segment, at the step of the segment; samples in segments shorter than
    the filter give NaN. Returns one row per sample with the columns of FEATURES.
    """
    t = np.asarray(time_s, dtype=float)
    alt = np.asarray(alt_ft, dtype=float) * np.ones_like(t)
    gs = np.asarray(gs_kt, dtype=float) * np.ones_like(t)
    sat = np.asarray(sat_k, dtype=float) * np.ones_like(t)
    if mach is not None:
        m = np.asarray(mach, dtype=float) * np.ones_like(t)
    elif tas_kt is not None:
        m = atmos.mach_from_tas(tas_kt, sat)
    elif cas_kt is not None:
        m = atmos.mach_from_cas(cas_kt, alt)
    else:
        raise ValueError("one of cas_kt, tas_kt or mach is needed")
    seg, step = track_segments(t, track_id)
    tas_ms = m * atmos.sound_speed(sat)
    dhp = _sg_by_segment(alt * FT, seg, step, window_s)
    dgs = _sg_by_segment(gs * KT, seg, step, window_s)
    _, sin_gamma, acc_in = energy_inputs(
        dhp, gs * KT, dgs, tas_ms, sat, atmos.isa_temperature(alt)
    )
    return pd.DataFrame(
        {
            "mach": m,
            "ln_delta": np.log(atmos.isa_delta(alt)),
            "ln_m": np.log(np.asarray(mass_kg, dtype=float) * np.ones_like(t)),
            "sin_gamma": sin_gamma,
            "acc_in": acc_in,
            "ln_theta": np.log(sat / T0),
        }
    )


# ---------------------------------------------------------------- ONNX
class _ExportNet(torch.nn.Module):
    """Same function as FuelFlowNet, output shape (n, 1) without a Squeeze node.

    For the 'fixed' and 'learn' settings the graph takes the same six inputs and
    applies the temperature rule inside: constant selection matrices (MatMul) split x
    into the five network inputs and ln theta, so the graph needs no Slice or Gather
    node.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        n = len(FEATURES)
        eye = torch.eye(n)
        self.register_buffer("sel_net", eye[:, : n - 1].clone())
        self.register_buffer("sel_theta", eye[:, n - 1 :].clone())

    def forward(self, x):
        m = self.m
        if m.temperature == "free":
            return m.y_mean + m.y_std * m.net((x - m.x_mean) / m.x_std)
        z = m.net((x @ self.sel_net - m.x_mean) / m.x_std)
        return m.theta_exp * (x @ self.sel_theta) + (m.y_mean + m.y_std * z)


def _softplus_to_log1pexp(onnx_model):
    """Rewrite Softplus as Log(1 + Exp(x)) (for importers without Softplus)."""
    from onnx import helper, numpy_helper

    g = onnx_model.graph
    names = {v.name for v in g.initializer}
    names |= {s for node in g.node for s in (*node.input, *node.output)}

    def fresh(label):
        while label in names:
            label += "_"
        names.add(label)
        return label

    nodes = []
    for i, node in enumerate(g.node):
        if node.op_type != "Softplus" or node.domain:
            nodes.append(node)
            continue
        one = fresh(f"sp_one_{i}")
        g.initializer.append(
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), one)
        )
        e, s = fresh(f"sp_exp_{i}"), fresh(f"sp_sum_{i}")
        nodes += [
            helper.make_node("Exp", [node.input[0]], [e]),
            helper.make_node("Add", [one, e], [s]),
            helper.make_node("Log", [s], node.output),
        ]
    del g.node[:]
    g.node.extend(nodes)
    return onnx_model


def max_preactivation(model: FuelFlowNet, x) -> float:
    """Largest input of a Softplus unit of the network at the model inputs x (n, 6)."""
    m = model.cpu().float().eval()
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    z = (x if m.temperature == "free" else x[:, :-1]) - m.x_mean
    z = z / m.x_std
    top = -np.inf
    with torch.no_grad():
        for mod in m.net:
            if isinstance(mod, torch.nn.Softplus):
                top = max(top, float(z.max()))
            z = mod(z)
    return top


def export_onnx(model: FuelFlowNet, path, casadi=False, opset=17, n_check=1000, seed=0):
    """Write the network as ONNX and compare it with PyTorch.

    ONNX input 'x' (n, 6) float32 in the order of FEATURES, ln theta included, for
    every temperature setting; output 'ln_fuel_flow' (n, 1), ln of fuel flow in kg/h.
    For 'fixed' and 'learn' the graph computes c ln theta inside (c is a constant of
    the graph). The graph has no Squeeze node. The aircraft settings of the model are
    stored as the metadata entry 'neuralfuel_aircraft' (JSON), the valid envelope
    (neuralfuel.support) as 'neuralfuel_envelope' (JSON) if the model has one.
    casadi=True also writes Softplus as Log(1 + Exp(x)), which the CasADi ONNX importer
    can read. This form is smooth for an optimizer, but exp overflows in float32 for a
    large pre-activation: casadi=True raises ValueError (before it writes a file) when
    the largest hidden pre-activation at 4,096 states inside the envelope (and at the
    check inputs) exceeds MAX_PREACTIVATION.
    Returns a dict with the operators and the largest differences on n_check random
    inputs around the standardization means.
    """
    import onnx
    import onnxruntime as ort

    model = model.cpu().float().eval()
    path = Path(path)
    gen = torch.Generator().manual_seed(seed)
    mean, sd = model.x_mean, model.x_std
    if model.temperature != "free":  # ln theta is not standardized; about 230 K
        mean = torch.cat([mean, torch.tensor([-0.22])])
        sd = torch.cat([sd, torch.tensor([0.05])])
    x = mean + sd * torch.randn(n_check, len(FEATURES), generator=gen)
    info = {}
    if casadi:
        states = [x.numpy()]
        if model.envelope is not None:
            states.append(random_inputs(model.envelope, 4096, seed=seed))
        top = max_preactivation(model, np.concatenate(states))
        info["max_preactivation"] = top
        if not top <= MAX_PREACTIVATION:
            raise ValueError(
                f"the largest hidden pre-activation inside the envelope is {top:.1f}: "
                "Softplus as log(1 + exp(x)) can overflow in float32 above "
                f"{MAX_PREACTIVATION:g}; export without --casadi (casadi_function "
                "rewrites Softplus in float64 itself)"
            )
    torch.onnx.export(
        _ExportNet(model),
        (x[:2],),
        str(path),
        input_names=["x"],
        output_names=["ln_fuel_flow"],
        dynamic_axes={"x": {0: "n"}, "ln_fuel_flow": {0: "n"}},
        opset_version=opset,
        dynamo=False,
    )
    m = onnx.load(str(path))
    if casadi:
        m = _softplus_to_log1pexp(m)
    props = {}
    if model.aircraft is not None:
        props["neuralfuel_aircraft"] = dumps(model.aircraft.as_dict(), indent=None)
    if model.envelope is not None:
        props["neuralfuel_envelope"] = dumps(model.envelope, indent=None)
    if props:
        onnx.helper.set_model_props(m, props)
    onnx.save(m, str(path))
    onnx.checker.check_model(m)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"x": x.numpy()})[0].ravel().astype(float)
    with torch.no_grad():
        y_torch = model(x).numpy().astype(float)
    return {
        "path": str(path),
        "temperature": model.temperature,
        "theta_exp": model.exponent(),
        "opset": m.opset_import[0].version,
        "ops": sorted({n.op_type for n in m.graph.node}),
        "n_check": n_check,
        "max_abs_diff_ln_ff": float(np.abs(y_onnx - y_torch).max()),
        "max_rel_diff_ff": float(np.abs(np.exp(y_onnx - y_torch) - 1).max()),
        **info,
    }


def onnx_metadata(path) -> dict:
    """The aircraft settings and the valid envelope stored in an exported ONNX file
    (None for an entry that the file does not have)."""
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    props = {p.key: p.value for p in m.metadata_props}
    return {
        k: json.loads(props[f"neuralfuel_{k}"]) if f"neuralfuel_{k}" in props else None
        for k in ("aircraft", "envelope")
    }


def external_tensors(graph) -> list:
    """Names of the tensors of an ONNX graph (initializers and attributes, subgraphs
    included) whose values are stored in an external file."""
    from onnx import AttributeProto, TensorProto

    def ext(t):
        return t.data_location == TensorProto.EXTERNAL or len(t.external_data) > 0

    out = [t.name for t in graph.initializer if ext(t)]
    out += [t.values.name for t in graph.sparse_initializer if ext(t.values)]
    for n in graph.node:
        for a in n.attribute:
            if a.type == AttributeProto.TENSOR and ext(a.t):
                out.append(a.t.name or n.name)
            out += [t.name or n.name for t in a.tensors if ext(t)]
            if a.type == AttributeProto.GRAPH:
                out += external_tensors(a.g)
            for g in a.graphs:
                out += external_tensors(g)
    return out


def casadi_function(path, expand=False):
    """Import an exported ONNX file into CasADi symbolically (needs the optional
    dependency casadi, dependency group 'casadi').

    Returns a casadi.Function 'ln_fuel_flow' of x (1, 6), the model inputs in the
    order of FEATURES, to ln of fuel flow in kg/h (1, 1). The file is read without
    external data: a graph with tensors in an external file raises ValueError before
    CasADi is called. The CasADi importer has no
    Softplus operator, so Softplus is rewritten as Log(1 + Exp(x)) first; the batch
    dimension is fixed to 1 (use Function.map for many states). expand=True converts
    the imported graph to scalar expressions (SX); for these dense layers the matrix
    graph (MX, the default) evaluated faster in the tests of the package.
    """
    import tempfile

    import casadi as ca
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    ext = external_tensors(m.graph)
    if ext:  # CasADi would read the files that these tensors name
        raise ValueError(
            f"{Path(path).name}: the ONNX graph has {len(ext)} tensors with external "
            "data; casadi_function reads only a self-contained file (export_onnx)"
        )
    m = _softplus_to_log1pexp(m)
    for v in (m.graph.input[0], m.graph.output[0]):
        d = v.type.tensor_type.shape.dim[0]
        d.ClearField("dim_param")
        d.dim_value = 1
    onnx.checker.check_model(m)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "model.onnx"
        onnx.save(m, str(p))
        net = ca.GraphBuilder(str(p)).create("net", {"symbolic": True})
    x = ca.MX.sym("x", 1, len(FEATURES))
    f = ca.Function("ln_fuel_flow", [x], [net(x)], ["x"], ["ln_fuel_flow"])
    return f.expand() if expand else f


def casadi_check(model: FuelFlowNet, onnx_path, x, expand=False, repeat=5):
    """Compare the CasADi import of onnx_path (casadi_function) with the PyTorch
    model at the model inputs x (n, 6).

    Returns the largest absolute differences of ln F, of its gradient and of its
    Hessian with respect to x (PyTorch in float64 as the reference), and the CasADi
    evaluation time per state (s) of the function and of the gradient with the
    Hessian, from a map over the n states (best of `repeat` runs).
    """
    import copy
    import time

    import casadi as ca
    from torch.func import grad, hessian, vmap

    x = np.asarray(x, dtype=float)
    n = len(x)
    f = casadi_function(onnx_path, expand)
    xs = (ca.SX if expand else ca.MX).sym("x", 1, len(FEATURES))
    y = f(xs)
    h, g = ca.hessian(y, xs)
    fun = ca.Function("f", [xs], [y]).map(n)
    der = ca.Function("d", [xs], [ca.vec(g), ca.vec(h)]).map(n)
    xa = ca.DM(x.reshape(1, -1))

    def best(fn):
        t = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            out = fn(xa)
            t.append(time.perf_counter() - t0)
        return out, min(t) / n

    y_ca, t_f = best(fun)
    (g_ca, h_ca), t_h = best(der)
    y_ca = np.asarray(y_ca).ravel()
    g_ca = np.asarray(g_ca).T
    h_ca = np.asarray(h_ca).T.reshape(n, len(FEATURES), len(FEATURES))
    m = copy.deepcopy(model).cpu().double().eval()
    xt = torch.tensor(x, dtype=torch.float64)

    def fx(v):
        return m(v)

    with torch.no_grad():
        y_t = m(xt).numpy()
    g_t = vmap(grad(fx))(xt).detach().numpy()
    h_t = vmap(hessian(fx))(xt).detach().numpy()
    return {
        "states": n,
        "expand_to_sx": expand,
        "max_abs_diff_ln_ff": float(np.abs(y_ca - y_t).max()),
        "max_abs_diff_gradient": float(np.abs(g_ca - g_t).max()),
        "max_abs_diff_hessian": float(np.abs(h_ca - h_t).max()),
        "max_abs_gradient": float(np.abs(g_t).max()),
        "max_abs_hessian": float(np.abs(h_t).max()),
        "time_per_function_eval_s": t_f,
        "time_per_hessian_eval_s": t_h,
    }
