"""Aggregation of contributions into a new shared model.

A contribution is a model that continued the training of the same shared model (the
base) on local data, normally with the protection against forgetting of
neuralfuel.train. Its declared region is the envelope of its local training data
(the 'local_envelope' of its run record).

Two methods, both for one round (repeat them for more rounds):

  average   weight averaging: the trainable parameters of the contributions are
            averaged, with equal weights by default, or with given weights (for
            example the number of training blocks of each contribution, as in the
            stage-2 experiments). The buffers (standardization constants, a fixed
            temperature exponent) are copied from the base. All contributions must
            have the architecture, temperature rule, aircraft settings and
            standardization constants of the base.
  combine   output combination: a student model, initialized with the base, is
            trained on points with target values and no flight data. The target at a
            point is the mean ln fuel flow of the contributions whose declared region
            holds the point; at a point outside every declared region it is the base.
            Half of the points are collocation states over the aircraft envelope
            (with acc_in uniform in +-0.03), half are drawn inside the declared regions
            in equal parts. The student has the full step budget without early
            stopping; the state with the lowest loss on a separate point set is kept.
            The phase rule and the constraint settings come from the base; the
            envelopes are checked for compatibility before the training.

Both methods need the declared region of every contribution. The result gets the
union of the envelope of the base and of the declared regions
(neuralfuel.support.union_envelope): a box per phase, which can span gaps between the
regions. The block and window counts of the union are sums: over many rounds they count
the training exposures (a region used in two rounds counts twice), not unique
observations. A contribution must differ from the base and from the other
contributions (weights hash). `record_problems` (and `run_record_problems`, the same as
one list) is the one check of a run record, used by `contribute`, `check-model` and
`merge`: it tells whether a run record belongs to a contribution and to the base. The
record of an aggregation holds the weights hashes of the base and of every
contribution; `save_merged` writes it into the meta entry of the weights file and, with
the union envelope and the quality check of the merged model, into the merge record
<name>.merge.json.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .losses import envelope, model_loss_config, sample_collocation
from .model import FuelFlowNet, save, state_hash
from .prep import FEATURES
from .privacy import dumps
from .support import in_envelope, input_states, phase_of, random_inputs, union_envelope

BUFFERS = ("x_mean", "x_std", "y_mean", "y_std")


def differences(base: FuelFlowNet, model: FuelFlowNet) -> list:
    """Reasons why model cannot be aggregated with base (empty if compatible)."""
    out = []
    if (model.hidden, model.layers) != (base.hidden, base.layers):
        out.append(
            f"architecture {model.layers}x{model.hidden} differs from "
            f"{base.layers}x{base.hidden}"
        )
    if list(model.features) != list(base.features):
        out.append("inputs differ")
    if model.temperature != base.temperature:
        out.append(
            f"temperature rule '{model.temperature}' differs from '{base.temperature}'"
        )
    elif model.temperature == "fixed" and model.exponent() != base.exponent():
        out.append("temperature exponent differs")
    a = model.aircraft.as_dict() if model.aircraft else None
    b = base.aircraft.as_dict() if base.aircraft else None
    if a != b:
        out.append("aircraft settings differ")
    for k in BUFFERS:
        if not torch.equal(getattr(model, k).cpu(), getattr(base, k).cpu()):
            out.append(f"standardization constants differ ({k})")
    if not out and set(model.state_dict()) != set(base.state_dict()):
        out.append("parameter names differ")
    return out


def check_compatible(base, models):
    """Raise ValueError if a model differs from the base (see `differences`), is
    the base itself or is given twice."""
    for i, m in enumerate(models):
        d = differences(base, m)
        if d:
            raise ValueError(f"contribution {i + 1}: " + "; ".join(d))
    d = duplicates(base, models)
    if d:
        raise ValueError("; ".join(d))


def duplicates(base, models) -> list:
    """Reasons why the contributions are not distinct: a contribution equal to the
    base, or the same weights given twice (empty if distinct)."""
    out, seen = [], {state_hash(base): "the base"}
    for i, m in enumerate(models):
        h = state_hash(m)
        if h in seen:
            out.append(f"contribution {i + 1} has the same weights as {seen[h]}")
        else:
            seen[h] = f"contribution {i + 1}"
    return out


def record_problems(model, rec, base=None, meta=None, need_base=True) -> dict:
    """Reasons why the run record `rec` does not belong to the model `model`, by kind
    (empty lists if it belongs):

      weights  the weights hash of rec, and of the meta entry of the model file (meta,
               neuralfuel.model.read_meta) where it has one, differ from the model
      base     rec names no base model (only with need_base), or names another base
               than `base` (with base), or the meta entry names another base than rec
      region   a continued model without the declared region (local_envelope); block
               and window counts of rec that differ from the declared region; an
               envelope of the model that is not the union of the envelope of the base
               and the declared region (with base), or not the envelope of rec
    """
    out = {"weights": [], "base": [], "region": []}
    if rec is None:
        out["weights"].append("no run record")
        return out
    meta = meta or {}
    h = state_hash(model)
    if rec.get("weights_sha256") != h:
        out["weights"].append(
            "the run record belongs to other weights (weights_sha256)"
        )
    if meta.get("weights_sha256", h) != h:
        out["weights"].append("the meta entry of the model file names other weights")
    init = rec.get("init_weights_sha256")
    if init is None and need_base:
        out["base"].append("the run record names no base model (trained from scratch)")
    elif init is not None and base is not None and init != state_hash(base):
        out["base"].append(
            "the run record names another base model (init_weights_sha256)"
        )
    if "init_weights_sha256" in meta and meta["init_weights_sha256"] != init:
        out["base"].append(
            "the model file and the run record name different base models "
            "(init_weights_sha256)"
        )
    local = rec.get("local_envelope")
    if local is None:
        if init is not None:
            out["region"].append(
                "the run record of a continued model has no declared region "
                "(local_envelope)"
            )
        return out
    counts = (
        ("n_train_blocks", local.get("phases", {}).get("all", {}).get("blocks")),
        ("n_train_windows", local.get("cruise", {}).get("windows")),
    )
    for k, v in counts:
        if k in rec and rec[k] != v:
            out["region"].append(f"{k} of the run record differs from local_envelope")
    try:
        if base is not None and init is not None:
            expected = union_envelope(base.envelope, local)
            what = "the union of the base envelope and the declared region"
        elif init is None:
            expected, what = local, "the declared region (a model from scratch)"
        else:
            expected, what = rec.get("envelope", model.envelope), "envelope of rec"
    except ValueError as e:
        out["region"].append(f"the declared region does not fit the base: {e}")
        return out
    if model.envelope != expected:
        out["region"].append(f"the envelope of the model is not {what}")
    return out


def run_record_problems(model, rec, base=None, meta=None, need_base=True) -> list:
    """The reasons of `record_problems` as one list (empty if the record belongs)."""
    p = record_problems(model, rec, base, meta, need_base)
    return p["weights"] + p["base"] + p["region"]


def finite_state(model) -> bool:
    return all(torch.isfinite(v).all() for v in model.state_dict().values())


def _record(method, base, models, weights):
    return {
        "method": method,
        "base_weights_sha256": state_hash(base),
        "contributions": [
            {"weights_sha256": state_hash(m), "weight": float(w)}
            for m, w in zip(models, weights, strict=False)
        ],
    }


def average(base: FuelFlowNet, models, regions, weights=None):
    """Weight average of the contributions `models` (all continued from `base`).

    regions: the declared region of every contribution (the 'local_envelope' of its
    run record). weights: one finite positive number per contribution (normalized
    here); default equal. Returns (model, record).
    """
    if not models:
        raise ValueError("no contributions")
    if len(regions) != len(models) or any(r is None for r in regions):
        raise ValueError("average needs the declared region of every contribution")
    check_compatible(base, models)
    w = np.ones(len(models)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(models) or not (np.all(np.isfinite(w)) and np.all(w > 0)):
        raise ValueError("weights: one finite positive number per contribution")
    w = w / w.max()  # cannot overflow; then the sum is between 1 and len(w)
    w = w / w.sum()
    if not np.all(w > 0):
        raise ValueError("weights: the ratio of the largest to the smallest is too big")
    # trainable parameters only; the buffers (standardization, a fixed temperature
    # exponent) stay those of the base
    names = [k for k, _ in base.named_parameters()]
    states = [m.state_dict() for m in models]
    out = copy.deepcopy(base).cpu()
    avg = {k: v.detach().cpu().clone() for k, v in out.state_dict().items()}
    for k in names:
        avg[k] = sum(
            float(wi) * s[k].detach().cpu() for wi, s in zip(w, states, strict=True)
        )
    out.load_state_dict(avg)
    out.eval()
    if not finite_state(out):
        raise ValueError("the merged parameters are not finite")
    out.envelope = union_envelope(base.envelope, *regions)
    rec = _record("average", base, models, w)
    rec["weights_sha256"] = state_hash(out)
    return out, rec


# ---------------------------------------------------------------- output combination
def _predict_ln(model, x):
    m = copy.deepcopy(model).cpu().double().eval()
    with torch.no_grad():
        return m(torch.as_tensor(np.asarray(x, dtype=float))).numpy()


def combine_points(ac, regions, n, seed, loss_cfg=None):
    """n points (n, 6): half collocation states over the aircraft envelope (of the
    constraint settings loss_cfg) with acc_in uniform in [-0.03, 0.03], half inside
    the declared regions in equal parts."""
    from .losses import LossConfig

    gen = torch.Generator()
    gen.manual_seed(seed)
    p = sample_collocation(n // 2, gen, "cpu", envelope(ac, loss_cfg or LossConfig()))
    p = p.numpy().astype(float)
    acc = np.random.default_rng(seed).uniform(-0.03, 0.03, len(p))
    x = [np.column_stack([p[:, 0], p[:, 1], p[:, 2], p[:, 3], acc, p[:, 4]])]
    per = (n - len(p)) // len(regions)
    for k, reg in enumerate(regions):
        x.append(random_inputs(reg, per, seed=seed + k, phase="all"))
    return np.concatenate(x)


def combine_targets(x, base, models, regions):
    """ln F targets: mean of the contributions over the declared regions that hold x,
    the base where no region holds x."""
    a = np.stack([in_envelope(r, x).astype(float) for r in regions])
    f = np.stack([_predict_ln(m, x) for m in models])
    a0 = np.clip(1 - a.sum(0), 0, None)
    f0 = _predict_ln(base, x)
    return (np.sum(a * f, 0) + a0 * f0) / (a.sum(0) + a0)


def _pseudo_blocks(x, y, split, level_fpm):
    """Table with the columns that neuralfuel.train needs, for points x and targets
    y."""
    s = input_states(x)
    d = pd.DataFrame(x, columns=FEATURES)
    d["ln_ff"] = y
    d["mach"], d["alt_ft"], d["mass_kg"] = s["mach"], s["alt_ft"], s["mass_kg"]
    d["delta"], d["dT_isa"] = s["delta"], s["dT_isa_k"]
    d["phase"] = phase_of(s["vs_fpm"], level_fpm)
    d["split"] = split
    return d


def combine(
    base: FuelFlowNet,
    models,
    regions,
    cfg=None,
    n_points=1 << 20,
    n_val=1 << 16,
    log=print,
):
    """Output combination of the contributions `models` with their declared regions
    (envelopes, one per model). cfg: TrainConfig of the student (seed, steps,
    device); the protection term and early stopping are off. Returns (model, record).
    """
    from .train import TrainConfig, train

    if not models:
        raise ValueError("no contributions")
    if len(regions) != len(models) or any(r is None for r in regions):
        raise ValueError("combine needs the declared region of every contribution")
    check_compatible(base, models)
    if base.envelope is None:
        raise ValueError("combine needs a base with a valid envelope")
    union = union_envelope(base.envelope, *regions)  # ValueError if incompatible
    rule = base.envelope["phase_rule"]
    cfg = copy.deepcopy(cfg) if cfg is not None else TrainConfig()
    cfg.loss = model_loss_config(base)
    cfg.loss.teacher = 0.0
    cfg.patience_steps = 10**9
    ac = base.aircraft
    xtr = combine_points(ac, regions, n_points, 11 + cfg.seed, cfg.loss)
    xva = combine_points(ac, regions, n_val, 99 + cfg.seed, cfg.loss)
    ytr = combine_targets(xtr, base, models, regions)
    yva = combine_targets(xva, base, models, regions)
    lv = rule["level_fpm"]
    blocks = pd.concat(
        [_pseudo_blocks(xtr, ytr, "train", lv), _pseudo_blocks(xva, yva, "val", lv)],
        ignore_index=True,
    )
    windows = blocks.iloc[:0]
    student = copy.deepcopy(base)
    log(
        f"student starts from the base; {len(xtr)} training points, {len(xva)} "
        "validation points, no flight data"
    )
    model, trec = train(
        blocks,
        windows,
        cfg,
        init=student,
        log=log,
        log_style="points",
        phase_rule=rule,
    )
    model.envelope = union
    rec = _record("combine", base, models, np.full(len(models), 1.0 / len(models)))
    rec.update(
        points=len(xtr),
        validation_points=len(xva),
        val_loss=trec["val_loss"],
        best_step=trec["best_step"],
        steps_run=trec["steps_run"],
        weights_sha256=state_hash(model),
    )
    return model, rec


def merge_record_path(path) -> Path:
    """<name>.merge.json next to the weights file <name>.pt."""
    p = Path(path)
    return p.with_name(p.stem + ".merge.json")


def save_merged(model, rec, path, quality=None):
    """Write the merged model with the aggregation record in its meta entry, and the
    merge record <name>.merge.json next to it: the aggregation record, the union
    envelope and `quality` (the result of neuralfuel.quality.check_model of the
    merged model against the base). Returns (weights file, merge record file)."""
    meta = {"weights_sha256": rec["weights_sha256"], "aggregation": rec}
    save(model, path, meta=meta)
    mr = merge_record_path(path)
    full = {**rec, "envelope": model.envelope, "quality_check": quality}
    mr.write_text(dumps(full))
    return Path(path), mr
