"""Training loop: AdamW, cosine learning rate, early stopping on the validation blocks.

The training budget is a number of update steps (default 19,500, the number that the
full A320 dataset gets in about 150 passes at batch 4096), so a small dataset gets as
many updates as a large one. The cosine schedule runs over the steps, the validation
loss is computed every `eval_every` steps and training stops early after
`patience_steps` steps without a better validation loss. The older budget in passes
over the data (epochs) is kept for the run records written with it: `max_epochs`
replaces the step settings by max_epochs passes, with the schedule stepped once per
pass, validation after each pass and `patience` in passes.

Protection against forgetting: when training continues from given weights (`init`),
a frozen copy of these weights is the teacher of the loss term `teacher`
(neuralfuel.losses): outside the envelope of the local training data the new model
must stay close to the given model. The term uses a fixed set of points (half over
the aircraft envelope, half inside the valid envelope of the given model) and a batch
of n_colloc of them per step, drawn with a separate random generator, so the other
random numbers are those of a run without protection. Weight 0 (`loss.teacher=0`)
switches the protection off.

Training stops with ValueError at the first non-finite training or validation loss.

RUN_RECORD is the format of the run record (the keys that `train` writes, for
neuralfuel.privacy.allowed). `check_config` tells whether the settings of a
TrainConfig are in range.
"""

import copy
import dataclasses
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import __version__
from .aircraft import SIZES, Aircraft, aircraft, size_of
from .drag import Unavailable
from .losses import (
    DRAG_RISE,
    SHAPE_TERMS,
    LossConfig,
    constraint_settings,
    drag_rise_penalty,
    drag_rise_setup,
    drag_rise_states,
    envelope,
    range_boundary_penalty,
    sample_collocation,
    set_constraints,
    shape_penalties,
    teacher_penalty,
    teacher_points,
    teacher_weights,
)
from .model import (
    MAX_HIDDEN,
    MAX_LAYERS,
    TEMPERATURE_RULES,
    FuelFlowNet,
    save,
    state_hash,
)
from .prep import FEATURES, PrepConfig
from .privacy import (
    CODE,
    CODE_OR_EMPTY,
    NUM,
    NUM_KEY,
    SHA256,
    VERSION,
    Text,
    dumps,
    enum,
    number_text,
    settings_spec,
)
from .support import ENVELOPE, data_envelope, union_envelope


@dataclass
class TrainConfig:
    seed: int = 0
    max_steps: int = 19500  # budget in update steps
    patience_steps: int = 3900  # early stopping: steps without a better val loss
    eval_every: int = 130  # steps between validation losses
    max_epochs: int | None = None  # budget in passes instead of steps (older runs)
    patience: int | None = None  # with max_epochs: patience in passes (default 30)
    lr: float = 2e-3
    batch: int = 4096
    weight_decay: float = 1e-6
    hidden: int = 64  # 64: size 'standard', 32: size 'xs' (neuralfuel.aircraft)
    layers: int = 3
    temperature: str = "fixed"  # fixed, learn or free (see neuralfuel.model)
    theta_exp: float = 0.5  # exponent c of theta (fixed, or start value of learn)
    aircraft: Aircraft = field(default_factory=lambda: aircraft("a320"))
    device: str = "auto"
    threads: int = 4
    loss: LossConfig = field(default_factory=LossConfig)


# the weights of the loss terms (LossConfig): finite and 0 or more
LOSS_WEIGHTS = (
    "points",
    "window",
    *SHAPE_TERMS,
    "range_boundary",
    "drag_rise",
    "teacher",
)


def check_config(cfg: TrainConfig) -> None:
    """Raise ValueError for a setting out of range."""
    for k in ("max_steps", "patience_steps", "eval_every", "batch", "threads"):
        v = getattr(cfg, k)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"{k} must be an integer of 1 or more, got {v!r}")
    for k in ("max_epochs", "patience"):
        v = getattr(cfg, k)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            raise ValueError(f"{k} must be an integer of 1 or more, got {v!r}")
    if not 1 <= cfg.hidden <= MAX_HIDDEN or not 1 <= cfg.layers <= MAX_LAYERS:
        raise ValueError(
            f"hidden must be 1 to {MAX_HIDDEN} and layers 1 to {MAX_LAYERS}, got "
            f"{cfg.hidden} and {cfg.layers}"
        )
    if cfg.temperature not in TEMPERATURE_RULES:
        raise ValueError(f"temperature must be one of {TEMPERATURE_RULES}")
    if not (math.isfinite(cfg.lr) and cfg.lr >= 0 and cfg.weight_decay >= 0):
        raise ValueError("lr and weight_decay must be 0 or more")
    lc = cfg.loss
    for k in ("n_window_batch", "n_colloc", "n_teacher_points"):
        if getattr(lc, k) < 1:
            raise ValueError(f"loss.{k} must be 1 or more")
    for f in dataclasses.fields(lc):
        v = getattr(lc, f.name)
        vals = v if isinstance(v, tuple) else (v,)
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError(f"loss.{f.name} must be finite")
        if f.name in ("eps", "dT_isa") and not (len(v) == 2 and v[0] < v[1]):
            raise ValueError(f"loss.{f.name} must be (low, high)")
    for k in LOSS_WEIGHTS:
        if getattr(lc, k) < 0:  # a negative weight reverses the term
            raise ValueError(f"loss.{k} must be 0 or more, got {getattr(lc, k)!r}")
    if lc.min_cas_kt >= cfg.aircraft.vmo_kt or lc.mach_min >= cfg.aircraft.mmo:
        raise ValueError(
            "loss.min_cas_kt and loss.mach_min must be below aircraft.vmo_kt and "
            "aircraft.mmo"
        )


INIT_TEXT = ("from scratch", "continued from given weights")
NO_ENVELOPE_WARNING = (
    "the given weights have no valid envelope; the protection uses only points over "
    "the aircraft envelope"
)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def budget(cfg: TrainConfig, steps_per_epoch: int):
    """(total steps, validation interval, patience in steps, schedule length,
    steps per schedule unit) of the step or the epoch budget."""
    if cfg.max_epochs is not None:
        pat = 30 if cfg.patience is None else cfg.patience
        spe = steps_per_epoch
        return cfg.max_epochs * spe, spe, pat * spe, cfg.max_epochs, spe
    return cfg.max_steps, cfg.eval_every, cfg.patience_steps, cfg.max_steps, 1


def train(
    blocks,
    windows,
    cfg: TrainConfig | None = None,
    init=None,
    log=print,
    phase_rule=None,
    log_style="data",
    prep_config=None,
):
    """Train on the 'train' rows, stop early on the 'val' rows.

    blocks, windows: tables from neuralfuel.prep.prepare (column 'split').
    init: a FuelFlowNet to continue from. Training works on a copy; init is not
    changed. Its standardization constants, its temperature rule and its aircraft
    settings are kept (cfg.temperature, cfg.theta_exp and cfg.aircraft are set from
    it; a file without aircraft settings gets cfg.aircraft), so several sites that
    continue from the same weights share the same constants. The constraint settings
    of init (FuelFlowNet.constraints) also replace those of cfg.loss, including the
    settings of the drag-rise term P7 (weight, Mach band, CL bound and table grid):
    a continued model has P7 exactly when init has stored P7 settings.
    The table rebuilt from them must have the stored table_sha256; otherwise
    ValueError (another aerox version gives other values).
    P7 (from scratch: cfg.loss.drag_rise > 0, on by default): the table of the aerox
    drag slope is built once (neuralfuel.drag) and the term uses its own random
    generator, so the other random numbers are those of a run without it. The model
    stores the P7 settings in constraints['drag_rise'] and the run record in
    'drag_rise'. When P7 is not available for the type (drag.Unavailable: no aerox
    drag model with wave drag, no wing area, no cruise CL in the envelope), or its
    weight is 0, the training runs without it: the run record then has
    'drag_rise': None and the reason in 'drag_rise_note', and the log says so.
    With init and cfg.loss.teacher > 0, a frozen copy of init is the teacher of the
    protection term (see the module docstring).
    phase_rule: the phase settings {'level_fpm', 'cruise_alt_ft'} of the preparation
    that made the blocks (the 'config' of prep.json). With None, the defaults of
    PrepConfig are used; give the settings when prep ran with other values.
    prep_config: the PrepConfig of the preparation (the 'config' of prep.json); the run
    record keeps it as 'prep', and its phase settings are the default of phase_rule.
    The valid envelope of the returned model (model.envelope, neuralfuel.support)
    describes the 'train' blocks and windows of this run, with these phase settings.
    A continued model gets the union of the envelope of init and that of the local
    data; the run record also holds the local envelope alone ('local_envelope').
    log_style: 'data' prints the epoch (passes over the local data) in the progress
    lines; 'points' (the student of neuralfuel.aggregate.combine) leaves it out.
    Returns (model on the CPU, run record).
    """
    cfg = cfg or TrainConfig()
    if prep_config is not None and phase_rule is None:
        phase_rule = {
            "level_fpm": prep_config.level_fpm,
            "cruise_alt_ft": prep_config.cruise_alt_ft,
        }
    check_config(cfg)
    torch.set_num_threads(cfg.threads)
    device = pick_device(cfg.device)
    lc = cfg.loss
    tr, va = blocks[blocks.split == "train"], blocks[blocks.split == "val"]
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("training needs blocks in the 'train' and 'val' splits")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    teacher = None
    if init is not None and lc.teacher > 0:
        teacher = copy.deepcopy(init).cpu().eval()
    init_hash = state_hash(init) if init is not None else None
    init_envelope = copy.deepcopy(init.envelope) if init is not None else None
    if init is None:
        ac = cfg.aircraft.check()
        model = FuelFlowNet.from_data(
            tr, cfg.hidden, cfg.layers, cfg.temperature, cfg.theta_exp, ac
        )
    else:
        model = copy.deepcopy(init)
        cfg.temperature = model.temperature
        if model.temperature != "free":
            cfg.theta_exp = model.exponent()
        if model.aircraft is None:
            model.aircraft = cfg.aircraft.check()
        cfg.aircraft = model.aircraft
        cfg.hidden, cfg.layers = model.hidden, model.layers
        set_constraints(lc, model.constraints)
    stored_drag = ((init.constraints or {}) if init is not None else {}).get(
        "drag_rise"
    )
    model.constraints = constraint_settings(lc)
    ac = model.aircraft
    env = envelope(ac, lc)
    model = model.to(device)

    def tens(d, cols):
        return torch.tensor(
            d[cols].to_numpy(dtype=float), dtype=torch.float32, device=device
        )

    xtr, ytr = tens(tr, FEATURES), tens(tr, "ln_ff")
    xva, yva = tens(va, FEATURES), tens(va, "ln_ff")
    wtr, wva = windows[windows.split == "train"], windows[windows.split == "val"]
    use_w = lc.window > 0 and len(wtr) > 0
    xw, yw = tens(wtr, FEATURES), tens(wtr, "ln_ff")
    xwv, ywv = tens(wva, FEATURES), tens(wva, "ln_ff")
    use_shape = any(getattr(lc, k) > 0 for k in SHAPE_TERMS)
    pc = PrepConfig()
    rule = {"level_fpm": pc.level_fpm, "cruise_alt_ft": pc.cruise_alt_ft}
    rule.update({k: float(v) for k, v in (phase_rule or {}).items() if k in rule})
    local_env = data_envelope(tr, wtr, ac.wing_area_m2, **rule)
    drag, drag_note = None, None  # P7: (settings, table on the device, generator)
    if stored_drag:
        settings, table = drag_rise_setup(ac, lc, None, stored_drag)
    elif init is not None:
        drag_note = "the given model has no P7 settings"
    elif lc.drag_rise == 0:
        drag_note = "loss.drag_rise is 0"
    else:
        try:
            settings, table = drag_rise_setup(ac, lc, local_env)
        except Unavailable as e:
            drag_note = str(e)
    if drag_note is None:
        model.constraints["drag_rise"] = settings
        gen3 = torch.Generator(device=device)
        gen3.manual_seed(7000 + cfg.seed)
        drag = (settings, torch.tensor(table, dtype=torch.float32, device=device), gen3)
    prot, prot_rec = None, {"teacher_weight": 0.0}
    if teacher is not None:
        prot, prot_rec = protection(teacher, init_envelope, local_env, ac, lc, device)
        gen2 = torch.Generator(device=device)
        gen2.manual_seed(5000 + cfg.seed)
    if init is not None and log_style == "data":
        log(
            f"continued from the model {init_hash[:12]}: protection against "
            f"forgetting {'on' if prot is not None else 'off'} (teacher weight "
            f"{prot_rec['teacher_weight']:g})"
        )
    if drag_note is not None:
        log(f"drag-rise term P7 off: {drag_note}")
    if log_style == "data":
        log(
            "columns: step = update steps, epoch = passes over the local training "
            "data, val = mean squared error of ln fuel flow on the validation blocks, "
            "best = lowest val so far"
        )
    else:
        log(
            "columns: step = update steps, val = mean squared error of ln fuel flow "
            "against the targets on the validation points, best = lowest val so far"
        )
    gen = torch.Generator(device=device)
    gen.manual_seed(1000 + cfg.seed)
    n = len(xtr)
    spe = math.ceil(n / cfg.batch)
    total, every, patience, t_max, unit = budget(cfg, spe)
    lr_min = 0.01

    def lr_factor(k):  # cosine from 1 to lr_min over t_max schedule units
        t = min(k // unit, t_max)
        return lr_min + (1 - lr_min) * 0.5 * (1 + math.cos(math.pi * t / t_max))

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    best, best_state, best_step, hist = float("inf"), None, 0, []
    step, tot, n_tot, done = 0, {}, 0, False
    t0 = time.time()
    while not done:
        model.train()
        perm = torch.randperm(n, device=device, generator=gen)
        for i in range(spe):
            idx = perm[i * cfg.batch : (i + 1) * cfg.batch]
            loss_d = F.mse_loss(model(xtr[idx]), ytr[idx])
            loss = lc.points * loss_d
            terms = {"points": loss_d}
            if use_w:
                j = torch.randint(
                    len(xw), (lc.n_window_batch,), generator=gen, device=device
                )
                terms["window"] = F.mse_loss(model(xw[j]), yw[j])
                loss = loss + lc.window * terms["window"]
            if use_shape:
                p = sample_collocation(lc.n_colloc, gen, device, env)
                for k, v in shape_penalties(model, p, lc).items():
                    terms[k] = v
                    loss = loss + getattr(lc, k) * v
            if lc.range_boundary > 0:
                terms["range_boundary"] = range_boundary_penalty(
                    model, lc.n_colloc, gen, device, lc, ac
                )
                loss = loss + lc.range_boundary * terms["range_boundary"]
            if drag is not None:
                q = drag_rise_states(
                    lc.n_colloc, drag[2], device, ac, lc, drag[0]["cl_max"]
                )
                terms["drag_rise"] = drag_rise_penalty(
                    model, q, drag[1], drag[0], ac.wing_area_m2
                )
                loss = loss + lc.drag_rise * terms["drag_rise"]
            if prot is not None:
                j = torch.randint(
                    len(prot[0]), (lc.n_colloc,), generator=gen2, device=device
                )
                terms["teacher"] = teacher_penalty(
                    model, prot[0][j], prot[1][j], prot[2][j]
                )
                loss = loss + lc.teacher * terms["teacher"]
            if not math.isfinite(loss.item()):
                bad = [k for k, v in terms.items() if not math.isfinite(v.item())]
                raise ValueError(
                    f"non-finite training loss at update step {step + 1} (terms: "
                    f"{', '.join(bad) or 'total'}); check the prepared data for "
                    "non-finite inputs or targets"
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            for k, v in terms.items():
                tot[k] = tot.get(k, 0.0) + v.item()
            n_tot += 1
            if step % every and step < total:
                continue
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(xva), yva).item()
            if not math.isfinite(val):
                raise ValueError(
                    f"non-finite validation loss at update step {step}; check the "
                    "'val' blocks of the prepared data"
                )
            model.train()
            hist.append(
                {
                    "step": step,
                    "epoch": step / spe,
                    "val": val,
                    **{k: v / n_tot for k, v in tot.items()},
                }
            )
            tot, n_tot = {}, 0
            if model.temperature == "learn":
                hist[-1]["theta_exp"] = model.exponent()
            if val < best:
                best, best_step = val, step
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if len(hist) % 10 == 1 or step >= total:
                ep = f" epoch {step / spe:7.1f}" if log_style == "data" else ""
                log(f"step {step:6d}{ep} val {val:.6f} best {best:.6f}")
            if step >= total or step - best_step >= patience:
                done = True
                break
    run_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    model.envelope = (
        local_env if init is None else union_envelope(init_envelope, local_env)
    )
    config = asdict(cfg)
    config["device"] = cfg.device.split(":")[0]  # auto, cpu or cuda, no index
    rec = {
        "neuralfuel_version": __version__,
        "config": config,
        "init": INIT_TEXT[init is not None],
        "init_weights_sha256": init_hash,
        "protection": prot_rec,
        "features": list(FEATURES),
        "temperature": model.temperature,
        "theta_exp": model.exponent(),
        "aircraft": ac.as_dict(),
        "size": size_of(model.hidden, model.layers),
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_train_blocks": len(tr),
        "n_aircraft": int(tr["tail"].nunique()) if "tail" in tr else 1,
        "n_val_blocks": len(va),
        "n_train_windows": len(wtr),
        "n_val_windows": len(wva),
        "budget": "epochs" if cfg.max_epochs is not None else "steps",
        "steps_per_epoch": spe,
        "val_loss": best,
        "best_step": best_step,
        "steps_run": step,
        "best_epoch": best_step / spe,
        "epochs_run": step / spe,
        "run_time_s": run_time,
        "device": device.type,  # the category only: no product name, no index
        "torch": torch.__version__,
        "python": platform.python_version(),
        "envelope": model.envelope,
        "local_envelope": local_env,
        "prep": asdict(prep_config) if prep_config is not None else None,
    }
    rec["drag_rise"] = drag[0] if drag is not None else None
    if drag_note is not None:
        rec["drag_rise_note"] = drag_note
    with torch.no_grad():
        rec["train_loss"] = F.mse_loss(model(xtr), ytr).item()
        if len(xwv):
            rec["val_window_loss"] = F.mse_loss(model(xwv), ywv).item()
    model = model.cpu()
    rec["weights_sha256"] = state_hash(model)
    rec["history"] = hist
    return model, rec


def protection(teacher, shared_env, local_env, ac, lc, device):
    """Points x, weights w and teacher outputs f of the protection term (tensors on
    the device), and its part of the run record."""
    rec = {
        "teacher_weight": lc.teacher,
        "points": lc.n_teacher_points,
        "points_in_shared_envelope": 0,
        "warning": None,
    }
    if shared_env is None:
        rec["warning"] = NO_ENVELOPE_WARNING
    x = teacher_points(ac, lc, shared_env)
    if shared_env is not None:
        rec["points_in_shared_envelope"] = len(x) - len(x) // 2
    w = teacher_weights(x, local_env)
    rec["share_inside_local_envelope"] = float(1.0 - w.mean())
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    teacher = teacher.to(device)
    with torch.no_grad():
        f = torch.cat([teacher(xt[i : i + 65536]) for i in range(0, len(xt), 65536)])
    teacher.cpu()
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    return (xt, wt, f), rec


def write_run(model, rec, out_dir):
    """Write model.pt and run.json into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "weights_sha256": rec["weights_sha256"],
        "init_weights_sha256": rec.get("init_weights_sha256"),
    }
    save(model, out / "model.pt", meta=meta)
    (out / "run.json").write_text(dumps(rec))
    return out / "model.pt", out / "run.json"


# the strings of the settings dataclasses (format of neuralfuel.privacy.allowed)
SETTING_TEXTS = {
    "temperature": enum(*TEMPERATURE_RULES),
    "device": enum("auto", "cpu", "cuda"),
    "type": CODE,
    "engine": CODE_OR_EMPTY,
}
AIRCRAFT = settings_spec(Aircraft, SETTING_TEXTS)
# the notes of a training without P7 (train, losses.drag_rise_setup,
# drag.aerox_drag_model); a note of another form (for example with the text of an
# aerox error) is not shared
_TYPE = r"'[a-z0-9_]{1,16}'"
DRAG_RISE_NOTE = Text(
    "|".join(
        (
            enum(
                "the given model has no P7 settings",
                "loss.drag_rise is 0",
                "the aircraft settings have no wing area",
                "the envelope has no largest lift coefficient of the cruise windows "
                "(no steady cruise windows)",
            ).pattern,
            f"aerox has no type {_TYPE}",
            f"the aerox drag model of {_TYPE} has no wave drag",
            f"aerox has no wing area for {_TYPE}",
            number_text(
                "the Mach band is empty: loss.drag_rise_mach_min ({}) is not below "
                "aircraft.mmo ({})"
            ).pattern,
        )
    )
)
HISTORY_TERMS = (
    "points",
    "window",
    *SHAPE_TERMS,
    "range_boundary",
    "drag_rise",
    "teacher",
)
# the keys of the run record (format of neuralfuel.privacy.allowed); 'seed_val_losses'
# is added by 'release'
RUN_RECORD = {
    "neuralfuel_version": VERSION,
    "config": settings_spec(TrainConfig, SETTING_TEXTS),
    "init": enum(*INIT_TEXT),
    "init_weights_sha256": SHA256,
    "protection": {
        "teacher_weight": NUM,
        "points": NUM,
        "points_in_shared_envelope": NUM,
        "warning": enum(NO_ENVELOPE_WARNING),
        "share_inside_local_envelope": NUM,
    },
    "features": [enum(*FEATURES)],
    "temperature": enum(*TEMPERATURE_RULES),
    "theta_exp": NUM,
    "aircraft": AIRCRAFT,
    "size": enum(*SIZES, "custom"),
    "n_params": NUM,
    "n_train_blocks": NUM,
    "n_aircraft": NUM,
    "n_val_blocks": NUM,
    "n_train_windows": NUM,
    "n_val_windows": NUM,
    "budget": enum("steps", "epochs"),
    "steps_per_epoch": NUM,
    "val_loss": NUM,
    "best_step": NUM,
    "steps_run": NUM,
    "best_epoch": NUM,
    "epochs_run": NUM,
    "run_time_s": NUM,
    "device": enum("cpu", "cuda"),
    "torch": VERSION,
    "python": VERSION,
    "envelope": ENVELOPE,
    "local_envelope": ENVELOPE,
    "prep": settings_spec(PrepConfig),
    "train_loss": NUM,
    "val_window_loss": NUM,
    "weights_sha256": SHA256,
    "drag_rise": DRAG_RISE,
    "drag_rise_note": DRAG_RISE_NOTE,
    "history": [
        {"step": NUM, "epoch": NUM, "val": NUM, "theta_exp": NUM}
        | dict.fromkeys(HISTORY_TERMS, NUM)
    ],
    "seed_val_losses": {NUM_KEY: NUM},
}
