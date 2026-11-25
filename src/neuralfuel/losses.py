"""Loss terms. Every term has a weight in LossConfig; weight 0 switches it off.

points            MSE of ln fuel flow on the 10 s blocks
window            MSE of ln fuel flow on steady 300 s windows (random batch per step)
mach_convexity    P1: fuel flow convex in Mach, f_MM + f_M^2 >= 0 (f = ln F), only at
                  powered states (eps >= eps_powered)
mass_monotone     P2: fuel flow increases with mass, f_ln m >= 0, only at powered
                  states (eps >= eps_powered): below a limit descent angle the
                  required thrust T = D + W eps decreases with mass until idle
energy_monotone   P3: fuel flow increases with energy rate, f_eps >= 0
range_boundary    P6: specific air range decreases at the maximum operating Mach (MMO)
                  of the aircraft type: M f_M >= p6_min at cruise altitudes, eps = 0
drag_rise         P7: in steady level flight F = TSFC x D and TSFC does not decrease
                  with Mach, so f_M >= g with g = d ln D / dM at fixed W/delta from the
                  aerox drag model (neuralfuel.drag); at level-flight states (eps = 0)
                  with Mach from drag_rise_mach_min to the MMO and CL up to
                  drag_rise_cl_factor x the largest CL of the cruise windows of the
                  envelope. Only for a type with an aerox drag model with wave drag
                  (see `drag_rise_setup`); otherwise the training runs without it

teacher           protection against forgetting, only when training continues from
                  given weights: mean of w(x) (f(x) - f_teacher(x))^2 on a fixed set
                  of points (teacher_points), with f_teacher the frozen given model
                  and w(x) = 0 inside the envelope of the local training data, 1
                  outside

P1 to P3 use random collocation states, uniform in the envelope with CAS inside its
limits (`sample_collocation` draws until it has the requested number of states, or
raises ValueError when the CAS limits leave too little of the envelope). Each is the
mean squared scaled violation. The envelope combines the
aircraft-type settings (MMO, VMO, mass and altitude ranges; neuralfuel.aircraft) with
the type-independent ranges of LossConfig (see `envelope`).
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import drag
from .aircraft import Aircraft
from .atmos import T0, cas_from_mach_t, isa_delta_t, isa_temperature_t
from .privacy import NUM, STR
from .support import in_envelope, random_inputs


@dataclass
class Envelope:
    """Flight envelope of the collocation states (see `envelope`)."""

    mach: tuple
    alt_ft: tuple
    mass_kg: tuple
    eps: tuple
    dT_isa: tuple
    cas_kt: tuple


@dataclass
class LossConfig:
    points: float = 1.0
    window: float = 0.1
    mach_convexity: float = 0.001
    mass_monotone: float = 0.001
    energy_monotone: float = 0.001
    range_boundary: float = 0.001
    drag_rise: float = 0.01  # P7 (only where available, see drag_rise_setup)
    n_window_batch: int = 512
    n_colloc: int = 4096
    scale_mach: float = 0.1  # P1 residual is scaled by scale_mach^2
    scale_eps: float = 0.02  # P3 residual is scaled by scale_eps
    eps_powered: float = -0.005
    p6_alt_min_ft: float = 30000.0  # P6 altitudes: from here to the aircraft maximum
    p6_min: float = 1.1
    drag_rise_mach_min: float = 0.70  # P7 Mach band: from here to the MMO
    drag_rise_cl_factor: float = 1.1  # P7 CL bound: factor x largest cruise CL
    # type-independent parts of the envelope
    mach_min: float = 0.45
    min_cas_kt: float = 180.0
    eps: tuple = (-0.08, 0.08)
    dT_isa: tuple = (-15.0, 25.0)
    # protection against forgetting (training from given weights only, see
    # teacher_points): weight, size and random seed of the fixed point set
    teacher: float = 10.0
    n_teacher_points: int = 262144
    teacher_seed: int = 777


# settings of LossConfig that define the constraints (envelope of the collocation
# states and the limits); a trained model stores them (FuelFlowNet.constraints)
CONSTRAINT_SETTINGS = (
    "mach_min",
    "min_cas_kt",
    "eps",
    "dT_isa",
    "eps_powered",
    "p6_alt_min_ft",
    "p6_min",
)


# settings of P7 that a model trained with it stores (FuelFlowNet.constraints
# ['drag_rise'], see `drag_rise_setup`), with their format (neuralfuel.privacy)
DRAG_RISE = {
    "weight": NUM,
    "mach_min": NUM,
    "mach_max": NUM,
    "cl_factor": NUM,
    "cl_max": NUM,
    "aerox_version": STR,
    "table": {
        "type": STR,
        "wing_area_m2": NUM,
        "mach": NUM,
        "cl": NUM,
        "ln_delta": NUM,
        "dmach": NUM,
    },
    "table_sha256": STR,
}
# LossConfig fields of P7 and their keys in the stored settings
DRAG_RISE_FIELDS = {
    "drag_rise": "weight",
    "drag_rise_mach_min": "mach_min",
    "drag_rise_cl_factor": "cl_factor",
}


def constraint_settings(cfg: LossConfig) -> dict:
    """The CONSTRAINT_SETTINGS of cfg (JSON-ready dict)."""
    d = asdict(cfg)
    return {
        k: list(d[k]) if isinstance(d[k], tuple) else d[k] for k in CONSTRAINT_SETTINGS
    }


def model_loss_config(model) -> LossConfig:
    """LossConfig with the constraint settings stored in the model (the defaults for
    a model without them)."""
    cfg = LossConfig()
    set_constraints(cfg, getattr(model, "constraints", None))
    return cfg


def set_constraints(cfg: LossConfig, stored) -> None:
    """Set the constraint settings stored in a model in cfg. The P7 settings travel
    with the model: a model without stored P7 settings sets the P7 weight to 0."""
    for k, v in (stored or {}).items():
        if k in CONSTRAINT_SETTINGS:
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    dr = (stored or {}).get("drag_rise")
    if dr:
        for k, key in DRAG_RISE_FIELDS.items():
            setattr(cfg, k, float(dr[key]))
    else:
        cfg.drag_rise = 0.0


def envelope(ac: Aircraft, cfg: LossConfig) -> Envelope:
    """Envelope of the collocation states: Mach from mach_min to MMO, the altitude
    and mass ranges of the type, CAS from min_cas_kt to VMO."""
    return Envelope(
        mach=(cfg.mach_min, ac.mmo),
        alt_ft=tuple(ac.alt_ft),
        mass_kg=tuple(ac.mass_kg),
        eps=tuple(cfg.eps),
        dT_isa=tuple(cfg.dT_isa),
        cas_kt=(cfg.min_cas_kt, ac.vmo_kt),
    )


def p6_band(ac: Aircraft, cfg: LossConfig) -> tuple:
    """Altitude band (ft) of the range-boundary condition."""
    if cfg.p6_alt_min_ft >= ac.alt_ft[1]:
        raise ValueError(
            "loss.p6_alt_min_ft must be below the aircraft maximum altitude"
        )
    return (cfg.p6_alt_min_ft, ac.alt_ft[1])


SHAPE_TERMS = ("mach_convexity", "mass_monotone", "energy_monotone")


def state_vector(mach, alt_ft, mass_kg, eps, sat_k):
    """(M, ln delta, ln m, eps, ln theta) from flight conditions (torch)."""
    return torch.stack(
        [
            mach,
            torch.log(isa_delta_t(alt_ft)),
            torch.log(mass_kg),
            eps,
            torch.log(sat_k / T0),
        ],
        dim=-1,
    )


COLLOCATION_BATCHES = 64  # draws of 4 n states before the envelope counts as empty


def sample_collocation(
    n, gen, device, env: Envelope, dtype=torch.float32, cl_max=None, wing_area_m2=None
):
    """n states, uniform in the envelope, CAS inside its limits (and, with cl_max, the
    lift coefficient of level flight with the wing area not above cl_max). The states
    are drawn in batches of 4 n until n are inside; ValueError if COLLOCATION_BATCHES
    batches do not give n states."""
    keys = ("mach", "alt_ft", "mass_kg", "eps", "dT_isa")
    lo = torch.tensor([getattr(env, k)[0] for k in keys], device=device)
    hi = torch.tensor([getattr(env, k)[1] for k in keys], device=device)
    out, have = [], 0
    for _ in range(COLLOCATION_BATCHES):
        u = torch.rand(4 * n, 5, generator=gen, device=device, dtype=dtype)
        mach, alt, m, eps, dt = (lo + u * (hi - lo)).unbind(1)
        delta = isa_delta_t(alt)
        cas = cas_from_mach_t(mach, delta)
        ok = (cas >= env.cas_kt[0]) & (cas <= env.cas_kt[1])
        if cl_max is not None:
            cl = drag.level_cl(mach, torch.log(delta), torch.log(m), wing_area_m2)
            ok &= cl <= cl_max
        theta = (isa_temperature_t(alt) + dt) / T0
        p = torch.stack(
            [mach, torch.log(delta), torch.log(m), eps, torch.log(theta)], dim=1
        )
        out.append(p[ok])
        have += len(out[-1])
        if have >= n:
            return torch.cat(out)[:n]
    raise ValueError(
        f"could not draw {n} collocation states: the CAS limits "
        f"{env.cas_kt[0]:g} to {env.cas_kt[1]:g} kt leave too little of the envelope "
        f"(Mach {env.mach[0]:g} to {env.mach[1]:g}, {env.alt_ft[0]:g} to "
        f"{env.alt_ft[1]:g} ft)"
    )


def shape_penalties(model, p, cfg: LossConfig):
    """P1 to P3 at the states p; only the terms with weight > 0."""
    active = [k for k in SHAPE_TERMS if getattr(cfg, k) > 0]
    p = p.detach().requires_grad_(True)
    f = model.ln_ff_state(p)
    g = torch.autograd.grad(f.sum(), p, create_graph=True)[0]
    out = {}
    powered = p[:, 3] >= cfg.eps_powered
    if "mach_convexity" in active:
        fmm = torch.autograd.grad(g[:, 0].sum(), p, create_graph=True)[0][:, 0]
        v = F.relu(-(fmm + g[:, 0] ** 2) * cfg.scale_mach**2)[powered]
        out["mach_convexity"] = (v**2).mean() if v.numel() else g.sum() * 0
    if "mass_monotone" in active:
        v = F.relu(-g[:, 2])[powered]
        out["mass_monotone"] = (v**2).mean() if v.numel() else g.sum() * 0
    if "energy_monotone" in active:
        out["energy_monotone"] = (F.relu(-g[:, 3] * cfg.scale_eps) ** 2).mean()
    return out


def range_boundary_penalty(model, n, gen, device, cfg: LossConfig, ac: Aircraft):
    """P6: hinge on M f_M >= p6_min at M = MMO, eps = 0, random altitude in the P6
    band and random mass and ISA deviation in the envelope."""
    env = envelope(ac, cfg)
    lo, hi = p6_band(ac, cfg)
    u = torch.rand(n, 3, generator=gen, device=device)
    alt = lo + u[:, 0] * (hi - lo)
    m = env.mass_kg[0] + u[:, 1] * (env.mass_kg[1] - env.mass_kg[0])
    dt = env.dT_isa[0] + u[:, 2] * (env.dT_isa[1] - env.dT_isa[0])
    p = state_vector(
        torch.full_like(alt, ac.mmo),
        alt,
        m,
        torch.zeros_like(alt),
        isa_temperature_t(alt) + dt,
    ).requires_grad_(True)
    g = torch.autograd.grad(model.ln_ff_state(p).sum(), p, create_graph=True)[0]
    return (F.relu(cfg.p6_min - ac.mmo * g[:, 0]) ** 2).mean()


def drag_rise_setup(ac: Aircraft, cfg: LossConfig, envelope_=None, stored=None):
    """(settings, table) of P7. The stored settings of a model (`stored`, from
    FuelFlowNet.constraints['drag_rise']) keep their CL bound and table grid;
    otherwise the CL bound is drag_rise_cl_factor x the largest CL of the cruise
    windows of the valid envelope `envelope_`. drag.Unavailable (a ValueError, with
    the reason) if aerox has no drag model with wave drag for the type, if the wing
    area or the cruise CL is missing, or if the Mach band of P7 is empty. ValueError
    if the table rebuilt from stored settings does not have their table_sha256 (also
    when they have none): the installed aerox gives other values."""
    drag.aerox_drag_model(ac.type)
    if not np.isfinite(ac.wing_area_m2):
        raise drag.Unavailable("the aircraft settings have no wing area")
    if not cfg.drag_rise_mach_min < ac.mmo:
        raise drag.Unavailable(
            f"the Mach band is empty: loss.drag_rise_mach_min "
            f"({cfg.drag_rise_mach_min:g}) is not below aircraft.mmo ({ac.mmo:g})"
        )
    if stored:
        cl_max, spec = float(stored["cl_max"]), dict(stored["table"])
    else:
        if cfg.drag_rise_cl_factor <= 0:
            raise ValueError("loss.drag_rise_cl_factor must be positive")
        max_cl = ((envelope_ or {}).get("cruise") or {}).get("max_cl")
        if max_cl is None:
            raise drag.Unavailable(
                "the envelope has no largest lift coefficient of the cruise windows "
                "(no steady cruise windows)"
            )
        cl_max = cfg.drag_rise_cl_factor * float(max_cl)
        spec = drag.table_spec(ac, cfg.drag_rise_mach_min, cl_max)
    if stored and not stored.get("table_sha256"):
        raise ValueError(
            "the drag-slope table of P7 of the shared model cannot be checked: its "
            "P7 settings (constraints.drag_rise) have no table_sha256"
        )
    table = drag.build_table(spec)
    if stored and drag.table_hash(table) != stored["table_sha256"]:
        raise ValueError(
            "the drag-slope table of P7 of the shared model cannot be reproduced: "
            f"the table built with the installed aerox {drag.aerox_version()} differs "
            f"from the table of the model, built with aerox "
            f"{stored.get('aerox_version')} (table_sha256 differs). Install aerox "
            f"{stored.get('aerox_version')} to continue the training"
        )
    settings = {
        "weight": cfg.drag_rise,
        "mach_min": cfg.drag_rise_mach_min,
        "mach_max": ac.mmo,
        "cl_factor": cfg.drag_rise_cl_factor,
        "cl_max": cl_max,
        "aerox_version": drag.aerox_version(),
        "table": spec,
        "table_sha256": drag.table_hash(table),
    }
    return settings, table


def drag_rise_states(n, gen, device, ac: Aircraft, cfg: LossConfig, cl_max):
    """n level-flight states (eps = 0) of P7: Mach uniform from drag_rise_mach_min to
    the MMO, altitude, mass and ISA deviation uniform in the envelope, CAS inside its
    limits, lift coefficient not above cl_max."""
    env = envelope(ac, cfg)
    env.mach = (cfg.drag_rise_mach_min, ac.mmo)
    env.eps = (0.0, 0.0)
    return sample_collocation(
        n, gen, device, env, cl_max=cl_max, wing_area_m2=ac.wing_area_m2
    )


def drag_rise_penalty(model, p, table, settings, wing_area_m2):
    """P7 at the states p: mean of max(0, g - f_M)^2, g from the table."""
    p = p.detach().requires_grad_(True)
    fm = torch.autograd.grad(model.ln_ff_state(p).sum(), p, create_graph=True)[0][:, 0]
    with torch.no_grad():
        cl = drag.level_cl(p[:, 0], p[:, 1], p[:, 2], wing_area_m2)
        g = drag.lookup(table, settings["table"], p[:, 0], cl, p[:, 1])
    return (F.relu(g - fm) ** 2).mean()


def still_air_inputs(p):
    """(n, 5) states (M, ln delta, ln m, eps, ln theta) -> (n, 6) model inputs with
    sin_gamma = eps and acc_in = 0."""
    p = np.asarray(p, dtype=float)
    return np.column_stack([p[:, :4], np.zeros(len(p)), p[:, 4]])


def teacher_points(ac: Aircraft, cfg: LossConfig, shared_envelope=None):
    """Fixed point set of the teacher term, (n, 6) model inputs (numpy).

    Half are collocation states over the aircraft envelope (`sample_collocation`,
    still air); the other half are drawn inside the valid envelope of the given model
    (`support.random_inputs`, the 'all' ranges, kept if inside the envelope of their
    phase), so that the low altitudes and the energy-rate inputs of the shared data
    are included. Without an envelope (a model not made by `train`) all points are
    collocation states.
    """
    n = cfg.n_teacher_points
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.teacher_seed)
    p = sample_collocation(n, gen, "cpu", envelope(ac, cfg)).numpy()
    x = still_air_inputs(p)
    if shared_envelope is None:
        return x
    half = len(x) // 2
    m = len(x) - half
    xs = random_inputs(shared_envelope, m, seed=cfg.teacher_seed, phase="all")
    xs = xs[np.random.default_rng(cfg.teacher_seed).permutation(len(xs))[:m]]
    return np.concatenate([x[:half], xs])


def teacher_weights(x, local_envelope):
    """w(x): 0 inside the envelope of the local training data, 1 outside."""
    return 1.0 - in_envelope(local_envelope, x).astype(float)


def teacher_penalty(model, x, w, f_teacher):
    """mean of w (f(x) - f_teacher)^2 at the inputs x."""
    return (w * (model(x) - f_teacher) ** 2).mean()
