import numpy as np
import pytest

from neuralfuel import atmos
from neuralfuel.atmos import FT, G0, KT
from neuralfuel.prep import energy_inputs, segment_ids, sg_derivative


def test_isa():
    assert atmos.isa_delta(0.0) == pytest.approx(1.0)
    assert atmos.isa_temperature(0.0) == pytest.approx(288.15)
    h_trop_ft = 11000 / FT
    assert atmos.isa_delta(h_trop_ft) == pytest.approx(atmos.DELTA_TROP, rel=1e-12)
    assert atmos.isa_temperature(40000.0) == pytest.approx(216.65)
    # ISA pressure ratio at 36,089 ft is 0.2234
    assert atmos.isa_delta(36089.0) == pytest.approx(0.2234, abs=2e-4)


def test_airspeed_conversions():
    # at sea level in ISA, CAS equals TAS
    m = atmos.mach_from_cas(250.0, 0.0)
    assert m * atmos.A0 / KT == pytest.approx(250.0, rel=1e-9)
    alt = np.array([5000.0, 20000.0, 35000.0, 39000.0])
    cas = np.array([250.0, 290.0, 270.0, 240.0])
    m = atmos.mach_from_cas(cas, alt)
    np.testing.assert_allclose(atmos.cas_from_mach(m, alt), cas, rtol=1e-10)
    # 280 kt CAS at FL350 is about M 0.82
    assert atmos.mach_from_cas(280.0, 35000.0) == pytest.approx(0.82, abs=0.005)
    sat = np.array([270.0, 250.0, 220.0, 216.65])
    tas = m * atmos.sound_speed(sat) / KT
    np.testing.assert_allclose(atmos.mach_from_tas(tas, sat), m, rtol=1e-12)
    tat = sat * (1 + 0.2 * m**2)
    np.testing.assert_allclose(atmos.sat_from_tat(tat, m), sat, rtol=1e-12)
    np.testing.assert_allclose(atmos.mach_from_tas_tat(tas, tat), m, rtol=1e-10)


def test_segments_and_derivative():
    fc = np.array([0] * 30 + [1] * 10 + [1] * 25)
    t = np.r_[np.arange(30), np.arange(10), np.arange(12, 37)].astype(float)
    seg = segment_ids(fc, t)
    assert seg.max() == 2 and (seg[:30] == 0).all() and (seg[30:40] == 1).all()
    x = 3.0 * t + 7.0
    d = sg_derivative(x, seg, 21, 2)
    np.testing.assert_allclose(d[:30], 3.0, rtol=1e-12)
    assert np.isnan(d[30:40]).all()  # segment shorter than the filter
    np.testing.assert_allclose(d[40:], 3.0, rtol=1e-12)
    # quadratic: exact derivative with an order-2 filter
    q = 0.5 * t[:30] ** 2
    np.testing.assert_allclose(
        sg_derivative(q, np.zeros(30, int), 21, 2), t[:30], atol=1e-9
    )


def test_energy_inputs():
    # steady climb at 2000 ft/min in ISA, no wind, no acceleration
    v = 230.0  # m/s
    dhp = 2000 * FT / 60
    t_isa = 250.0
    dhg, sg, acc = energy_inputs(dhp, v, 0.0, v, t_isa, t_isa)
    assert dhg == pytest.approx(dhp)
    assert sg == pytest.approx(dhp / v)
    assert acc == 0.0
    # warmer than ISA: geometric rate is larger by SAT / T_ISA
    dhg, sg, _ = energy_inputs(dhp, v, 0.0, v, t_isa + 10, t_isa)
    assert dhg == pytest.approx(dhp * 260 / 250)
    # level acceleration of 0.5 m/s^2 with 20 m/s tailwind
    _, sg, acc = energy_inputs(0.0, v + 20, 0.5, v, t_isa, t_isa)
    assert sg == 0.0
    assert acc == pytest.approx((v + 20) / v * 0.5 / G0)


def test_derive_matches_surveillance_builder(syn_samples):
    """The ADS-B input builder gives the same inputs as the QAR pipeline at 1 Hz."""
    from neuralfuel.inference import surveillance_inputs
    from neuralfuel.prep import derive

    one = syn_samples[syn_samples.flight_id == syn_samples.flight_id.iloc[0]]
    s, _, _ = derive(one)
    x = surveillance_inputs(
        s.time_s, s.alt_ft, s.gs_kt, s.mass_kg, s.sat_k, mach=s.mach
    )
    for c in ["mach", "sin_gamma", "acc_in"]:
        np.testing.assert_allclose(x[c], s[c], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(x.ln_theta, np.log(s.theta), rtol=1e-12)


@pytest.mark.parametrize("speed", ["mach", "cas_kt", "tas_kt"])
def test_surveillance_inputs_broadcast_constants(speed):
    """Constant values (scalars) give the same inputs as arrays of the time shape."""
    from neuralfuel.inference import surveillance_inputs

    t = np.arange(0.0, 120.0, 4.0)
    alt = 30000.0 + 25.0 * t
    sat = 240.0
    value = {"mach": 0.74, "cas_kt": 280.0, "tas_kt": 440.0}[speed]
    full = np.full(len(t), 1.0)
    a = surveillance_inputs(t, alt, 420.0, 64000.0, sat, **{speed: value})
    b = surveillance_inputs(
        t, alt, 420.0 * full, 64000.0 * full, sat * full, **{speed: value * full}
    )
    assert len(a) == len(t)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=0)
    c = surveillance_inputs(t, 33000.0, 420.0, 64000.0, sat, **{speed: value})
    assert len(c) == len(t)
    np.testing.assert_allclose(c.sin_gamma, 0.0, atol=1e-12)
    np.testing.assert_allclose(c.acc_in, 0.0, atol=1e-12)


def test_surveillance_inputs_cadence_per_track():
    """The step comes from the differences inside each track, not from one global
    step; a gap starts a new segment."""
    from neuralfuel.inference import surveillance_inputs

    t = np.arange(0.5, 60.5)  # 1 s steps from 0.5 s
    x = surveillance_inputs(t, 30000 + 10 * t, 420.0, 62000.0, 230.0, mach=0.75)
    assert x.sin_gamma.notna().all()
    t = np.r_[np.arange(100.0), np.arange(0.0, 120.0, 4.0)]  # 1 s and 4 s tracks
    ids = np.repeat(["a", "b"], [100, 30])
    x = surveillance_inputs(
        t, 30000 + 10 * t, 420.0, 62000.0, 230.0, mach=0.75, track_id=ids
    )
    assert x.sin_gamma.groupby(ids).count().to_dict() == {"a": 100, "b": 30}
    # the same climb rate (10 ft/s) in both tracks
    np.testing.assert_allclose(
        x.sin_gamma[:100].mean(), x.sin_gamma[100:].mean(), rtol=2e-3
    )
    # a gap of 30 s in a 1 s track: two segments, each long enough for the filter
    t = np.r_[np.arange(40.0), np.arange(70.0, 110.0)]
    x = surveillance_inputs(t, 30000 + 10 * t, 420.0, 62000.0, 230.0, mach=0.75)
    assert x.sin_gamma.notna().all()


def test_collocation_gives_the_requested_states():
    import torch

    from neuralfuel.aircraft import aircraft
    from neuralfuel.losses import LossConfig, envelope, sample_collocation

    ac = aircraft("a320")
    ac.vmo_kt = 195.0  # CAS from 180 to 195 kt: a small part of the box
    env = envelope(ac, LossConfig())
    p = sample_collocation(4096, torch.Generator().manual_seed(0), "cpu", env)
    assert len(p) == 4096
    ac.vmo_kt = 180.0001
    with pytest.raises(ValueError, match="CAS limits"):
        sample_collocation(
            512, torch.Generator().manual_seed(0), "cpu", envelope(ac, LossConfig())
        )


@pytest.mark.parametrize("eps_neg", [-0.005 - 1e-3, 0.01])
def test_mass_monotone_only_at_powered_states(eps_neg):
    """P2: a negative mass slope below eps_powered is not a violation, in the loss and
    in the validation; at a powered state it is."""
    import torch

    from neuralfuel.aircraft import aircraft
    from neuralfuel.losses import LossConfig, shape_penalties
    from neuralfuel.validate import ValidationConfig, constraints, hessian, state

    class Stub(torch.nn.Module):
        def ln_ff_state(self, p):
            slope = torch.where(p[:, 3] < eps_neg, -0.5, 0.5).to(p.dtype)
            # the zero term keeps every derivative in the autograd graph
            return 2 * p[:, 0] + slope * p[:, 2] + 0 * (p**2).sum(1)

    lc, ac = LossConfig(), aircraft("a320")
    p = state(0.75, 35000.0, 60000.0, [-0.04, -0.01, 0.0, 0.02], 230.0)
    loss = shape_penalties(Stub(), p.float(), lc)["mass_monotone"]
    _, g, h = hessian(Stub(), p)
    c = constraints(Stub(), p, np.ones(len(p), bool), g, h, ValidationConfig(), lc, ac)
    pct = [c[r]["mass_monotone_violation_pct"] for r in ("all", "inside")]
    if eps_neg < lc.eps_powered:
        assert loss.item() == 0
        assert pct == [0.0, 0.0]
    else:  # the state at eps = 0 is powered and has the negative slope
        assert loss.item() == pytest.approx(0.25 / 2)
        assert pct == [50.0, 50.0]
    # no powered state: no P2 loss and no P2 rate
    q = p[:2]
    assert shape_penalties(Stub(), q.float(), lc)["mass_monotone"].item() == 0
    _, g, h = hessian(Stub(), q)
    c = constraints(Stub(), q, np.ones(2, bool), g, h, ValidationConfig(), lc, ac)
    assert c["all"]["mass_monotone_violation_pct"] is None
