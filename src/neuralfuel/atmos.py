"""ISA atmosphere and airspeed conversions (numpy, and torch for the training grids).

Units: altitude in ft (pressure altitude), speeds in kt, temperatures in K.
"""

import math

import numpy as np
import torch

GAMMA = 1.4
R_GAS = 287.05287  # J/(kg K)
G0 = 9.80665  # m/s^2
P0 = 101325.0  # Pa
T0 = 288.15  # K
A0 = math.sqrt(GAMMA * R_GAS * T0)  # m/s
KT = 0.514444  # m/s per kt
FT = 0.3048  # m per ft
LAPSE = 0.0065  # K/m
H_TROP = 11000.0  # m
T_TROP = 216.65  # K
EXP_P = G0 / (LAPSE * R_GAS)
DELTA_TROP = (T_TROP / T0) ** EXP_P


# ---------------------------------------------------------------- numpy
def isa_temperature(alt_ft):
    """ISA static temperature (K) at pressure altitude (ft)."""
    h = np.asarray(alt_ft, dtype=float) * FT
    return np.where(h <= H_TROP, T0 - LAPSE * h, T_TROP)


def isa_delta(alt_ft):
    """ISA pressure ratio p / p0 at pressure altitude (ft)."""
    h = np.asarray(alt_ft, dtype=float) * FT
    low = (1 - LAPSE * np.minimum(h, H_TROP) / T0) ** EXP_P
    high = DELTA_TROP * np.exp(-G0 * (h - H_TROP) / (R_GAS * T_TROP))
    return np.where(h <= H_TROP, low, high)


def alt_from_delta(delta):
    """Pressure altitude (ft) from the ISA pressure ratio (inverse of isa_delta)."""
    d = np.asarray(delta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        low = T0 / LAPSE * (1 - d ** (1 / EXP_P))
        high = H_TROP - np.log(d / DELTA_TROP) * R_GAS * T_TROP / G0
    return np.where(d >= DELTA_TROP, low, high) / FT


def sound_speed(sat_k):
    """Speed of sound (m/s) at static air temperature (K)."""
    return np.sqrt(GAMMA * R_GAS * np.asarray(sat_k, dtype=float))


def mach_from_cas(cas_kt, alt_ft):
    """Mach from calibrated airspeed and pressure altitude (subsonic, compressible)."""
    vc = np.asarray(cas_kt, dtype=float) * KT
    qc_p0 = (1 + 0.2 * (vc / A0) ** 2) ** 3.5 - 1
    return np.sqrt(5 * ((qc_p0 / isa_delta(alt_ft) + 1) ** (2 / 7) - 1))


def cas_from_mach(mach, alt_ft):
    """Calibrated airspeed (kt) from Mach and pressure altitude."""
    m = np.asarray(mach, dtype=float)
    qc_p0 = isa_delta(alt_ft) * ((1 + 0.2 * m**2) ** 3.5 - 1)
    return A0 * np.sqrt(5 * ((qc_p0 + 1) ** (2 / 7) - 1)) / KT


def mach_from_tas(tas_kt, sat_k):
    """Mach from true airspeed (kt) and static air temperature (K)."""
    return np.asarray(tas_kt, dtype=float) * KT / sound_speed(sat_k)


def mach_from_tas_tat(tas_kt, tat_k):
    """Mach from true airspeed (kt) and total air temperature (K), recovery factor 1."""
    v2 = (np.asarray(tas_kt, dtype=float) * KT) ** 2
    return np.sqrt(v2 / (GAMMA * R_GAS * np.asarray(tat_k, dtype=float) - 0.2 * v2))


def sat_from_tat(tat_k, mach):
    """Static air temperature (K) from total air temperature, recovery factor 1."""
    return np.asarray(tat_k, dtype=float) / (
        1 + 0.2 * np.asarray(mach, dtype=float) ** 2
    )


# ---------------------------------------------------------------- torch
def isa_delta_t(alt_ft):
    h = alt_ft * FT
    low = (1 - LAPSE * torch.clamp(h, max=H_TROP) / T0) ** EXP_P
    high = DELTA_TROP * torch.exp(
        -G0 * (torch.clamp(h, min=H_TROP) - H_TROP) / (R_GAS * T_TROP)
    )
    return torch.where(h <= H_TROP, low, high)


def isa_temperature_t(alt_ft):
    return torch.clamp(T0 - LAPSE * alt_ft * FT, min=T_TROP)


def cas_from_mach_t(mach, delta):
    qc = delta * ((1 + 0.2 * mach**2) ** 3.5 - 1)
    return A0 * torch.sqrt(5 * ((qc + 1) ** (2 / 7) - 1)) / KT
