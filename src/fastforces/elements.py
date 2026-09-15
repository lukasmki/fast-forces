"""Element-indexed default parameters, in ASE units (eV, Angstrom).

These are *seeds*, not fitted values.  The bonded fit refines the force
constants against the reference calculator; the nonbonded parameters here are
subtracted from the reference as a fixed baseline and are never refined, so
they set the residual the bonded terms have to absorb.

Provenance of each table is documented at its definition.  Where an earlier
hand-checked H2O2 force field pinned a value for H or O, that value wins over
the generic table -- it came out of a real fit and the generic tables are
calibrated to reproduce it.
"""

import numpy as np
from ase import units
from ase.data import covalent_radii

# 1 kJ/mol in eV, for the literature tables below that are quoted in kJ/mol.
KJ_MOL = units.kJ / units.mol

# ---------------------------------------------------------------------------
# ACKS2
# ---------------------------------------------------------------------------

# Mulliken electronegativity chi = (IP + EA)/2 and hardness eta = (IP - EA)/2,
# in eV, from the NIST atomic ionization energies and electron affinities.
# Noble gases and the alkaline earths have no bound anion; the negative EA
# values used there are the usual extrapolated placeholders.
MULLIKEN_CHI: dict[int, float] = {
    1: 7.18,
    2: 12.04,
    3: 3.01,
    4: 4.41,
    5: 4.29,
    6: 6.26,
    7: 7.23,
    8: 7.54,
    9: 10.41,
    10: 10.18,
    11: 2.84,
    12: 3.62,
    13: 3.21,
    14: 4.77,
    15: 5.62,
    16: 6.22,
    17: 8.29,
    18: 7.38,
    35: 7.59,
    53: 6.76,
}
MULLIKEN_ETA: dict[int, float] = {
    1: 6.42,
    2: 12.55,
    3: 2.39,
    4: 4.91,
    5: 4.01,
    6: 5.00,
    7: 7.30,
    8: 6.08,
    9: 7.01,
    10: 11.38,
    11: 2.30,
    12: 4.02,
    13: 2.77,
    14: 3.38,
    15: 4.87,
    16: 4.14,
    17: 4.68,
    18: 8.38,
    35: 4.23,
    53: 3.70,
}

# The ACKS2 softness parameters have no tabulated experimental analogue.  Both
# are taken linear in the covalent radius, with the two coefficients fixed by
# the H and O values of that hand-checked H2O2 force field:
#
#     H: r_cov 0.31 -> amp 2.09682, decay 0.26887
#     O: r_cov 0.66 -> amp 3.88211, decay 0.43584
#
# A two-point calibration is a placeholder, not a model.  It exists so that
# every element gets a finite, size-ordered value; fitting these against
# reference charges is future work.
SOFT_AMP_SLOPE, SOFT_AMP_INTERCEPT = 5.10123, 0.51544
SOFT_DECAY_SLOPE, SOFT_DECAY_INTERCEPT = 0.47686, 0.12104

# `soft_scale` rides along in the DynamicTopology format but is not consumed by
# `ACKS2`; the two example values agree to within 2%, so one constant covers it.
SOFT_SCALE_DEFAULT = 0.18

# Values that came out of a real fit, and so override the generic tables above.
ACKS2_OVERRIDES: dict[int, dict[str, float]] = {
    1: {
        "mu": 1.879652976989746,
        "eta": 7.284643173217773,
        "soft_amp": 2.0968151092529297,
        "soft_decay": 0.26886698603630066,
        "soft_scale": 0.18287299573421478,
    },
    8: {
        "mu": 8.120335578918457,
        "eta": 3.7445240020751953,
        "soft_amp": 3.882110118865967,
        "soft_decay": 0.4358389973640442,
        "soft_scale": 0.17855200171470642,
    },
}

ACKS2_FIELDS = ("mu", "eta", "soft_amp", "soft_decay", "soft_scale")

# ---------------------------------------------------------------------------
# Lennard-Jones
# ---------------------------------------------------------------------------

# sigma in Angstrom, epsilon in kJ/mol; OPLS-AA where it defines the element and
# UFF elsewhere.  H and O are the example's values, which is why they sit below
# the OPLS aliphatic-H and TIP3P-O radii.
LJ_SIGMA: dict[int, float] = {
    1: 1.96,
    2: 2.65,
    3: 2.13,
    4: 2.45,
    5: 3.64,
    6: 3.40,
    7: 3.25,
    8: 2.96,
    9: 3.12,
    10: 2.89,
    11: 2.66,
    12: 2.69,
    13: 4.01,
    14: 3.83,
    15: 3.74,
    16: 3.56,
    17: 3.47,
    18: 3.40,
    35: 3.73,
    53: 4.01,
}
LJ_EPS_KJ: dict[int, float] = {
    1: 0.184,
    2: 0.0850,
    3: 0.0765,
    4: 0.360,
    5: 0.397,
    6: 0.360,
    7: 0.711,
    8: 0.711,
    9: 0.255,
    10: 0.290,
    11: 0.0577,
    12: 0.0468,
    13: 2.11,
    14: 1.68,
    15: 0.837,
    16: 1.046,
    17: 1.108,
    18: 0.996,
    35: 1.506,
    53: 2.092,
}

# ---------------------------------------------------------------------------
# Morse well depths
# ---------------------------------------------------------------------------

# Homonuclear single-bond dissociation energies, eV.  Used through the
# Pauling-style geometric mean when a pair is not tabulated below.
HOMONUCLEAR_D: dict[int, float] = {
    1: 4.52,
    3: 1.10,
    4: 0.62,
    5: 3.08,
    6: 3.61,
    7: 1.70,
    8: 1.52,
    9: 1.61,
    11: 0.75,
    12: 0.11,
    13: 1.40,
    14: 3.39,
    15: 2.08,
    16: 2.75,
    17: 2.51,
    35: 1.99,
    53: 1.54,
}

# Measured heteronuclear single-bond dissociation energies, eV.  Keys are
# unordered element pairs.
PAIR_D: dict[frozenset[int], float] = {
    frozenset((1, 6)): 4.28,
    frozenset((1, 7)): 4.05,
    frozenset((1, 8)): 4.81,
    frozenset((1, 9)): 5.87,
    frozenset((1, 14)): 3.24,
    frozenset((1, 15)): 3.30,
    frozenset((1, 16)): 3.66,
    frozenset((1, 17)): 4.43,
    frozenset((1, 35)): 3.76,
    frozenset((1, 53)): 3.17,
    frozenset((6, 7)): 3.17,
    frozenset((6, 8)): 3.74,
    frozenset((6, 9)): 5.03,
    frozenset((6, 14)): 3.31,
    frozenset((6, 15)): 2.68,
    frozenset((6, 16)): 2.80,
    frozenset((6, 17)): 3.44,
    frozenset((6, 35)): 2.90,
    frozenset((6, 53)): 2.34,
    frozenset((7, 8)): 2.10,
    frozenset((8, 14)): 4.72,
    frozenset((8, 15)): 3.51,
}

# Last resort when neither the pair nor both homonuclear entries are known.
FALLBACK_D = 4.0


def _lookup(table: dict[int, float], numbers: np.ndarray, what: str) -> np.ndarray:
    numbers = np.asarray(numbers, dtype=int)
    missing = sorted({int(z) for z in numbers if int(z) not in table})
    if missing:
        raise KeyError(f"no {what} default for atomic number(s) {missing}")
    return np.array([table[int(z)] for z in numbers], dtype=float)


def acks2_defaults(numbers: np.ndarray) -> dict[str, np.ndarray]:
    """Per-atom ACKS2 parameters, in eV and Angstrom."""
    numbers = np.asarray(numbers, dtype=int)
    mu = _lookup(MULLIKEN_CHI, numbers, "electronegativity")
    eta = _lookup(MULLIKEN_ETA, numbers, "hardness")
    r = np.array([covalent_radii[int(z)] for z in numbers], dtype=float)
    params = {
        "mu": mu,
        "eta": eta,
        "soft_amp": SOFT_AMP_SLOPE * r + SOFT_AMP_INTERCEPT,
        "soft_decay": SOFT_DECAY_SLOPE * r + SOFT_DECAY_INTERCEPT,
        "soft_scale": np.full(len(numbers), SOFT_SCALE_DEFAULT),
    }
    for i, z in enumerate(numbers):
        for field, value in ACKS2_OVERRIDES.get(int(z), {}).items():
            params[field][i] = value
    return params


def lj_defaults(numbers: np.ndarray) -> dict[str, np.ndarray]:
    """Per-atom Lennard-Jones parameters: sigma in Angstrom, eps in eV."""
    numbers = np.asarray(numbers, dtype=int)
    return {
        "sigma": _lookup(LJ_SIGMA, numbers, "LJ sigma"),
        "eps": _lookup(LJ_EPS_KJ, numbers, "LJ epsilon") * KJ_MOL,
    }


def morse_well_depth(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Seed Morse well depths `D` for a batch of element pairs, in eV.

    Tabulated pairs win; otherwise the Pauling-style geometric mean of the two
    homonuclear values; otherwise `FALLBACK_D`.  These are single-bond values,
    so a double or triple bond starts out too shallow -- the nonlinear refine in
    `fit` is what corrects that, seeded from here.
    """
    z1 = np.atleast_1d(np.asarray(z1, dtype=int))
    z2 = np.atleast_1d(np.asarray(z2, dtype=int))
    out = np.empty(len(z1), dtype=float)
    for i, (a, b) in enumerate(zip(z1, z2, strict=True)):
        pair = PAIR_D.get(frozenset((int(a), int(b))))
        if pair is not None:
            out[i] = pair
        elif int(a) in HOMONUCLEAR_D and int(b) in HOMONUCLEAR_D:
            out[i] = np.sqrt(HOMONUCLEAR_D[int(a)] * HOMONUCLEAR_D[int(b)])
        else:
            out[i] = FALLBACK_D
    return out


def defaults_for(numbers: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Every per-atom nonbonded default, keyed by the term type that uses it."""
    return {"atom": acks2_defaults(numbers), "lennardjones": lj_defaults(numbers)}
