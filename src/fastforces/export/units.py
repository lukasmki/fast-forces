"""The one place the export conventions are written down.

Both export formats are OpenMM-flavoured: lengths in nm, energies in kJ/mol.
The calculator works in ASE units (Angstrom, eV).  Every conversion between the
two lives in this table, so the two exporters cannot drift from each other.

`UNIT_POWERS[term][parameter]` is `(length_power, energy_power)`, the dimension
of that parameter.  A bond force constant is an energy over a length squared,
so `(-2, 1)`.
"""

from ase import units

# Angstrom -> nm, and eV -> kJ/mol.
LENGTH = 0.1
ENERGY = 1.0 / (units.kJ / units.mol)

# Slot names each term uses for its atom indices in the DynamicTopology format.
SLOT_PREFIX: dict[str, str] = {"atom": "p", "lennardjones": "p", "reference": "a"}
SLOT_START: dict[str, int] = {"atom": 0, "lennardjones": 0, "reference": 1}

# `(length_power, energy_power)` per parameter.
#
# The ACKS2 `atom` parameters are the exception to the whole scheme: the example
# file stores them in eV and Angstrom, unconverted, and `ACKS2.CCOUL` is 14.4
# eV*Angstrom, so the format simply carries them in the calculator's own units.
UNIT_POWERS: dict[str, dict[str, tuple[int, int]]] = {
    "atom": {
        "mu": (0, 0),
        "eta": (0, 0),
        "soft_amp": (0, 0),
        "soft_decay": (0, 0),
        "soft_scale": (0, 0),
    },
    "lennardjones": {"sigma": (1, 0), "eps": (0, 1)},
    "bond": {"r0": (1, 0), "k": (-2, 1), "D": (0, 1), "c": (0, 0)},
    "angle": {"theta0": (0, 0), "k": (0, 1)},
    "bondbond": {"r1_0": (1, 0), "r2_0": (1, 0), "k": (-2, 1)},
    "bondangle": {"theta0": (0, 0), "r0": (1, 0), "k": (-1, 1)},
    "angleangle": {"theta1_0": (0, 0), "theta2_0": (0, 0), "k": (0, 1)},
    "dihedralangle": {"k": (0, 1), "theta0": (0, 0), "n": (0, 0), "phi0": (0, 0)},
    "dihedralbond": {"k": (-1, 1), "r0": (1, 0), "n": (0, 0), "phi0": (0, 0)},
    "dihedralangleangle": {
        "k": (0, 1),
        "theta0_1": (0, 0),
        "theta0_2": (0, 0),
        "n": (0, 0),
        "phi0": (0, 0),
    },
    "periodicdihedral": {"k": (0, 1), "n": (0, 0), "phi0": (0, 0)},
    "reference": {"E0": (0, 1)},
}

# The order terms are written in, matching the example file.
TERM_ORDER: tuple[str, ...] = (
    "atom",
    "lennardjones",
    "bond",
    "angle",
    "bondbond",
    "bondangle",
    "angleangle",
    "dihedralangle",
    "dihedralbond",
    "dihedralangleangle",
    "periodicdihedral",
    "reference",
)


def factor(term: str, parameter: str) -> float:
    """Multiplier taking one parameter from ASE units to OpenMM units."""
    length_power, energy_power = UNIT_POWERS[term][parameter]
    return (LENGTH**length_power) * (ENERGY**energy_power)


def to_openmm(term: str, parameter: str, value):
    return value * factor(term, parameter)


def from_openmm(term: str, parameter: str, value):
    return value / factor(term, parameter)


def slot_names(term: str, n_slots: int) -> list[str]:
    prefix = SLOT_PREFIX.get(term, "p")
    start = SLOT_START.get(term, 1)
    return [f"{prefix}{start + i}" for i in range(n_slots)]
