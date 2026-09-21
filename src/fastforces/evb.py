"""The two-state EVB surface: two diabatic force fields and one coupling.

The Hamiltonian is a 2x2 whose diagonal is the two states' own energies and
whose off-diagonal is the fitted coupling, and the surface is its lower
eigenvalue:

    E = Hbar - sqrt(dH**2 + V**2),    Hbar = (H1 + H2)/2,  dH = (H1 - H2)/2

**The coupling is a function of the geometry alone, not of the diagonal.**  That
is the whole difference from the `sqrt((1+h) H1 H2)` form, and it is what makes
the surface fittable: `V` is pinned at the transition state by inverting this
same equation there (`coupling.fit_amplitude`), so the barrier is reproduced by
construction rather than by tuning a hardness.  It also means a common shift of
both diagonals shifts `E` by exactly that constant and nothing else, so where
the topology-independent terms are added is free -- they are put on the diagonal
here because that is where they belong physically.

Forces are Hellmann-Feynman.  For a symmetric matrix with a non-degenerate
ground state, `dE/dx = c^T (dH/dx) c` with `c` the ground-state eigenvector, so
the two diabatic force fields and the coupling's own gradient combine with the
state weights `c**2` and the interference term `2*c1*c2`.  Nothing else is
needed and no finite difference appears anywhere.

Degenerate diabats with zero coupling are the one place this breaks down -- the
eigenvector is then undefined and the surface has a real conical intersection --
and it is not a case `fit` produces: a fitted amplitude is nonzero at the
transition state by construction, and where the coupling has quenched to `eps`
the diabats are far from degenerate.
"""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .forcefield.acks2 import ACKS2
from .forcefield.lj import LennardJones
from .forcefield.qforce import QForce
from .forcefield.zbl import ZBL


def _topology_independent(pos, numbers, pbc, cell, terms, acks2, lj, zbl):
    """ACKS2 + Lennard-Jones + ZBL: the part both states share exactly.

    Every one of the three is a function of the geometry and the elements only
    -- `LennardJones` takes no exclusions and `ZBL` never had any -- so both
    states produce the same number and it is computed once.  Note that the
    combined system's ACKS2 energy is *not* the sum of its fragments': the
    charges equilibrate over whatever they are handed.  That is a real
    difference and it is why this is evaluated on the combined system here, the
    same way `coupling.fit` evaluates the diabats it fits the amplitude to.
    """
    energy, forces, virial = 0.0, np.zeros_like(pos), np.zeros((3, 3))
    if "atom" in terms:
        de, df, dw = acks2(pos, pbc, cell, terms)
        energy, forces, virial = energy + de, forces + df, virial + dw
    if "lennardjones" in terms:
        de, df, dw = lj(pos, pbc, cell, terms)
        energy, forces, virial = energy + de, forces + df, virial + dw
    de, df, dw = zbl(pos, numbers, pbc, cell)
    return energy + de, forces + df, virial + dw


class EVB(Calculator):
    """`FastForces` for two diabatic states, mixed by a fitted coupling.

    `states` is a pair of `Parameters` over the *same* atom indices -- what
    `reaction.state_parameters` builds -- and `coupling` is the `Coupling` that
    `coupling.fit` returns for the same reaction.

    `results["statevec"]` is the squared ground-state eigenvector: how much of
    each diabatic state the current geometry is made of, which is the number
    that says where along the reaction the system sits.
    """

    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "charges",
        "statevec",
    ]

    def __init__(
        self, atoms=None, states=None, coupling=None, bond_form: str = "morse", **kwargs
    ):
        super().__init__(atoms=atoms, **kwargs)
        if states is None or len(states) != 2:
            raise ValueError("EVB needs exactly two diabatic states")
        if coupling is None:
            raise ValueError("EVB needs a Coupling; see `fastforces.coupling.fit`")
        self.states = list(states)
        self.coupling = coupling
        self.qforce = QForce(bond_form=bond_form)
        self.acks2 = ACKS2()
        self.lj = LennardJones()
        self.zbl = ZBL()
        self._bonded = [p.bonded_terms() for p in self.states]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = atoms if atoms is not None else self.atoms

        pos = atoms.get_positions()
        pbc, cell = atoms.pbc, np.array(atoms.cell)
        numbers = atoms.get_atomic_numbers()

        shared_e, shared_f, shared_w = _topology_independent(
            pos, numbers, pbc, cell, self.states[0].terms, self.acks2, self.lj, self.zbl
        )

        diagonal = np.zeros(2)
        state_forces = np.zeros((2,) + pos.shape)
        state_virials = np.zeros((2, 3, 3))
        for i, params in enumerate(self.states):
            energy, forces, virial = self.qforce(pos, pbc, cell, self._bonded[i])
            diagonal[i] = energy + params.e0 + shared_e
            state_forces[i] = forces + shared_f
            state_virials[i] = virial + shared_w

        v, coupling_forces, coupling_virial = self.coupling(pos, pbc, cell)

        hamiltonian = np.array([[diagonal[0], v], [v, diagonal[1]]])
        values, vectors = np.linalg.eigh(hamiltonian)
        c = vectors[:, 0]
        weights = c**2

        energy = float(values[0])
        forces = (
            weights[0] * state_forces[0]
            + weights[1] * state_forces[1]
            + 2.0 * c[0] * c[1] * coupling_forces
        )
        virial = (
            weights[0] * state_virials[0]
            + weights[1] * state_virials[1]
            + 2.0 * c[0] * c[1] * coupling_virial
        )

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "statevec": weights,
            "hamiltonian": hamiltonian,
        }
        volume = atoms.get_volume() if atoms.cell.rank == 3 else 0.0
        if volume > 0.0:
            self.results["stress"] = full_3x3_to_voigt_6_stress(
                0.5 * (virial + virial.T) / volume
            )
        if "atom" in self.states[0].terms and self.acks2.Q is not None:
            charges = np.zeros(len(atoms))
            charges[self.states[0].terms["atom"]["atoms"][:, 0]] = self.acks2.Q
            self.results["charges"] = charges


def diabatic_energies(atoms: Atoms, states, bond_form: str = "morse") -> tuple:
    """Each state's own energy at one geometry, without mixing them.

    This is what `coupling.fit_amplitude` inverts, and it has to be the number
    the running surface puts on its diagonal -- the same nonbonded sum over the
    same combined system, the same `E0` -- or the fitted amplitude reproduces a
    barrier nobody will evaluate.  Computing it here, through the code `EVB`
    itself uses, is what keeps the two from drifting.
    """

    pos = atoms.get_positions()
    pbc, cell = atoms.pbc, np.array(atoms.cell)
    numbers = atoms.get_atomic_numbers()
    qforce = QForce(bond_form=bond_form)
    shared, _, _ = _topology_independent(
        pos, numbers, pbc, cell, states[0].terms, ACKS2(), LennardJones(), ZBL()
    )
    out = []
    for params in states:
        energy, _, _ = qforce(pos, pbc, cell, params.bonded_terms())
        out.append(float(energy + params.e0 + shared))
    return tuple(out)


def diabatic_forces(atoms: Atoms, states, bond_form: str = "morse"):
    """`(energies, forces)` per state; `forces` is `(n_states, n_atoms, 3)`."""
    pos = atoms.get_positions()
    pbc, cell = atoms.pbc, np.array(atoms.cell)
    numbers = atoms.get_atomic_numbers()
    qforce = QForce(bond_form=bond_form)
    shared_e, shared_f, _ = _topology_independent(
        pos, numbers, pbc, cell, states[0].terms, ACKS2(), LennardJones(), ZBL()
    )
    energies, forces = [], []
    for params in states:
        energy, force, _ = qforce(pos, pbc, cell, params.bonded_terms())
        energies.append(float(energy + params.e0 + shared_e))
        forces.append(force + shared_f)
    return np.array(energies), np.array(forces)
