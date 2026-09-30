"""The two-state EVB surface: two diabatic force fields and one coupling.

The Hamiltonian is a 2x2 whose diagonal is the two states' own energies and
whose off-diagonal is the fitted coupling, and the surface is its lower
eigenvalue:

    E = Hbar - sqrt(dH**2 + V**2),    Hbar = (H1 + H2)/2,  dH = (H1 - H2)/2

**The coupling is a function of the geometry alone, not of the diagonal.**  That
is the whole difference from the `sqrt((1+h) H1 H2)` form, and it is what makes
the surface fittable: `V` is pinned at the transition state by inverting this
same equation there (`coupling.fit_amplitude`), so the barrier is reproduced by
construction rather than by tuning a hardness.

**Each diagonal is DynamicTopology's own single-topology energy**,
`DynamicTopology.forcefield.evaluate`, and this is exactly the surface
DynamicTopology's `System` produces for a reaction that forms one EVB block on
its own.  The argument is that with one block and no environment, `System`'s
diagonal *is* each state's own evaluation, under either electrostatic term:

  * Under fragment ACKS2 each state minimizes its own functional -- its own
    reference charges `q0` (so the hopping charge sits where that state's
    bonding puts it) and a softness acting only within that state's molecules
    -- less each of its molecules' isolated minimum.  Other blocks would couple
    in through their mean charges, but there are none, so a state's
    electrostatics depends on its own topology and the geometry alone: what
    `evaluate` returns for it.
  * Under point charges each state carries its own template charges, and within
    one block `System`'s weight matrix is `sum_s w_s q_s q_s^T` -- linear in the
    weights, with no second block to couple to.

Forces are Hellmann-Feynman: `dE/dx = c^T (dH/dx) c` with `c` the ground-state
eigenvector, so the two diabatic gradients and the coupling's own combine with
the state weights `c**2` and the interference term `2*c1*c2`.  Degenerate
diabats with zero coupling are the one place this breaks down -- a real conical
intersection -- and it is not a case `fit` produces.
"""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from DynamicTopology.forcefield.evaluate import evaluate_term_dict

from .calculator import pack_results, term_dict_for


def _diagonals(atoms, states, term_dicts, bond_form):
    """Each state's `Evaluation` at `atoms`.

    `term_dicts` are the states' prebuilt `term_dict_for`, or None to build them
    here -- which a caller evaluating many geometries should not leave to this.
    """
    if term_dicts is None:
        term_dicts = [term_dict_for(p) for p in states]
    return [evaluate_term_dict(atoms, td, bond_form) for td in term_dicts]


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
        kinds = {p.global_electrostatics() for p in states}
        if len(kinds) > 1:
            raise ValueError(
                f"the two states disagree about their electrostatics ({kinds}); "
                "a dataset is evaluated under one `global_params.electrostatics`"
            )
        self.states = list(states)
        self.coupling = coupling
        self.bond_form = bond_form
        self._term_dicts = [term_dict_for(p) for p in self.states]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = atoms if atoms is not None else self.atoms

        diagonal = _diagonals(atoms, self.states, self._term_dicts, self.bond_form)
        v, coupling_forces, coupling_virial = self.coupling(
            atoms.get_positions(), atoms.pbc, np.array(atoms.cell)
        )

        hamiltonian = np.array(
            [[diagonal[0].energy, v], [v, diagonal[1].energy]], dtype=float
        )
        values, vectors = np.linalg.eigh(hamiltonian)
        c = vectors[:, 0]
        weights = c**2
        interference = 2.0 * c[0] * c[1]

        energy = float(values[0])
        forces = (
            weights[0] * diagonal[0].forces
            + weights[1] * diagonal[1].forces
            + interference * coupling_forces
        )
        virial = (
            weights[0] * diagonal[0].virial
            + weights[1] * diagonal[1].virial
            + interference * coupling_virial
        )

        self.results = pack_results(atoms, energy, forces, virial)
        self.results["statevec"] = weights
        self.results["hamiltonian"] = hamiltonian
        if self.states[0].electrostatics() is not None:
            # The ground state's weight-averaged charges under either term --
            # each state solves its own under fragment ACKS2 -- which is the
            # `qbar` DynamicTopology couples blocks through.
            self.results["charges"] = (
                weights[0] * diagonal[0].charges + weights[1] * diagonal[1].charges
            )


def diabatic_energies(
    atoms: Atoms, states, bond_form: str = "morse", term_dicts=None
) -> tuple:
    """Each state's own energy at one geometry, without mixing them.

    This is what `coupling.fit_amplitude` inverts, and it has to be the number
    the running surface puts on its diagonal -- DynamicTopology's own
    single-topology sum, which is what `EVB` calls too -- or the fitted
    amplitude reproduces a barrier nobody will evaluate.  `term_dicts`, the
    states' `term_dict_for`, saves rebuilding them on a scan over geometries.
    """
    return tuple(e.energy for e in _diagonals(atoms, states, term_dicts, bond_form))


def diabatic_forces(atoms: Atoms, states, bond_form: str = "morse", term_dicts=None):
    """`(energies, forces)` per state; `forces` is `(n_states, n_atoms, 3)`."""
    evaluations = _diagonals(atoms, states, term_dicts, bond_form)
    return (
        np.array([e.energy for e in evaluations]),
        np.array([e.forces for e in evaluations]),
    )
