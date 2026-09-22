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

**What is on the diagonal, and what is not.**  Two things depend on the bonding
pattern: the bonded terms, and the exclusions of `forcefield/exclusions.py`.
Everything else -- the whole-system ZBL, 12-6 and Coulomb sums -- is a function
of the positions and the elements alone and takes the same value on both states,
so it adds a common shift that moves the eigenvalue by exactly that constant and
leaves the eigenvectors untouched.

The Coulomb exclusion is the one that cannot simply be added and subtracted, and
it is handled in the two complementary pieces `exclusions.py` describes.  The
charges are solved once from the *unmasked* kernel, so they are identical on
both states; each state's diagonal then carries
`-ccoul * sum_{(i,j) in excl(s)} Q_i Q_j K_ij`, which is what lets the exclusion
decide which bonding pattern is lower; and the energy and forces are evaluated
once through the screen `S = 1 - sum_s w_s M^s`, which is the same quantity
summed the other way round.  That identity is not an approximation: each state's
correction is linear in its own mask, the contraction is linear in `S`, and the
adjoint that carries the charge response is linear in it too, so
`sum_s w_s dE_s/dr` collapses into the single screened evaluation exactly.
`S` is then held fixed under the derivative, as Hellmann-Feynman prescribes for
eigenvector components.
"""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .forcefield import electrostatic_evaluator, exclusions
from .forcefield.lj import LennardJones
from .forcefield.qforce import QForce
from .forcefield.zbl import ZBL


def _whole_system(pos, numbers, pbc, cell, terms, lj, zbl):
    """Lennard-Jones + ZBL over every pair: the part both states share exactly.

    Neither takes any topology, so both produce the same number on both states
    and it is computed once.  Electrostatics is *not* here, although its pair
    sum is equally topology-free: its exclusion is a screen on that sum rather
    than a separate subtraction, so the sum cannot be evaluated until the
    weights are known.  `_electrostatics` does that half.
    """
    energy, forces, virial = 0.0, np.zeros_like(pos), np.zeros((3, 3))
    if "lennardjones" in terms:
        de, df, dw = lj(pos, pbc, cell, terms)
        energy, forces, virial = energy + de, forces + df, virial + dw
    de, df, dw = zbl(pos, numbers, pbc, cell)
    return energy + de, forces + df, virial + dw


def _electrostatics(pos, pbc, cell, terms, electrostatic):
    """`(indices, Q, K, E_unscreened)` for the shared charge problem, or `None`.

    The charges come from the unmasked kernel over the combined system, so they
    are a function of the positions and the elements alone and are identical on
    every state -- which they have to be.  Solving them per state instead makes
    the surface depend on the arbitrary choice of reference state, measured at
    0.88 eV.

    Note that the combined system's ACKS2 energy is not the sum of its
    fragments': the charges equilibrate over whatever they are handed.  That is
    a real difference, and it is why this is evaluated on the combined system,
    the same way `coupling.fit` evaluates the diabats it fits the amplitude to.
    """
    if electrostatic is None:
        return None
    indices, Q, K = electrostatic.charges_and_kernel(pos, pbc, cell, terms)
    unscreened = 0.5 * electrostatic.CCOUL * float(Q @ (K @ Q))
    return indices, Q, K, unscreened


def _coulomb_screen(masks, weights, indices):
    """`S = 1 - sum_s w_s M^s` restricted to the electrostatic block's atoms.

    `None` when nothing is to be screened, which the evaluators read as the
    plain contraction.  A state with no mask contributes nothing to the sum,
    which is the right reading of "this state excludes no pairs".
    """
    if not exclusions.EXCLUDE_COULOMB or all(m is None for m in masks):
        return None
    sub = [None if m is None else m[np.ix_(indices, indices)] for m in masks]
    return exclusions.screen(sub, weights, len(indices))


def _diagonals(pos, numbers, pbc, cell, states, bonded, qforce, charges, shared):
    """Each state's `H_ss` and its gradient, minus the electrostatic pair sum.

    The electrostatic part is left out of the forces on purpose: it is screened
    rather than subtracted, so it is evaluated once with the collapsed `S`
    instead of once per state.  The *energy* does carry it -- both the shared
    unscreened contraction and this state's own correction -- because the
    diagonal is what the eigenproblem sees, and the correction is precisely what
    lets the exclusion decide which bonding pattern is lower.
    """
    shared_e, shared_f, shared_w = shared
    terms = states[0].terms
    sigma, eps = exclusions.lj_parameters(terms, len(numbers))

    diagonal = np.zeros(len(states))
    forces = np.zeros((len(states),) + pos.shape)
    virials = np.zeros((len(states), 3, 3))
    for i, params in enumerate(states):
        energy, force, virial = qforce(pos, pbc, cell, bonded[i])
        energy += params.e0 + shared_e
        force = force + shared_f
        virial = virial + shared_w

        mask = params.exclusions
        if mask is not None:
            de, df, dw = exclusions.additive(
                pos, numbers, pbc, cell, mask, sigma, eps
            )
            energy, force, virial = energy + de, force + df, virial + dw

        if charges is not None:
            indices, Q, K, unscreened = charges
            energy += unscreened
            if mask is not None and exclusions.EXCLUDE_COULOMB:
                energy += exclusions.coulomb_correction(
                    Q, K, mask[np.ix_(indices, indices)]
                )

        diagonal[i] = energy
        forces[i] = force
        virials[i] = virial
    return diagonal, forces, virials


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
        self.lj = LennardJones()
        self.zbl = ZBL()
        # One shared electrostatic evaluator: both states are built over the
        # same atoms with the same per-atom block, so they cannot disagree
        # about which term it is.
        self.electrostatics = self.states[0].electrostatics()
        self.electrostatic = electrostatic_evaluator(self.states[0])
        self._bonded = [p.bonded_terms() for p in self.states]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = atoms if atoms is not None else self.atoms

        pos = atoms.get_positions()
        pbc, cell = atoms.pbc, np.array(atoms.cell)
        numbers = atoms.get_atomic_numbers()
        terms = self.states[0].terms

        shared = _whole_system(pos, numbers, pbc, cell, terms, self.lj, self.zbl)
        charges = _electrostatics(pos, pbc, cell, terms, self.electrostatic)
        diagonal, state_forces, state_virials = _diagonals(
            pos, numbers, pbc, cell, self.states, self._bonded, self.qforce,
            charges, shared,
        )

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

        # The electrostatic pair sum, once, through the collapsed screen.  Its
        # energy is already inside `diagonal` in the unscreened-plus-correction
        # form, so only the gradient is taken from here; the two agree exactly,
        # which `tests/test_evb.py` pins.
        if charges is not None:
            screen = _coulomb_screen(
                [p.exclusions for p in self.states], weights, charges[0]
            )
            _, df, dw = self.electrostatic(pos, pbc, cell, terms, screen)
            forces = forces + df
            virial = virial + dw

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
        if self.electrostatic is not None and self.electrostatic.Q is not None:
            charge_array = np.zeros(len(atoms))
            block = self.states[0].terms[self.electrostatics]
            charge_array[block["atoms"][:, 0]] = self.electrostatic.Q
            self.results["charges"] = charge_array


def diabatic_energies(atoms: Atoms, states, bond_form: str = "morse") -> tuple:
    """Each state's own energy at one geometry, without mixing them.

    This is what `coupling.fit_amplitude` inverts, and it has to be the number
    the running surface puts on its diagonal -- the same nonbonded sums over the
    same combined system, the same exclusions, the same `E0` -- or the fitted
    amplitude reproduces a barrier nobody will evaluate.  It goes through
    `_diagonals`, which is the function `EVB` itself uses, so the two cannot
    drift.
    """
    pos = atoms.get_positions()
    pbc, cell = atoms.pbc, np.array(atoms.cell)
    numbers = atoms.get_atomic_numbers()
    terms = states[0].terms
    shared = _whole_system(pos, numbers, pbc, cell, terms, LennardJones(), ZBL())
    charges = _electrostatics(
        pos, pbc, cell, terms, electrostatic_evaluator(states[0])
    )
    energies, _, _ = _diagonals(
        pos,
        numbers,
        pbc,
        cell,
        states,
        [p.bonded_terms() for p in states],
        QForce(bond_form=bond_form),
        charges,
        shared,
    )
    return tuple(float(e) for e in energies)


def diabatic_forces(atoms: Atoms, states, bond_form: str = "morse"):
    """`(energies, forces)` per state; `forces` is `(n_states, n_atoms, 3)`.

    Unlike `EVB`, which screens the electrostatic sum once with the collapsed
    `S`, this screens it per state with that state's own `1 - M^s` -- because
    there are no weights here, and `dH_ss/dr` is what is being asked for.  The
    two agree where it matters: the electrostatic force is linear in the screen,
    so `sum_s w_s` of these is the single screened evaluation `EVB` makes.
    """
    pos = atoms.get_positions()
    pbc, cell = atoms.pbc, np.array(atoms.cell)
    numbers = atoms.get_atomic_numbers()
    terms = states[0].terms
    qforce = QForce(bond_form=bond_form)
    electrostatic = electrostatic_evaluator(states[0])

    shared = _whole_system(pos, numbers, pbc, cell, terms, LennardJones(), ZBL())
    charges = _electrostatics(pos, pbc, cell, terms, electrostatic)
    bonded = [p.bonded_terms() for p in states]
    energies, forces, _ = _diagonals(
        pos, numbers, pbc, cell, states, bonded, qforce, charges, shared
    )

    if charges is not None:
        for i, params in enumerate(states):
            weights = [1.0 if j == i else 0.0 for j in range(len(states))]
            screen = _coulomb_screen(
                [p.exclusions for p in states], weights, charges[0]
            )
            _, df, _ = electrostatic(pos, pbc, cell, terms, screen)
            forces[i] = forces[i] + df
    return energies, forces
