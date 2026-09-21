"""Fixed point-charge electrostatics: the same kernel as ACKS2, no solve.

**What this term is for.**  `ACKS2` re-solves a `2n+2` linear system at every
geometry to get its charges, and then pays for it twice: once in the solve, and
once in the `dQ/dr` adjoint that `ACKS2.compute_response_forces` exists to
supply.  That machinery buys a real physical effect -- charges that redistribute
as bonds stretch and as molecules approach -- and it is what a reaction needs.
It is not what a fixed topology needs, and the price is steep: the response term
is the single easiest piece of the whole force field to get subtly wrong, and
the charges it produces come from element defaults that have never been fit to
anything (`elements.py` says so outright).

So this is the other end of the trade.  Each atom carries a charge as an
ordinary per-atom parameter, `q`, and the energy

    E = CCOUL/2 * sum_ij q_i q_j K_ij

is a function of the geometry alone.  There is no linear system, no adjoint, and
`ewald.coulomb_sum` is the entire gradient rather than half of it.

`q` is not *fitted*, any more than the ACKS2 block it replaces is: like every
nonbonded parameter here it is part of the baseline the bonded terms are fit
against.  `fit._coulomb_block` seeds it from the reference calculation's
Mulliken populations, which is the crudest population analysis available and is
explicitly a starting point -- fitting charges against reference electrostatics
is the same future work `elements.py` records for the ACKS2 softness.

**The kernel is deliberately the same one.**  `erf(GAMMA r) / r`, smeared at
`GAMMA = 2.0` per Angstrom, not bare `1/r`.  Three reasons, and the third is the
one that settles it:

  * Bare `1/r` between fixed charges at a bond length is tens of eV with a
    matching gradient, and nothing in this force field is set up to absorb that
    -- `ZBL` and `LennardJones` take no exclusions, so the bonded pairs are
    already carrying two unscreened nonbonded terms apiece.
  * `export.openmm` already bakes frozen charges into `erf(beta*r)/r` with
    `beta = 2.0`, so this is the form the exported system has always had.  A
    term using a different kernel would export to something that is not itself.
  * The Ewald machinery in `ewald.py` is written for this kernel.  Reusing it
    means the periodic lattice sum, the forces and the virial are the ones that
    `tests/test_forcefield.py` already checks against finite differences, not a
    second implementation to keep in step.

**Charge neutrality is the caller's problem.**  `ewald` omits the `k = 0`
reciprocal term, which is only legitimate when `sum_i q_i = 0` -- see its module
docstring.  `ACKS2` imposes that as a hard constraint; this term carries
whatever it is handed.  Mulliken charges sum to the total molecular charge by
construction, so a neutral molecule satisfies it exactly and an ion does not, and
a periodic evaluation of an ion is missing the neutralizing background.  That is
warned about once per evaluator rather than raised: an isolated ion is a
perfectly ordinary thing to evaluate under open boundaries, where the `k = 0`
term does not exist at all.
"""

import warnings

import numpy as np

from .ewald import CCOUL, KernelCache, coulomb_sum

# Net charge, in e, above which a periodic evaluation is worth complaining
# about.  Loose enough that float error in a set of charges that is meant to
# sum to zero never trips it, tight enough that a real +1 does.
NEUTRALITY_TOL = 1e-6


class Coulomb:
    """Smeared Coulomb between fixed per-atom charges.

    `q` is in units of the elementary charge and `CCOUL` is in eV*Angstrom, so
    the energy, forces and virial come back in eV and Angstrom -- the units
    `Parameters.terms` holds, with nothing converted anywhere in here.

    Two index spaces meet here, as in `ACKS2` and `LennardJones`, and confusing
    them is silent: `q` arrives in *term order* -- the order the `coulomb` terms
    were collected -- while `pos` and the returned forces are in *global* atom
    order.  `indices[k]` is the global index of the k-th term.  Everything is
    built in term order and the forces are scattered back at the very end.

    Atoms with no `coulomb` term are simply absent from the sum, which is how a
    field can charge a solute and leave a bath neutral.
    """

    CCOUL = CCOUL

    def __init__(self):
        self.Q = None
        self.kernels = KernelCache()
        self._warned = False

    def _check_neutrality(self, Q, pbc):
        """Warn once if a periodic sum is being asked for on a charged system."""
        if self._warned or not np.all(pbc):
            return
        net = float(np.sum(Q))
        if abs(net) > NEUTRALITY_TOL:
            warnings.warn(
                f"periodic Coulomb sum with a net charge of {net:+.4f} e: the "
                "Ewald k=0 term is omitted, so the energy is missing the "
                "neutralizing background (see forcefield/ewald.py)",
                stacklevel=3,
            )
            self._warned = True

    def __call__(
        self, pos: np.ndarray, pbc: np.ndarray, cell: np.ndarray, term_dict: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        pbc = np.asarray(pbc, dtype=bool)
        params = term_dict.get("coulomb")
        if params is None:
            raise KeyError("No `coulomb` parameters set")
        indices = params["atoms"][:, 0]
        Q = np.asarray(params["kwargs"]["q"], dtype=float)

        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            F = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(F + 0.5)) @ cell

        # Both axes in term order, so the charge vector lines up with them.
        sub = np.ix_(indices, indices)
        vecs = vecs[sub]
        rij = np.sqrt(np.sum(vecs * vecs, -1))

        self._check_neutrality(Q, pbc)
        kernel = self.kernels.get(pos[indices], vecs, rij, pbc, cell)
        e_tot, f_tot, w_tot = coulomb_sum(Q, kernel, self.CCOUL)

        # Published the way `ACKS2.Q` is, so the calculator can report charges
        # from either term without knowing which one it has.
        self.Q = Q

        # Scatter term-ordered forces back to global atom order.  The virial
        # needs no scatter: it is a single 3x3 sum over pairs, not a per-atom
        # quantity, so term order and global order give the same matrix.
        forces = np.zeros_like(pos)
        forces[indices] = f_tot
        return e_tot, forces, w_tot
