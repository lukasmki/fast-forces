"""The ASE calculator that evaluates a fitted force field."""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .forcefield import electrostatic_evaluator
from .forcefield.lj import LennardJones
from .forcefield.qforce import QForce
from .forcefield.zbl import ZBL


class FastForces(Calculator):
    """`QForce + electrostatics + LennardJones + ZBL`, plus the reference offset.

    The four evaluators divide the interaction between them with no overlap:
    `QForce` covers everything bonded, one electrostatic term the charges, `ZBL`
    the screened nuclear repulsion at short range, and `LennardJones` the
    repulsion and dispersion at long range.

    Which electrostatic term is a property of the force field, not of the
    calculator: an `atom` block means `ACKS2`, with its charges re-solved at
    every geometry, and a `coulomb` block means `Coulomb`, with the charges
    carried as fitted parameters.  `Parameters.electrostatics` picks between
    them and rejects a field carrying both.

    The last two are the ones with no topology at all: neither takes exclusions,
    so both are evaluated over every pair including bonded ones.  They hand over
    to each other rather than overlapping -- `zbl.taper` switches ZBL off at
    1.5 A and `lj.switch` switches the 12-6 on at 2.2 A -- and what they
    contribute at a bond length is absorbed by the fitted Morse depths, which is
    why `fit` pre-compresses `r0`.

    All four read `params.terms` as it stands -- eV and Angstrom, the units the
    fit works in -- so nothing is converted anywhere in the evaluation path.
    Only `QForce` is handed a filtered view, `bonded_terms()`, and only to keep
    it off the `reference` block: `compute_reference` would evaluate `E0` a
    second time.  It is built once here rather than per call, because a
    calculator's parameters do not change over its life.

    The per-atom terms are passed through in global index order, so the
    term-order versus global-order distinction both electrostatic evaluators
    document stays trivial here.  It is still real: reorder those terms and the
    charges follow the terms, not the atoms.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "charges"]

    def __init__(self, atoms=None, params=None, bond_form: str = "morse", **kwargs):
        super().__init__(atoms=atoms, **kwargs)
        if params is None:
            raise ValueError("FastForces needs a Parameters object")
        self.params = params
        self.qforce = QForce(bond_form=bond_form)
        self.lj = LennardJones()
        self.zbl = ZBL()
        # Built once, and kept: both electrostatic evaluators cache state
        # across calls, and a calculator's parameters do not change over its
        # life, so the choice between them cannot either.
        self.electrostatics = params.electrostatics()
        self.electrostatic = electrostatic_evaluator(params)
        self._bonded = params.bonded_terms()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = atoms if atoms is not None else self.atoms

        pos = atoms.get_positions()
        pbc = atoms.pbc
        cell = np.array(atoms.cell)
        terms = self.params.terms

        energy, forces, virial = self.qforce(pos, pbc, cell, self._bonded)

        if self.electrostatic is not None:
            de, df, dw = self.electrostatic(pos, pbc, cell, terms)
            energy, forces, virial = energy + de, forces + df, virial + dw
        if "lennardjones" in terms:
            de, df, dw = self.lj(pos, pbc, cell, terms)
            energy, forces, virial = energy + de, forces + df, virial + dw
        de, df, dw = self.zbl(pos, atoms.get_atomic_numbers(), pbc, cell)
        energy, forces, virial = energy + de, forces + df, virial + dw

        energy = energy + self.params.e0

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }
        # Every term now returns `dE/d(strain)`, so the stress is available for
        # the first time.  It is only meaningful with a cell: without one the
        # volume it divides by is zero, and ASE's own convention is that an
        # isolated molecule has no stress rather than an infinite one.
        volume = atoms.get_volume() if atoms.cell.rank == 3 else 0.0
        if volume > 0.0:
            # Symmetrized because the analytic virials are built from outer
            # products that are symmetric only up to rounding, and ASE's Voigt
            # packing reads the upper triangle and would silently keep the
            # asymmetry.
            self.results["stress"] = full_3x3_to_voigt_6_stress(
                0.5 * (virial + virial.T) / volume
            )
        # Both evaluators publish their charges as `.Q`, in term order, so
        # this reads the same either way -- solved for by `ACKS2`, carried as
        # parameters by `Coulomb`.
        if self.electrostatic is not None and self.electrostatic.Q is not None:
            charges = np.zeros(len(atoms))
            indices = terms[self.electrostatics]["atoms"][:, 0]
            charges[indices] = self.electrostatic.Q
            self.results["charges"] = charges


def evaluate(atoms: Atoms, params, bond_form: str = "morse"):
    """`(energy, forces)` for one frame, without disturbing its calculator.

    Frames from `sampling` carry a `SinglePointCalculator` holding the reference
    labels; attaching `FastForces` to them would overwrite it, so this evaluates
    on a copy.
    """
    work = atoms.copy()
    work.calc = FastForces(work, params, bond_form=bond_form)
    return work.get_potential_energy(), work.get_forces()
