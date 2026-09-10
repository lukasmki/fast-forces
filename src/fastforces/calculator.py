"""The ASE calculator that evaluates a fitted force field."""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .forcefield.acks2 import ACKS2
from .forcefield.lj import LennardJones
from .forcefield.qforce import QForce
from .forcefield.zbl import ZBL


class FastForces(Calculator):
    """`QForce + ACKS2 + LennardJones + ZBL`, plus the reference offset.

    The four evaluators divide the interaction between them with no overlap:
    `QForce` covers everything bonded, `ACKS2` the electrostatics with its
    charges re-solved at every geometry, `LennardJones` the dispersion outside
    the bonded exclusions, and `ZBL` the screened nuclear repulsion over every
    pair.  The constant `E0` puts the total on the reference method's energy
    scale.

    The `atom` terms are passed through in global index order, so the term-order
    versus global-order distinction `ACKS2` documents stays trivial here.  It is
    still real: reorder those terms and the charges follow the terms, not the
    atoms.
    """

    implemented_properties = ["energy", "free_energy", "forces", "charges"]

    def __init__(self, atoms=None, params=None, bond_form: str = "morse", **kwargs):
        super().__init__(atoms=atoms, **kwargs)
        if params is None:
            raise ValueError("FastForces needs a Parameters object")
        self.params = params
        self.qforce = QForce(bond_form=bond_form)
        self.acks2 = ACKS2()
        self.lj = LennardJones(params.exclusions)
        self.zbl = ZBL()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = atoms if atoms is not None else self.atoms

        pos = atoms.get_positions()
        pbc = atoms.pbc
        cell = np.array(atoms.cell)
        terms = self.params.terms

        energy, forces = self.qforce(pos, pbc, cell, self.params.bonded_terms())

        if "atom" in terms:
            de, df = self.acks2(pos, pbc, cell, terms)
            energy, forces = energy + de, forces + df
        if "lennardjones" in terms:
            de, df = self.lj(pos, pbc, cell, terms)
            energy, forces = energy + de, forces + df
        # The virial is dropped: `stress` is not in `implemented_properties`,
        # and neither `ACKS2` nor `LennardJones` returns one to add it to.
        de, df, _ = self.zbl(pos, atoms.get_atomic_numbers(), pbc, cell)
        energy, forces = energy + de, forces + df

        energy = energy + self.params.e0

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }
        if "atom" in terms and self.acks2.Q is not None:
            charges = np.zeros(len(atoms))
            charges[terms["atom"]["atoms"][:, 0]] = self.acks2.Q
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
