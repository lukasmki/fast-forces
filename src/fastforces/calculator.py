"""The ASE calculator that evaluates a fitted force field."""

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from DynamicTopology.forcefield.evaluate import evaluate_term_dict, term_dict


def term_dict_for(params) -> dict:
    """`params` as DynamicTopology's vectorized `term_dict`, exclusions derived.

    Built once per force field rather than per call: the exclusions are a
    function of the bond graph, which a force field does not change over its
    life.
    """
    return term_dict(Atoms(numbers=params.numbers), params.to_terms())


class FastForces(Calculator):
    """One fitted force field, evaluated by DynamicTopology.

    The energy is `DynamicTopology.forcefield.evaluate` -- the single-topology
    sum `System` puts on a diabat: `QForce` for everything bonded, the reference
    shift and the ZBL and 12-6 exclusions; the electrostatics; and the
    whole-system `ZBL` and switched 12-6.  So a
    field fitted and checked here is scored exactly as the reactive simulation
    will score that molecule.

    Which electrostatic term is a property of the force field: an `atom` block
    means fragment ACKS2, with its charges re-solved at every geometry around
    the block's reference charges `q0`, and a `charge` block means fixed
    charges.  DynamicTopology's `evaluate` selects the matching term from the
    block present.  One molecule on its own scores zero under ACKS2 -- its
    energy is taken relative to its own isolated minimum -- so the ACKS2 part
    of this is nonzero only between molecules.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "charges"]

    def __init__(self, atoms=None, params=None, bond_form: str = "morse", **kwargs):
        super().__init__(atoms=atoms, **kwargs)
        if params is None:
            raise ValueError("FastForces needs a Parameters object")
        self.params = params
        self.bond_form = bond_form
        self._term_dict = term_dict_for(params)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = atoms if atoms is not None else self.atoms

        result = evaluate_term_dict(atoms, self._term_dict, self.bond_form)
        self.results = pack_results(atoms, result.energy, result.forces, result.virial)
        if self.params.electrostatics() is not None:
            self.results["charges"] = result.charges


def pack_results(atoms: Atoms, energy: float, forces, virial) -> dict:
    """ASE `results` for one evaluation: energy, forces, and stress if periodic.

    Stress only with a cell: without one the volume it divides by is zero, and
    ASE's own convention is that an isolated molecule has no stress rather than
    an infinite one.  Symmetrized because the analytic virials are built from
    outer products that are symmetric only up to rounding, and ASE's Voigt
    packing reads the upper triangle and would silently keep the asymmetry.
    """
    results = {"energy": energy, "free_energy": energy, "forces": forces}
    volume = atoms.get_volume() if atoms.cell.rank == 3 else 0.0
    if volume > 0.0:
        results["stress"] = full_3x3_to_voigt_6_stress(
            0.5 * (virial + virial.T) / volume
        )
    return results


def evaluate(atoms: Atoms, params, bond_form: str = "morse"):
    """`(energy, forces)` for one frame, without disturbing its calculator.

    Frames from `sampling` carry a `SinglePointCalculator` holding the reference
    labels; attaching `FastForces` to them would overwrite it, so this evaluates
    on a copy.
    """
    work = atoms.copy()
    work.calc = FastForces(work, params, bond_form=bond_form)
    return work.get_potential_energy(), work.get_forces()


def breakdown(atoms: Atoms, params, bond_form: str = "morse"):
    """The full `DynamicTopology.forcefield.evaluate.Evaluation` for one frame.

    Energy, forces and virial, and the four parts they are the sum of.
    """
    return evaluate_term_dict(atoms, term_dict_for(params), bond_form)


__all__ = ["FastForces", "breakdown", "evaluate", "pack_results", "term_dict_for"]
