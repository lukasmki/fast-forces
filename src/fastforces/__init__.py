"""Fast automatic parameterization of force fields.

import fastforces as ff
from tblite.ase import TBLite

atoms = ff.build("CC#N")
params = ff.parameterize(atoms, calc_factory=lambda atm: TBLite(atm))
atoms.calc = ff.FastForces(atoms, params)

A reaction is the same call one level up: `parameterize_reaction` fits a force
field per fragment, finds the saddle with Sella and fits the EVB off-diagonal
coupling that joins the two diabatic states.

rxn = ff.parameterize_reaction(
    "[O+:1]([H:2])([H:3])[H:4].[O:5]([H:6])[H:7]"
    ">>[O:1]([H:3])[H:4].[O+:5]([H:2])([H:6])[H:7]",
    calc_factory=lambda atm: TBLite(atm),
)
atoms.calc = rxn.calculator(atoms)

A whole dataset -- molecules, reactions and how to fit them -- is one manifest
file, fitted by `fast-forces fit-manifest Water.json` or `ff.manifest.run("Water.json")`.
"""

import numpy as np
from ase import Atoms

from . import elements, io, sampling, topology
from .calculator import FastForces, evaluate
from .fit import FitConfig, as_parameters, fit
from .params import Parameters
from .topology import enumerate_terms, perceive
from . import coupling, evb, reaction  # noqa: E402  (needs `sampling` bound first)
from . import manifest  # noqa: E402
from .coupling import Coupling
from .evb import EVB
from .reaction import Reaction, ReactionParameters
from .reaction import parameterize as parameterize_reaction

__all__ = [
    "Coupling",
    "EVB",
    "FastForces",
    "FitConfig",
    "Parameters",
    "Reaction",
    "ReactionParameters",
    "as_parameters",
    "build",
    "coupling",
    "elements",
    "enumerate_terms",
    "evaluate",
    "evb",
    "fit",
    "fit_from_file",
    "io",
    "manifest",
    "parameterize",
    "parameterize_reaction",
    "perceive",
    "reaction",
    "sampling",
    "topology",
]


def build(smiles: str, seed: int = 42) -> Atoms:
    """An `Atoms` for `smiles`, carrying the SMILES and perceived connectivity.

    Both survive into the training file, which is what lets the fit be
    reproduced from that file alone: the connectivity fixes the topology and the
    SMILES drives conformer generation.
    """
    from molify import smiles2atoms

    atoms = smiles2atoms(smiles, seed=seed)
    # molify keys `info` with a str enum; normalize so the keys survive extxyz.
    info = {str(getattr(k, "value", k)): v for k, v in atoms.info.items()}
    atoms.info = info
    atoms.info["smiles"] = smiles
    if "connectivity" in atoms.info:
        atoms.info["connectivity"] = [
            [int(i), int(j), float(order)] for i, j, order in atoms.info["connectivity"]
        ]
    return atoms


def parameterize(
    atoms: Atoms,
    calc_factory,
    config: FitConfig | None = None,
    training_set: str = "training.xyz",
    initial=None,
) -> Parameters:
    """Fit a force field for `atoms` against the calculator `calc_factory` makes.

    Generates the reference data, writes it all to `training_set`, and fits.
    The written file is self-contained: `fit_from_file` reproduces the result
    from it without touching the reference calculator again.

    `initial` starts the fit from an existing force field instead of from the
    element table -- a `Parameters`, a term dict, or a path to a jsonl file.
    It is not written into `training_set`: the training file describes the
    reference data, and a starting point is an argument to the fit, not part of
    what is being fit to.
    """
    config = config or FitConfig()

    equilibrium = sampling.optimize(atoms, calc_factory, fmax=config.fmax)
    _, _, _, hessian_frames = sampling.hessian(
        equilibrium, calc_factory, delta=config.hessian_delta
    )
    hessian_matrix = io.rebuild_hessian(equilibrium, hessian_frames)

    frames = [equilibrium, *hessian_frames]
    frames += sampling.normal_mode_frames(
        equilibrium,
        calc_factory,
        hessian_matrix,
        n_frames=config.n_mode_frames,
        temperature=config.temperature,
        seed=config.seed,
    )
    frames += sampling.conformer_frames(
        equilibrium, calc_factory, n_conformers=config.n_conformers, seed=config.seed
    )
    dihedrals = sampling.rotatable_dihedrals(equilibrium)
    if dihedrals:
        frames += sampling.torsion_frames(
            equilibrium, calc_factory, dihedrals, step_deg=config.torsion_step_deg
        )

    meta = {
        "method": str(atoms.info.get("method", type(calc_factory(atoms)).__name__)),
        "smiles": atoms.info.get("smiles"),
        "charge": int(round(float(np.sum(atoms.get_initial_charges())))),
        "spin": int(atoms.info.get("spin", 0)),
        "fit_config": config.to_dict(),
    }
    io.write_training_set(training_set, frames, meta=meta)

    return fit_from_file(training_set, config=config, initial=initial)


def fit_from_file(
    path: str, config: FitConfig | None = None, initial=None
) -> Parameters:
    """Fit from a training file alone -- no reference calculator required.

    `initial` is passed through to `fit`; see it for what a starting point
    changes.
    """
    data = io.read_training_set(path)
    if config is None:
        stored = data.meta.get("fit_config")
        config = FitConfig(**stored) if stored else FitConfig()
    graph = perceive(data.equilibrium)
    return fit(
        data, enumerate_terms(data.equilibrium, graph), config=config, initial=initial
    )
