Fitting a molecule
==================

Everything in memory is in ASE units: positions in Å, energies in eV, forces in
eV/Å. Only the ``.jsonl`` files on disk use nm and kJ/mol, and the conversion
happens on read and write, inside DynamicTopology's ``io.units``.

From SMILES to a force field
----------------------------

.. code-block:: python

   import fastforces as ff
   from tblite.ase import TBLite

   atoms = ff.build("CC#N")
   params = ff.parameterize(
       atoms,
       calc_factory=lambda atm: TBLite(atm, method="GFN2-xTB", verbosity=0),
       training_set="acetonitrile.xyz",
   )
   print(params.fit_report())

:func:`fastforces.build` embeds the SMILES in 3D and stores the SMILES and the
perceived connectivity on ``atoms.info``. Both are written into the training
file, which is what lets the fit be reproduced from that file alone.

:func:`fastforces.parameterize` then runs the whole pipeline against the
reference calculator:

1. relax to the equilibrium geometry (``fmax``);
2. compute a finite-difference Hessian (``hessian_delta``);
3. sample thermal normal-mode displacements (``n_mode_frames`` at
   ``temperature``);
4. generate conformers with ``openconf`` (``n_conformers``);
5. scan every rotatable dihedral (``torsion_step_deg``);
6. write all of those frames to ``training_set`` and fit every bonded term to
   them.

The nonbonded terms (ACKS2 or fixed charges, tapered ZBL, switched 12-6) are
*not* fitted. They come from element defaults and are subtracted from the
reference as a fixed baseline, and the bonded terms fit what is left. The bond
parameters are solved nonlinearly. Every other force constant multiplies a
function of the geometry alone, so all of them together are one bounded linear
least-squares solve. The fit alternates the two until the residual stops moving
(``cycle_tol``).

Any calculator
--------------

``calc_factory`` is any callable that takes an ``Atoms`` and returns a fresh ASE
calculator. It is called with the atoms, so calculators that need the system at
construction time work too. Anything that provides energies and forces can be
fitted against: tblite, a DFT code, an ML potential.

.. code-block:: python

   from fastforces.calculators.pyscf import PySCFCalculator

   params = ff.parameterize(
       atoms,
       calc_factory=lambda atm: PySCFCalculator(xc="wb97x-v", basis="def2-svp"),
   )

The bundled :class:`~fastforces.calculators.tblite.TBLiteCalculator` is tblite's
calculator with the partial charges left on each frame. A fixed-charge fit
(below) needs those charges.

Configuring the fit
-------------------

:class:`~fastforces.fit.FitConfig` holds everything that decides the result. It
is written into the training file with the frames.

.. code-block:: python

   config = ff.FitConfig(temperature=300.0, n_conformers=10, electrostatics="fixed")
   params = ff.parameterize(atoms, calc_factory, config=config)

=====================  ===========  =============================================
field                  default      meaning
=====================  ===========  =============================================
``temperature``        ``500.0``    K, for the normal-mode displacements
``n_mode_frames``      ``40``       number of normal-mode frames
``n_conformers``       ``20``       number of ``openconf`` conformers
``torsion_step_deg``   ``15.0``     dihedral scan step
``hessian_delta``      ``0.01``     Å, finite-difference displacement
``fmax``               ``1e-3``     eV/Å, relaxation convergence
``energy_weight``      ``None``     energy vs. force weight; ``None`` balances them
``regularization``     ``1e-3``     Tikhonov penalty, scaled per column
``seed``               ``0``        sampling seed
``bond_form``          ``"morse"``  ``"morse"`` or ``"harmonic"``
``electrostatics``     ``"acks2"``  ``"acks2"`` or ``"fixed"`` (see below)
``n_cycles``           ``200``      cap on nonlinear/linear alternations
``cycle_tol``          ``1e-4``     convergence tolerance of the alternation
=====================  ===========  =============================================

``electrostatics`` picks the electrostatic model the fitted field carries:

``"acks2"``
   ACKS2 charge equilibration, from element defaults. The charges are
   re-solved at every geometry.

``"fixed"``
   One charge per atom, taken from the reference calculation's Mulliken
   populations. This is DynamicTopology's
   ``global_params.electrostatics = "pointcharge"``.

Neither model is fitted. Both are part of the baseline, so a field refit under
the other setting is a different force field, not the same one with new
numbers.

Refitting from the training file
--------------------------------

The training file (extended XYZ) contains everything needed to reproduce the
fit: the frames with their reference energies and forces, the connectivity, the
reference method and the ``FitConfig``. The Hessian is rebuilt from the
displaced frames. Refitting needs no reference calculator:

.. code-block:: python

   params = ff.fit_from_file("acetonitrile.xyz")                 # the stored config
   params = ff.fit_from_file("acetonitrile.xyz", config=config)  # override it

Use :mod:`fastforces.io` to inspect the file directly:

.. code-block:: python

   data = ff.io.read_training_set("acetonitrile.xyz")
   print(data.summary())
   data.equilibrium, data.hessian, data.of_kind("torsion")

Starting from an existing force field
-------------------------------------

:func:`~fastforces.parameterize`, :func:`~fastforces.fit_from_file` and
:func:`~fastforces.fit.fit` take ``initial=``: a
:class:`~fastforces.params.Parameters`, a term dict, or a path to a ``.jsonl``
file.

.. code-block:: python

   params = ff.parameterize(atoms, calc_factory, initial="previous.jsonl")

Terms are matched by their atom slots. Anything the supplied field does not
cover keeps its ordinary default. The supplied bond parameters are where the
nonlinear solve starts, and the supplied equilibrium values and nonbonded
baseline are held fixed. The remaining force constants come from a linear solve
that has no starting point. Refitting a converged field from its own output is
idempotent to within ``cycle_tol``.

Running a simulation
--------------------

:class:`~fastforces.calculator.FastForces` is an ASE calculator for one fitted
topology:

.. code-block:: python

   from ase import units
   from ase.md.langevin import Langevin

   atoms.calc = ff.FastForces(atoms, params)
   atoms.get_potential_energy()   # eV
   atoms.get_forces()             # eV/Å
   atoms.get_charges()            # ACKS2 or fixed charges

   Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.01).run(1000)

To score frames without replacing the calculator that holds their reference
labels, use :func:`~fastforces.calculator.evaluate`, which returns
``(energy, forces)``. :func:`~fastforces.calculator.breakdown` returns the full
DynamicTopology ``Evaluation``, with the bonded, electrostatic, ZBL and 12-6
parts broken out.

Saving and exporting
--------------------

.. code-block:: python

   params.to_jsonl("acetonitrile.jsonl")        # DynamicTopology's format (nm, kJ/mol)
   params = ff.Parameters.from_jsonl("acetonitrile.jsonl")

   # needs `uv sync --extra openmm`
   params.to_openmm_xml("acetonitrile.xml", positions=atoms.get_positions())
   system = params.to_openmm_system(positions=atoms.get_positions())

``positions`` fixes the geometry the ACKS2 charges are frozen at in the OpenMM
export. Without it, the exported system has no electrostatics.
:meth:`~fastforces.params.Parameters.to_terms` gives the term list
DynamicTopology holds in memory, for handing a field to it directly.

Examples
--------

The ``quickstart/`` directory has numbered, runnable scripts, each standalone:

.. code-block:: sh

   uv run python quickstart/01_quickstart.py

===============================  ==================================================
script                           shows
===============================  ==================================================
``01_quickstart.py``             SMILES to a simulation-ready force field
``02_topology.py``               bond-graph perception, symmetry typing, terms
``03_training_set.py``           the training file, and refitting from it alone
``04_export_formats.py``         jsonl and OpenMM XML, with an OpenMM cross-check
``05_any_calculator.py``         one molecule fitted against three references
``06_energy_decomposition.py``   what each evaluator contributes, and charges
``07_molecular_dynamics.py``     NVE and Langevin dynamics
``08_validation.py``             scoring the fit on geometries it never saw
``09_starting_point.py``         fitting from an existing field; idempotency
``10_reaction.py``               a reaction SMILES to an EVB surface
``11_fixed_charges.py``          fixed point charges instead of ACKS2
===============================  ==================================================
