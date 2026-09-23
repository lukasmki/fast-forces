Command line
============

Installing the package provides a ``fast-forces`` command (``uv run
fast-forces`` from the repository). Every subcommand takes ``--help``.

.. code-block:: sh

   fast-forces fit CC#N -o acetonitrile.jsonl
   fast-forces fit-manifest examples/hydrogen-combustion/hydrogen-combustion.json --dry-run
   fast-forces refit datasets/HCombustion/HCombustion.json --force-constants
   fast-forces label -i rxn.xyz -o rxn.xyz
   fast-forces import-qforce -i qforce.xml -o mol.jsonl

============================  ===================================================
command                       does
============================  ===================================================
:ref:`fit <cli-fit>`          fit one molecule from a SMILES string (tblite)
:ref:`fit-manifest <cli-fm>`  fit every molecule and reaction a manifest lists
:ref:`refit <cli-refit>`      refit an existing dataset's bonds and couplings
:ref:`label <cli-label>`      compute reference atomization energies (PySCF)
:ref:`import-qforce <cli-q>`  convert q-force XML to ``.jsonl`` term rows
============================  ===================================================

The first two fit from a reference calculator. ``refit``, ``label`` and
``import-qforce`` are dataset maintenance tools that used to live in
DynamicTopology's ``scripts/`` directory as ``fit.py``, ``compute.py`` and
``convert.py``.

.. _cli-fit:

fast-forces fit
---------------

.. program:: fast-forces fit

.. code-block:: sh

   fast-forces fit SMILES [OPTIONS]

Fits one molecule against tblite, running the full
:func:`~fastforces.parameterize` pipeline, and prints the fit report.

.. option:: -o, --output <path>

   ``.jsonl`` output. Default ``forcefield.jsonl``.

.. option:: --xml <path>

   Also write an OpenMM XML system (needs the ``openmm`` extra).

.. option:: --training-set <path>

   Where to write the training set. Default ``training.xyz``.

.. option:: --method <str>

   tblite method. Default ``GFN2-xTB``.

.. option:: --temperature <K>

   Normal-mode sampling temperature. Default ``500.0``.

.. option:: --mode-frames <int>

   Number of normal-mode frames. Default ``40``.

.. option:: --initial <path>

   ``.jsonl`` force field to start from, instead of the element table.

.. _cli-fm:

fast-forces fit-manifest
------------------------

.. program:: fast-forces fit-manifest

.. code-block:: sh

   fast-forces fit-manifest MANIFEST [--dry-run]

Fits every molecule and then every reaction in a manifest, and writes each
entry to its ``path``. See :doc:`manifest` for the file format. Exits 2 if the
manifest is invalid and 1 if any entry failed.

.. option:: --dry-run

   Validate the manifest and print what would be fitted, without fitting.

.. _cli-refit:

fast-forces refit
-----------------

.. program:: fast-forces refit

.. code-block:: sh

   fast-forces refit MANIFEST [OPTIONS]

This is the dataset-level half of the pipeline. ``fit-manifest`` fits each
molecule from a reference calculator. ``refit`` instead takes the dataset as it
stands, with templates on disk and stationary points in each reaction's
``.xyz``, and refits what depends on the dataset as a whole: the bonds, then one
coupling per reaction. It writes the same files.

For each reaction:

* the **amplitude** is fitted by inverting the 2×2 secular equation at the
  transition state. This needs a reference energy on that frame (see
  ``label``), and falls back to ``--amplitude`` if there is none;
* the **width** is fitted from the endpoint geometries, so that the coupling is
  switched off at the reactant and product minima.

Inverting the secular equation has a real root only where the reference barrier
lies below both diabats. ``--force-constants`` refits the bonds first so that it
does.

.. warning::

   ``--force-constants`` is not idempotent, although a coupling-only refit is.
   Re-running it over already-fitted output moves the result again, and
   ``--max-k-scale`` compounds. Refit once from the previous state rather than
   iterating.

Coupling options:

.. option:: --eps <float>

   Coupling decay at the nearer endpoint. Default ``0.001``.

.. option:: --bimol-cutoff <Å>

   Separation past which a bimolecular channel is no longer enumerated. Must
   match the value the simulation runs with. Default ``4.0``.

.. option:: --amplitude <eV>

   Amplitude for transition states with no reference energy. Without it, such
   reactions are reported and skipped.

.. option:: --rmsd-width

   Measure atom-transfer widths in RMSD rather than in the transferring atom's
   triangle. Reproduces a pre-``fit_threebody`` baseline.

.. option:: --refit-manual

   Also overwrite amplitudes marked ``provenance: manual``. By default only
   their widths are refitted.

.. option:: --margin <eV>

   How far below the reference barrier a diabat must sit for the channel to
   count as fittable. Default ``0.02``.

.. option:: -n, --dry-run

   Report without writing.

Bond options:

.. option:: --bonds

   Rescale each template's Morse well depths so its bonds carry its full
   atomization energy. Do this before fitting couplings.

.. option:: --force-constants

   Refit the Morse force constants as well as the depths. Implies ``--bonds``.

.. option:: --fit-mode <asymptote|k|asymptote-k>

   ``asymptote`` (the default) fits each bond type's asymptote height ``h`` and
   leaves the force constants alone. ``k`` stiffens the bonds instead.
   ``asymptote-k`` does both.

.. option:: --max-asymptote <eV>

   Upper bound on the per-bond asymptote height. Default ``10.0``.

.. option:: --max-k-scale <float>

   Hard bound on each force-constant scale, relative to the ``.jsonl`` as it
   stands. Default ``2.0``; ``1.0`` freezes the force constants.

.. option:: --frequency-weight <float>

   Pull back towards the original force constants. Default ``0.005``.

.. option:: --max-wavenumber <cm^-1>

   Stretching modes above this are penalized. It encodes the MD timestep: a dt
   of 0.5 fs needs everything under about 4450. Default ``4400.0``.

.. option:: --curvature-weight <float>

   How much a mode over ``--max-wavenumber`` costs; ``0`` removes the cap.
   Default ``0.01``.

.. _cli-label:

fast-forces label
-----------------

.. program:: fast-forces label

.. code-block:: sh

   fast-forces label -i rxn.xyz -o rxn.xyz -b aug-cc-pvtz -c 1 --atom-cache atoms.json

Labels structures with ωB97X-V atomization energies,
``E(molecule) - sum_i E(atom_i)``, computed with PySCF (negative for a bound
molecule). These are the ``energy=`` values a DynamicTopology dataset carries on
every template and reaction frame, which ``refit`` fits the Morse depths and
coupling amplitudes against. Output is an extxyz with the energies in eV.

.. option:: -i, --input <file>

   Structures to label (required).

.. option:: -o, --output <path>

   Labelled extxyz output (required).

.. option:: -b, --basis <str>

   Basis set. Default ``cc-pvtz``.

.. option:: -c, --charge <int>

   Total charge, applied to every structure. Default ``0``.

.. option:: -s, --spin <int>

   Unpaired electrons (``2S``, not the multiplicity), applied to every
   structure. Default ``0``.

.. option:: --atom-cache <path>

   JSON file that persists free-atom energies between runs.

.. option:: --grid-level <int>, --nlc-grid-level <int>

   DFT integration grids. Defaults ``3`` and ``1``.

.. option:: --density-fit

   Use RI for the two-electron integrals: faster, with a small fitting error.

.. option:: --nthreads <int>

   PySCF thread count.

.. option:: -v, --verbose <int>

   PySCF verbosity. Default ``0``.

.. _cli-q:

fast-forces import-qforce
-------------------------

.. program:: fast-forces import-qforce

.. code-block:: sh

   fast-forces import-qforce -i qforce_xml_dir/ -o molecules/

Reads q-force's OpenMM-style ``<Forces>`` XML into ``.jsonl`` term rows, in
q-force's own units (nm, kJ/mol), which are also the units ``.jsonl`` stores.
Forces with unnamed parameters are dropped, q-force's ``Coulomb`` among them.
The 12-6 comes out as q-force's ``A``/``B`` rather than ``sigma``/``eps``. An
imported file is therefore a starting point for ``refit``, not a finished
template.

.. option:: -i, --input <path>

   q-force XML file or directory (required).

.. option:: -o, --output <path>

   ``.jsonl`` file or directory (required).
