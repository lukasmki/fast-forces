Fitting a dataset from a manifest
=================================

A whole DynamicTopology dataset (every molecule, every reaction, and how to fit
them) is described by one JSON manifest. It is the same file DynamicTopology
loads. The ``fit_config`` block is read only by fast-forces, and DynamicTopology
ignores it.

.. code-block:: json

   {
       "name": "Water",
       "molecules": [
           {"id": 1, "smiles": "O",      "path": "molecules/h2o"},
           {"id": 2, "smiles": "[OH3+]", "path": "molecules/h3o"}
       ],
       "reactions": [
           {"id": 1, "smiles": "[OH3+].O>>O.[OH3+]", "path": "reactions/h3o-h2o-transfer"}
       ],
       "global_params": {"bond_asymptote": 1.0, "electrostatics": "pointcharge"},
       "fit_config": {"calculator": {"name": "tblite"}, "workdir": "training",
                      "electrostatics": "mulliken"}
   }

Fit it from the command line or from Python:

.. code-block:: sh

   fast-forces fit-manifest Water.json --dry-run   # validate and print the plan
   fast-forces fit-manifest Water.json             # fit, overwriting the outputs

.. code-block:: python

   import fastforces as ff

   manifest = ff.manifest.load("Water.json")   # raises ManifestError if invalid
   print(manifest.plan())
   outcomes = ff.manifest.run(manifest)        # or ff.manifest.run("Water.json")
   failed = [o for o in outcomes if not o.ok]

``run`` accepts ``calc_factory=`` to use a reference method other than the named
backends. If one entry fails, ``run`` records the failure and carries on, so one
reaction with no gas-phase saddle does not cost you the rest of the dataset.
The exit status of ``fit-manifest`` is 1 if any entry failed, and 2 if the
manifest is invalid.

Worked examples are in ``examples/hydrogen-combustion/`` and
``examples/proton-transfer/``.

Outputs
-------

Each entry's ``path`` is an output stem, relative to the manifest's directory:

============  ==============================================  =========================================
entry         file                                            contents
============  ==============================================  =========================================
molecule      ``<path>.jsonl``                                the force field
molecule      ``<path>.xyz``                                  equilibrium frame and metadata
reaction      ``<path>.xyz``                                  reactant, transition state and product
reaction      ``<path>.jsonl``                                the coupling
reaction      ``<path>-reactant.jsonl``, ``-product.jsonl``   the two diabatic states
============  ==============================================  =========================================

Every ``.jsonl`` is in eV and Å, the units DynamicTopology holds parameters in.

The full training sets are written under ``fit_config.workdir``, mirroring the
same paths. A rerun reuses them. A molecule whose training set exists is refit
without calling the reference calculator, and a reaction whose stationary
points exist skips the saddle search. Delete a file to regenerate it. As a
result, changes to the *sampling* settings (``n_mode_frames``,
``temperature``) only take effect once the training set is deleted, while
changes to the *fitting* settings take effect immediately. A molecule training
set computed with a different ``calculator`` is refused rather than refit, since
its energies are on another method's zero.

A reaction's ``<path>.xyz`` is also an input. If one is already there, for
example a published dataset's stationary points, its reactant, TS and product
are used as the reaction's geometry and the saddle search is skipped. Only the
geometries are used:

- The atoms are renumbered onto the mapped SMILES by matching elements and the
  bonds of both ends. A file in another atom order is accepted, and a file of
  a different reaction is refused.
- Each frame is relabelled with the reference calculator at the entry's spin,
  so the barrier is on the same zero as the fragment fits.

The labelled frames are cached like a searched path, and later runs reuse the
cache while it holds the same geometries. Because a searched path is written
to ``<path>.xyz`` too, regenerating one means deleting both files. An explicit
``frames`` file outranks the output file, and the output file outranks the
cache. ``--dry-run`` shows which source each reaction will use.

Molecules are fitted before reactions. A reaction takes its fragments from the
molecule fits wherever the manifest lists that molecule, so a water molecule
shared by three proton transfers is fitted once. Any fragment the manifest does
not list is fitted anyway, into ``<workdir>/fragments``.

``fit_config``
--------------

``fit_config`` accepts any :class:`~fastforces.fit.FitConfig` field (see
:doc:`usage`), plus:

================  ======================  ===================================================
key               default                 meaning
================  ======================  ===================================================
``calculator``    ``{"name": "tblite"}``  reference backend, see below
``workdir``       ``"training"``          where the full training sets go
``embed_seed``    ``42``                  seed for conformer embedding and the saddle guess
``eps``           ``1e-3``                coupling decay at the nearer endpoint
================  ======================  ===================================================

``calculator`` names a backend. Every other key in the block is passed to that
backend's constructor:

``{"name": "tblite", "method": "GFN2-xTB"}``
   tblite. With ``electrostatics: "mulliken"``, the reference charges are its
   xTB charges.

``{"name": "pyscf", "xc": "...", "basis": "...", "pcm": "IEF-PCM", "pcm_eps": 78.36}``
   PySCF DFT (ωB97X-V/cc-pVTZ by default). ``pcm`` adds implicit solvent,
   which is the wrong reference for a dataset meant to run in explicit solvent;
   see the ``PySCFCalculator`` docstring.

Charge and spin come from each geometry, so one manifest can fit cations,
anions and radicals against the same method.

Entries
-------

``smiles``
   A molecule SMILES, or a ``reactants>>products`` reaction SMILES. Reaction
   SMILES need no atom map. Heavy atoms correspond by order of appearance, and
   hydrogens are assigned to change as few bonds as possible
   (:func:`~fastforces.reaction.map_atoms`). So ``[OH3+].O>>O.[OH3+]`` is the
   Grotthuss proton transfer.

``spin``
   ``2S``, the number of unpaired electrons, used only during fitting. It is
   read from the SMILES radicals if omitted. On a reaction it sets the saddle
   search and both endpoint relaxations. By default every reactant radical is
   coupled high-spin, so ``[O].[OH]`` runs at 3.

``spin_r``, ``spin_ts``, ``spin_p``
   Override the reactant, transition state and product spins individually. A
   fission is fitted from its reactant alone, so it accepts only ``spin`` or
   ``spin_r``.

``frames``
   An extxyz of reactant, TS and product that skips the saddle search. A
   barrierless gas-phase channel needs this. Unlike an output ``<path>.xyz``,
   it is taken as it stands: it has to be in the reaction's atom order and
   carry energies from the manifest's reference method.

``amplitude``
   Fix the coupling amplitude (eV) instead of fitting it.

.. code-block:: json

   {"id": 2, "smiles": "[H:1][H:2].[O:3]>>[O:3][H:2].[H:1]", "path": "reactions/rxn_02",
    "spin": 2, "spin_p": 0}

Everything is validated at load time, before any reference calculation runs:
unknown keys, duplicate outputs, reaction SMILES that cannot be mapped, and
spins the electron count does not allow. A cached training set or ``frames``
file computed at a different spin is refused rather than reused.

``global_params``
-----------------

``global_params`` is DynamicTopology's own block (the taper radii, exclusion
depth, Morse asymptote, charge width and electrostatics model; see
``DynamicTopology.forcefield.params.ForceFieldParams``). It is *applied*: every
entry is fitted under these values, so the dataset is fitted on exactly the
surface it will be simulated on. Values it leaves out take DynamicTopology's
defaults.

``global_params.electrostatics`` and ``fit_config.electrostatics`` are the two
halves of one choice:

``global_params.electrostatics``
   How a simulation uses the charges: ``"acks2"`` (the default), fragment ACKS2
   equilibrated around each atom's reference charge ``q0``, or
   ``"pointcharge"``, the charges held fixed.

``fit_config.electrostatics``
   Where the charges come from: ``"mulliken"`` (the reference calculator's
   populations), ``"esp"`` (Merz-Kollman charges fitted to the reference
   density; needs the ``pyscf`` calculator) or ``"neutral"`` (zero on every
   atom, the default).

Under ``acks2`` the reference charges are what keep an ion's formal charge on
it, so a charged molecule under ``"neutral"`` is refused when it is fitted.
``pointcharge`` with ``"neutral"`` is refused when the manifest is loaded, since
it would put no charge anywhere.
