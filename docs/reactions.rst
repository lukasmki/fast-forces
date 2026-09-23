Fitting a reaction
==================

:func:`fastforces.parameterize_reaction` (``fastforces.reaction.parameterize``)
takes an atom-mapped reaction SMILES and returns a two-state EVB surface. It
contains a force field for each side of the arrow over the same atom indices, a
transition state located with `Sella <https://github.com/zadorlab/sella>`_, and
the off-diagonal coupling that joins the two diabatic states.

.. code-block:: python

   import fastforces as ff
   from tblite.ase import TBLite

   rxn = ff.parameterize_reaction(
       "[Cl-:1].[C:2]([H:3])([H:4])([H:5])[Cl:6]"
       ">>[Cl:1][C:2]([H:3])([H:4])([H:5]).[Cl-:6]",
       calc_factory=lambda atm: TBLite(atm),
       workdir="sn2",
   )
   print(rxn.report())
   print(rxn.energies())      # reference vs. EVB at reactant, TS and product

   atoms = rxn.frames[1].copy()
   atoms.calc = rxn.calculator(atoms)   # an ASE calculator that crosses the barrier
   rxn.write("sn2-chloride")            # .xyz, coupling .jsonl, one .jsonl per state

Write every hydrogen out explicitly and give it a map number. An implicit
hydrogen has no index, and in a transfer reaction the atom that moves is usually
a hydrogen. (The manifest fitter is more lenient; see :doc:`manifest`.)

The three stages
----------------

1. **Fragments.** Each distinct molecule on either side gets its own
   :func:`fastforces.parameterize` against the same reference calculator. This
   is the expensive stage. It is cached per molecule, so a species that
   appears on both sides is fitted once.
2. **Stationary points.** A saddle guess is built from the bonds that change
   and refined with Sella at ``order=1``. The two endpoints are relaxed down
   from the saddle along its imaginary mode, so all three frames lie on one
   reaction path in one atom order.
3. **Coupling.** The change in connectivity picks the coupling form:

   ==============  =====================  ================================
   form            used for               width measured in
   ==============  =====================  ================================
   ``threebody``   atom transfer          the transferring atom's triangle
   ``twobody``     barrierless fission    the breaking bond's length
   ``rmsd``        anything else          RMSD to the TS geometry
   ==============  =====================  ================================

   The amplitude comes from inverting the 2×2 secular equation at the saddle
   against the reference barrier, so the fit reproduces the barrier by
   construction. For a fission, the amplitude comes from the diabatic crossing
   instead. The width is set so the coupling has decayed to ``eps`` at the
   nearer endpoint.

The endpoints are not fitted, which makes them the real test of the surface.
The coupling has switched off there, so what remains is the two diabatic force
fields and the unfitted nonbonded baseline between them.

Reusing and supplying stages
----------------------------

Either expensive stage can be passed in instead of recomputed:

``fragments=``
   Existing fits, as :func:`~fastforces.reaction.fit_fragments` returns them.
   Any molecule the dict lacks is still fitted.

``frames=``
   ``[reactant, transition state, product]``, or the reactant alone for a
   fission. They must be in the reaction's atom order and carry reference
   energies. Use this for a constrained path, a higher-level calculation, or a
   channel with no gas-phase saddle.

All stages must share one reference method. Each fragment's ``E0`` puts its
diabatic energy on that method's absolute scale, and the barrier the amplitude
is fitted to must be on the same scale.

A channel whose diabats never separate has no saddle.
``[H3O]+ + H2O`` is a single well in the gas phase, for example. In that case
:func:`~fastforces.reaction.stationary_points` raises
:class:`~fastforces.reaction.ReactionError` instead of fitting a coupling to a
path that does not exist. Supply the frames yourself, or call
:func:`fastforces.coupling.fit_threebody` directly.

Spin
----

``spins=`` gives ``2S`` (the number of unpaired electrons, *not* the
multiplicity) per frame kind; see :func:`~fastforces.reaction.frame_spins`.
Frames it leaves out default to every reactant radical coupled high-spin. It
applies only to frames computed in this call. Supplied ``frames`` keep the spin
they were computed at, and fragments are fitted at their own.

The EVB calculator
------------------

:class:`~fastforces.evb.EVB` is the calculator ``rxn.calculator`` returns. Its
energy is the lower eigenvalue of

.. math::

   H = \begin{pmatrix} H_1 & V \\ V & H_2 \end{pmatrix},

where each diagonal entry is DynamicTopology's single-topology energy for that
state, and ``V`` is the fitted coupling, a function of the geometry alone.
``results["statevec"]`` is the squared ground-state eigenvector: how much of
each diabatic state the current geometry contains.
