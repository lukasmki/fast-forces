Top-level package
=================

.. module:: fastforces

``import fastforces as ff`` gives the whole single-molecule workflow:

.. code-block:: python

   atoms = ff.build("CC#N")
   params = ff.parameterize(atoms, calc_factory=lambda atm: TBLite(atm))
   atoms.calc = ff.FastForces(atoms, params)

Pipeline
--------

.. autofunction:: fastforces.build

.. autofunction:: fastforces.parameterize

.. autofunction:: fastforces.fit_from_file

Re-exports
----------

These names are importable from ``fastforces`` directly, and each is documented
on its own module's page.

* :mod:`fastforces.calculator`: ``FastForces``, ``evaluate``
* :mod:`fastforces.fit`: ``FitConfig``, ``fit``, ``as_parameters``
* :mod:`fastforces.params`: ``Parameters``
* :mod:`fastforces.topology`: ``perceive``, ``enumerate_terms``
* :mod:`fastforces.reaction`: ``Reaction``, ``ReactionParameters``,
  ``parameterize_reaction`` (``reaction.parameterize``)
* :mod:`fastforces.coupling`: ``Coupling``
* :mod:`fastforces.evb`: ``EVB``

The submodules ``coupling``, ``elements``, ``evb``, ``io``, ``manifest``,
``reaction``, ``sampling`` and ``topology`` are imported with the package, so
``ff.manifest.run(...)`` works without a separate import.
