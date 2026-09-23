fast-forces
===========

**fast-forces** parameterizes force fields automatically. Give it a SMILES
string and any ASE calculator that returns energies and forces, and it generates
its own reference data, fits every bonded term against it and returns a force
field you can simulate with straight away. The same pipeline extends to
reactions (two diabatic force fields, a Sella transition state and a fitted EVB
coupling) and to whole datasets described by one JSON manifest.

.. code-block:: python

   import fastforces as ff
   from tblite.ase import TBLite

   atoms = ff.build("CC#N")
   params = ff.parameterize(atoms, calc_factory=lambda atm: TBLite(atm))
   atoms.calc = ff.FastForces(atoms, params)

   print(atoms.get_potential_energy())  # eV

.. note::

   fast-forces fits parameters for
   `DynamicTopology <https://github.com/lukasmki/DynamicTopology>`_, the
   reactive MD package, and carries no force field of its own. Every energy it
   fits against is ``DynamicTopology.forcefield.evaluate``, the same
   single-topology sum DynamicTopology puts on each diabatic state. A template
   is therefore scored during the fit exactly as the simulation will score it.
   The datasets fast-forces writes load straight into DynamicTopology.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   usage
   reactions
   manifest
   cli

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
