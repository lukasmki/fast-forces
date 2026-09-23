Calculators and export
======================

Evaluating a fitted force field
-------------------------------

.. automodule:: fastforces.calculator

Reference calculators
---------------------

Backends the CLI and manifest fitter can select by name. In Python, any ASE
calculator works as a reference.

.. automodule:: fastforces.calculators.tblite

.. automodule:: fastforces.calculators.pyscf

OpenMM export
-------------

Requires the ``openmm`` extra (``uv sync --extra openmm``).

.. automodule:: fastforces.export.openmm
