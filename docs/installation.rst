Installation
============

fast-forces needs Python 3.13 or newer and is managed with
`uv <https://docs.astral.sh/uv/>`_. It depends on DynamicTopology as an
editable path dependency (``../DynamicTopology``), so check the two repositories out
side by side:

.. code-block:: text

   Projects/
   ├── DynamicTopology/     # the force field and reactive MD
   └── fast-forces/     # this package

Then, from ``fast-forces/``:

.. code-block:: sh

   uv sync                  # runtime dependencies and the dev group
   uv run fast-forces --help

Because DynamicTopology is installed editable, changes to it are picked up
without reinstalling.

Optional: OpenMM export
-----------------------

Exporting a fitted force field to an OpenMM ``System`` needs the ``openmm``
extra:

.. code-block:: sh

   uv sync --extra openmm

Pass ``--extra openmm`` on every later ``uv sync`` too. A plain ``uv sync``
removes packages that are not declared for the current sync, OpenMM included.

Reference calculators
---------------------

Two reference backends are installed with the package, and the CLI and manifest
fitter can use either by name:

* **tblite** (GFN2-xTB and relatives): fast and the default. It is what the
  quickstart scripts use.
* **PySCF**: DFT (ωB97X-V/cc-pVTZ by default), optionally with implicit
  solvent. It is also used by ``fast-forces label``.

In Python, any ASE calculator works. See :doc:`usage`.

Building these docs
-------------------

.. code-block:: sh

   uv run make -C docs html     # output in docs/_build/html
