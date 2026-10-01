Installation
============

fast-forces needs Python 3.13 or newer and is managed with
`uv <https://docs.astral.sh/uv/>`_. It depends on DynamicTopology, which uv
fetches from `GitHub <https://github.com/lukasmki/DynamicTopology>`_ at the
commit pinned in ``uv.lock``, so no separate checkout is needed. From
``fast-forces/``:

.. code-block:: sh

   uv sync                  # runtime dependencies and the dev group
   uv run fast-forces --help

To move to the latest DynamicTopology ``main``:

.. code-block:: sh

   uv lock --upgrade-package dynamictopology
   uv sync

Developing against a local DynamicTopology
------------------------------------------

To pick up uncommitted changes to a DynamicTopology checkout, install it
editable over the pinned copy and skip the sync that would undo it:

.. code-block:: sh

   uv pip install -e ../DynamicTopology
   uv run --no-sync fast-forces --help

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
