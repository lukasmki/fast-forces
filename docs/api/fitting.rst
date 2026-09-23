Fitting a molecule
==================

The single-topology pipeline, in the order it runs. :mod:`~fastforces.topology`
decides which terms exist, :mod:`~fastforces.elements` seeds them,
:mod:`~fastforces.sampling` generates the reference frames, :mod:`~fastforces.io`
stores them, :mod:`~fastforces.fit` fits the bonded terms, and
:mod:`~fastforces.params` holds and writes the result.

Topology and terms
------------------

.. automodule:: fastforces.topology

Element defaults
----------------

.. automodule:: fastforces.elements

Reference sampling
------------------

.. automodule:: fastforces.sampling

Training set
------------

.. automodule:: fastforces.io

Fit
---

.. automodule:: fastforces.fit

Parameters
----------

.. automodule:: fastforces.params
