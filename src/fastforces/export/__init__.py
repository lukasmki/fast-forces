"""Export of fitted parameters to OpenMM.

The `.jsonl` format is DynamicTopology's own, in eV and Angstrom, and is written
by `Parameters.to_jsonl`.  The conversion to OpenMM's nm and kJ/mol here goes
through `DynamicTopology.io.units`.
"""
