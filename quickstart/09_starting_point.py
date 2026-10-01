"""09 -- Starting a fit from an existing force field.

`fit`, `fit_from_file` and `parameterize` all take `initial=`: a `Parameters`,
a term dict, or a path to a jsonl file.  Terms are matched to the topology by
their atom slots, so anything the supplied field does not cover keeps its
ordinary default -- the geometry for an equilibrium value, the element table for
a bond or a nonbonded parameter.

What a starting point can move is exactly what the fit holds fixed:

  * the equilibrium values (bond `r0` excepted, which is always measured), the
    bond depths `D` and asymptotes, and the nonbonded baseline are not fit, so
    supplying them replaces what would be measured or looked up;
  * every force constant, the bonds' included, comes from one bounded *global*
    least squares, which has no starting point -- a supplied `k` does not reach
    the result at all.

The second half of this script asks the obvious follow-up question: is fitting
idempotent?  Hand a fit its own output back and see whether it returns it
unchanged.

    uv run python quickstart/09_starting_point.py
"""

import numpy as np

import fastforces as ff
from _common import OUTPUT, banner, fitted

banner(__doc__)

# The explicit `FitConfig()` matters here.  A training file records the config
# it was written with, and `fit_from_file` prefers that recorded one so the file
# reproduces its own fit.  Passing a config overrides it.
atoms, params, path = fitted("OO", config=ff.FitConfig(), name="h2o2")
jsonl = OUTPUT / "h2o2_starting_point.jsonl"
params.to_jsonl(str(jsonl))

print(f"fitted from the element table: {params!r}")
print(f"  E_rmse {params.report['energy_rmse_eV']:.6f} eV")
print(f"  F_rmse {params.report['force_rmse_eV_A']:.6f} eV/A\n")

# --- the same fit, started from that field ---------------------------------
again = ff.fit_from_file(str(path), config=ff.FitConfig(), initial=str(jsonl))
print(f"restarted from {jsonl.name}: {again.report['seeded']} classes seeded")
print(f"  E_rmse {again.report['energy_rmse_eV']:.6f} eV")
print(f"  F_rmse {again.report['force_rmse_eV_A']:.6f} eV/A\n")


def drift(a, b):
    """Every parameter's change, scaled by its own size."""
    rows = []
    for term, block in a.terms.items():
        for name, value in block["kwargs"].items():
            value = np.asarray(value, dtype=float)
            other = np.asarray(b.terms[term]["kwargs"][name], dtype=float)
            scale = max(float(np.abs(value).max()), 1e-12)
            rows.append((float(np.abs(other - value).max() / scale), f"{term}.{name}"))
    return sorted(rows, reverse=True)


print("largest parameter changes from one restart:")
for change, name in drift(params, again)[:5]:
    print(f"  {name:24s} {change:10.2e}")
print()

# --- a supplied force constant goes nowhere ---------------------------------
# Triple every `k` in the starting point: the linear solve never reads them.
scrambled = ff.as_parameters(str(jsonl))
for block in scrambled.terms.values():
    if "k" in block["kwargs"]:
        block["kwargs"]["k"] = 3.0 * np.asarray(block["kwargs"]["k"])
inert = ff.fit_from_file(str(path), config=ff.FitConfig(), initial=scrambled)
change, name = drift(params, inert)[0]
print(f"restarted with every k tripled: largest change {change:.2e} ({name})")

print("""
Both restarts return the field they started from, to round-off.  There is no
convergence tolerance to be idempotent *to*: every force constant is one linear
solve, and what a starting point supplies is exactly what the first fit held
fixed.

`E0` is not in that solve either.  It is a fixed shift of the molecule's whole
surface, there so that molecules sit on one absolute scale and a reactant and a
product state carry the reference offset between them.  Every energy residual
is mean-centered, and `E0` is read off the leftover mean afterwards, plus the
`-D` per bond that the
Morse bond carries at its minimum and the harmonic spring it was fitted as does
not.  A Morse bond `D*(1 - exp(-a*dr))**2 - D` with `a = sqrt(k / 2D)` is
`k*dr**2/2 - D` to second order, which is why the harmonic `k` is the Morse `k`
-- and why `D`, which the frames near equilibrium barely see, is not fitted to
them but scaled from the element table until `E0` is the free atoms: the bonds
then carry the atomization energy.""")
