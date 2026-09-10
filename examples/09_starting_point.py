"""09 -- Starting a fit from an existing force field.

`fit`, `fit_from_file` and `parameterize` all take `initial=`: a `Parameters`,
a term dict, or a path to a jsonl file.  Terms are matched to the topology by
their atom slots, so anything the supplied field does not cover keeps its
ordinary default -- the geometry for an equilibrium value, the element table for
a bond or a nonbonded parameter.

What a starting point actually moves differs by block, and the difference is
structural rather than a tuning choice:

  * the bond parameters start a *local* nonlinear solve, so supplying them
    changes where the fit lands;
  * the equilibrium values and the nonbonded baseline are held fixed, not fit,
    so supplying them replaces what would be measured or looked up;
  * every other force constant comes from a bounded *global* least squares,
    which has no starting point -- supplied values reach the result only through
    the single bond refinement that runs before the first linear solve.

The second half of this script asks the obvious follow-up question: is fitting
idempotent?  Hand a fit its own output back and see whether it returns it
unchanged.  It does -- and the one design decision that makes it true is worth
seeing, because it was not always.

    uv run python examples/09_starting_point.py
"""

import numpy as np

import fastforces as ff
from _common import OUTPUT, banner, fitted

banner(__doc__)

# The explicit `FitConfig()` matters here.  A training file records the config
# it was written with, and `fit_from_file` prefers that recorded one so the file
# reproduces its own fit -- which means a file written before `n_cycles` was
# raised would quietly keep the old budget.  Passing a config overrides it.
atoms, params, path = fitted("OO", config=ff.FitConfig(), name="h2o2")
jsonl = OUTPUT / "h2o2_starting_point.jsonl"
params.to_jsonl(str(jsonl))

print(f"fitted from the element table: {params!r}")
print(f"  E_rmse {params.report['energy_rmse_eV']:.6f} eV")
print(f"  F_rmse {params.report['force_rmse_eV_A']:.6f} eV/A")
print(f"  cycles {params.report['cycles']}\n")

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
    print(f"  {name:24s} {change:8.2%}")
print()

# --- iterate it -------------------------------------------------------------
# An idempotent fit makes this table one row repeated, give or take `cycle_tol`.
# Watch the cycle count too: the first fit works, and every restart after it
# finds nothing left to do and stops at the two-cycle floor.
print("iterating the restart:")
print(
    f"  {'':>4} {'cycles':>7} {'E0 / eV':>12} {'sum_b D_b':>11} "
    f"{'O-O k':>9} {'F_rmse':>10}"
)
current = params
for i in range(4):
    D = np.asarray(current.terms["bond"]["kwargs"]["D"], dtype=float)
    k = np.asarray(current.terms["bond"]["kwargs"]["k"], dtype=float)
    print(
        f"  {i:>4} {current.report['cycles']:>7} {current.e0:12.6f} {D.sum():11.6f} "
        f"{k.max():9.2f} {current.report['force_rmse_eV_A']:10.6f}"
    )
    current = ff.fit_from_file(str(path), config=ff.FitConfig(), initial=current)

print(
    """
Every restart after the first stops at two cycles, and what it predicts does not
move: the force RMSE repeats to five figures.  The parameters repeat to within
`cycle_tol`, which is a few percent on the shallowest cross terms and a fraction
of a percent on the well-determined ones -- tighten `cycle_tol` from its default
1e-4 to 1e-8 and that drift falls by ~80x for 40 extra cycles.  Drift that
shrinks with the tolerance is a fixed point being approached; before `E0` came
out of the fit, a 64x larger budget moved the parameters *further*.

It is true because `E0` is not a fitted parameter.  Every energy residual in the
fit is mean-centered -- shifting the whole energy profile by a constant costs
nothing anywhere in the loop -- and `E0` is read off the leftover mean once the
fit has converged, as the very last thing `fit` does.

Fitting `E0` instead put a flat direction straight through the parameter space.
`_bond_morse` evaluates

    E = D * (1 - exp(-a*dr))**2 - D,   a = sqrt(k / 2D)

and expanding for small `dr` gives `k*dr**2/2` -- the `D` cancels.  So near
equilibrium `D` reaches the energy only through the trailing constant `-D` per
bond, and a constant per bond is exactly what `E0` is.  The nonlinear block owns
`D`, the linear block owned `E0`, and they traded: `D` up, `E0` up to
compensate, for 620 cycles on this molecule, with the residual barely moving and
no fixed point in reach.  Restarting an unfinished fit resumed the walk instead
of repeating it.

Centering deletes that direction rather than penalizing it, and the fit is both
faster and better for it -- 21 cycles instead of 620, force RMSE 0.258 instead
of 0.280 eV/A.

What it costs is visible in the `sum_b D_b` column above: 600 eV across three
bonds is 200 each, every one of them pinned at its upper bound.  `D` now reaches
the energy only through the anharmonicity it describes, and `D -> infinity` is
the harmonic limit -- so where the reference data does not pin the anharmonicity
down, `D` saturates and stops meaning a dissociation energy.  The fit is better
for it, but a bond in that state will not describe dissociation if you pull it
apart in MD.  Worth checking before you do."""
)
