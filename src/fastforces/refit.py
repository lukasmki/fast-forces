"""Refit an existing DynamicTopology dataset: bonds, then one coupling per reaction.

    fast-forces refit datasets/HCombustion/HCombustion.json [--force-constants]

The dataset-level half of the pipeline, and the counterpart of `manifest.run`,
which fits each molecule from a reference calculator: this takes the dataset as
it stands -- templates on disk, stationary points in each reaction's `.xyz` --
and refits what depends on the dataset as a whole.  It was DynamicTopology's
`scripts/fit.py`, and it writes the same files.

Each reaction's amplitude and width come from the stationary points already
stored in its `rxn_*.xyz` (reactant / transition state / product):

  amplitude  fitted by inverting the 2x2 secular equation at the transition
             state, which needs a reference energy on that frame.  Falls back
             to --amplitude when the frame carries none, so a dataset with
             geometries but no computed energies still gets correct widths.
  width      fitted from the endpoint geometries so the coupling is switched
             off at the reactant and product minima.  Geometry only.

Inverting the secular equation only has a real root where the reference barrier
lies below both diabats, which with plain two-parameter Morse it does not always:
the form is too deep at stretched geometries.  --force-constants refits the
bonds first so that they are; see `refine`.

Three routes:

  --fit-mode asymptote    the default.  Fits each bond type's asymptote height
                          `h`, leaving every force constant where q-force put
                          it.  19 of 19 HCombustion channels, fastest mode
                          unchanged at 3748 cm^-1, in ~16 s.
  --fit-mode k            buys the depth by stiffening the bonds instead.
  --fit-mode asymptote-k  both; spends up to 1.19x in `k` for margins the hinge
                          does not value.

Channels routed to `fit_twobody` (fissions) take their amplitude from the
diabatic crossing, not from the margin, so a fission reported infeasible below
still gets a coupling; what the bond fit moves for them is where the crossing is.

`w total` in the table below is the honest number and `--max-wavenumber` is what
prices it; see `refine.DEFAULT_MAX_WAVENUMBER`.  The frequency table
is printed in cm^-1, where the cost is legible.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated
import json

import typer

from ase import Atoms, io
from ase.data import atomic_masses, atomic_numbers

from DynamicTopology.core import ReactionSet
from DynamicTopology.forcefield.exclusions import with_exclusions
from DynamicTopology.forcefield.params import ForceFieldParams
from DynamicTopology.io.json import read_jsonl, write_jsonl

from .coupling import (
    DEFAULT_BIMOL_CUTOFF,
    DEFAULT_EPS,
    CouplingError,
    fit_rmsd,
    fit_threebody,
    fit_twobody,
)
from .reaction import classify
from .refine import (
    DEFAULT_CURVATURE_WEIGHT,
    DEFAULT_FREQUENCY_WEIGHT,
    DEFAULT_MARGIN,
    DEFAULT_MAX_ASYMPTOTE,
    DEFAULT_MAX_SCALE,
    DEFAULT_MAX_WAVENUMBER,
    DissociationFitError,
    bonded_energy,
    diabatic_energy,
    fit_dissociation_energies,
    fit_force_constants,
    frequency,
    install_templates,
    scale_factor,
    total_wavenumber,
)

# 1 cm^-1 is 29.9793 THz^-1, so a mode's period in fs is this over its wavenumber.
FS_CM = 33356.4


def _masses(variable) -> list[float]:
    """The two masses (amu) of a bond variable, for its wavenumber in a report."""
    return [atomic_masses[atomic_numbers[element]] for element in variable.elements]


def _endpoint_bonds(frames) -> tuple[frozenset, frozenset]:
    """A reaction's two endpoint bond sets, as `reaction.classify` takes them.

    Read from the endpoint connectivity, which every dataset frame ships.  The
    channel they classify to decides the coupling form: a `fission` has no
    transition state to couple through, so it gets `coupling.fit_twobody` and
    its diabatic crossing; a `transfer` keeps the barrier inversion with the
    width measured in the moving atom's triangle; everything else, the RMSD
    form.
    """
    from DynamicTopology.core.topology import Topology

    def bonds(atoms):
        return frozenset(
            frozenset(edge) for edge in Topology.from_atoms(atoms).graph.edges()
        )

    return bonds(frames[0]), bonds(frames[-1])


def load_templates(
    manifest_path: Path, manifest: dict, params: ForceFieldParams
) -> list[tuple[str, Atoms, list]]:
    """`(name, atoms, terms)` per molecule entry.

    `params` is the dataset's own global parameters, taken from the loaded
    `ReactionSet` and passed explicitly rather than read from the active set:
    the exclusions derived here have to be the ones the manifest asks for, and
    saying so at the call site is cheaper to verify than an ordering argument.
    """
    templates = []
    for entry in manifest["molecules"]:
        stem = manifest_path.parent / entry["path"]
        atoms = io.read(stem.with_suffix(".xyz"))
        # The same exclusions `ReactionSet.load` derives, and for the reason
        # `refine` gives for routing the 12-6 through them: this
        # has to be the *same* sum the calculator will evaluate, or a template
        # stops reproducing the energy it was fitted to.  Fitting against an
        # unexcluded intramolecular ZBL while the calculator excludes it is not
        # a small disagreement -- it is 5.2 eV per O-H bond, and it is what
        # pins `r0` a quarter of an Angstrom inside the real bond length.
        # Stripped again by `strip_exclusions` before anything is written, since
        # the term files carry parameters and the exclusions are derived.
        templates.append(
            (
                stem.name,
                atoms,
                with_exclusions(
                    read_jsonl(stem.with_suffix(".jsonl")),
                    atoms.get_atomic_numbers(),
                    params=params,
                ),
            )
        )
    return templates


def strip_exclusions(terms: list) -> list:
    """`terms` without the derived exclusions, for writing back to a `.jsonl`."""
    return [term for term in terms if not term["type"].endswith("exclusion")]


def load_reactions(
    manifest_path: Path, manifest: dict
) -> list[tuple[str, list[Atoms]]]:
    """`(name, frames)` per reaction entry, frames in reactant/TS/product order."""
    return [
        (
            (manifest_path.parent / entry["path"]).name,
            io.read(
                (manifest_path.parent / entry["path"]).with_suffix(".xyz"), index=":"
            ),
        )
        for entry in manifest["reactions"]
    ]


def report_force_constants(fit, max_wavenumber: float = 0.0) -> None:
    """What the fit moved, and what it cost in wavenumbers.

    `w total` is the number the timestep is set by, so any row over
    `max_wavenumber` is flagged: those are the modes that would have to be
    integrated, and a fit that leaves one there has not bought a larger step
    however good its margins look.
    """
    if fit.mode in ("asymptote", "asymptote-k"):
        print(
            f"{'bond':<22}{'h (eV)':>8}{'k scale':>9}"
            f"{'w before':>10}{'w total':>9}{'ratio':>7}  over cap"
        )
        k_scales = fit.k_scales or [1.0] * len(fit.variables)
        for variable, height, k_scale, curvature in zip(
            fit.variables, fit.scales, k_scales, fit.curvatures
        ):
            masses = _masses(variable)
            label = f"{variable.template} {'-'.join(variable.elements)}"
            before_w = frequency(variable.k, *masses)
            after_w = total_wavenumber(curvature, *masses)
            over = (
                f"  +{after_w - max_wavenumber:.0f}"
                if max_wavenumber > 0.0 and after_w > max_wavenumber
                else ""
            )
            print(
                f"{label:<22}{height:>8.3f}{k_scale:>9.3f}{before_w:>10.0f}"
                f"{after_w:>9.0f}{after_w / before_w:>7.2f}{over}"
            )
        _report_timestep(fit, max_wavenumber)
        _report_margins(fit)
        return

    print(
        f"{'bond':<22}{'scale':>8}{'k before':>12}{'k after':>12}"
        f"{'w before':>11}{'w after':>10}{'ratio':>8}"
    )
    for variable, scale in zip(fit.variables, fit.scales):
        # The depth scale the inner solve chose moves `a = sqrt(k/2D)` too, but
        # the frequency is `sqrt(k/mu)` and depends on `k` alone.
        after = variable.k * scale
        masses = _masses(variable)
        before_w = frequency(variable.k, *masses)
        after_w = frequency(after, *masses)
        label = f"{variable.template} {'-'.join(variable.elements)}"
        print(
            f"{label:<22}{scale:>8.3f}{variable.k:>12.0f}{after:>12.0f}"
            f"{before_w:>11.0f}{after_w:>10.0f}{after_w / before_w:>8.2f}"
        )
    # Reported in whichever direction moved furthest: a force constant that
    # dropped by half is as much of a change as one that doubled.
    worst = max((max(s, 1.0 / s) for s in fit.scales), default=1.0) ** 0.5
    print(f"\nworst frequency drift {worst:.2f}x")
    _report_margins(fit)


def _report_timestep(fit, max_wavenumber: float) -> None:
    """The worst stretching mode, and the timestep that follows from it.

    Velocity Verlet wants roughly 15 steps per vibrational period, so the
    fastest mode on the surface is what a production run's timestep has to be
    chosen against.  Printed here because it is the point of the cap and it is
    otherwise a calculation the reader has to do themselves.
    """
    masses = [
        _masses(variable)
        for variable in fit.variables
    ]
    modes = [
        (total_wavenumber(curvature, *pair), variable)
        for curvature, pair, variable in zip(fit.curvatures, masses, fit.variables)
        if curvature is not None
    ]
    if not modes:
        return
    worst, variable = max(modes, key=lambda item: item[0])
    label = f"{variable.template} {'-'.join(variable.elements)}"
    period = FS_CM / worst if worst > 0 else float("inf")
    print(
        f"fastest mode {worst:.0f} cm^-1 ({label}), period {period:.2f} fs "
        f"-> {period / 15.0:.3f} fs at 15 steps/period"
    )
    if max_wavenumber > 0.0:
        over = sum(1 for value, _ in modes if value > max_wavenumber)
        print(f"{over} of {len(modes)} bond types over the {max_wavenumber:.0f} cap")


def _report_margins(fit) -> None:
    """Per-reaction feasibility, which every mode has to report the same way."""
    print(f"\n{'reaction':<16}{'margin before':>16}{'margin after':>14}  feasible")
    for name, after_margin in fit.margins.items():
        before_margin = fit.margins_before[name]
        print(
            f"{name:<16}{before_margin:>16.4f}{after_margin:>14.4f}"
            f"  {'yes' if after_margin > 0.0 else 'NO'}"
        )
    feasible = sum(value > 0.0 for value in fit.margins.values())
    print(f"\n{feasible} of {len(fit.margins)} channels fittable")


def _fit_rmsd(frames, **kwargs) -> list:
    """`coupling.fit_rmsd` as the term list a reaction `.jsonl` is written from."""
    return fit_rmsd(frames, **kwargs).to_terms()


class FitMode(str, Enum):
    asymptote = "asymptote"
    k = "k"
    asymptote_k = "asymptote-k"


def main(
    manifest_file: Annotated[
        Path,
        typer.Argument(
            metavar="MANIFEST",
            exists=True,
            dir_okay=False,
            help="the dataset manifest (.json)",
        ),
    ],
    eps: float = DEFAULT_EPS,
    bimol_cutoff: Annotated[
        float,
        typer.Option(
            help="separation (A) past which a bimolecular channel stops being "
            "enumerated. Must match what the calculation will run with: "
            "`fit_twobody` quenches a fission's coupling here, because past it "
            "there is no state left to couple to. Raising it is what a channel "
            "reported as cutoff-limited below needs."
        ),
    ] = DEFAULT_BIMOL_CUTOFF,
    amplitude: Annotated[
        float | None,
        typer.Option(
            help="amplitude (eV) to use where a transition state has no reference "
            "energy; without it such reactions are reported and skipped. A channel "
            "whose barrier cannot be inverted is decoupled (A = 0) rather than "
            "given this value."
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("-n", "--dry-run")] = False,
    rmsd_width: Annotated[
        bool,
        typer.Option(
            "--rmsd-width",
            help="keep the RMSD width for atom-transfer channels instead of "
            "measuring it in the transferring atom's triangle. The amplitude is "
            "identical either way; this only changes which coordinates switch the "
            "coupling off, so it is the way to reproduce a pre-`fit_threebody` "
            "baseline.",
        ),
    ] = False,
    bonds: Annotated[
        bool,
        typer.Option(
            "--bonds",
            help="first rescale each molecule template's Morse well depths so "
            "its bonds carry its full atomization energy, leaving no constant "
            "shift. Do this before fitting couplings: the shift is what makes "
            "reactant and product disagree about the energy of a broken bond.",
        ),
    ] = False,
    force_constants: Annotated[
        bool,
        typer.Option(
            "--force-constants",
            help="refit the Morse force constants as well as the depths, so the "
            "diabats rise above the reference barriers and the amplitudes become "
            "invertible. Implies --bonds. Paid for in vibrational frequencies; see "
            "--frequency-weight and --max-k-scale.",
        ),
    ] = False,
    fit_mode: Annotated[
        FitMode,
        typer.Option(
            help="`asymptote` fits a per-bond asymptote height `h`, leaving every "
            "force constant as q-force fitted it. `k` buys the depth by stiffening "
            "the bonds instead; `asymptote-k` fits both."
        ),
    ] = FitMode.asymptote,
    max_asymptote: Annotated[
        float,
        typer.Option(
            help="upper bound (eV) on the per-bond asymptote height in the "
            "`asymptote` modes; the lower bound is `bond_asymptote`. Setting it "
            "there freezes `h`, which is the vacuity check: plain Morse."
        ),
    ] = DEFAULT_MAX_ASYMPTOTE,
    max_k_scale: Annotated[
        float,
        typer.Option(
            help="hard bound on each force-constant scale. Its square root is the "
            "cap on frequency drift; 1.0 freezes the force constants entirely. "
            "Relative to the force constants in the .jsonl files as they stand, so "
            "re-running over already-fitted output compounds the bound."
        ),
    ] = DEFAULT_MAX_SCALE,
    frequency_weight: Annotated[
        float,
        typer.Option(
            help="how hard the fit is pulled back towards q-force's force constants"
        ),
    ] = DEFAULT_FREQUENCY_WEIGHT,
    max_wavenumber: Annotated[
        float,
        typer.Option(
            help="stretching modes above this (cm^-1) cost the objective. This is "
            "the timestep expressed as a force-field property: ~15 steps per period "
            "means a dt of 0.5 fs needs everything under 4450."
        ),
    ] = DEFAULT_MAX_WAVENUMBER,
    curvature_weight: Annotated[
        float,
        typer.Option(
            help="how much a mode over --max-wavenumber costs. 0 removes the cap "
            "entirely, which is the vacuity check for it."
        ),
    ] = DEFAULT_CURVATURE_WEIGHT,
    refit_manual: Annotated[
        bool,
        typer.Option(
            "--refit-manual",
            help="overwrite amplitudes marked `provenance: manual` in the existing "
            "term files. Without it they are kept and only their width is refitted.",
        ),
    ] = False,
    margin: Annotated[
        float,
        typer.Option(
            help="how far below the reference barrier (eV) a diabat must sit before "
            "the channel counts as fittable"
        ),
    ] = DEFAULT_MARGIN,
) -> None:
    manifest_path = manifest_file.resolve()
    manifest = json.loads(manifest_path.read_text())

    molecule_stems = [
        manifest_path.parent / entry["path"] for entry in manifest["molecules"]
    ]

    # Built once and refitted in place.  The coupling fit below reads its
    # diabats from this object, so a --dry-run reports the couplings the bond
    # fit would actually produce instead of the ones already on disk.
    reaction_set = ReactionSet(manifest_path)

    if force_constants:
        templates = load_templates(manifest_path, manifest, reaction_set.params)
        fit = fit_force_constants(
            reaction_set,
            templates,
            load_reactions(manifest_path, manifest),
            margin=margin,
            frequency_weight=frequency_weight,
            max_scale=max_k_scale,
            mode=fit_mode.value,
            max_wavenumber=max_wavenumber,
            curvature_weight=curvature_weight,
            max_asymptote=max_asymptote,
        )
        report_force_constants(fit, max_wavenumber)
        if not dry_run:
            for (name, _, _), stem in zip(templates, molecule_stems):
                write_jsonl(
                    stem.with_suffix(".jsonl"),
                    strip_exclusions(fit.terms[name]),
                    exist_ok=True,
                )
        print()

    elif bonds:
        print(f"{'molecule':<16}{'scale':>10}{'E_bonded':>12}{'E_reference':>13}")
        templates = load_templates(manifest_path, manifest, reaction_set.params)
        fitted_terms = []
        for (name, atoms, terms), stem in zip(templates, molecule_stems):
            try:
                fitted = fit_dissociation_energies(atoms, terms)
            except DissociationFitError as error:
                print(f"{name:<16}{'-':>10}  FAILED: {error}")
                fitted_terms.append(terms)
                continue
            print(
                f"{name:<16}{scale_factor(atoms, terms, fitted):>10.4f}"
                f"{bonded_energy(atoms, fitted):>12.5f}"
                f"{atoms.get_potential_energy():>13.5f}"
            )
            fitted_terms.append(fitted)
            if not dry_run:
                write_jsonl(
                    stem.with_suffix(".jsonl"), strip_exclusions(fitted), exist_ok=True
                )
        install_templates(reaction_set, templates, fitted_terms)
        print()

    print(f"{'reaction':<16}{'A (eV)':>12}{'a (1/A^2)':>12}{'r0 (A)':>9}  source")
    fitted = skipped = decoupled = preserved = 0
    barrierless: list[str] = []
    cutoff_limited: list[tuple[str, float, float]] = []
    for entry in manifest["reactions"]:
        stem = manifest_path.parent / entry["path"]
        frames = io.read(stem.with_suffix(".xyz"), index=":")
        transition = frames[len(frames) // 2]

        # A hand-set amplitude is a decision, not a stale fit, so a rerun must
        # not silently discard it.  Only the width is recomputed -- it is pure
        # geometry, and the stored one is generally wrong anyway: a channel that
        # reached `manual` by way of the decoupled path carries the width for
        # `NOMINAL_AMPLITUDE`, not for the amplitude someone then wrote in.
        existing = stem.with_suffix(".jsonl")
        manual = None
        if not refit_manual and existing.exists():
            stored = read_jsonl(existing)
            if stored and stored[0].get("provenance") == "manual":
                manual = stored[0]["kwargs"]["A"]

        reactant_bonds, product_bonds = _endpoint_bonds(frames)
        kind, spec = classify(reactant_bonds, product_bonds, len(frames[0]))

        try:
            if manual is not None:
                terms = _fit_rmsd(frames, amplitude=manual, eps=eps)
                terms[0]["provenance"] = "manual"
                source = "kept (manual; --refit-manual to replace)"
                preserved += 1
            elif kind == "fission":
                # A fission has no saddle, so the reference barrier this channel
                # carries is an energy at an arbitrary point on a monotone path
                # and inverting it is what returned `A = 0` however the diabats
                # were fitted.  Ask the diabats where they cross instead: that
                # needs no transition state and no reference energy, and it is
                # checked *before* the branch below so a stored TS energy does
                # not pull the channel back onto a fit that cannot succeed.
                pair, moving = spec
                bonded = 0 if frozenset(pair) in reactant_bonds else -1
                bonded_frame = frames[bonded]
                separated_frame = frames[-1 if bonded == 0 else 0]

                def diabats(positions, a=bonded_frame, b=separated_frame):
                    return (
                        diabatic_energy(reaction_set, a, positions),
                        diabatic_energy(reaction_set, b, positions),
                    )

                terms = fit_twobody(
                    bonded_frame.positions,
                    pair,
                    list(moving),
                    diabats,
                    eps=eps,
                    bimol_cutoff=bimol_cutoff,
                ).to_terms()
                source = "fitted from the diabatic crossing (bond length)"
                # A channel quenched against the cutoff rather than against its
                # own reactant has its crossing close to the edge of the region
                # where its recombination is enumerated at all, so its amplitude
                # is smaller than the surface would otherwise support.  Worth
                # reporting, because the fix is a larger cutoff and not a refit.
                if terms[0].get("limited_by") == "cutoff":
                    cutoff_limited.append(
                        (stem.name, terms[0]["kwargs"]["r0"], terms[0]["kwargs"]["A"])
                    )
            elif transition.calc is not None:
                energies = (
                    diabatic_energy(reaction_set, frames[0], transition.positions),
                    diabatic_energy(reaction_set, frames[-1], transition.positions),
                )
                if kind == "transfer" and not rmsd_width:
                    # Same amplitude, from the same reference barrier; only the
                    # width is measured in the transferring atom's own triangle
                    # instead of in the RMSD to the whole transition state.
                    terms = fit_threebody(
                        frames, spec, diabatic_energies=energies, eps=eps
                    ).to_terms()
                    source = "fitted from TS energy (transfer triangle)"
                else:
                    terms = _fit_rmsd(frames, diabatic_energies=energies, eps=eps)
                    source = "fitted from TS energy"
            elif amplitude is not None:
                terms = _fit_rmsd(frames, amplitude=amplitude, eps=eps)
                source = "width only (--amplitude)"
            else:
                print(
                    f"{stem.name:<16}{'-':>12}{'-':>12}  SKIPPED: no reference energy "
                    "on the transition state; run `fast-forces label` over the "
                    "reaction files, or pass --amplitude"
                )
                skipped += 1
                continue
        except CouplingError as error:
            # The barrier could not be inverted, so there is no amplitude to
            # write.  Decoupling says exactly that: A = 0 contributes no
            # stabilization, so `EVBBasis` never admits the state and the
            # Hamiltonian is not driven by a number nobody fitted.  The width
            # still comes from geometry and is kept, so filling the amplitude in
            # later needs no refit.
            fallback = 0.0 if amplitude is None else amplitude
            terms = _fit_rmsd(frames, amplitude=fallback, eps=eps)
            source = f"{terms[0]['provenance']} ({error.args[0][:52]}...)"
            decoupled += 1
            # One bond changed, in one direction: a fission or recombination
            # (or a ring opening), which has no saddle to invert a barrier at.
            if len(reactant_bonds ^ product_bonds) == 1:
                barrierless.append(stem.name)

        kwargs = terms[0]["kwargs"]
        # `r0` is the bond-length form's crossing; the RMSD form is centred on a
        # geometry rather than a distance and has none.
        centre = f"{kwargs['r0']:>9.3f}" if "r0" in kwargs else f"{'-':>9}"
        print(
            f"{stem.name:<16}{kwargs['A']:>12.4f}{kwargs['a']:>12.2f}{centre}  {source}"
        )
        if not dry_run:
            write_jsonl(
                stem.with_suffix(".jsonl"), terms, exist_ok=True
            )
        fitted += 1

    if cutoff_limited:
        print(
            f"\n{len(cutoff_limited)} fission channel(s) have their coupling width "
            f"set by --bimol-cutoff ({bimol_cutoff:.1f} A) rather than by their "
            "own reactant, because the crossing is nearer the cutoff than the "
            "reactant minimum. The coupling has to be off where the state stops "
            "being enumerated, so the amplitude is capped below what the surface "
            "would otherwise support:"
        )
        for name, centre, limited in cutoff_limited:
            print(
                f"    {name:<12} crossing {centre:.3f} A, "
                f"{bimol_cutoff - centre:.3f} A inside the cutoff, "
                f"A = {limited:+.4f} eV"
            )
        print(
            "  Raising --bimol-cutoff (and `System`'s to match) is what these "
            "need; refitting the force field will not move them."
        )

    verb = "would write" if dry_run else "wrote"
    print(
        f"\n{verb} {fitted} reaction term files "
        f"({fitted - decoupled - preserved} with a fitted amplitude, "
        f"{decoupled} decoupled, {preserved} manual and kept); {skipped} skipped"
    )
    if decoupled:
        underfit = decoupled - len(barrierless)
        if underfit:
            print(
                f"\n{underfit} channel(s) carry A = 0 because the force field's "
                "diabatic energies at the transition-state geometry lie below the "
                "reference barrier, so no real coupling reproduces it. The fix is a "
                "better diabatic force field -- try --force-constants, or a looser "
                "--frequency-weight if it is already on."
            )
        if barrierless:
            # Separated because the advice above is wrong for these, and following
            # it costs a pipeline run to learn so.
            print(
                f"\n{len(barrierless)} of those are barrierless, and inverting a "
                f"reference barrier was never going to fit them: "
                f"{', '.join(barrierless)}. A bond fission has no saddle -- the "
                "bound diabat is the ground state along the whole path -- so a "
                "perfect reactant diabat puts the reference barrier exactly on it "
                "and the discriminant vanishes identically. No refit reaches it. "
                "Such a channel should have gone to `coupling.fit_twobody`, "
                "which fits a Gaussian in the breaking bond's length centred on "
                "where its two diabats cross and needs no barrier at all; reaching "
                "this message means `reaction.classify` did not recognise it, or that fit "
                "raised as well -- its own error text says which."
            )
    if skipped:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
