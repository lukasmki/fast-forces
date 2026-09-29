"""PySCF-backed ASE calculator for Born-Oppenheimer and Car-Parrinello MD."""

import warnings
from typing import cast

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import CalculationFailed, Calculator, all_changes
from pyscf import dft, gto, lib, scf


def ase_to_pyscf(
    atoms: Atoms,
    basis: str = "cc-pvtz",
    charge: int | None = None,
    spin: int | None = None,
    verbose: int | None = None,
) -> gto.Mole:
    Z = atoms.get_atomic_numbers()
    R = atoms.get_positions()

    molstr = "\n".join(
        f"{Z[i]} {R[i][0]} {R[i][1]} {R[i][2]}\n" for i in range(len(atoms))
    )
    # PySCF's own default verbosity unless one is asked for.
    options = {} if verbose is None else {"verbose": verbose}
    mol: gto.Mole = gto.M(
        atom=molstr,
        basis=basis,
        charge=charge,
        spin=spin,
        **options,
    )
    return mol


def pyscf_to_ase(mol: gto.Mole) -> Atoms:
    return Atoms(
        symbols=mol.elements,
        positions=mol.atom_coords(unit="Angstrom"),
    )


def lowdin(ov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric ``S^{1/2}`` and ``S^{-1/2}`` from a single diagonalization.

    The Löwdin basis is where the idempotency constraint ``D S D = D`` loses its
    metric and becomes an ordinary projector condition, so the Car-Parrinello
    electronic dynamics is propagated there.
    """
    w, v = np.linalg.eigh(0.5 * (ov + ov.T))
    return (v * w**0.5) @ v.T, (v * w**-0.5) @ v.T


def bo(mf: scf.hf.SCF, ov=None, dm=None):
    mol: gto.Mole = mf.mol
    ao_idx = np.asarray([x[0] for x in mol.ao_labels(fmt=False)])

    # density matrix
    if dm is None:
        dm: np.ndarray = mf.make_rdm1()
    if dm.ndim == 2:
        dma, dmb = 0.5 * dm, 0.5 * dm
    else:
        dma, dmb = dm

    # overlap matrix
    if ov is None:
        ov: np.ndarray = mf.get_ovlp()

    # bond order
    DS: np.ndarray = (dma + dmb) @ ov
    RS: np.ndarray = (dma - dmb) @ ov
    Bu = DS * DS.T + RS * RS.T
    B = np.zeros((mol.natm, mol.natm))
    np.add.at(B, (ao_idx[:, None], ao_idx[None, :]), Bu)

    return B


# Mayer bond order a pair needs to count as bonded.  Every pair of atoms has a
# positive one, so without a cut every molecule is fully connected; the
# nonbonded pairs of H2O, HO2 and H2O2 sit below 0.03 and their bonds above
# 0.75, and 0.5 is the conventional line between them.
BOND_THRESHOLD = 0.5


def perceive_bonds(bond_order: np.ndarray, threshold: float = BOND_THRESHOLD) -> list:
    """`[(i, j, order), ...]` for every pair bonded in the `bo` matrix.

    `order` is rounded to 1, 2 or 3: a connectivity is a molecular graph, and
    `molify` -- which turns it into an RDKit molecule -- takes only integer
    bond orders.  The fractional values stay in the frame's `bond-order` array.
    """
    bonds = []
    n = len(bond_order)
    for i in range(n):
        for j in range(i + 1, n):
            if (bij := bond_order[i, j]) >= threshold:
                bonds.append((i, j, float(min(max(round(bij), 1), 3))))
    return bonds


def mulliken(mf: scf.hf.SCF, ov=None, dm=None):
    """Mulliken partial charges, one per atom, in elementary charges.

    `q_A = Z_A - sum_{mu in A} (D S)_{mu mu}`: the atomic number less the
    electron population Mulliken assigns to the basis functions centred on that
    atom.  Built the same way as `bo` above and from the same two matrices, so
    the two share the AO-to-atom map and the alpha/beta handling rather than
    each deriving them.

    Mulliken is the crudest population analysis there is -- it splits the
    overlap population straight down the middle and is notoriously
    basis-dependent, which with `cc-pvtz` is not a small caveat.  It is here
    because it costs one diagonal of a matrix product that `bo` already forms,
    and because the fixed-charge `charge` term needs *some* per-atom charge to
    start from.  See DynamicTopology's `forcefield/pointcharge.py`.

    The charges sum to the total molecular charge by construction, which is what
    a template's `charge` terms have to do: DynamicTopology's point charges move
    the excess charge with the proton only if every template carries its own.
    """
    mol: gto.Mole = mf.mol
    ao_idx = np.asarray([x[0] for x in mol.ao_labels(fmt=False)])

    if dm is None:
        dm = mf.make_rdm1()
    dm = np.asarray(dm)
    total = dm if dm.ndim == 2 else dm[0] + dm[1]

    if ov is None:
        ov = mf.get_ovlp()

    # Only the diagonal of `D S` is needed, so it is never formed.
    population = np.einsum("ij,ji->i", total, np.asarray(ov))
    charges = np.asarray(mol.atom_charges(), dtype=float)
    np.subtract.at(charges, ao_idx, population)
    return charges


class PySCFCalculator(Calculator):
    """Density-fitted PySCF UKS calculator with two evaluation modes.

    Born-Oppenheimer (``cp=False``, the default)
        A converged SCF is run at every geometry; energy and forces come from
        the PySCF gradient scanner, which warm-starts each SCF from the
        previous step's density.

    Car-Parrinello (``cp=True``)
        Once a density matrix has been installed via :meth:`set_dm`, no SCF is
        run.  The energy, the Fock matrix (the electronic gradient driving the
        fictitious dynamics) and the nuclear gradient are all evaluated at the
        supplied -- generally *unconverged* -- ``D``.  Until :meth:`set_dm` is
        called the calculator falls back to the Born-Oppenheimer path, so a CP
        run can bootstrap itself.

    Both paths write the bond orders and the Mulliken charges onto the caller's
    `Atoms` -- as the `bond-order` and `mulliken` arrays -- alongside the energy
    and forces they return, because both are by-products of the density matrix
    that is already in hand.

    The nuclear gradient on the CP path uses the Car-Parrinello energy-weighted
    density matrix ``W = C L C^T`` with ``L = C^T F C``, where ``C`` is recovered
    from ``D`` by diagonalizing it in the Löwdin basis.  It reduces to the
    ordinary SCF gradient when ``D`` is converged (``L`` diagonal).

    Implicit solvation (``pcm``)
        ``pcm`` names a PySCF polarizable continuum model -- ``"IEF-PCM"``,
        ``"C-PCM"``, ``"COSMO"`` or ``"SS(V)PE"`` -- and ``pcm_eps`` the
        dielectric constant of the continuum, water's by default.  The reaction
        field is solved self-consistently with the density, and both evaluation
        modes carry it: the Born-Oppenheimer path through PySCF's own solvent
        gradient, the Car-Parrinello path by adding the solvent potential to its
        Fock matrix and the solvent gradient to its forces, which PySCF would
        otherwise only fold in inside ``get_fock`` and the gradient ``kernel``
        -- neither of which that path calls.

        What a continuum buys is screening: a bare ion is several eV more
        stable in it (hydroxide by ~4 eV at PBE/6-31G), which puts charged
        and neutral fragments on a footing closer to solution.  What it does
        not buy is a barrier for the hydronium-water transfer.  With IEF-PCM
        the shared-proton geometry is still a single well, exactly as in the
        gas phase, because the asymmetry that localizes the proton in liquid
        water comes from specific solvation of the two oxygens, which a
        uniform dielectric cannot supply.  That channel still needs supplied
        frames -- a constrained O-O path -- rather than a saddle search.  So
        does the hydroxide-water one: Sella finds a saddle, and both downhill
        relaxations from it leave the proton shared.

        **It is also the wrong reference for a DynamicTopology dataset that is
        meant to run in explicit solvent.**  Each fragment's `E0` then carries
        its continuum solvation, while the force field adds the *unscreened*
        Coulomb between molecules on top, so a charged pair is stabilized
        twice.  At B3LYP/6-31+G*, H3O+ . H2O at an O-O of 2.75 A binds by 0.45
        eV in PCM and 1.30 eV in the gas phase, and the fixed-charge diabat
        gives 1.44 eV: the gas-phase number, as it should be.  Fitted against
        PCM the transfer endpoints came out 0.8-1.0 eV low, and the H3O+ . OH-
        contact pair fell 3.5 eV *below* the water dimer, where the reference
        puts it 1.7 eV above -- no coupling reproduces that, and MD on it would
        autoionize.  Against the gas phase, the same manifest fits every
        channel (`examples/proton-transfer`).  PCM is for an implicit-solvent
        simulation, whose intermolecular electrostatics are screened too.
    """

    implemented_properties = ["energy", "forces", "charges"]

    def __init__(
        self,
        charge: int | None = 0,
        spin: int | None = 0,
        xc: str = "HYB_GGA_XC_WB97X_V",
        basis: str = "cc-pvtz",
        verbose: int = 0,
        threads: int | None = None,
        level_shift: float | tuple[float, float] = 0.0,
        max_cycle: int = 200,
        cp: bool = False,
        pcm: str | None = None,
        pcm_eps: float = 78.3553,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.basis: str = basis
        self.charge: int | None = charge
        self.spin: int | None = spin
        self.cp: bool = cp
        self.pcm: str | None = pcm
        self.pcm_eps: float = pcm_eps
        # the connectivity last written onto a frame; see `calculate`
        self._perceived: list | None = None

        self.energy_pipe = (
            gto.M()
            .set(verbose=verbose)
            .apply(dft.UKS, xc=xc)
            # PySCF's own limit is 50 cycles, which open-shell pairs -- a
            # triplet OH + OH at a saddle guess -- routinely need more than,
            # and an SCF that runs out now raises (`_calculate_bo`).
            .set(conv_tol=1e-6, level_shift=level_shift, max_cycle=max_cycle)
            .density_fit()
        )
        if pcm is not None:
            self.energy_pipe = self.energy_pipe.PCM()
            self.energy_pipe.with_solvent.method = pcm
            self.energy_pipe.with_solvent.eps = pcm_eps
        self.forces_scanner = self.energy_pipe.nuc_grad_method().as_scanner()
        # the scanner owns its own mean-field object; everything below drives it
        self.mf: scf.hf.SCF = self.forces_scanner.base
        if threads is not None:
            self.threads = lib.num_threads(n=threads)

        # Car-Parrinello electronic state: AO-basis density matrix, per spin
        self.dm: np.ndarray | None = None

        # private
        self._fock: list[np.ndarray] | None = None
        self._synced: np.ndarray | None = None
        self._lowdin: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # geometry / electronic state plumbing used by the CP integrator
    # ------------------------------------------------------------------

    def sync(self, atoms: Atoms) -> gto.Mole:
        """Point the PySCF objects at ``atoms``' current geometry."""
        positions = atoms.get_positions()
        if self._synced is None or not np.array_equal(positions, self._synced):
            mol = ase_to_pyscf(
                atoms, basis=self.basis, charge=self.charge, spin=self.spin
            )
            self.forces_scanner.reset(mol)
            self._synced = positions.copy()
            self._fock = None
            self._lowdin = None
        return self.mf.mol

    def get_ovlp(self) -> np.ndarray:
        """AO overlap matrix ``S`` at the geometry set by :meth:`sync`."""
        return self.mf.get_ovlp()

    def get_lowdin(self) -> tuple[np.ndarray, np.ndarray]:
        """``(S^{1/2}, S^{-1/2})`` at the geometry set by :meth:`sync`.

        Cached because the CP integrator needs it at both ends of every step;
        :meth:`sync` drops the cache when the geometry moves.
        """
        if self._lowdin is None:
            self._lowdin = lowdin(self.get_ovlp())
        return self._lowdin

    @property
    def nelec(self) -> tuple[int, int]:
        """Alpha and beta electron counts at the synced geometry."""
        return self.mf.mol.nelec

    def scf_dm(self, atoms: Atoms) -> np.ndarray:
        """Converge an SCF at ``atoms`` and return its density matrix.

        This is how a Car-Parrinello run bootstraps its electronic state.
        """
        self.sync(atoms)
        self.mf.kernel()
        if not self.mf.converged:
            warnings.warn(
                "SCF did not converge while initializing the electronic state",
                stacklevel=2,
            )
        return np.asarray(self.mf.make_rdm1())

    def set_dm(self, dm) -> None:
        """Install the density matrix used by the next CP evaluation."""
        self.dm = np.asarray(dm)
        # the results cache is keyed on the geometry alone, so invalidate it by
        # hand -- the electronic state can change while the nuclei stand still
        self.results = {}
        self.atoms = None

    @property
    def fock(self) -> list[np.ndarray] | None:
        """Fock matrices built by the most recent CP evaluation, per spin."""
        return self._fock

    # ------------------------------------------------------------------
    # ASE interface
    # ------------------------------------------------------------------

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        # deliberately not self.atoms: the bond orders below are written onto
        # the caller's object, not the calculator's private copy
        atoms = cast("Atoms", atoms if atoms is not None else self.atoms)

        if self.cp and self.dm is not None:
            energy, gradient, dm = self._calculate_cp(atoms)
        else:
            energy, gradient, dm = self._calculate_bo(atoms)
        forces = -gradient

        # bond order
        bond_order = bo(self.mf, dm=dm)
        atoms.set_array("bond-order", bond_order, bond_order.dtype)

        # Mulliken charges, written onto the frame as well as returned: that is
        # what carries them through `sampling.snapshot`, which keeps whatever
        # arrays the calculator wrote, into the training file and on to the fit.
        charges = mulliken(self.mf, dm=dm)
        atoms.set_array("mulliken", charges, float)

        # A connectivity already on the frame is its topology -- from the
        # SMILES, or the diabatic state a reaction frame stands for -- and is
        # an input, not something a single point gets to redefine.  Only a
        # frame without one gets it perceived from the bond orders, and one
        # this calculator perceived earlier (the same list object: `copy`
        # deep-copies `info`) is re-perceived, so a relaxation or an MD run
        # follows the bonds as they change.
        stated = atoms.info.get("connectivity")
        if stated is None or stated is self._perceived:
            self._perceived = perceive_bonds(bond_order)
            atoms.info["connectivity"] = self._perceived

        self.results = {
            "energy": energy * units.Hartree,
            "forces": forces * units.Hartree / units.Bohr,
            "charges": charges,
        }

    def _calculate_bo(self, atoms: Atoms):
        """Converged-SCF energy and gradient (Hartree, Hartree/Bohr)."""
        mol = ase_to_pyscf(atoms, basis=self.basis, charge=self.charge, spin=self.spin)
        energy, gradient = self.forces_scanner(mol)
        # the scanner reset the mean-field object behind our back
        self._synced = atoms.get_positions().copy()
        self._fock = None
        self._lowdin = None
        # The scanner returns whatever it had when it ran out of cycles and
        # PySCF does not raise: an open-shell O + OH pair at 10 A came back from
        # PBE0/def2-SVP 88 eV above its own fragments with nothing to say so.
        # Refining that density with second-order SCF is no rescue -- it
        # converged to a state 610 eV up whose forces disagreed with its own
        # finite differences -- so the point fails, as a tblite one does,
        # rather than entering a training set.
        if not self.mf.converged:
            raise CalculationFailed(
                f"SCF not converged ({mol.natm} atoms, 2S={self.spin})"
            )
        return energy, gradient, self.mf.make_rdm1()

    def _calculate_cp(self, atoms: Atoms):
        """Non-self-consistent energy and gradient at the installed ``D``."""
        self.sync(atoms)
        dm = cast("np.ndarray", self.dm)
        veff = self.mf.get_veff(dm=dm)
        energy = self.mf.energy_tot(dm=dm, vhf=veff)
        # bare Kohn-Sham matrix; deliberately not mf.get_fock, which would fold
        # in the SCF level shift and damping -- those must not enter the EOM
        fock = np.asarray(self.mf.get_hcore()) + np.asarray(veff)
        if self.pcm is not None:
            # `energy_tot` already counts the solvent through `veff.e_solvent`;
            # its potential rides alongside on `veff.v_solvent` and has to be
            # added here by hand, or the EOM would propagate the gas-phase
            # density on the solvated surface.
            fock = fock + np.asarray(veff.v_solvent)
        self._fock = [fock[0], fock[1]]

        # PySCF's gradient wants MOs, so recover them from D.  In the Löwdin
        # basis an idempotent D is a projector, and its eigenvectors of
        # eigenvalue 1 span the occupied space; the largest N_sigma of them are
        # the occupied orbitals even when D has drifted slightly off the
        # manifold.  The full eigenvector set is kept -- PySCF expects square
        # per-spin MO arrays with the occupancy carried by mo_occ, and slicing
        # to the occupied columns would make the two spin channels ragged for
        # an open-shell system.
        s_half, s_minus_half = self.get_lowdin()
        nao = s_half.shape[0]
        mo_coeff, mo_occ, mo_energy = [], [], []
        for d, fock_s, nocc in zip(dm, self._fock, self.nelec, strict=True):
            p = s_half @ d @ s_half
            _, vecs = np.linalg.eigh(0.5 * (p + p.T))
            # descending occupancy, so the occupied block comes first
            c = s_minus_half @ vecs[:, ::-1]

            # Rotate the occupied block so that the Lagrange multiplier matrix
            # L = C^T F C is diagonal.  That rotation leaves both dm and
            # W = C L C^T invariant, so PySCF's stock gradient -- which assumes
            # a diagonal L when it builds W from mo_energy -- yields exactly the
            # Car-Parrinello force.  Virtual orbital energies never enter W.
            occ_block = c[:, :nocc]
            lam = occ_block.T @ fock_s @ occ_block
            eps, u = np.linalg.eigh(0.5 * (lam + lam.T))
            c[:, :nocc] = occ_block @ u

            occ = np.zeros(nao)
            occ[:nocc] = 1.0
            energies = np.zeros(nao)
            energies[:nocc] = eps

            mo_coeff.append(c)
            mo_occ.append(occ)
            mo_energy.append(energies)

        mo_coeff = np.asarray(mo_coeff)
        mo_occ = np.asarray(mo_occ)
        mo_energy = np.asarray(mo_energy)

        # the DF gradient reads some of this off the mean-field object rather
        # than the keywords, so plant it there too
        self.mf.mo_coeff = mo_coeff
        self.mf.mo_occ = mo_occ
        self.mf.mo_energy = mo_energy

        gradient = (
            self.forces_scanner.grad_elec(
                mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ
            )
            + self.forces_scanner.grad_nuc()
        )
        if self.pcm is not None:
            # what the solvent gradient `kernel` adds on the BO path, at this
            # (unconverged) density rather than the converged one
            gradient = gradient + self.mf.with_solvent.grad(dm[0] + dm[1])
        return energy, gradient, dm
