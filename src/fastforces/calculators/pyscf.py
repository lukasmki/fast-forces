"""PySCF-backed ASE calculator for Born-Oppenheimer and Car-Parrinello MD."""

import warnings
from typing import cast

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from pyscf import dft, gto, lib, scf


def ase_to_pyscf(
    atoms: Atoms,
    basis: str = "cc-pvtz",
    charge: int | None = None,
    spin: int | None = None,
) -> gto.Mole:
    Z = atoms.get_atomic_numbers()
    R = atoms.get_positions()

    molstr = "\n".join(
        f"{Z[i]} {R[i][0]} {R[i][1]} {R[i][2]}\n" for i in range(len(atoms))
    )
    mol: gto.Mole = gto.M(
        atom=molstr,
        basis=basis,
        charge=charge,
        spin=spin,
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

    The nuclear gradient on the CP path uses the Car-Parrinello energy-weighted
    density matrix ``W = C L C^T`` with ``L = C^T F C``, where ``C`` is recovered
    from ``D`` by diagonalizing it in the Löwdin basis.  It reduces to the
    ordinary SCF gradient when ``D`` is converged (``L`` diagonal).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        charge: int | None = 0,
        spin: int | None = 0,
        xc: str = "HYB_GGA_XC_WB97X_V",
        basis: str = "cc-pvtz",
        verbose: int = 0,
        threads: int | None = None,
        level_shift: float | tuple[float, float] = 0.0,
        cp: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.basis: str = basis
        self.charge: int | None = charge
        self.spin: int | None = spin
        self.cp: bool = cp

        self.energy_pipe = (
            gto.M()
            .set(verbose=verbose)
            .apply(dft.UKS, xc=xc)
            .set(conv_tol=1e-6, level_shift=level_shift)
            .density_fit()
        )
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

        conn = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if (bij := bond_order[i, j]) > 0:
                    conn.append((i, j, bij))
        atoms.info["connectivity"] = conn

        self.results = {
            "energy": energy * units.Hartree,
            "forces": forces * units.Hartree / units.Bohr,
        }

    def _calculate_bo(self, atoms: Atoms):
        """Converged-SCF energy and gradient (Hartree, Hartree/Bohr)."""
        mol = ase_to_pyscf(atoms, basis=self.basis, charge=self.charge, spin=self.spin)
        energy, gradient = self.forces_scanner(mol)
        # the scanner reset the mean-field object behind our back
        self._synced = atoms.get_positions().copy()
        self._fock = None
        self._lowdin = None
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
        return energy, gradient, dm
