"""07 -- Running dynamics with the fitted field.

`FastForces` is an ordinary ASE calculator, so every ASE driver works with it
unchanged.  This runs NVE to check energy conservation, then Langevin at 300 K,
and times both against the reference method the field was fit to.

    uv run python examples/07_molecular_dynamics.py
"""

import time

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

import fastforces as ff

from _common import banner, fitted, gfn2

banner(__doc__)

atoms, params, _ = fitted("CC#N", name="acetonitrile")
# The dynamics below leave `atoms` wherever the trajectory ended; keep the
# optimized structure for the timing run, which the reference method needs to
# converge its SCF on.
equilibrium = atoms.copy()
atoms.calc = ff.FastForces(atoms, params)

# ---------------------------------------------------------------- NVE
thermalize_momenta(atoms, temperature_K=300, rng=np.random.default_rng(0))
dynamics = VelocityVerlet(atoms, timestep=0.5 * units.fs, logfile=None)
start = atoms.get_total_energy()
history = []
for _ in range(20):
    dynamics.run(25)
    history.append(atoms.get_total_energy() - start)

print("NVE, 0.5 fs timestep, 500 steps:")
print(f"  total energy drift  {history[-1]:+.4f} eV")
print(f"  max excursion       {max(abs(h) for h in history):.4f} eV")
print(f"  kinetic energy      {atoms.get_kinetic_energy():.4f} eV")
print("  (drift small against the kinetic energy means the analytic forces")
print("   really are the gradient of the energy, including ACKS2's response)\n")

# ------------------------------------------------------------ Langevin
thermalize_momenta(atoms, temperature_K=300, rng=np.random.default_rng(1))
dynamics = Langevin(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=300,
    friction=0.02,
    fixcm=False,
    logfile=None,
)
temperatures = []
for _ in range(800):
    dynamics.run(25)
    temperatures.append(atoms.get_temperature())

print("Langevin, 300 K, 20 ps:")
print(f"  mean temperature    {np.mean(temperatures[100:]):.0f} K")
print(f"  std                 {np.std(temperatures[100:]):.0f} K")
print("  (a six-atom system fluctuates by ~100 K instantaneously, so a short run")
print("   averaging near rather than exactly at 300 K is sampling noise)\n")


# ------------------------------------------------------------ timing
def time_steps(calculator, n=50):
    work = equilibrium.copy()
    work.calc = calculator
    thermalize_momenta(work, temperature_K=300, rng=np.random.default_rng(2))
    driver = VelocityVerlet(work, timestep=0.5 * units.fs, logfile=None)
    driver.run(1)  # warm up
    start = time.perf_counter()
    driver.run(n)
    return (time.perf_counter() - start) / n


fast = time_steps(ff.FastForces(atoms, params))
reference = time_steps(gfn2())
print("cost per MD step:")
print(f"  FastForces          {fast * 1e3:8.3f} ms")
print(f"  GFN2-xTB reference  {reference * 1e3:8.3f} ms")
print(f"  speedup             {reference / fast:8.1f}x")
print("""
The gap widens with system size: the fitted field is a fixed number of local
terms plus one small dense solve for the charges, while the reference cost
grows with the electronic structure.""")
