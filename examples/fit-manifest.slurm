#!/bin/bash
#SBATCH --job-name=fit-manifest
#SBATCH --account=m2834
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=04:00:00
#SBATCH --output=fit-manifest-%j.out
#
# Fit a whole manifest on one Perlmutter CPU node (2 x 64-core Milan).
#
#     sbatch examples/fit-manifest.slurm examples/hydrogen-combustion/hydrogen-combustion.json
#
# One task holding the node, because the parallelism is fastforces' own: it
# forks a worker per reference calculation (`fastforces.parallel`), so Slurm
# only has to hand it the cores.  32 workers x 4 threads fills the 128 physical
# cores; PySCF stops gaining from threads well before a whole node (16-32 is the
# best a single calculation does), so many narrow workers beat a few wide ones.
# Molecules are fitted side by side, then reactions, each with its share.
#
# Every reference calculation is independent, so the training sets are the ones
# a serial fit writes -- bit for bit at the same per-calculation thread count.
# A cached training set is reused, so a resubmission after a timeout picks up
# from the molecules already done.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MANIFEST="${1:?usage: sbatch examples/fit-manifest.slurm MANIFEST.json}"

source .venv/bin/activate
export FASTFORCES_WORKERS="${FASTFORCES_WORKERS:-32}"
export FASTFORCES_THREADS="${FASTFORCES_THREADS:-4}"
# No OMP_PROC_BIND / OMP_PLACES: every forked worker inherits the whole node's
# CPU mask, so binding would pin each worker's threads to the *same* first few
# cores.  Left unbound, the kernel spreads the 128 threads over the node.
unset OMP_PROC_BIND OMP_PLACES

fast-forces fit-manifest "$MANIFEST"
