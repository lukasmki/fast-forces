"""The energy terms.

Deliberately almost empty: there is no registry and no re-export here, and
every consumer imports the concrete evaluator it wants.  The one exception is
below, and it exists because electrostatics is the only place a force field has
a *choice* of evaluator rather than a fixed set -- `atom` means `ACKS2`,
`coulomb` means `Coulomb` -- and four callers would otherwise each branch on the
term name for themselves.
"""

from .acks2 import ACKS2
from .coulomb import Coulomb

# Which evaluator each electrostatic term name is evaluated by.
ELECTROSTATICS = {"atom": ACKS2, "coulomb": Coulomb}


def electrostatic_evaluator(params):
    """A fresh evaluator for whichever electrostatic term `params` carries.

    `None` when it carries neither, which is a legitimate force field: `ZBL`
    and `LennardJones` still supply the nonbonded baseline.  Both evaluators
    cache state across calls -- solved charges, the Ewald setup -- so a caller
    keeps the one it is given rather than building one per frame.
    """
    term = params.electrostatics()
    return None if term is None else ELECTROSTATICS[term]()
