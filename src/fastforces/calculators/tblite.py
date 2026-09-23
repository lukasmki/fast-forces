"""tblite's ASE calculator, with its partial charges left on the frame."""

import numpy as np
from ase.calculators.calculator import all_changes
from tblite.ase import TBLite


class TBLiteCalculator(TBLite):
    """xTB, writing its atomic charges onto the frame as `mulliken`.

    This is what makes a tblite reference drive a fixed-charge fit:
    `fit._charge_block` reads the equilibrium frame's `mulliken` array, and
    `sampling.label` carries whatever a calculator writes onto a frame into the
    training file.  `calculators.pyscf.PySCFCalculator` writes the same array
    from its density matrix; tblite has the charges in `results["charges"]`
    after every single point, and this only moves them to where the fit looks.

    The array is named for PySCF's analysis, but what GFN2-xTB reports is also a
    Mulliken population of its minimal basis -- a different approximation to the
    same ill-defined quantity, and it sums to the total charge the same way.
    """

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # The caller's object, not `self.atoms`: the array has to land on the
        # frame `sampling.label` is about to freeze, as PySCF's bond orders do.
        target = atoms if atoms is not None else self.atoms
        target.set_array("mulliken", np.asarray(self.results["charges"], float), float)
