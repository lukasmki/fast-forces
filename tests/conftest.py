import numpy as np
import pytest

SCRATCH = None


@pytest.fixture(scope="session")
def tblite_factory():
    tblite = pytest.importorskip("tblite.ase")
    return lambda atoms: tblite.TBLite(method="GFN2-xTB", verbosity=0)


@pytest.fixture(scope="session")
def h2o2_fit(tmp_path_factory, tblite_factory):
    """One end-to-end H2O2 parameterization, shared by the slow tests."""
    import fastforces as ff

    path = tmp_path_factory.mktemp("h2o2") / "training.xyz"
    atoms = ff.build("OO")
    params = ff.parameterize(
        atoms,
        tblite_factory,
        config=ff.FitConfig(n_mode_frames=30, n_conformers=0),
        training_set=str(path),
    )
    return atoms, params, str(path)


@pytest.fixture
def random_positions():
    rng = np.random.default_rng(7)
    # a loose cloud: far enough apart that no term hits its clamp, close enough
    # that every term is nonzero
    return rng.normal(scale=1.6, size=(8, 3))
