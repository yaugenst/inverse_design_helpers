import pytest
from numpy.random import default_rng


@pytest.fixture(scope="session")
def rng():
    seed = 98276534394
    return default_rng(seed)
