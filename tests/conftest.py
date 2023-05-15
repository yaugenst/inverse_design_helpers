import pytest
from numpy.random import Generator, default_rng


@pytest.fixture(scope="session")
def rng() -> Generator:
    seed = 98276534394
    return default_rng(seed)
