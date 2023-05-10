import autograd.numpy as np
from numpy.testing import assert_allclose

from inverse_design_helpers.autograd.differential_operators import (
    grad_dict,
    value_and_grad_dict,
)


def mult(x, y):
    return np.sum(x * y)


def test_grad_dict(rng):
    x = rng.random(10)
    y = rng.random(10)
    gd = grad_dict(mult)(x, y)

    assert set(gd.keys()) == {"x", "y"}
    assert_allclose(gd["x"], y)
    assert_allclose(gd["y"], x)


def test_value_and_grad_dict(rng):
    x = rng.random(10)
    y = rng.random(10)
    v, gd = value_and_grad_dict(mult)(x, y)

    assert set(gd.keys()) == {"x", "y"}
    assert v == mult(x, y)
    assert_allclose(gd["x"], y)
    assert_allclose(gd["y"], x)
