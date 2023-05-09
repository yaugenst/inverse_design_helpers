import numpy as np
import pytest
from autograd.test_util import check_grads

import inverse_design_helpers.autograd.scipy.ndimage as ndimage


@pytest.fixture
def rng():
    seed = 98276534394
    return np.random.default_rng(seed)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "mode",
    [
        "reflect",
        "constant",
        pytest.param("nearest", marks=pytest.mark.xfail),
        pytest.param("mirror", marks=pytest.mark.xfail),
        "wrap",
    ],
)
def test_gaussian_filter(rng, dim, mode):
    x = rng.random(np.arange(10, 10 + dim))
    sigma = np.arange(1, 1 + dim)
    check_grads(ndimage.gaussian_filter, modes=["fwd", "rev"], order=2)(
        x, sigma, mode=mode
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "mode",
    [
        "reflect",
        "constant",
        pytest.param("nearest", marks=pytest.mark.xfail),
        pytest.param("mirror", marks=pytest.mark.xfail),
        "wrap",
    ],
)
def test_gaussian_laplace(rng, dim, mode):
    x = rng.random(np.arange(10, 10 + dim))
    sigma = np.arange(1, 1 + dim)
    check_grads(ndimage.gaussian_laplace, modes=["fwd", "rev"], order=2)(
        x, sigma, mode=mode
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "mode",
    [
        "reflect",
        "constant",
        "nearest",
        pytest.param("mirror", marks=pytest.mark.xfail),
        "wrap",
    ],
)
def test_laplace(rng, dim, mode):
    x = rng.random(np.arange(10, 10 + dim))
    check_grads(ndimage.laplace, modes=["fwd", "rev"], order=2)(x, mode=mode)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("reflect", marks=pytest.mark.xfail),
        "constant",
        pytest.param("nearest", marks=pytest.mark.xfail),
        pytest.param("mirror", marks=pytest.mark.xfail),
        "wrap",
    ],
)
def test_prewitt(rng, dim, mode):
    x = rng.random(np.arange(10, 10 + dim))
    check_grads(ndimage.prewitt, modes=["fwd", "rev"], order=2)(x, mode=mode)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("reflect", marks=pytest.mark.xfail),
        "constant",
        pytest.param("nearest", marks=pytest.mark.xfail),
        pytest.param("mirror", marks=pytest.mark.xfail),
        "wrap",
    ],
)
def test_sobel(rng, dim, mode):
    x = rng.random(np.arange(10, 10 + dim))
    check_grads(ndimage.sobel, modes=["fwd", "rev"], order=2)(x, mode=mode)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "mode",
    [
        "reflect",
        "constant",
        "nearest",
        pytest.param("mirror", marks=pytest.mark.xfail),
        "wrap",
    ],
)
def test_uniform_filter(rng, dim, mode):
    x = rng.random(np.arange(10, 10 + dim))
    check_grads(ndimage.uniform_filter, modes=["fwd", "rev"], order=2)(x, mode=mode)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_fourier_ellipsoid(rng, dim):
    x = rng.random(np.arange(10, 10 + dim))
    size = np.arange(1, 1 + dim)
    check_grads(ndimage.fourier_ellipsoid, modes=["fwd", "rev"], order=2)(x, size)
