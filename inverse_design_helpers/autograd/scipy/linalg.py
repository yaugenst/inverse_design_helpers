import autograd.numpy as np
import scipy.linalg
from autograd.extend import defvjp, primitive
from numpy.typing import NDArray


def _transpose_banded(l_and_u: tuple[int, int], ab: NDArray) -> NDArray:
    i, j = ab.shape
    a, b = np.unravel_index(np.arange(i * j), ab.shape)
    n = a + b - l_and_u[0]
    At = np.where(
        (n >= 0) & (n < j),
        ab[i - 1 - a, np.clip(n, 0, j - 1)],
        np.zeros_like(ab.ravel()),
    )
    return np.reshape(At, ab.shape)


solve_banded = primitive(scipy.linalg.solve_banded)


def solve_banded_ab_vjp(ans, l_and_u, ab, b, **kwargs):
    i, j = ab.shape
    a, b = np.unravel_index(np.arange(i * j), (i, j))
    n = a + b - l_and_u[1]

    def _vjp(g):
        s = solve_banded(l_and_u[::-1], _transpose_banded(l_and_u, ab), g, **kwargs)
        Ag = np.where(
            (n >= 0) & (n < j),
            -s[np.clip(n, 0, j - 1)] * ans[b],
            np.zeros_like(ab.ravel()),
        )
        return np.reshape(Ag, (i, j))

    return _vjp


def solve_banded_b_vjp(ans, l_and_u, ab, b, **kwargs):
    def _vjp(g):
        return solve_banded(l_and_u[::-1], _transpose_banded(l_and_u, ab), g, **kwargs)

    return _vjp


defvjp(solve_banded, None, solve_banded_ab_vjp, solve_banded_b_vjp)
