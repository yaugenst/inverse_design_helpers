import scipy.ndimage
from autograd.extend import defjvp, defvjp, primitive

gaussian_filter = primitive(scipy.ndimage.gaussian_filter)
gaussian_laplace = primitive(scipy.ndimage.gaussian_laplace)
laplace = primitive(scipy.ndimage.laplace)
prewitt = primitive(scipy.ndimage.prewitt)
sobel = primitive(scipy.ndimage.sobel)
uniform_filter = primitive(scipy.ndimage.uniform_filter)
fourier_ellipsoid = primitive(scipy.ndimage.fourier_ellipsoid)
fourier_gaussian = primitive(scipy.ndimage.fourier_gaussian)
fourier_uniform = primitive(scipy.ndimage.fourier_uniform)
fourier_shift = primitive(scipy.ndimage.fourier_shift)

defjvp(gaussian_filter, "same")
defjvp(gaussian_laplace, "same")
defjvp(laplace, "same")
defjvp(prewitt, "same")
defjvp(sobel, "same")
defjvp(uniform_filter, "same")
defjvp(fourier_ellipsoid, "same")

defvjp(
    gaussian_filter,
    lambda ans, x, *args, **kwargs: lambda g: gaussian_filter(g, *args, **kwargs),
)
defvjp(
    gaussian_laplace,
    lambda ans, x, *args, **kwargs: lambda g: gaussian_laplace(g, *args, **kwargs),
)
defvjp(
    laplace,
    lambda ans, x, *args, **kwargs: lambda g: laplace(g, *args, **kwargs),
)
defvjp(
    prewitt,
    lambda ans, x, *args, **kwargs: lambda g: prewitt(-g, *args, **kwargs),
)
defvjp(
    sobel,
    lambda ans, x, *args, **kwargs: lambda g: sobel(-g, *args, **kwargs),
)
defvjp(
    uniform_filter,
    lambda ans, x, *args, **kwargs: lambda g: uniform_filter(g, *args, **kwargs),
)
defvjp(
    fourier_ellipsoid,
    lambda ans, x, *args, **kwargs: lambda g: fourier_ellipsoid(g, *args, **kwargs),
)

defvjp(
    fourier_gaussian,
    lambda ans, x, *args, **kwargs: lambda g: fourier_gaussian(g, *args, **kwargs),
)

defvjp(
    fourier_uniform,
    lambda ans, x, *args, **kwargs: lambda g: fourier_uniform(g, *args, **kwargs),
)

defvjp(
    fourier_shift,
    lambda ans, x, *args, **kwargs: lambda g: fourier_shift(g, *args, **kwargs),
)
