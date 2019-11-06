import numpy as np
# import sw_06_cv_functions as sw6
from . import sw_06_cv_functions as sw6
import pytest

def test_conv2d():
    X = np.array([
        [1, 1, 2, 4, 5],
        [5, 6, 7, 8, 6],
        [3, 2, 1, 0, 7],
        [5, 1, 9, 3, 6],
        [1, 2, 3, 4, 8],
    ])
    kernel = np.asarray([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    expected = np.asarray([
        [6, 7, 8],
        [2, 1, 0],
        [1, 9, 3],
    ])
    result =sw6.conv2d(X, kernel, padding=0)
    assert np.array_equal(result, expected)


def test_separable_conv2d():
    X = np.array([
        [1, 1, 2, 4, 5],
        [5, 6, 7, 8, 6],
        [3, 2, 1, 0, 7],
        [5, 1, 9, 3, 6],
        [1, 2, 3, 4, 8],
    ])
    kernel = np.asarray([0, 1, 0])
    expected = np.asarray([
        [6, 7, 8],
        [2, 1, 0],
        [1, 9, 3],
    ])
    result = sw6.separable_conv2d(X, kernel, padding=0)
    assert np.array_equal(result, expected)


import numpy
def _gaussian_kernel1d(sigma, order, radius):
    """
    REFERENCE IMPLEMENTATION IN SCIPY (COPIED FROM https://github.com/scipy/scipy/blob/5681835ec51b728fa0ea6237d46aa8032b9e1400/scipy/ndimage/filters.py#L136)
    This is only used to verify that mine works.
    
    Computes a 1D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = numpy.arange(order + 1)
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius+1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = numpy.zeros(order + 1)
        q[0] = 1
        D = numpy.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = numpy.diag(numpy.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


@pytest.mark.parametrize("sigma,kernel_size", [
    (1, 5),
    (2, 10),
])
def test_gaussian_kernel_1d(sigma, kernel_size):
    radius = kernel_size // 2
    expected = _gaussian_kernel1d(sigma, 0, radius)
    result = sw6.gaussian_kernel_1d(sigma, kernel_size)
    assert np.allclose(result, expected)

@pytest.mark.parametrize("sigma,kernel_size", [
    (1, 5),
    (2, 10),
])
def test_gaussian_derivative(sigma, kernel_size):

    expected = _gaussian_kernel1d(sigma, 1, kernel_size//2)
    result = sw6.gaussian_derivative_1d(sigma, kernel_size)
    assert np.allclose(result, expected)
    
# test_gaussian_derivative(1, 5)