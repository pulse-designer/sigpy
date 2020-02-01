import unittest
import numpy as np
from sigpy import interp, config

if config.cupy_enabled:
    import cupy as cp

if __name__ == '__main__':
    unittest.main()


class TestInterp(unittest.TestCase):

    def test_linear_kernel(self):
        x = [-2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2]
        y = [0, 0, 0.5, 0.9, 1, 0.9, 0.5, 0, 0]
        for i in range(len(x)):
            assert y[i] == interp.linear_kernel(x[i])

    def test_get_kaiser_bessel_kernel(self):
        x = [-2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2]
        for beta in range(10):
            kernel = interp.get_kaiser_bessel_kernel(beta)
            for i in range(len(x)):
                if abs(x[i]) > 1:
                    np.testing.assert_allclose(0, kernel(x[i]))
                else:
                    np.testing.assert_allclose(
                        np.i0(beta * (1 - x[i]**2)**0.5), kernel(x[i]),
                        rtol=1e-7, atol=1e-7)

    def test_interpolate(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)

        batch = 2
        for xp in xps:
            for dtype in [np.float32, np.complex64]:
                for ndim in [1, 2, 3]:
                    with self.subTest(ndim=ndim, xp=xp, dtype=dtype):
                        shape = [3] + [1] * (ndim - 1)
                        coord = xp.array([[0.1] + [0] * (ndim - 1),
                                          [1.1] + [0] * (ndim - 1),
                                          [2.1] + [0] * (ndim - 1)])

                        input = xp.array([[0, 1.0, 0]] * batch, dtype=dtype)
                        input = input.reshape([batch] + shape)
                        output = interp.interpolate(input, coord)
                        output_expected = xp.array([[0.1, 0.9, 0]] * batch)
                        xp.testing.assert_allclose(output, output_expected,
                                                   atol=1e-7)

    def test_gridding(self):
        xps = [np]
        if config.cupy_enabled:
            xps.append(cp)

        batch = 2
        for xp in xps:
            for dtype in [np.float32, np.complex64]:
                for ndim in [1, 2, 3]:
                    with self.subTest(ndim=ndim, xp=xp, dtype=dtype):
                        shape = [3] + [1] * (ndim - 1)
                        coord = xp.array([[0.1] + [0] * (ndim - 1),
                                          [1.1] + [0] * (ndim - 1),
                                          [2.1] + [0] * (ndim - 1)])

                        input = xp.array([[0, 1.0, 0]] * batch, dtype=dtype)
                        output = interp.gridding(input, [batch] + shape, coord)
                        output_expected = xp.array(
                            [[0, 0.9, 0.1]] * batch).reshape([batch] + shape)
                        xp.testing.assert_allclose(output, output_expected,
                                                   atol=1e-7)
