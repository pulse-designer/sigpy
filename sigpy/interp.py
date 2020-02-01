# -*- coding: utf-8 -*-
"""Interpolation functions.
"""
import math
import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['interpolate', 'gridding',
           'linear_kernel', 'get_kaiser_bessel_kernel']


def linear_kernel(x):
    r"""Linear spline kernel.

    Computes :math:`f(x) = 1 - |x|` for x between -1 and 1.
    Otherwise, returns 0.

    Args:
        x (float): input.

    Returns:
       float: output.

    """
    abs_x = abs(x)
    if abs_x > 1:
        return 0
    else:
        return 1 - abs_x


def get_kaiser_bessel_kernel(beta):
    r"""Create Kaiser-Bessel kernel function with given beta.

    The function computes :math:`f(x) = I_0(\beta \sqrt{1 - x^2})`
    for x between -1 and 1, where :math:`I_0` is the modified
    Bessel function of the first kind. Otherwise, returns 0.

    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.

    Args:
        beta (float): Kaiser-Bessel smoothness parameter.

    Returns:
       function: Kaiser-Bessel function.

    References:
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """
    def kaiser_bessel_kernel(x):
        if abs(x) > 1:
            return 0

        x = beta * (1 - x**2)**0.5
        t = x / 3.75
        if x < 3.75:
            return 1 + 3.5156229 * t**2 + 3.0899424 * t**4 +\
                1.2067492 * t**6 + 0.2659732 * t**8 +\
                0.0360768 * t**10 + 0.0045813 * t**12
        else:
            return x**-0.5 * math.exp(x) * (
                0.39894228 + 0.01328592 * t**-1 +
                0.00225319 * t**-2 - 0.00157565 * t**-3 +
                0.00916281 * t**-4 - 0.02057706 * t**-5 +
                0.02635537 * t**-6 - 0.01647633 * t**-7 +
                0.00392377 * t**-8)

    return kaiser_bessel_kernel


def interpolate(input, coord, width=2, kernel=linear_kernel):
    """Interpolation from array to points specified by coordinates.

    Args:
        input (array): Input array of shape [..., ny, nx]
        coord (array): Coordinate array of shape [..., ndim]
        width (float): Interpolation kernel width.
        kernel (function): interpolation kernel function.

    Returns:
        output (array): Output array of coord.shape[:-1]

    """
    xp = backend.get_array_module(input)
    ndim = coord.shape[-1]

    batch_shape = input.shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    input = input.reshape([batch_size] + list(input.shape[-ndim:]))
    coord = coord.reshape([npts, ndim])
    output = xp.zeros([batch_size, npts], dtype=input.dtype)

    _interpolate = _get_interpolate(ndim, xp, kernel, npts, batch_size)
    _interpolate(output, input, width, coord)

    return output.reshape(batch_shape + pts_shape)


def gridding(input, shape, coord, width=2, kernel=linear_kernel):
    """Gridding of points specified by coordinates to array.

    Args:
        input (array): Input array.
        shape (array of ints): Output shape.
        coord (array): Coordinate array of shape [..., ndim]
        width (float): Interpolation kernel width.
        kernel (function): interpolation kernel function.

    Returns:
        output (array): Output array.

    """
    xp = backend.get_array_module(input)
    ndim = coord.shape[-1]

    batch_shape = shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    xp = backend.get_array_module(input)

    input = input.reshape([batch_size, npts])
    coord = coord.reshape([npts, ndim])
    output = xp.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype)

    isreal = np.issubdtype(input.dtype, np.floating)
    _gridding = _get_gridding(ndim, xp, kernel, npts, batch_size, isreal)
    _gridding(output, input, width, coord)

    return output.reshape(shape)


def _get_interpolate(ndim, xp, kernel, npts, batch_size):
    if ndim > 3 or ndim < 1:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    if xp == np:
        if ndim == 1:
            _interpolate = _get_interpolate1(kernel)
        elif ndim == 2:
            _interpolate = _get_interpolate2(kernel)
        elif ndim == 3:
            _interpolate = _get_interpolate3(kernel)
    else:  # pragma: no cover
        threads = config.numba_cuda_threads
        blocks = math.ceil(npts * batch_size / threads)
        if ndim == 1:
            _interpolate = _get_interpolate1_cuda(kernel)[blocks, threads]
        elif ndim == 2:
            _interpolate = _get_interpolate2_cuda(kernel)[blocks, threads]
        elif ndim == 3:
            _interpolate = _get_interpolate3_cuda(kernel)[blocks, threads]

    return _interpolate


def _get_gridding(ndim, xp, kernel, npts, batch_size, isreal):
    if ndim > 3 or ndim < 1:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    if xp == np:
        if ndim == 1:
            _gridding = _get_gridding1(kernel)
        elif ndim == 2:
            _gridding = _get_gridding2(kernel)
        elif ndim == 3:
            _gridding = _get_gridding3(kernel)
    else:  # pragma: no cover
        threads = config.numba_cuda_threads
        blocks = math.ceil(npts * batch_size / threads)

        if isreal:
            if ndim == 1:
                _gridding = _get_gridding1_cuda(kernel)[blocks, threads]
            elif ndim == 2:
                _gridding = _get_gridding2_cuda(kernel)[blocks, threads]
            elif ndim == 3:
                _gridding = _get_gridding3_cuda(kernel)[blocks, threads]
        else:
            if ndim == 1:
                _gridding = _get_gridding1_cuda_complex(kernel)[
                    blocks, threads]
            elif ndim == 2:
                _gridding = _get_gridding2_cuda_complex(kernel)[
                    blocks, threads]
            elif ndim == 3:
                _gridding = _get_gridding3_cuda_complex(kernel)[
                    blocks, threads]

    return _gridding


def _get_interpolate1(kernel):
    kernel = nb.jit(kernel, nopython=True)

    @nb.jit(nopython=True)  # pragma: no cover
    def _interpolate1(output, input, width, coord):
        batch_size, nx = input.shape
        npts = coord.shape[0]

        for b in range(batch_size):
            for i in range(npts):
                kx = coord[i, -1]
                x0 = math.ceil(kx - width / 2)
                x1 = math.floor(kx + width / 2)
                for x in range(x0, x1 + 1):
                    w = kernel((x - kx) / (width / 2))
                    output[b, i] += w * input[b, x % nx]

    return _interpolate1


def _get_interpolate2(kernel):
    kernel = nb.jit(kernel, nopython=True)

    @nb.jit(nopython=True)  # pragma: no cover
    def _interpolate2(output, input, width, coord):
        batch_size, ny, nx = input.shape
        npts = coord.shape[0]

        for b in range(batch_size):
            for i in range(npts):
                kx, ky = coord[i, -1], coord[i, -2]
                x0, y0 = (math.ceil(kx - width / 2),
                          math.ceil(ky - width / 2))
                x1, y1 = (math.floor(kx + width / 2),
                          math.floor(ky + width / 2))
                for y in range(y0, y1 + 1):
                    wy = kernel((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * kernel((x - kx) / (width / 2))
                        output[b, i] += w * input[b, y % ny, x % nx]

    return _interpolate2


def _get_interpolate3(kernel):
    kernel = nb.jit(kernel, nopython=True)

    @nb.jit(nopython=True)  # pragma: no cover
    def _interpolate3(output, input, width, coord):
        batch_size, nz, ny, nx = input.shape
        npts = coord.shape[0]

        for b in range(batch_size):
            for i in range(npts):
                kx, ky, kz = coord[i, -1], coord[i, -2], coord[i, -3]
                x0, y0, z0 = (math.ceil(kx - width / 2),
                              math.ceil(ky - width / 2),
                              math.ceil(kz - width / 2))
                x1, y1, z1 = (math.floor(kx + width / 2),
                              math.floor(ky + width / 2),
                              math.floor(kz + width / 2))
                for z in range(z0, z1 + 1):
                    wz = kernel((z - kz) / (width / 2))
                    for y in range(y0, y1 + 1):
                        wy = wz * kernel((y - ky) / (width / 2))
                        for x in range(x0, x1 + 1):
                            w = wy * kernel((x - kx) / (width / 2))
                            output[b, i] += w * input[
                                b, z % nz, y % ny, x % nx]

    return _interpolate3


def _get_gridding1(kernel):
    kernel = nb.jit(kernel, nopython=True)

    @nb.jit(nopython=True)  # pragma: no cover
    def _gridding1(output, input, width, coord):
        batch_size, nx = output.shape
        npts = coord.shape[0]

        for b in range(batch_size):
            for i in range(npts):
                kx = coord[i, -1]
                x0 = math.ceil(kx - width / 2)
                x1 = math.floor(kx + width / 2)
                for x in range(x0, x1 + 1):
                    w = kernel((x - kx) / (width / 2))
                    output[b, x % nx] += w * input[b, i]

    return _gridding1


def _get_gridding2(kernel):
    kernel = nb.jit(kernel, nopython=True)

    @nb.jit(nopython=True)  # pragma: no cover
    def _gridding2(output, input, width, coord):
        batch_size, ny, nx = output.shape
        npts = coord.shape[0]

        for b in range(batch_size):
            for i in range(npts):
                kx, ky = coord[i, -1], coord[i, -2]
                x0, y0 = (math.ceil(kx - width / 2),
                          math.ceil(ky - width / 2))
                x1, y1 = (math.floor(kx + width / 2),
                          math.floor(ky + width / 2))
                for y in range(y0, y1 + 1):
                    wy = kernel((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * kernel((x - kx) / (width / 2))
                        output[b, y % ny, x % nx] += w * input[b, i]

    return _gridding2


def _get_gridding3(kernel):
    kernel = nb.jit(kernel, nopython=True)

    @nb.jit(nopython=True)  # pragma: no cover
    def _gridding3(output, input, width, coord):
        batch_size, nz, ny, nx = output.shape
        npts = coord.shape[0]

        for b in range(batch_size):
            for i in range(npts):
                kx, ky, kz = coord[i, -1], coord[i, -2], coord[i, -3]
                x0, y0, z0 = (math.ceil(kx - width / 2),
                              math.ceil(ky - width / 2),
                              math.ceil(kz - width / 2))
                x1, y1, z1 = (math.floor(kx + width / 2),
                              math.floor(ky + width / 2),
                              math.floor(kz + width / 2))
                for z in range(z0, z1 + 1):
                    wz = kernel((z - kz) / (width / 2))
                    for y in range(y0, y1 + 1):
                        wy = wz * kernel((y - ky) / (width / 2))
                        for x in range(x0, x1 + 1):
                            w = wy * kernel((x - kx) / (width / 2))
                            val = w * input[b, i]
                            output[b, z % nz, y % ny, x % nx] += val

    return _gridding3


if config.numba_cuda_enabled:
    import numba.cuda as nbc

    def _get_interpolate1_cuda(kernel):
        kernel = nbc.jit(kernel, device=True)

        @nbc.jit()  # pragma: no cover
        def _interpolate1_cuda(output, input, width, coord):
            batch_size, nx = input.shape
            npts = coord.shape[0]

            pos = nbc.grid(1)
            b = pos // npts
            pos -= b * npts
            i = pos
            if i < npts and b < batch_size:
                kx = coord[i, -1]
                x0 = math.ceil(kx - width / 2)
                x1 = math.floor(kx + width / 2)
                for x in range(x0, x1 + 1):
                    w = kernel((x - kx) / (width / 2))
                    output[b, i] += w * input[b, x % nx]

        return _interpolate1_cuda

    def _get_interpolate2_cuda(kernel):
        kernel = nbc.jit(kernel, device=True)

        @nbc.jit()  # pragma: no cover
        def _interpolate2_cuda(output, input, width, coord):
            batch_size, ny, nx = input.shape
            npts = coord.shape[0]

            pos = nbc.grid(1)
            b = pos // npts
            pos -= b * npts
            i = pos
            if i < npts and b < batch_size:
                kx, ky = coord[i, -1], coord[i, -2]
                x0, y0 = (math.ceil(kx - width / 2),
                          math.ceil(ky - width / 2))
                x1, y1 = (math.floor(kx + width / 2),
                          math.floor(ky + width / 2))
                for y in range(y0, y1 + 1):
                    wy = kernel((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * kernel((x - kx) / (width / 2))
                        output[b, i] += w * input[b, y % ny, x % nx]

        return _interpolate2_cuda

    def _get_interpolate3_cuda(kernel):
        kernel = nbc.jit(kernel, device=True)

        @nbc.jit()  # pragma: no cover
        def _interpolate3_cuda(output, input, width, coord):
            batch_size, nz, ny, nx = input.shape
            npts = coord.shape[0]

            pos = nbc.grid(1)
            b = pos // npts
            pos -= b * npts
            i = pos
            if i < npts and b < batch_size:
                kx, ky, kz = coord[i, -1], coord[i, -2], coord[i, -3]
                x0, y0, z0 = (math.ceil(kx - width / 2),
                              math.ceil(ky - width / 2),
                              math.ceil(kz - width / 2))
                x1, y1, z1 = (math.floor(kx + width / 2),
                              math.floor(ky + width / 2),
                              math.floor(kz + width / 2))
                for z in range(z0, z1 + 1):
                    wz = kernel((z - kz) / (width / 2))
                    for y in range(y0, y1 + 1):
                        wy = wz * kernel((y - ky) / (width / 2))
                        for x in range(x0, x1 + 1):
                            w = wy * kernel((x - kx) / (width / 2))
                            val = w * input[b, z % nz, y % ny, x % nx]
                            output[b, i] += val

        return _interpolate3_cuda

    def _get_gridding1_cuda(kernel):
        kernel = nbc.jit(kernel, device=True)

        @nbc.jit()  # pragma: no cover
        def _gridding1_cuda(output, input, width, coord):
            batch_size, nx = output.shape
            npts = coord.shape[0]

            pos = nbc.grid(1)
            b = pos // npts
            pos -= b * npts
            i = pos
            if i < npts and b < batch_size:
                kx = coord[i, -1]
                x0 = math.ceil(kx - width / 2)
                x1 = math.floor(kx + width / 2)
                for x in range(x0, x1 + 1):
                    w = kernel((x - kx) / (width / 2))
                    val = w * input[b, i]
                    nbc.atomic.add(output, (b, x % nx), val)

        return _gridding1_cuda

    def _get_gridding2_cuda(kernel):
        kernel = nbc.jit(kernel, device=True)

        @nbc.jit()  # pragma: no cover
        def _gridding2_cuda(output, input, width, coord):
            batch_size, ny, nx = output.shape
            npts = coord.shape[0]

            pos = nbc.grid(1)
            b = pos // npts
            pos -= b * npts
            i = pos
            if i < npts and b < batch_size:
                kx, ky = coord[i, -1], coord[i, -2]
                x0, y0 = (math.ceil(kx - width / 2),
                          math.ceil(ky - width / 2))
                x1, y1 = (math.floor(kx + width / 2),
                          math.floor(ky + width / 2))
                for y in range(y0, y1 + 1):
                    wy = kernel((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * kernel((x - kx) / (width / 2))
                        val = w * input[b, i]
                        nbc.atomic.add(output, (b, y % ny, x % nx), val)

        return _gridding2_cuda

    def _get_gridding3_cuda(kernel):
        kernel = nbc.jit(kernel, device=True)

        @nbc.jit()  # pragma: no cover
        def _gridding3_cuda(output, input, width, coord):
            batch_size, nz, ny, nx = output.shape
            npts = coord.shape[0]

            pos = nbc.grid(1)
            b = pos // npts
            pos -= b * npts
            i = pos
            if i < npts and b < batch_size:
                kx, ky, kz = coord[i, -1], coord[i, -2], coord[i, -3]
                x0, y0, z0 = (math.ceil(kx - width / 2),
                              math.ceil(ky - width / 2),
                              math.ceil(kz - width / 2))
                x1, y1, z1 = (math.floor(kx + width / 2),
                              math.floor(ky + width / 2),
                              math.floor(kz + width / 2))
                for z in range(z0, z1 + 1):
                    wz = kernel((z - kz) / (width / 2))
                    for y in range(y0, y1 + 1):
                        wy = wz * kernel((y - ky) / (width / 2))
                        for x in range(x0, x1 + 1):
                            w = wy * kernel((x - kx) / (width / 2))
                            val = w * input[b, i]
                            nbc.atomic.add(
                                output, (b, z % nz, y % ny, x % nx), val)

        return _gridding3_cuda

    def _get_gridding1_cuda_complex(kernel):
        kernel = nbc.jit(kernel, device=True)

        @nbc.jit()  # pragma: no cover
        def _gridding1_cuda_complex(output, input, width, coord):
            batch_size, nx = output.shape
            npts = coord.shape[0]

            pos = nbc.grid(1)
            b = pos // npts
            pos -= b * npts
            i = pos
            if i < npts and b < batch_size:
                kx = coord[i, -1]
                x0 = math.ceil(kx - width / 2)
                x1 = math.floor(kx + width / 2)
                for x in range(x0, x1 + 1):
                    w = kernel((x - kx) / (width / 2))
                    val = w * input[b, i]
                    nbc.atomic.add(output.real, (b, x % nx), val.real)
                    nbc.atomic.add(output.imag, (b, x % nx), val.imag)

        return _gridding1_cuda_complex

    def _get_gridding2_cuda_complex(kernel):
        kernel = nbc.jit(kernel, device=True)

        @nbc.jit()  # pragma: no cover
        def _gridding2_cuda_complex(output, input, width, coord):
            batch_size, ny, nx = output.shape
            npts = coord.shape[0]

            pos = nbc.grid(1)
            b = pos // npts
            pos -= b * npts
            i = pos
            if i < npts and b < batch_size:
                kx, ky = coord[i, -1], coord[i, -2]
                x0, y0 = (math.ceil(kx - width / 2),
                          math.ceil(ky - width / 2))
                x1, y1 = (math.floor(kx + width / 2),
                          math.floor(ky + width / 2))
                for y in range(y0, y1 + 1):
                    wy = kernel((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * kernel((x - kx) / (width / 2))
                        val = w * input[b, i]
                        nbc.atomic.add(
                            output.real, (b, y % ny, x % nx), val.real)
                        nbc.atomic.add(
                            output.imag, (b, y % ny, x % nx), val.imag)

        return _gridding2_cuda_complex

    def _get_gridding3_cuda_complex(kernel):
        kernel = nbc.jit(kernel, device=True)

        @nbc.jit()  # pragma: no cover
        def _gridding3_cuda_complex(output, input, width, coord):
            batch_size, nz, ny, nx = output.shape
            npts = coord.shape[0]

            pos = nbc.grid(1)
            b = pos // npts
            pos -= b * npts
            i = pos
            if i < npts and b < batch_size:
                kx, ky, kz = coord[i, -1], coord[i, -2], coord[i, -3]
                x0, y0, z0 = (math.ceil(kx - width / 2),
                              math.ceil(ky - width / 2),
                              math.ceil(kz - width / 2))
                x1, y1, z1 = (math.floor(kx + width / 2),
                              math.floor(ky + width / 2),
                              math.floor(kz + width / 2))
                for z in range(z0, z1 + 1):
                    wz = kernel((z - kz) / (width / 2))
                    for y in range(y0, y1 + 1):
                        wy = wz * kernel((y - ky) / (width / 2))
                        for x in range(x0, x1 + 1):
                            w = wy * kernel((x - kx) / (width / 2))
                            val = w * input[b, i]
                            nbc.atomic.add(
                                output.real,
                                (b, z % nz, y % ny, x % nx), val.real)
                            nbc.atomic.add(
                                output.imag,
                                (b, z % nz, y % ny, x % nx), val.imag)

        return _gridding3_cuda_complex
