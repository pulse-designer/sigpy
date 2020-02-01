# -*- coding: utf-8 -*-
"""Interpolation functions.
"""
import math
import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['interpolate', 'gridding']


def interpolate(input, width, coord):
    """Interpolation from array to points specified by coordinates.

    Args:
        input (array): Input array of shape [..., ny, nx]
        width (float): Interpolation kernel width.
        coord (array): Coordinate array of shape [..., ndim]

    Returns:
        output (array): Output array of coord.shape[:-1]

    """
    ndim = coord.shape[-1]

    batch_shape = input.shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    xp = backend.get_array_module(input)

    input = input.reshape([batch_size] + list(input.shape[-ndim:]))
    coord = coord.reshape([npts, ndim])
    output = xp.zeros([batch_size, npts], dtype=input.dtype)

    _interpolate = _select_interpolate(ndim, xp)
    if xp == np:
        _interpolate(output, input, width, coord)
    else:  # pragma: no cover
        blocks = math.ceil(npts * batch_size / config.numba_cuda_threads)
        _interpolate[blocks, config.numba_cuda_threads](
            output, input, width, coord)

    return output.reshape(batch_shape + pts_shape)


def gridding(input, shape, width, coord):
    """Gridding of points specified by coordinates to array.

    Args:
        input (array): Input array.
        shape (array of ints): Output shape.
        width (float): Interpolation kernel width.
        coord (array): Coordinate array of shape [..., ndim]

    Returns:
        output (array): Output array.

    """
    ndim = coord.shape[-1]

    batch_shape = shape[:-ndim]
    batch_size = util.prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = util.prod(pts_shape)

    xp = backend.get_array_module(input)
    isreal = np.issubdtype(input.dtype, np.floating)

    input = input.reshape([batch_size, npts])
    coord = coord.reshape([npts, ndim])
    output = xp.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype)

    _gridding = _select_gridding(ndim, xp, isreal)
    if xp == np:
        _gridding(output, input, width, coord)
    else:  # pragma: no cover
        blocks = math.ceil(npts * batch_size / config.numba_cuda_threads)
        _gridding[blocks, config.numba_cuda_threads](
            output, input, width, coord)

    return output.reshape(shape)


def _select_interpolate(ndim, xp):
    if ndim == 1:
        if xp == np:
            _interpolate = _interpolate1
        else:  # pragma: no cover
            _interpolate = _interpolate1_cuda
    elif ndim == 2:
        if xp == np:
            _interpolate = _interpolate2
        else:  # pragma: no cover
            _interpolate = _interpolate2_cuda
    elif ndim == 3:
        if xp == np:
            _interpolate = _interpolate3
        else:  # pragma: no cover
            _interpolate = _interpolate3_cuda
    else:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    return _interpolate


def _select_gridding(ndim, xp, isreal):
    if ndim == 1:
        if xp == np:
            _gridding = _gridding1
        else:  # pragma: no cover
            if isreal:
                _gridding = _gridding1_cuda
            else:
                _gridding = _gridding1_cuda_complex
    elif ndim == 2:
        if xp == np:
            _gridding = _gridding2
        else:  # pragma: no cover
            if isreal:
                _gridding = _gridding2_cuda
            else:
                _gridding = _gridding2_cuda_complex
    elif ndim == 3:
        if xp == np:
            _gridding = _gridding3
        else:  # pragma: no cover
            if isreal:
                _gridding = _gridding3_cuda
            else:
                _gridding = _gridding3_cuda_complex
    else:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    return _gridding


@nb.jit(nopython=True)  # pragma: no cover
def linear_interpolate(x):
    if x > 1:
        return 0
    else:
        return 1 - abs(x)


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _interpolate1(output, input, width, coord):
    batch_size, nx = input.shape
    npts = coord.shape[0]

    for b in range(batch_size):
        for i in range(npts):
            kx = coord[i, -1]
            x0 = math.ceil(kx - width / 2)
            x1 = math.floor(kx + width / 2)
            for x in range(x0, x1 + 1):
                w = linear_interpolate((x - kx) / (width / 2))
                output[b, i] += w * input[b, x % nx]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
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
                wy = linear_interpolate((y - ky) / (width / 2))
                for x in range(x0, x1 + 1):
                    w = wy * linear_interpolate((x - kx) / (width / 2))
                    output[b, i] += w * input[b, y % ny, x % nx]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
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
                wz = linear_interpolate((z - kz) / (width / 2))
                for y in range(y0, y1 + 1):
                    wy = wz * linear_interpolate((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * linear_interpolate((x - kx) / (width / 2))
                        output[b, i] += w * input[b, z % nz, y % ny, x % nx]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _gridding1(output, input, width, coord):
    batch_size, nx = output.shape
    npts = coord.shape[0]

    for b in range(batch_size):
        for i in range(npts):
            kx = coord[i, -1]
            x0 = math.ceil(kx - width / 2)
            x1 = math.floor(kx + width / 2)
            for x in range(x0, x1 + 1):
                w = linear_interpolate((x - kx) / (width / 2))
                output[b, x % nx] += w * input[b, i]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
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
                wy = linear_interpolate((y - ky) / (width / 2))
                for x in range(x0, x1 + 1):
                    w = wy * linear_interpolate((x - kx) / (width / 2))
                    output[b, y % ny, x % nx] += w * input[b, i]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
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
                wz = linear_interpolate((z - kz) / (width / 2))
                for y in range(y0, y1 + 1):
                    wy = wz * linear_interpolate((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * linear_interpolate((x - kx) / (width / 2))
                        output[b, z % nz, y % ny, x % nx] += w * input[b, i]


if config.numba_cuda_enabled:
    import numba.cuda as nbc

    @nbc.jit(device=True)  # pragma: no cover
    def linear_interpolate_cuda(x):
        if x > 1:
            return 0
        else:
            return 1 - abs(x)

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
                w = linear_interpolate_cuda((x - kx) / (width / 2))
                output[b, i] += w * input[b, x % nx]

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
                wy = linear_interpolate_cuda((y - ky) / (width / 2))
                for x in range(x0, x1 + 1):
                    w = wy * linear_interpolate_cuda((x - kx) / (width / 2))
                    output[b, i] += w * input[b, y % ny, x % nx]

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
                wz = linear_interpolate_cuda((z - kz) / (width / 2))
                for y in range(y0, y1 + 1):
                    wy = wz * linear_interpolate_cuda((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * linear_interpolate_cuda((x - kx) / (width / 2))
                        output[b, i] += w * input[b, z % nz, y % ny, x % nx]

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
                w = linear_interpolate_cuda((x - kx) / (width / 2))
                val = w * input[b, i]
                nbc.atomic.add(output, (b, x % nx), val)

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
                wy = linear_interpolate_cuda((y - ky) / (width / 2))
                for x in range(x0, x1 + 1):
                    w = wy * linear_interpolate_cuda((x - kx) / (width / 2))
                    val = w * input[b, i]
                    nbc.atomic.add(output, (b, y % ny, x % nx), val)

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
                wz = linear_interpolate_cuda((z - kz) / (width / 2))
                for y in range(y0, y1 + 1):
                    wy = wz * linear_interpolate_cuda((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * linear_interpolate_cuda((x - kx) / (width / 2))
                        val = w * input[b, i]
                        nbc.atomic.add(
                            output, (b, z % nz, y % ny, x % nx), val)

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
                w = linear_interpolate_cuda((x - kx) / (width / 2))
                val = w * input[b, i]
                nbc.atomic.add(output.real, (b, x % nx), val.real)
                nbc.atomic.add(output.imag, (b, x % nx), val.imag)

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
                wy = linear_interpolate_cuda((y - ky) / (width / 2))
                for x in range(x0, x1 + 1):
                    w = wy * linear_interpolate_cuda((x - kx) / (width / 2))
                    val = w * input[b, i]
                    nbc.atomic.add(output.real, (b, y % ny, x % nx), val.real)
                    nbc.atomic.add(output.imag, (b, y % ny, x % nx), val.imag)

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
                wz = linear_interpolate_cuda((z - kz) / (width / 2))
                for y in range(y0, y1 + 1):
                    wy = wz * linear_interpolate_cuda((y - ky) / (width / 2))
                    for x in range(x0, x1 + 1):
                        w = wy * linear_interpolate_cuda((x - kx) / (width / 2))
                        val = w * input[b, i]
                        nbc.atomic.add(
                            output.real, (b, z % nz, y % ny, x % nx), val.real)
                        nbc.atomic.add(
                            output.imag, (b, z % nz, y % ny, x % nx), val.imag)
