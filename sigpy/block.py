# -*- coding: utf-8 -*-
"""Block reshape functions.

"""
import math
import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['array_to_blocks', 'blocks_to_array']


def array_to_blocks(input, blk_shape, blk_strides):
    """Extract blocks from an array in a sliding window manner.

    Args:
        input (array): input array of shape [..., N_1, ..., N_D]
        blk_shape (tuple): block shape of length D, with D <= 4.
        blk_strides (tuple): block strides of length D.

    Returns:
        array: array of shape [...] + num_blks + blk_shape, where
            num_blks = (N - blk_shape + blk_strides) // blk_strides.

    Example:

        >>> input = np.array([0, 1, 2, 3, 4, 5])
        >>> print(array_to_blocks(input, [2], [2]))
        [[0, 1],
         [2, 3],
         [4, 5]]

    """
    if len(blk_shape) != len(blk_strides):
        raise ValueError('blk_shape must have the same length as blk_strides.')

    ndim = len(blk_shape)
    blk_shape = tuple(blk_shape)
    blk_strides = tuple(blk_strides)
    num_blks = tuple(
        (i - b + s) // s
        for i, b, s in zip(input.shape[-ndim:], blk_shape, blk_strides))
    batch_shape = input.shape[:-ndim]
    batch_size = util.prod(batch_shape)

    xp = backend.get_array_module(input)
    output = xp.zeros((batch_size, ) + num_blks + blk_shape, dtype=input.dtype)
    input = input.reshape((batch_size, ) + input.shape[-ndim:])

    _array_to_blocks = _select_array_to_blocks(ndim, xp)
    if xp == np:
        _array_to_blocks(
            output, input, batch_size, blk_shape, blk_strides, num_blks)
    else:
        n = batch_size * util.prod(blk_shape) * util.prod(num_blks)
        blocks = math.ceil(n / config.numba_cuda_threads)

        _array_to_blocks[blocks, config.numba_cuda_threads](
            output, input, batch_size, xp.asarray(blk_shape),
            xp.asarray(blk_strides), xp.asarray(num_blks))

    return output.reshape(batch_shape + num_blks + blk_shape)


def blocks_to_array(input, oshape, blk_shape, blk_strides):
    """Accumulate blocks into an array in a sliding window manner.

    Args:
        input (array): input array of shape [...] + num_blks + blk_shape
        oshape (tuple): output shape.
        blk_shape (tuple): block shape of length D.
        blk_strides (tuple): block strides of length D.

    Returns:
        array: array of shape oshape.

    """
    if len(blk_shape) != len(blk_strides):
        raise ValueError('blk_shape must have the same length as blk_strides.')

    ndim = len(blk_shape)
    blk_shape = tuple(blk_shape)
    blk_strides = tuple(blk_strides)
    num_blks = input.shape[-(2 * ndim):-ndim]
    batch_shape = tuple(oshape[:-ndim])
    batch_size = util.prod(batch_shape)

    xp = backend.get_array_module(input)
    output = xp.zeros((batch_size, ) + tuple(oshape[-ndim:]),
                      dtype=input.dtype)
    input = input.reshape((batch_size, ) + input.shape[-2 * ndim:])
    _blocks_to_array = _select_blocks_to_array(ndim, xp)
    if xp == np:
        _blocks_to_array(
            output, input, batch_size, blk_shape, blk_strides, num_blks)
    else:
        n = batch_size * util.prod(blk_shape) * util.prod(num_blks)
        blocks = math.ceil(n / config.numba_cuda_threads)

        _blocks_to_array[blocks, config.numba_cuda_threads](
            output, input, batch_size,
            xp.asarray(blk_shape), xp.asarray(blk_strides),
            xp.asarray(num_blks))

    return output.reshape(oshape)


def _select_array_to_blocks(ndim, xp):
    if ndim == 1:
        if xp == np:
            _array_to_blocks = _array_to_blocks1
        else:  # pragma: no cover
            _array_to_blocks = _array_to_blocks1_cuda
    elif ndim == 2:
        if xp == np:
            _array_to_blocks = _array_to_blocks2
        else:  # pragma: no cover
            _array_to_blocks = _array_to_blocks2_cuda
    elif ndim == 3:
        if xp == np:
            _array_to_blocks = _array_to_blocks3
        else:  # pragma: no cover
            _array_to_blocks = _array_to_blocks3_cuda
    else:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    return _array_to_blocks


def _select_blocks_to_array(ndim, xp):
    if ndim == 1:
        if xp == np:
            _blocks_to_array = _blocks_to_array1
        else:  # pragma: no cover
            _blocks_to_array = _blocks_to_array1_cuda
    elif ndim == 2:
        if xp == np:
            _blocks_to_array = _blocks_to_array2
        else:  # pragma: no cover
            _blocks_to_array = _blocks_to_array2_cuda
    elif ndim == 3:
        if xp == np:
            _blocks_to_array = _blocks_to_array3
        else:  # pragma: no cover
            _blocks_to_array = _blocks_to_array3_cuda
    else:
        raise ValueError(
            'Number of dimensions can only be 1, 2 or 3, got {}'.format(ndim))

    return _blocks_to_array


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _array_to_blocks1(output, input, batch_size,
                      blk_shape, blk_strides, num_blks):
    for b in range(batch_size):
        for nx in range(num_blks[-1]):
            for bx in range(blk_shape[-1]):
                ix = nx * blk_strides[-1] + bx
                if ix < input.shape[-1]:
                    output[b, nx, bx] = input[b, ix]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _array_to_blocks2(output, input, batch_size,
                      blk_shape, blk_strides, num_blks):
    for b in range(batch_size):
        for ny in range(num_blks[-2]):
            for nx in range(num_blks[-1]):
                for by in range(blk_shape[-2]):
                    for bx in range(blk_shape[-1]):
                        iy = ny * blk_strides[-2] + by
                        ix = nx * blk_strides[-1] + bx
                        if ix < input.shape[-1] and iy < input.shape[-2]:
                            output[b, ny, nx, by, bx] = input[b, iy, ix]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _array_to_blocks3(output, input, batch_size,
                      blk_shape, blk_strides, num_blks):
    for b in range(batch_size):
        for nz in range(num_blks[-3]):
            for ny in range(num_blks[-2]):
                for nx in range(num_blks[-1]):
                    for bz in range(blk_shape[-3]):
                        for by in range(blk_shape[-2]):
                            for bx in range(blk_shape[-1]):
                                iz = nz * blk_strides[-3] + bz
                                iy = ny * blk_strides[-2] + by
                                ix = nx * blk_strides[-1] + bx
                                if (ix < input.shape[-1] and
                                    iy < input.shape[-2] and
                                    iz < input.shape[-3]):
                                    output[b, nz, ny, nx, bz, by,
                                           bx] = input[b, iz, iy, ix]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _blocks_to_array1(output, input, batch_size,
                      blk_shape, blk_strides, num_blks):
    for b in range(batch_size):
        for ix in range(output.shape[-1]):
            rx = ix % blk_strides[-1]
            for bx in range(rx, blk_shape[-1], blk_strides[-1]):
                nx = (ix - bx) // blk_strides[-1]
                if nx >= 0 and nx < num_blks[-1]:
                    output[b, ix] += input[b, nx, bx]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _blocks_to_array2(output, input, batch_size,
                      blk_shape, blk_strides, num_blks):
    for b in range(batch_size):
        for iy in range(output.shape[-2]):
            for ix in range(output.shape[-1]):
                ry = iy % blk_strides[-2]
                rx = ix % blk_strides[-1]
                for by in range(ry, blk_shape[-2], blk_strides[-2]):
                    for bx in range(rx, blk_shape[-1], blk_strides[-1]):
                        ny = (iy - by) // blk_strides[-2]
                        nx = (ix - bx) // blk_strides[-1]
                        if nx >= 0 and nx < num_blks[-1] \
                           and ny >= 0 and ny < num_blks[-2]:
                            output[b, iy, ix] += input[b, ny, nx, by, bx]


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _blocks_to_array3(output, input, batch_size,
                      blk_shape, blk_strides, num_blks):
    for b in range(batch_size):
        for iz in range(output.shape[-3]):
            for iy in range(output.shape[-2]):
                for ix in range(output.shape[-1]):
                    rz = iz % blk_strides[-3]
                    ry = iy % blk_strides[-2]
                    rx = ix % blk_strides[-1]
                    for bz in range(rz, blk_shape[-3], blk_strides[-3]):
                        for by in range(ry, blk_shape[-2], blk_strides[-2]):
                            for bx in range(rx, blk_shape[-1],
                                            blk_strides[-1]):
                                nz = (iz - bz) // blk_strides[-3]
                                ny = (iy - by) // blk_strides[-2]
                                nx = (ix - bx) // blk_strides[-1]
                                if nx >= 0 and nx < num_blks[-1] \
                                   and ny >= 0 and ny < num_blks[-2] \
                                   and nz >= 0 and nz < num_blks[-3]:
                                    output[b, iz, iy, ix] += input[b, nz,
                                                                   ny, nx,
                                                                   bz, by, bx]


if config.numba_cuda_enabled:
    import numba.cuda as nbc

    @nbc.jit()  # pragma: no cover
    def _array_to_blocks1_cuda(output, input, batch_size,
                               blk_shape, blk_strides, num_blks):
        i = nbc.grid(1)
        b = i // num_blks[-1]
        i -= b * num_blks[-1]
        nx = i

        if b < batch_size and nx < num_blks[-1]:
            for bx in range(blk_shape[-1]):
                ix = nx * blk_strides[-1] + bx
                if ix < input.shape[-1]:
                    output[b, nx, bx] = input[b, ix]

    @nbc.jit()  # pragma: no cover
    def _array_to_blocks2_cuda(output, input, batch_size,
                               blk_shape, blk_strides, num_blks):
        i = nbc.grid(1)
        b = i // num_blks[-1] // num_blks[-2]
        i -= b * num_blks[-1] * num_blks[-2]
        ny = i // num_blks[-1]
        i -= ny * num_blks[-1]
        nx = i

        if b < batch_size and nx < num_blks[-1] \
           and ny < num_blks[-2]:
            for by in range(blk_shape[-2]):
                for bx in range(blk_shape[-1]):
                    iy = ny * blk_strides[-2] + by
                    ix = nx * blk_strides[-1] + bx
                    if ix < input.shape[-1] and iy < input.shape[-2]:
                        output[b, ny, nx, by, bx] = input[b, iy, ix]

    @nbc.jit()  # pragma: no cover
    def _array_to_blocks3_cuda(output, input, batch_size,
                               blk_shape, blk_strides, num_blks):
        i = nbc.grid(1)
        b = i // num_blks[-1] // num_blks[-2] // num_blks[-3]
        i -= b * num_blks[-1] * num_blks[-2] * num_blks[-3]
        nz = i // num_blks[-1] // num_blks[-2]
        i -= nz * num_blks[-1] * num_blks[-2]
        ny = i // num_blks[-1]
        i -= ny * num_blks[-1]
        nx = i

        if b < batch_size and nx < num_blks[-1] \
           and ny < num_blks[-2] \
           and nz < num_blks[-3]:
            for bz in range(blk_shape[-3]):
                for by in range(blk_shape[-2]):
                    for bx in range(blk_shape[-1]):
                        iz = nz * blk_strides[-3] + bz
                        iy = ny * blk_strides[-2] + by
                        ix = nx * blk_strides[-1] + bx
                        if (ix < input.shape[-1] and
                            iy < input.shape[-2] and
                            iz < input.shape[-3]):
                            output[b, nz, ny, nx, bz, by,
                                   bx] = input[b, iz, iy, ix]

    @nbc.jit()  # pragma: no cover
    def _blocks_to_array1_cuda(output, input, batch_size,
                               blk_shape, blk_strides, num_blks):
        i = nbc.grid(1)
        b = i // output.shape[-1]
        i -= b * output.shape[-1]
        ix = i

        if b < batch_size and ix < output.shape[-1]:
            rx = ix % blk_strides[-1]
            for bx in range(rx, blk_shape[-1], blk_strides[-1]):
                nx = (ix - bx) // blk_strides[-1]
                if nx >= 0 and nx < num_blks[-1]:
                    output[b, ix] += input[b, nx, bx]

    @nbc.jit()  # pragma: no cover
    def _blocks_to_array2_cuda(output, input, batch_size,
                               blk_shape, blk_strides, num_blks):
        i = nbc.grid(1)
        b = i // output.shape[-1] // output.shape[-2]
        i -= b * output.shape[-1] * output.shape[-2]
        iy = i // output.shape[-1]
        i -= iy * output.shape[-1]
        ix = i

        if b < batch_size and ix < output.shape[-1] and iy < output.shape[-2]:
            ry = iy % blk_strides[-2]
            rx = ix % blk_strides[-1]
            for by in range(ry, blk_shape[-2], blk_strides[-2]):
                for bx in range(rx, blk_shape[-1], blk_strides[-1]):
                    ny = (iy - by) // blk_strides[-2]
                    nx = (ix - bx) // blk_strides[-1]
                    if nx >= 0 and nx < num_blks[-1] \
                       and ny >= 0 and ny < num_blks[-2]:
                        output[b, iy, ix] += input[b, ny, nx, by, bx]

    @nbc.jit()  # pragma: no cover
    def _blocks_to_array3_cuda(output, input, batch_size,
                               blk_shape, blk_strides, num_blks):
        i = nbc.grid(1)
        b = i // output.shape[-1] // output.shape[-2] // output.shape[-3]
        i -= b * output.shape[-1] * output.shape[-2] * output.shape[-3]
        iz = i // output.shape[-1] // output.shape[-2]
        i -= iz * output.shape[-1] * output.shape[-2]
        iy = i // output.shape[-1]
        i -= iy * output.shape[-1]
        ix = i

        if b < batch_size and ix < output.shape[-1] and iy < output.shape[-2] \
           and iz < output.shape[-3]:
            rz = iz % blk_strides[-3]
            ry = iy % blk_strides[-2]
            rx = ix % blk_strides[-1]
            for bz in range(rz, blk_shape[-3], blk_strides[-3]):
                for by in range(ry, blk_shape[-2], blk_strides[-2]):
                    for bx in range(rx, blk_shape[-1],
                                    blk_strides[-1]):
                        nz = (iz - bz) // blk_strides[-3]
                        ny = (iy - by) // blk_strides[-2]
                        nx = (ix - bx) // blk_strides[-1]
                        if nx >= 0 and nx < num_blks[-1] \
                           and ny >= 0 and ny < num_blks[-2] \
                           and nz >= 0 and nz < num_blks[-3]:
                            output[b, iz, iy, ix] += input[b, nz,
                                                           ny, nx,
                                                           bz, by, bx]
