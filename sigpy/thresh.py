# -*- coding: utf-8 -*-
"""Thresholding functions.
"""
import numpy as np
import numba as nb

from sigpy import backend, config, util


__all__ = ['soft_thresh', 'hard_thresh', 'l1_proj', 'l2_proj']


def soft_thresh(lamda, input):
    r"""Soft threshold.

    Performs:

    .. math::
        (| x | - \lambda)_+  \text{sgn}(x)

    Args:
        lamda (float, or array): Threshold parameter.
        input (array)

    Returns:
        array: soft-thresholded result.

    """
    xp = backend.get_array_module(input)
    abs_input = xp.abs(input)
    output = input.copy()
    output[abs_input != 0] /= abs_input[abs_input != 0]
    mag = abs_input - lamda
    mag = (xp.abs(mag) + mag) / 2
    output *= mag

    return output


def hard_thresh(lamda, input):
    """Hard threshold.

    Args:
        lamda (float, or array): Threshold parameter.
        input (array)

    Returns:
        array: hard-thresholded result.

    """
    xp = backend.get_array_module(input)
    abs_input = xp.abs(input)
    output = input.copy()
    output[abs_input <= lamda] = 0
    return output


def l1_proj(eps, input):
    """Projection onto L1 ball.

    Args:
        eps (float, or array): L1 ball scaling.
        input (array)

    Returns:
        array: Result.

    References:
        J. Duchi, S. Shalev-Shwartz, and Y. Singer, "Efficient projections onto
        the l1-ball for learning in high dimensions" 2008.

    """
    xp = backend.get_array_module(input)
    shape = input.shape
    input = input.ravel()

    if xp.linalg.norm(input, 1) < eps:
        return input
    else:
        size = len(input)
        s = xp.sort(xp.abs(input))[::-1]
        st = (xp.cumsum(s) - eps) / (xp.arange(size) + 1)
        idx = xp.flatnonzero((s - st) > 0).max()
        return soft_thresh(st[idx], input.reshape(shape))


def l2_proj(eps, input, axes=None):
    """Projection onto L2 ball.

    Args:
        eps (float, or array): L2 ball scaling.
        input (array)

    Returns:
        array: Result.

    """
    axes = util._normalize_axes(axes, input.ndim)

    xp = backend.get_array_module(input)
    norm = xp.sum(xp.abs(input)**2, axis=axes, keepdims=True)**0.5
    mask = norm < eps
    output = mask * input + (1 - mask) * (eps * input / (norm + mask))

    return output
