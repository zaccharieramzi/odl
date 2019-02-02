# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Helpers for discretization-related functionality.

Many functions deal with interpolation of arrays, sampling of functions and
providing a single interface for the sampler by wrapping functions or
arrays of functions appropriately.
"""

from __future__ import absolute_import, division, print_function

import inspect
import sys
import warnings
from builtins import object
from functools import partial
from itertools import product

import numpy as np

from odl.util import (
    dtype_repr, is_complex_floating_dtype, is_real_dtype, is_string,
    is_valid_input_array, is_valid_input_meshgrid, out_shape_from_array,
    out_shape_from_meshgrid, vectorize, writable_array)

__all__ = (
    'point_collocation',
    'nearest_interpolator',
    'linear_interpolator',
    'per_axis_interpolator',
    'make_vec_func_for_sampling',
)

_SUPPORTED_INTERP_SCHEMES = ['nearest', 'linear']


def point_collocation(func, points, out=None, **kwargs):
    """Sample a function on a grid of points.

    This function represents the simplest way of discretizing a function.
    It does little more than calling the function on a single point or a
    set of points, and returning the result.

    Parameters
    ----------
    func : callable
        Function to be sampled. It is expected to work with single points,
        meshgrids and point arrays, and to support an optional ``out``
        argument.
        Usually, ``func`` is the return value of `make_vec_func_for_sampling`.
    points : point, meshgrid or array of points
        The point(s) where to sample.
    out : numpy.ndarray, optional
        Array to which the result should be written.
    kwargs :
        Additional arguments that are passed on to ``func``.

    Returns
    -------
    out : numpy.ndarray
        Array holding the values of ``func`` at ``points``. If ``out`` was
        given, the returned object is a reference to it.

    Examples
    --------
    Sample a 1D function:

    >>> from odl.discr.grid import sparse_meshgrid
    >>> domain = odl.IntervalProd(0, 5)
    >>> func = make_vec_func_for_sampling(lambda x: x ** 2, domain)
    >>> mesh = sparse_meshgrid([1, 2, 3])
    >>> point_collocation(func, mesh)
    array([ 1.,  4.,  9.])

    By default, inputs are checked against ``domain`` to be in bounds. This
    can be switched off by passing ``bounds_check=False``:

    >>> mesh = np.meshgrid([-1, 0, 4])
    >>> point_collocation(func, mesh, bounds_check=False)

    array([  1.,   0.,  16.])

    In two or more dimensions, the function to be sampled can be written as
    if its arguments were the components of a point, and an implicit loop
    around the call would iterate over all points:

    >>> domain = odl.IntervalProd([0, 0], [5, 5])
    >>> xs = [1, 2]
    >>> ys = [3, 4, 5]
    >>> mesh = sparse_meshgrid(xs, ys)
    >>> func = make_vec_func_for_sampling(lambda x: x[0] - x[1], domain)
    >>> point_collocation(func, mesh)
    array([[-2., -3., -4.],
           [-1., -2., -3.]])

    It is possible to return results that require broadcasting, and to use
    *optional* function parameters:

    >>> def f(x, c=0):
    ...     return x[0] + c
    >>> func = make_vec_func_for_sampling(f, domain)
    >>> point_collocation(func, mesh)  # uses default c=0
    array([[ 1.,  1.,  1.],
           [ 2.,  2.,  2.]])
    >>> point_collocation(func, mesh, c=2)
    array([[ 3.,  3.,  3.],
           [ 4.,  4.,  4.]])

    The ``point_collocation`` function also supports vector- and tensor-valued
    functions. They can be given either as a single function returning an
    array-like of results, or as an array-like of member functions:

    >>> domain = odl.IntervalProd([0, 0], [5, 5])
    >>> xs = [1, 2]
    >>> ys = [3, 4]
    >>> mesh = sparse_meshgrid(xs, ys)
    >>> def vec_valued(x):
    ...     return (x[0] - 1, 0, x[0] + x[1])  # broadcasting
    >>> # We must tell the wrapper that we want a 3-component function
    >>> func1 = make_vec_func_for_sampling(
    ...     vec_valued, domain, out_dtype=(float, (3,)))
    >>> point_collocation(func1, mesh)
    array([[[ 0.,  0.],
            [ 1.,  1.]],

           [[ 0.,  0.],
            [ 0.,  0.]],

           [[ 4.,  5.],
            [ 5.,  6.]]])
    >>> array_of_funcs = [  # equivalent to `vec_valued`
    ...     lambda x: x[0] - 1,
    ...     0,                   # constants are allowed
    ...     lambda x: x[0] + x[1]
    ... ]
    >>> func2 = make_vec_func_for_sampling(
    ...     array_of_funcs, domain, out_dtype=(float, (3,)))
    >>> point_collocation(func2, mesh)
    array([[[ 0.,  0.],
            [ 1.,  1.]],

           [[ 0.,  0.],
            [ 0.,  0.]],

           [[ 4.,  5.],
            [ 5.,  6.]]])

    Notes
    -----
    This operator expects its input functions to be written in a
    vectorization-conforming manner to ensure fast evaluation.
    See the `ODL vectorization guide`_ for a detailed introduction.

    See Also
    --------
    make_vec_func_for_sampling : wrap a function
    odl.discr.grid.RectGrid.meshgrid
    numpy.meshgrid

    References
    ----------
    .. _ODL vectorization guide:
       https://odlgroup.github.io/odl/guide/in_depth/vectorization_guide.html
    """
    if out is None:
        out = func(points, **kwargs)
    else:
        func(points, out=out, **kwargs)
    return out


def _normalize_interp(interp, ndim):
    """Turn interpolation type into a tuple with one entry per axis."""
    interp_in = interp
    if is_string(interp):
        interp = str(interp).lower()
        if interp not in _SUPPORTED_INTERP_SCHEMES:
            raise ValueError('`interp` {!r} not understood'.format(interp_in))
        interp_byaxis = (interp,) * ndim
    else:
        interp_byaxis = tuple(str(itp).lower() for itp in interp)
        if len(interp_byaxis) != ndim:
            raise ValueError(
                'length of `interp` ({}) does not match number of axes ({})'
                ''.format(len(interp_byaxis, ndim))
            )

    return interp_byaxis


def _all_interp_equal(interp_byaxis):
    """Whether all entries are equal, with ``False`` for length 0."""
    if len(interp_byaxis) == 0:
        return False
    return all(itp == interp_byaxis[0] for itp in interp_byaxis)


# TODO: fix docs
def nearest_interpolator(x, coord_vecs, variant='left'):
    """Nearest neighbor interpolation.

    Given points ``x[1] < x[2] < ... < x[N]``, and values ``f[1], ..., f[N]``,
    nearest neighbor interpolation at ``x`` is defined as ::

        I(x) = f[j]  with j such that |x - x[j]| is minimal.

    The ambiguity at the midpoints is resolved by preferring one of the
    neighbors. For higher dimensions, this rule is applied per
    component.

    The returned interpolator is the function ``x -> I(x)``.

    In higher dimensions, this principle is applied per axis, the
    only difference being the additional information about the ordering
    of the axes in the flat storage array (C- vs. Fortran ordering).

    Parameters
    ----------
    variant : {'left', 'right'}, optional
        Behavior variant at the midpoint between neighbors.

        - ``'left'``: favor left neighbor (default).
        - ``'right'``: favor right neighbor.

    Examples
    --------
    We test nearest neighbor interpolation with a non-scalar
    data type in 2d:

    >>> rect = odl.IntervalProd([0, 0], [1, 1])
    >>> fspace = odl.FunctionSpace(rect, out_dtype='U1')

    Partitioning the domain uniformly with no nodes on the boundary
    (will shift the grid points):

    >>> part = odl.uniform_partition_fromintv(rect, [4, 2])
    >>> part.grid.coord_vectors
    (array([ 0.125,  0.375,  0.625,  0.875]), array([ 0.25,  0.75]))

    Now we initialize the operator and test it with some points:

    >>> tspace = odl.tensor_space(part.shape, dtype='U1')
    >>> interp_op = NearestInterpolation(fspace, part, tspace)
    >>> values = np.array([['m', 'y'],
    ...                    ['s', 't'],
    ...                    ['r', 'i'],
    ...                    ['n', 'g']])
    >>> function = interp_op(values)
    >>> print(function([0.3, 0.6]))  # closest to index (1, 1) -> 3
    t
    >>> out = np.empty(2, dtype='U1')
    >>> pts = np.array([[0.3, 0.6],
    ...                 [1.0, 1.0]])
    >>> out = function(pts.T, out=out)  # returns original out
    >>> all(out == ['t', 'g'])
    True

    See Also
    --------
    LinearInterpolation : (bi-/tri-/...)linear interpolation

    Notes
    -----
    - **Important:** if called on a point array, the points are
      assumed to be sorted in ascending order in each dimension
      for efficiency reasons.
    - Nearest neighbor interpolation is the only scheme which works
      with data of non-numeric data type since it does not involve any
      arithmetic operations on the values, in contrast to other
      interpolation methods.
    - The distinction between left and right variants is currently
      made by changing ``<=`` to ``<`` at one place. This difference
      may not be noticable in some situations due to rounding errors.

    """
    # TODO(kohr-h): pass reasonable options on to the interpolator
    def nearest_interp(arg, out=None):
        """Interpolating function with vectorization."""
        if is_valid_input_meshgrid(arg, x.ndim):
            input_type = 'meshgrid'
        else:
            input_type = 'array'

        interpolator = _NearestInterpolator(
            coord_vecs,
            x,
            variant=variant,
            input_type=input_type
        )

        return interpolator(arg, out=out)

    return nearest_interp


# TODO: doc
def linear_interpolator(x, coord_vecs):
    """

    Parameters
    ----------
    x :

    coord_vecs :

    """
    # TODO: pass reasonable options on to the interpolator
    def linear_interp(arg, out=None):
        """Interpolating function with vectorization."""
        if is_valid_input_meshgrid(arg, x.ndim):
            input_type = 'meshgrid'
        else:
            input_type = 'array'

        interpolator = _LinearInterpolator(
            coord_vecs,
            x,
            input_type=input_type
        )

        return interpolator(arg, out=out)

    return linear_interp


# TODO: doc
def per_axis_interpolator(x, coord_vecs, schemes, nn_variants=None):
    """Interpolator with interpolation scheme per axis.

    Parameters
    ----------
    x :

    coord_vecs :

    schemes : string or sequence of strings
        Indicates which interpolation scheme to use for which axis.
        A single string is interpreted as a global scheme for all
        axes.
    nn_variants : string or sequence of strings, optional
        Which variant ('left' or 'right') to use in nearest neighbor
        interpolation for which axis. A single string is interpreted
        as a global variant for all axes.
        This option has no effect for schemes other than nearest
        neighbor.

    """
    schemes_in = schemes
    if is_string(schemes):
        scheme = str(schemes).lower()
        if scheme not in _SUPPORTED_INTERP_SCHEMES:
            raise ValueError('`schemes` {!r} not understood'
                             ''.format(schemes_in))
        schemes = [scheme] * x.ndim
    else:
        schemes = [str(scm).lower() if scm is not None else None
                   for scm in schemes]

    nn_variants_in = nn_variants
    if nn_variants is None:
        nn_variants = ['left' if scm == 'nearest' else None
                       for scm in schemes]
    else:
        if is_string(nn_variants):
            # Make list with `nn_variants` where `schemes == 'nearest'`,
            # else `None` (variants only applies to axes with nn
            # interpolation)
            nn_variants = [nn_variants if scm == 'nearest' else None
                           for scm in schemes]
            if str(nn_variants_in).lower() not in ('left', 'right'):
                raise ValueError('`nn_variants` {!r} not understood'
                                 ''.format(nn_variants_in))
        else:
            nn_variants = [str(var).lower() if var is not None else None
                           for var in nn_variants]

    for i in range(x.ndim):
        # Reaching a raise condition here only happens for invalid
        # sequences of inputs, single-input case has been checked above
        if schemes[i] not in _SUPPORTED_INTERP_SCHEMES:
            raise ValueError('`interp[{}]={!r}` not understood'
                             ''.format(schemes_in[i], i))
        if (schemes[i] == 'nearest' and
                nn_variants[i] not in ('left', 'right')):
            raise ValueError('`nn_variants[{}]={!r}` not understood'
                             ''.format(nn_variants_in[i], i))
        elif schemes[i] != 'nearest' and nn_variants[i] is not None:
            raise ValueError('in axis {}: `nn_variants` cannot be used '
                             'with `interp={!r}'
                             ''.format(i, schemes_in[i]))

    def per_axis_interp(arg, out=None):
        """Interpolating function with vectorization."""
        if is_valid_input_meshgrid(arg, x.ndim):
            input_type = 'meshgrid'
        else:
            input_type = 'array'

        interpolator = _PerAxisInterpolator(
            coord_vecs,
            x,
            schemes=schemes,
            nn_variants=nn_variants,
            input_type=input_type
        )

        return interpolator(arg, out=out)

    return per_axis_interp


class _Interpolator(object):

    """Abstract interpolator class.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.

    The init method does not convert to floating point to
    support arbitrary data type for nearest neighbor interpolation.

    Subclasses need to override ``_evaluate`` for concrete
    implementations.
    """

    def __init__(self, coord_vecs, values, input_type):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        """
        values = np.asarray(values)
        typ_ = str(input_type).lower()
        if typ_ not in ('array', 'meshgrid'):
            raise ValueError('`input_type` ({}) not understood'
                             ''.format(input_type))

        if len(coord_vecs) > values.ndim:
            raise ValueError('there are {} point arrays, but `values` has {} '
                             'dimensions'.format(len(coord_vecs),
                                                 values.ndim))
        for i, p in enumerate(coord_vecs):
            if not np.asarray(p).ndim == 1:
                raise ValueError('the points in dimension {} must be '
                                 '1-dimensional'.format(i))
            if values.shape[i] != len(p):
                raise ValueError('there are {} points and {} values in '
                                 'dimension {}'.format(len(p),
                                                       values.shape[i], i))

        self.coord_vecs = tuple(np.asarray(p) for p in coord_vecs)
        self.values = values
        self.input_type = input_type

    def __call__(self, x, out=None):
        """Do the interpolation.

        Parameters
        ----------
        x : `meshgrid` or `numpy.ndarray`
            Evaluation points of the interpolator
        out : `numpy.ndarray`, optional
            Array to which the results are written. Needs to have
            correct shape according to input ``x``.

        Returns
        -------
        out : `numpy.ndarray`
            Interpolated values. If ``out`` was given, the returned
            object is a reference to it.
        """
        ndim = len(self.coord_vecs)
        if self.input_type == 'array':
            # Make a (1, n) array from one with shape (n,)
            x = x.reshape([ndim, -1])
            out_shape = out_shape_from_array(x)
        else:
            if len(x) != ndim:
                raise ValueError('number of vectors in x is {} instead of '
                                 'the grid dimension {}'
                                 ''.format(len(x), ndim))
            out_shape = out_shape_from_meshgrid(x)

        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('`out` {!r} not a `numpy.ndarray` '
                                'instance'.format(out))
            if out.shape != out_shape:
                raise ValueError('output shape {} not equal to expected '
                                 'shape {}'.format(out.shape, out_shape))
            if out.dtype != self.values.dtype:
                raise ValueError('output dtype {} not equal to expected '
                                 'dtype {}'
                                 ''.format(out.dtype, self.values.dtype))

        indices, norm_distances = self._find_indices(x)
        return self._evaluate(indices, norm_distances, out)

    def _find_indices(self, x):
        """Find indices and distances of the given nodes.

        Can be overridden by subclasses to improve efficiency.
        """
        # find relevant edges between which xi are situated
        index_vecs = []
        # compute distance to lower edge in unity units
        norm_distances = []

        # iterate through dimensions
        for xi, cvec in zip(x, self.coord_vecs):
            idcs = np.searchsorted(cvec, xi) - 1

            idcs[idcs < 0] = 0
            idcs[idcs > cvec.size - 2] = cvec.size - 2
            index_vecs.append(idcs)

            norm_distances.append((xi - cvec[idcs]) /
                                  (cvec[idcs + 1] - cvec[idcs]))

        return index_vecs, norm_distances

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluation method, needs to be overridden."""
        raise NotImplementedError('abstract method')


class _NearestInterpolator(_Interpolator):

    """Nearest neighbor interpolator.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.

    This implementation is faster than the more generic one in the
    `_PerAxisPointwiseInterpolator`. Compared to the original code,
    support of ``'left'`` and ``'right'`` variants are added.
    """

    def __init__(self, coord_vecs, values, input_type, variant):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        variant : {'left', 'right'}
            Indicates which neighbor to prefer in the interpolation
        """
        super(_NearestInterpolator, self).__init__(
            coord_vecs, values, input_type)
        variant_ = str(variant).lower()
        if variant_ not in ('left', 'right'):
            raise ValueError("variant '{}' not understood".format(variant_))
        self.variant = variant_

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluate nearest interpolation."""
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            if self.variant == 'left':
                idx_res.append(np.where(yi <= .5, i, i + 1))
            else:
                idx_res.append(np.where(yi < .5, i, i + 1))
        idx_res = tuple(idx_res)
        if out is not None:
            out[:] = self.values[idx_res]
            return out
        else:
            return self.values[idx_res]


def _compute_nearest_weights_edge(idcs, ndist, variant):
    """Helper for nearest interpolation mimicing the linear case."""
    # Get out-of-bounds indices from the norm_distances. Negative
    # means "too low", larger than or equal to 1 means "too high"
    lo = (ndist < 0)
    hi = (ndist > 1)

    # For "too low" nodes, the lower neighbor gets weight zero;
    # "too high" gets 1.
    if variant == 'left':
        w_lo = np.where(ndist <= 0.5, 1.0, 0.0)
    else:
        w_lo = np.where(ndist < 0.5, 1.0, 0.0)

    w_lo[lo] = 0
    w_lo[hi] = 1

    # For "too high" nodes, the upper neighbor gets weight zero;
    # "too low" gets 1.
    if variant == 'left':
        w_hi = np.where(ndist <= 0.5, 0.0, 1.0)
    else:
        w_hi = np.where(ndist < 0.5, 0.0, 1.0)

    w_hi[lo] = 1
    w_hi[hi] = 0

    # For upper/lower out-of-bounds nodes, we need to set the
    # lower/upper neighbors to the last/first grid point
    edge = [idcs, idcs + 1]
    edge[0][hi] = -1
    edge[1][lo] = 0

    return w_lo, w_hi, edge


def _compute_linear_weights_edge(idcs, ndist):
    """Helper for linear interpolation."""
    # Get out-of-bounds indices from the norm_distances. Negative
    # means "too low", larger than or equal to 1 means "too high"
    lo = np.where(ndist < 0)
    hi = np.where(ndist > 1)

    # For "too low" nodes, the lower neighbor gets weight zero;
    # "too high" gets 2 - yi (since yi >= 1)
    w_lo = (1 - ndist)
    w_lo[lo] = 0
    w_lo[hi] += 1

    # For "too high" nodes, the upper neighbor gets weight zero;
    # "too low" gets 1 + yi (since yi < 0)
    w_hi = np.copy(ndist)
    w_hi[lo] += 1
    w_hi[hi] = 0

    # For upper/lower out-of-bounds nodes, we need to set the
    # lower/upper neighbors to the last/first grid point
    edge = [idcs, idcs + 1]
    edge[0][hi] = -1
    edge[1][lo] = 0

    return w_lo, w_hi, edge


def _create_weight_edge_lists(indices, norm_distances, schemes, variants):
    # Precalculate indices and weights (per axis)
    low_weights = []
    high_weights = []
    edge_indices = []
    for i, (idcs, yi, scm, var) in enumerate(
            zip(indices, norm_distances, schemes, variants)):
        if scm == 'nearest':
            w_lo, w_hi, edge = _compute_nearest_weights_edge(
                idcs, yi, var)
        elif scm == 'linear':
            w_lo, w_hi, edge = _compute_linear_weights_edge(
                idcs, yi)
        else:
            raise ValueError("scheme '{}' at index {} not supported"
                             "".format(scm, i))

        low_weights.append(w_lo)
        high_weights.append(w_hi)
        edge_indices.append(edge)

    return low_weights, high_weights, edge_indices


class _PerAxisInterpolator(_Interpolator):

    """Interpolator where the scheme is set per axis.

    This allows to use e.g. nearest neighbor interpolation in the
    first dimension and linear in dimensions 2 and 3.
    """

    def __init__(self, coord_vecs, values, input_type, schemes, nn_variants):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        schemes : sequence of strings
            Indicates which interpolation scheme to use for which axis
        nn_variants : sequence of strings
            Which variant ('left' or 'right') to use in nearest neighbor
            interpolation for which axis.
            This option has no effect for schemes other than nearest
            neighbor.
        """
        super(_PerAxisInterpolator, self).__init__(
            coord_vecs, values, input_type)
        self.schemes = schemes
        self.nn_variants = nn_variants

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluate linear interpolation.

        Modified for in-place evaluation and treatment of out-of-bounds
        points by implicitly assuming 0 at the next node."""
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        if out is None:
            out_shape = out_shape_from_meshgrid(norm_distances)
            out_dtype = self.values.dtype
            out = np.zeros(out_shape, dtype=out_dtype)
        else:
            out[:] = 0.0

        # Weights and indices (per axis)
        low_weights, high_weights, edge_indices = _create_weight_edge_lists(
            indices, norm_distances, self.schemes, self.nn_variants)

        # Iterate over all possible combinations of [i, i+1] for each
        # axis, resulting in a loop of length 2**ndim
        for lo_hi, edge in zip(product(*([['l', 'h']] * len(indices))),
                               product(*edge_indices)):
            weight = 1.0
            # TODO: determine best summation order from array strides
            for lh, w_lo, w_hi in zip(lo_hi, low_weights, high_weights):

                # We don't multiply in-place to exploit the cheap operations
                # in the beginning: sizes grow gradually as following:
                # (n, 1, 1, ...) -> (n, m, 1, ...) -> ...
                # Hence, it is faster to build up the weight array instead
                # of doing full-size operations from the beginning.
                if lh == 'l':
                    weight = weight * w_lo
                else:
                    weight = weight * w_hi
            out += np.asarray(self.values[edge]) * weight[vslice]
        return np.array(out, copy=False, ndmin=1)


class _LinearInterpolator(_PerAxisInterpolator):

    """Linear (i.e. bi-/tri-/multi-linear) interpolator.

    Convenience class.
    """

    def __init__(self, coord_vecs, values, input_type):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        """
        super(_LinearInterpolator, self).__init__(
            coord_vecs,
            values,
            input_type,
            schemes=['linear'] * len(coord_vecs),
            nn_variants=[None] * len(coord_vecs),
        )


def _check_func_out_arg(func):
    """Check if ``func`` has an (optional) ``out`` argument.

    Also verify that the signature of ``func`` has no ``*args`` since
    they make argument propagation a huge hassle.

    Parameters
    ----------
    func : callable
        Object that should be inspected.

    Returns
    -------
    has_out : bool
        ``True`` if the signature has an ``out`` argument, ``False``
        otherwise.
    out_is_optional : bool
        ``True`` if ``out`` is present and optional in the signature,
        ``False`` otherwise.

    Raises
    ------
    TypeError
        If ``func``'s signature has ``*args``.
    """
    if sys.version_info.major > 2:
        spec = inspect.getfullargspec(func)
        kw_only = spec.kwonlyargs
    else:
        spec = inspect.getargspec(func)
        kw_only = ()

    if spec.varargs is not None:
        raise TypeError('*args not allowed in function signature')

    pos_args = spec.args
    pos_defaults = () if spec.defaults is None else spec.defaults

    has_out = 'out' in pos_args or 'out' in kw_only
    if 'out' in pos_args:
        has_out = True
        out_optional = (
            pos_args.index('out') >= len(pos_args) - len(pos_defaults)
        )
    elif 'out' in kw_only:
        has_out = out_optional = True
    else:
        has_out = out_optional = False

    return has_out, out_optional


def _func_out_type(func):
    """Check if ``func`` has (optional) output argument.

    This function is intended to work with all types of callables
    that are used as input to `make_vec_func_for_sampling`.
    """
    # Numpy Ufuncs and similar objects (e.g. Numba DUfuncs)
    if hasattr(func, 'nin') and hasattr(func, 'nout'):
        if func.nin != 1:
            raise ValueError(
                'ufunc {} takes {} input arguments, expected 1'
                ''.format(func.__name__, func.nin)
            )
        if func.nout > 1:
            raise ValueError(
                'ufunc {} returns {} outputs, expected 0 or 1'
                ''.format(func.__name__, func.nout)
            )
        has_out = out_optional = (func.nout == 1)
    elif inspect.isfunction(func):
        has_out, out_optional = _check_func_out_arg(func)
    elif callable(func):
        has_out, out_optional = _check_func_out_arg(func.__call__)
    else:
        raise TypeError('object {!r} not callable'.format(func))

    return has_out, out_optional


def make_vec_func_for_sampling(func_or_arr, domain, out_dtype='float64',
                               vectorized=True):
    """Return a vectorized function that can be used for sampling.

    For examples on this function's usage, see `point_collocation`.

    Parameters
    ----------
    func_or_arr : callable or array-like
        Either a callable object or an array or callables and constants.
    domain : IntervalProd
        Set in which inputs to the function are assumed to lie. It is used
        to determine the type of input (point/meshgrid/array) based on
        ``domain.ndim``, and (unless switched off) to check whether all
        inputs are in bounds.
    out_dtype : optional
        Data type of a *single* output of ``func_or_arr``, i.e., when
        called with a single point as input. In particular:

        - If ``func_or_arr`` is a scalar-valued function, ``out_dtype`` is
          expected to be a basic dtype with empty shape.
        - If ``func_or_arr`` is a vector- or tensor-valued function,
          ``out_dtype`` should be a shaped data type, e.g., ``(float, (3,))``
          for a vector-valued function with 3 components.
        - If ``func_or_arr`` is an array-like, ``out_dtype`` should be a
          shaped dtype whose shape matches that of ``func_or_arr``.

    vectorized : bool, optional
        Whether the provided function or functions support vectorized call.
        If ``False``, `numpy.vectorize` is used to add that support. Note
        that this results in very slow evaluation.

    Returns
    -------
    func : function
        Wrapper function that supports vectorized call and an optional
        ``out`` argument.
    """
    if out_dtype is None:
        # Don't let `np.dtype` convert `None` to `float64`
        raise TypeError('`out_dtype` cannot be `None`')

    out_dtype = np.dtype(out_dtype)
    val_shape = out_dtype.shape
    scalar_out_dtype = out_dtype.base

    # Provide default implementations of missing function signature types

    def _default_oop(func_ip, x, **kwargs):
        """Default out-of-place variant of an in-place-only function."""
        if is_valid_input_array(x, domain.ndim):
            scalar_out_shape = out_shape_from_array(x)
        elif is_valid_input_meshgrid(x, domain.ndim):
            scalar_out_shape = out_shape_from_meshgrid(x)
        else:
            raise TypeError('cannot use in-place method to implement '
                            'out-of-place non-vectorized evaluation')

        out_shape = val_shape + scalar_out_shape
        out = np.empty(out_shape, dtype=scalar_out_dtype)
        func_ip(x, out=out, **kwargs)
        return out

    def _default_ip(func_oop, x, out, **kwargs):
        """Default in-place variant of an out-of-place-only function."""
        result = np.array(func_oop(x, **kwargs), copy=False)
        if result.dtype == object:
            # Different shapes encountered, need to broadcast
            flat_results = result.ravel()
            if is_valid_input_array(x, domain.ndim):
                scalar_out_shape = out_shape_from_array(x)
            elif is_valid_input_meshgrid(x, domain.ndim):
                scalar_out_shape = out_shape_from_meshgrid(x)
            else:
                raise RuntimeError('bad input')

            bcast_results = [np.broadcast_to(res, scalar_out_shape)
                             for res in flat_results]
            # New array that is flat in the `out_shape` axes, reshape it
            # to the final `out_shape + scalar_shape`, using the same
            # order ('C') as the initial `result.ravel()`.
            result = np.array(bcast_results, dtype=scalar_out_dtype)
            result = result.reshape(val_shape + scalar_out_shape)

        # The following code is required to remove extra axes, e.g., when
        # the result has shape (2, 1, 3) but should have shape (2, 3).
        # For those cases, broadcasting doesn't apply.
        try:
            reshaped = result.reshape(out.shape)
        except ValueError:
            # This is the case when `result` must be broadcast
            out[:] = result
        else:
            out[:] = reshaped

        return out

    # Now prepare the in-place and out-of-place functions for the final
    # wrapping. If needed, vectorize.

    if callable(func_or_arr):
        # Got a (single) function, possibly need to vectorize
        func = func_or_arr

        if not vectorized:
            if hasattr(func, 'nin') and hasattr(func, 'nout'):
                warnings.warn(
                    '`func` {!r} is a ufunc-like object, use vectorized=True'
                    ''.format(func),
                    UserWarning
                )
            # Don't put this before the ufunc case. Ufuncs can't be inspected.
            has_out, _ = _func_out_type(func)
            if has_out:
                raise TypeError(
                    'non-vectorized `func` with `out` parameter not allowed'
                )
            if out_dtype is not None:
                otypes = [out_dtype.base]
            else:
                otypes = []

            func = vectorize(otypes=otypes)(func)

        # Get default implementations if necessary

        has_out, out_optional = _func_out_type(func)
        if not has_out:
            # Out-of-place-only
            func_ip = partial(_default_ip, func)
            func_oop = func
        elif out_optional:
            # Dual-use
            func_ip = func_oop = func
        else:
            # In-place-only
            func_ip = func
            func_oop = partial(_default_oop, func)

    else:
        # This is for the case that an array-like of callables is provided.
        # We need to convert this into a single function that returns an
        # array, and maybe we need to vectorize member functions.

        arr = func_or_arr
        if np.shape(arr) != val_shape:
            raise ValueError(
                'invalid `func_or_arr` {!r}: expected `None`, a callable or '
                'an array-like of callables whose shape matches '
                '`out_dtype.shape` {}'.format(val_shape))

        arr = np.array(arr, dtype=object, ndmin=1).ravel().tolist()
        if not vectorized:
            if is_real_dtype(out_dtype):
                otypes = ['float64']
            elif is_complex_floating_dtype(out_dtype):
                otypes = ['complex128']
            else:
                otypes = []

            # Vectorize, preserving scalars
            arr = [f if np.isscalar(f) else vectorize(otypes=otypes)(f)
                   for f in arr]

        def array_wrapper_func(x, out=None, **kwargs):
            """Function wrapping an array of callables and constants.

            This wrapper does the following for out-of-place
            evaluation (when ``out=None``):

            1. Collect the results of all function evaluations into
               a list, handling all kinds of sequence entries
               (normal function, ufunc, constant, etc.).
            2. Broadcast all results to the desired shape that is
               determined by the space's ``out_shape`` and the
               shape(s) of the input.
            3. Form a big array containing the final result.

            The in-place version is simpler because broadcasting
            happens automatically when assigning to the components
            of ``out``. Hence, we only have

            1. Assign the result of the evaluation of the i-th
               function to ``out_flat[i]``, possibly using the
               ``out`` parameter of the function.
            """
            if is_valid_input_meshgrid(x, domain.ndim):
                scalar_out_shape = out_shape_from_meshgrid(x)
            elif is_valid_input_array(x, domain.ndim):
                scalar_out_shape = out_shape_from_array(x)
            else:
                raise RuntimeError('bad input')

            if out is None:
                # Out-of-place evaluation

                # Collect results of member functions into a list.
                # Put simply, all that happens here is
                # `results.append(f(x))`, just for a bunch of cases
                # and with or without `out`.
                results = []
                for f in arr:
                    if np.isscalar(f):
                        # Constant function
                        results.append(f)
                    elif not callable(f):
                        raise TypeError(
                            'element {!r} of `func_or_arr` not callable'
                            ''.format(f)
                        )
                    elif hasattr(f, 'nin') and hasattr(f, 'nout'):
                        # ufunc-like object
                        results.append(f(x, **kwargs))
                    else:
                        has_out, _ = _func_out_type(f)
                        if has_out:
                            out = np.empty(
                                scalar_out_shape, dtype=scalar_out_dtype
                            )
                            f(x, out=out, **kwargs)
                            results.append(out)
                        else:
                            results.append(f(x, **kwargs))

                # Broadcast to required shape and convert to array.
                # This will raise an error if the shape of some member
                # array is wrong, since in that case the resulting
                # dtype would be `object`.
                bcast_results = []
                for res in results:
                    try:
                        reshaped = np.reshape(res, scalar_out_shape)
                    except ValueError:
                        bcast_results.append(
                            np.broadcast_to(res, scalar_out_shape))
                    else:
                        bcast_results.append(reshaped)

                out_arr = np.array(
                    bcast_results, dtype=scalar_out_dtype
                )

                return out_arr.reshape(val_shape + scalar_out_shape)

            else:
                # In-place evaluation

                # This is a precaution in case out is not contiguous
                with writable_array(out) as out_arr:
                    # Flatten tensor axes to work on one tensor
                    # component (= scalar function) at a time
                    out_comps = out.reshape((-1,) + scalar_out_shape)
                    for f, out_comp in zip(arr, out_comps):
                        if np.isscalar(f):
                            out_comp[:] = f
                        else:
                            has_out, _ = _func_out_type(f)
                            if has_out:
                                f(x, out=out_comp, **kwargs)
                            else:
                                out_comp[:] = f(x, **kwargs)

        func_ip = func_oop = array_wrapper_func

    return _make_dual_use_func(func_ip, func_oop, domain, out_dtype)


def _make_dual_use_func(func_ip, func_oop, domain, out_dtype):
    """Return a unifying wrapper function with optional ``out`` argument."""

    ndim = domain.ndim
    if out_dtype is None:
        # Don't let `np.dtype` convert `None` to `float64`
        raise TypeError('`out_dtype` cannot be `None`')

    out_dtype = np.dtype(out_dtype)
    val_shape = out_dtype.shape
    scalar_out_dtype = out_dtype.base

    tensor_valued = val_shape != ()

    def dual_use_func(x, out=None, **kwargs):
        """Wrapper function with optional ``out`` argument.

        This function closes over two other functions, one for in-place,
        the other for out-of-place evaluation. Its purpose is to unify their
        interfaces to a single one with optional ``out`` argument, and to
        automate all details of input/output checking, broadcasting and
        type casting.

        The closure also contains ``domain``, an `IntervalProd` where points
        should lie, and the expected ``out_dtype``.

        For usage examples, see `point_collocation`.

        Parameters
        ----------
        x : point, `meshgrid` or `numpy.ndarray`
            Input argument for the function evaluation. Conditions
            on ``x`` depend on its type:

            - point: must be castable to an element of the enclosed ``domain``.
            - meshgrid: length must be ``domain.ndim``, and the arrays must
              be broadcastable against each other.
            - array: shape must be ``(ndim, N)``, where ``ndim`` equals
              ``domain.ndim``.

        out : `numpy.ndarray`, optional
            Output argument holding the result of the function evaluation.
            Can only be used for vectorized functions. Its shape must be equal
            to ``out_dtype.shape + np.broadcast(*x).shape``.
        bounds_check : bool, optional
            If ``True``, check if all input points lie in ``domain``. This
            requires ``domain`` to implement `Set.contains_all`.
            Default: ``True``

        Returns
        -------
        out : `numpy.ndarray`
            Result of the function evaluation. If ``out`` was provided,
            the returned object is a reference to it.

        Raises
        ------
        TypeError
            If ``x`` is not a valid vectorized evaluation argument.

            If ``out`` is neither ``None`` nor a `numpy.ndarray` of
            adequate shape and data type.

        ValueError
            If ``bounds_check == True`` and some evaluation points fall
            outside the valid domain.
        """
        bounds_check = kwargs.pop('bounds_check', True)
        if bounds_check and not hasattr(domain, 'contains_all'):
            raise AttributeError(
                'bounds check not possible for domain {!r}, missing '
                '`contains_all()` method'
                ''.format(domain)
            )

        # Check for input type and determine output shape
        if is_valid_input_meshgrid(x, ndim):
            scalar_in = False
            scalar_out_shape = out_shape_from_meshgrid(x)
            scalar_out = False
            # Avoid operations on tuples like x * 2 by casting to array
            if ndim == 1:
                x = x[0][None, ...]
        elif is_valid_input_array(x, ndim):
            x = np.asarray(x)
            scalar_in = False
            scalar_out_shape = out_shape_from_array(x)
            scalar_out = False
        elif x in domain:
            x = np.atleast_2d(x).T  # make a (d, 1) array
            scalar_in = True
            scalar_out_shape = (1,)
            scalar_out = (out is None and not tensor_valued)
        else:
            # Unknown input
            txt_1d = ' or (n,)' if ndim == 1 else ''
            raise TypeError(
                'argument {!r} not a valid function input. '
                'Expected an element of the domain {domain!r}, an array-like '
                'with shape ({domain.ndim}, n){} or a length-{domain.ndim} '
                'meshgrid tuple.'
                ''.format(x, txt_1d, domain=domain)
            )

        # Check bounds if specified
        if bounds_check and not domain.contains_all(x):
            raise ValueError('input contains points outside the domain {!r}'
                             ''.format(domain))

        if scalar_in:
            out_shape = val_shape
        else:
            out_shape = val_shape + scalar_out_shape

        # Call the function and check out shape, before or after
        if out is None:

            # The out-of-place evaluation path

            if ndim == 1:
                try:
                    out = func_oop(x, **kwargs)
                except (TypeError, IndexError):
                    # TypeError is raised if a meshgrid was used but the
                    # function expected an array (1d only). In this case we try
                    # again with the first meshgrid vector.
                    # IndexError is raised in expressions like x[x > 0] since
                    # "x > 0" evaluates to 'True', i.e. 1, and that index is
                    # out of range for a meshgrid tuple of length 1 :-). To get
                    # the real errors with indexing, we check again for the
                    # same scenario (scalar output when not valid) as in the
                    # first case.
                    out = func_oop(x[0], **kwargs)

            else:
                # Here we don't catch exceptions since they are likely true
                # errors
                out = func_oop(x, **kwargs)

            if isinstance(out, np.ndarray) or np.isscalar(out):
                # Cast to proper dtype if needed, also convert to array if out
                # is a scalar.
                out = np.asarray(out, dtype=scalar_out_dtype)
                if scalar_in:
                    out = np.squeeze(out)
                elif ndim == 1 and out.shape == (1,) + out_shape:
                    out = out.reshape(out_shape)

                if out_shape != () and out.shape != out_shape:
                    # Broadcast the returned element, but not in the
                    # scalar case. The resulting array may be read-only,
                    # in which case we copy.
                    out = np.broadcast_to(out, out_shape)
                    if not out.flags.writeable:
                        out = out.copy()

            elif tensor_valued:
                # The out object can be any array-like of objects with shapes
                # that should all be broadcastable to scalar_out_shape.
                results = np.array(out)
                if results.dtype == object or scalar_in:
                    # Some results don't have correct shape, need to
                    # broadcast
                    bcast_res = []
                    for res in results.ravel():
                        if ndim == 1:
                            # As usual, 1d is tedious to deal with. This
                            # code deals with extra dimensions in result
                            # components that stem from using x instead of
                            # x[0] in a function.
                            # Without this, broadcasting fails.
                            shp = getattr(res, 'shape', ())
                            if shp and shp[0] == 1:
                                res = res.reshape(res.shape[1:])
                        bcast_res.append(
                            np.broadcast_to(res, scalar_out_shape))

                    out_arr = np.array(bcast_res, dtype=scalar_out_dtype)

                elif results.dtype != scalar_out_dtype:
                    raise ValueError(
                        'result is of dtype {}, expected {}'
                        ''.format(dtype_repr(results.dtype),
                                  dtype_repr(scalar_out_dtype))
                    )

                else:
                    out_arr = results

                out = out_arr.reshape(out_shape)

            else:
                # TODO(kohr-h): improve message
                raise RuntimeError('bad output of function call')

        else:

            # The in-place evaluation path

            if not isinstance(out, np.ndarray):
                raise TypeError(
                    'output must be a `numpy.ndarray` got {!r}'
                    ''.format(out)
                )
            if out_shape != (1,) and out.shape != out_shape:
                raise ValueError(
                    'output has shape, expected {} from input'
                    ''.format(out.shape, out_shape)
                )
            if out.dtype != scalar_out_dtype:
                raise ValueError(
                    '`out` is of dtype {}, expected {}'
                    ''.format(out.dtype, scalar_out_dtype)
                )

            if ndim == 1 and not tensor_valued:
                # TypeError for meshgrid in 1d, but expected array (see above)
                try:
                    func_ip(x, out, **kwargs)
                except TypeError:
                    func_ip(x[0], out, **kwargs)
            else:
                func_ip(x, out=out, **kwargs)

        # If we are to output a scalar, convert the result

        # Numpy < 1.12 does not implement __complex__ for arrays (in contrast
        # to __float__), so we have to fish out the scalar ourselves.
        if scalar_out:
            scalar = out.ravel()[0].item()
            if is_real_dtype:
                return float(scalar)
            else:
                return complex(scalar)
        else:
            return out

    return dual_use_func


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
