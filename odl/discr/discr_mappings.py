# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Discretization of functions and interpolation of discrete arrays."""

from __future__ import print_function, division, absolute_import
from builtins import object
from itertools import product
import numpy as np

from odl.util import (
    is_valid_input_meshgrid, out_shape_from_array, out_shape_from_meshgrid,
    is_string,
)


__all__ = (
    'point_collocation',
    'nearest_interpolator',
    'linear_interpolator',
    'per_axis_interpolator'
)

_SUPPORTED_INTERP_SCHEMES = ['nearest', 'linear']


def point_collocation(func, meshgrid, out=None, **kwargs):
    """Sample a function on a grid of points.

    This function represents the simplest way of discretizing a function.
    It does little more than calling the function on a sparse meshgrid
    and returning the result.

    Parameters
    ----------
    func : callable
        Function to be sampled. It must be vectorized, i.e., it must support
        evaluation with a (sparse) meshgrid of points. Optionally, the ``func``
        can accept an optional `numpy.ndarray` as ``out`` argument to which
        results can be written.
    meshgrid : TODO
        TODO
    out : numpy.ndarray
        TODO
    kwargs :
        Additional arguments that are passed on to ``func``.

    Examples
    --------
    Sample a 1D function:

    >>> mesh = np.meshgrid([1, 2, 3])
    >>> point_collocation(lambda x: x ** 2, mesh)

    In two or more dimensions, the function to be sampled can be written as
    if its arguments were the components of a point, and an implicit loop
    around the call would iterate over all points:

    >>> xs = [1, 2]
    >>> ys = [3, 4, 5]
    >>> mesh = np.meshgrid(xs, ys, sparse=True)
    >>> point_collocation(lambda x: x[0] - x[1], mesh)

    It is possible to use parametric functions and pass the parameters
    during operator call:

    >>> def plus_c(x, c=0):
    ...     return x[0] - x[1] + c
    >>> coll_op(plus_c)  # uses default c = 0
    rn((2, 3)).element(
        [[-2., -3., -4.],
         [-1., -2., -3.]]
    )
    >>> coll_op(plus_c, c=2)
    rn((2, 3)).element(
        [[ 0., -1., -2.],
         [ 1.,  0., -1.]]
    )

    Notes
    -----
    This operator expects its input functions to be written in a
    vectorization-conforming manner to ensure fast evaluation.
    See the `ODL vectorization guide`_ for a detailed introduction.

    See Also
    --------
    odl.discr.grid.RectGrid.meshgrid
    numpy.meshgrid

    References
    ----------
    .. _ODL vectorization guide:
       https://odlgroup.github.io/odl/guide/in_depth/vectorization_guide.html
    """
    if out is None:
        out = func(meshgrid, **kwargs)
    else:
        # TODO: wrap `func` if it doesn't support `out`
        func(meshgrid, out=out, **kwargs)
    return out


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


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
