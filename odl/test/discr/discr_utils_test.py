# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for `discr_mappings`."""

from __future__ import division
import numpy as np
import pytest

import odl
from odl.discr.grid import sparse_meshgrid
from odl.discr.discr_utils import (
    point_collocation, nearest_interpolator, linear_interpolator,
    per_axis_interpolator)
from odl.util.testutils import all_almost_equal, all_equal


def test_nearest_interpolation_1d_complex():
    """Test nearest neighbor interpolation in 1d with complex values."""
    part = odl.uniform_partition(0, 1, 5, nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.1, 0.3, 0.5, 0.7, 0.9]

    fvals = [0 + 1j, 1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j]
    interpolator = nearest_interpolator(fvals, part.coord_vectors)

    # Evaluate at single point
    val = interpolator(0.35)  # closest to index 1 -> 1 + 2j
    assert val == 1.0 + 2.0j
    # Input array, with and without output array
    pts = np.array([0.4, 0.0, 0.65, 0.95])
    true_arr = [1 + 2j, 0 + 1j, 3 + 4j, 4 + 5j]
    assert all_equal(interpolator(pts), true_arr)
    # Should also work with a (1, N) array
    pts = pts[None, :]
    assert all_equal(interpolator(pts), true_arr)
    out = np.empty(4, dtype='complex128')
    interpolator(pts, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    # Same as array for 1d
    mg = sparse_meshgrid([0.4, 0.0, 0.65, 0.95])
    true_mg = [1 + 2j, 0 + 1j, 3 + 4j, 4 + 5j]
    assert all_equal(interpolator(mg), true_mg)
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_nearest_interpolation_1d_variants():
    """Test nearest neighbor interpolation variants in 1d."""
    part = odl.uniform_partition(0, 1, 5, nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.1, 0.3, 0.5, 0.7, 0.9]

    fvals = [0, 1, 2, 3, 4]

    # 'left' variant
    interpolator = nearest_interpolator(
        fvals, part.coord_vectors, variant='left'
    )

    # Testing two midpoints and the extreme values
    pts = np.array([0.4, 0.8, 0.0, 1.0])
    true_arr = [1, 3, 0, 4]
    assert all_equal(interpolator(pts), true_arr)

    # 'right' variant
    interpolator = nearest_interpolator(
        fvals, part.coord_vectors, variant='right'
    )

    # Testing two midpoints and the extreme values
    pts = np.array([0.4, 0.8, 0.0, 1.0])
    true_arr = [2, 4, 0, 4]
    assert all_equal(interpolator(pts), true_arr)


def test_nearest_interpolation_2d_float():
    """Test nearest neighbor interpolation in 2d."""
    part = odl.uniform_partition([0, 0], [1, 1], (4, 2), nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    fvals = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7], dtype='float64'
    ).reshape(part.shape)
    interpolator = nearest_interpolator(fvals, part.coord_vectors)

    # Evaluate at single point
    val = interpolator([0.3, 0.6])  # closest to index (1, 1) -> 3
    assert val == 3.0
    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [1.0, 1.0]])
    true_arr = [3, 7]
    assert all_equal(interpolator(pts.T), true_arr)
    out = np.empty(2, dtype='float64')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0])
    # Indices: (1, 3) x (0, 1)
    true_mg = [[2, 3],
               [6, 7]]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_nearest_interpolation_2d_string():
    """Test nearest neighbor interpolation in 2d with string values."""
    part = odl.uniform_partition([0, 0], [1, 1], (4, 2), nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    fvals = np.array([c for c in 'mystring']).reshape(part.shape)
    interpolator = nearest_interpolator(fvals, part.coord_vectors)

    # Evaluate at single point
    val = interpolator([0.3, 0.6])  # closest to index (1, 1) -> 3
    assert val == 't'
    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [1.0, 1.0]])
    true_arr = ['t', 'g']
    assert all_equal(interpolator(pts.T), true_arr)
    out = np.empty(2, dtype='U1')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)
    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 1.0])
    # Indices: (1, 3) x (0, 1)
    true_mg = [['s', 't'],
               ['n', 'g']]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='U1')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_linear_interpolation_1d():
    """Test linear interpolation in 1d."""
    part = odl.uniform_partition(0, 1, 5, nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.1, 0.3, 0.5, 0.7, 0.9]

    fvals = np.array([1, 2, 3, 4, 5], dtype='float64')
    interpolator = linear_interpolator(fvals, part.coord_vectors)

    # Evaluate at single point
    val = interpolator(0.35)
    true_val = 0.75 * 2 + 0.25 * 3
    assert val == pytest.approx(true_val)

    # Input array, with and without output array
    pts = np.array([0.4, 0.0, 0.65, 0.95])
    true_arr = [2.5, 0.5, 3.75, 3.75]
    assert all_almost_equal(interpolator(pts), true_arr)


def test_linear_interpolation_2d():
    """Test linear interpolation in 2d."""
    part = odl.uniform_partition([0, 0], [1, 1], (4, 2), nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    fvals = np.arange(1, 9, dtype='float64').reshape(part.shape)
    interpolator = linear_interpolator(fvals, part.coord_vectors)

    # Evaluate at single point
    val = interpolator([0.3, 0.6])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val = ((1 - l1) * (1 - l2) * fvals[0, 0] +
                (1 - l1) * l2 * fvals[0, 1] +
                l1 * (1 - l2) * fvals[1, 0] +
                l1 * l2 * fvals[1, 1])
    assert val == pytest.approx(true_val)

    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [0.1, 0.25],
                    [1.0, 1.0]])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    l2 = (0.6 - 0.25) / (0.75 - 0.25)
    true_val_1 = ((1 - l1) * (1 - l2) * fvals[0, 0] +
                  (1 - l1) * l2 * fvals[0, 1] +
                  l1 * (1 - l2) * fvals[1, 0] +
                  l1 * l2 * fvals[1, 1])
    l1 = (0.125 - 0.1) / (0.375 - 0.125)
    # l2 = 0
    true_val_2 = (1 - l1) * fvals[0, 0]  # only lower left contributes
    l1 = (1.0 - 0.875) / (0.875 - 0.625)
    l2 = (1.0 - 0.75) / (0.75 - 0.25)
    true_val_3 = (1 - l1) * (1 - l2) * fvals[3, 1]  # lower left only
    true_arr = [true_val_1, true_val_2, true_val_3]
    assert all_equal(interpolator(pts.T), true_arr)

    out = np.empty(3, dtype='float64')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)

    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 0.75])
    # Indices: (1, 3) x (0, 1)
    lx1 = (0.3 - 0.125) / (0.375 - 0.125)
    lx2 = (1.0 - 0.875) / (0.875 - 0.625)
    ly1 = (0.4 - 0.25) / (0.75 - 0.25)
    # ly2 = 0
    true_val_11 = ((1 - lx1) * (1 - ly1) * fvals[0, 0] +
                   (1 - lx1) * ly1 * fvals[0, 1] +
                   lx1 * (1 - ly1) * fvals[1, 0] +
                   lx1 * ly1 * fvals[1, 1])
    true_val_12 = ((1 - lx1) * fvals[0, 1] + lx1 * fvals[1, 1])  # ly2 = 0
    true_val_21 = ((1 - lx2) * (1 - ly1) * fvals[3, 0] +
                   (1 - lx2) * ly1 * fvals[3, 1])  # high node 1.0, no upper
    true_val_22 = (1 - lx2) * fvals[3, 1]  # ly2 = 0, no upper for 1.0
    true_mg = [[true_val_11, true_val_12],
               [true_val_21, true_val_22]]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_per_axis_interpolation():
    """Test different interpolation schemes per axis."""
    part = odl.uniform_partition([0, 0], [1, 1], (4, 2), nodes_on_bdry=False)
    # Coordinate vectors are:
    # [0.125, 0.375, 0.625, 0.875], [0.25, 0.75]

    schemes = ['linear', 'nearest']
    variants = [None, 'right']
    fvals = np.arange(1, 9, dtype='float64').reshape(part.shape)
    interpolator = per_axis_interpolator(
        fvals, part.coord_vectors, schemes, variants
    )

    # Evaluate at single point
    val = interpolator([0.3, 0.5])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    # 0.5 equally far from both neighbors -> 'right' chooses 0.75
    true_val = (1 - l1) * fvals[0, 1] + l1 * fvals[1, 1]
    assert val == pytest.approx(true_val)

    # Input array, with and without output array
    pts = np.array([[0.3, 0.6],
                    [0.1, 0.25],
                    [1.0, 1.0]])
    l1 = (0.3 - 0.125) / (0.375 - 0.125)
    true_val_1 = (1 - l1) * fvals[0, 1] + l1 * fvals[1, 1]
    l1 = (0.125 - 0.1) / (0.375 - 0.125)
    true_val_2 = (1 - l1) * fvals[0, 0]  # only lower left contributes
    l1 = (1.0 - 0.875) / (0.875 - 0.625)
    true_val_3 = (1 - l1) * fvals[3, 1]  # lower left only
    true_arr = [true_val_1, true_val_2, true_val_3]
    assert all_equal(interpolator(pts.T), true_arr)

    out = np.empty(3, dtype='float64')
    interpolator(pts.T, out=out)
    assert all_equal(out, true_arr)

    # Input meshgrid, with and without output array
    mg = sparse_meshgrid([0.3, 1.0], [0.4, 0.85])
    # Indices: (1, 3) x (0, 1)
    lx1 = (0.3 - 0.125) / (0.375 - 0.125)
    lx2 = (1.0 - 0.875) / (0.875 - 0.625)
    true_val_11 = (1 - lx1) * fvals[0, 0] + lx1 * fvals[1, 0]
    true_val_12 = ((1 - lx1) * fvals[0, 1] + lx1 * fvals[1, 1])
    true_val_21 = (1 - lx2) * fvals[3, 0]
    true_val_22 = (1 - lx2) * fvals[3, 1]
    true_mg = [[true_val_11, true_val_12],
               [true_val_21, true_val_22]]
    assert all_equal(interpolator(mg), true_mg)
    out = np.empty((2, 2), dtype='float64')
    interpolator(mg, out=out)
    assert all_equal(out, true_mg)


def test_collocation_interpolation_identity():
    """Check if collocation is left-inverse to interpolation."""
    # Interpolation followed by collocation on the same grid should be
    # the identity
    part = odl.uniform_partition([0, 0], [1, 1], (4, 2))
    fvals = np.arange(1, 9, dtype='float64').reshape(part.shape)

    collocator = lambda f: point_collocation(f, part.meshgrid)
    interpolators = [
        nearest_interpolator(fvals, part.coord_vectors, variant='left'),
        nearest_interpolator(fvals, part.coord_vectors, variant='right'),
        linear_interpolator(fvals, part.coord_vectors),
        per_axis_interpolator(
            fvals, part.coord_vectors, schemes=['linear', 'nearest']
        ),
    ]

    for interpolator in interpolators:
        ident_values = collocator(interpolator)
        assert all_almost_equal(ident_values, fvals)


if __name__ == '__main__':
    odl.util.test_file(__file__)
