# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

from functools import partial

import numpy as np

import odl
from odl.discr.discr_utils import point_collocation
from odl.discr.grid import sparse_meshgrid
from odl.util.testutils import all_almost_equal, all_equal, simple_fixture

# --- Helper functions --- #


def _test_eq(x, y):
    """Test equality of x and y."""
    assert x == y
    assert not x != y
    assert hash(x) == hash(y)


def _test_neq(x, y):
    """Test non-equality of x and y."""
    assert x != y
    assert not x == y
    assert hash(x) != hash(y)


def _points(domain, num):
    """Helper to generate ``num`` points in ``domain``."""
    min_pt = domain.min_pt
    max_pt = domain.max_pt
    ndim = domain.ndim
    points = np.random.uniform(low=0, high=1, size=(ndim, num))
    for i in range(ndim):
        points[i, :] = min_pt[i] + (max_pt[i] - min_pt[i]) * points[i]
    return points


def _meshgrid(domain, shape):
    """Helper to generate a ``shape`` meshgrid of points in ``domain``."""
    min_pt = domain.min_pt
    max_pt = domain.max_pt
    ndim = domain.ndim
    coord_vecs = []
    for i in range(ndim):
        vec = np.random.uniform(low=min_pt[i], high=max_pt[i], size=shape[i])
        vec.sort()
        coord_vecs.append(vec)
    return sparse_meshgrid(*coord_vecs)


class FuncList(list):  # So we can set __name__
    pass


# --- pytest fixtures (general) --- #


out_dtype_params = ['float32', 'float64', 'complex64']
out_dtype = simple_fixture('out_dtype', out_dtype_params,
                           fmt=' {name} = {value!r} ')

out_shape = simple_fixture('out_shape', [(), (2,), (2, 3)])
domain_ndim = simple_fixture('domain_ndim', [1, 2])
vectorized = simple_fixture('vectorized', [True, False])


# --- pytest fixtures (scalar test functions) --- #


def func_nd_oop(x):
    return sum(x)


def func_nd_ip(x, out):
    out[:] = sum(x)


def func_nd_dual(x, out=None):
    if out is None:
        return sum(x)
    else:
        out[:] = sum(x)


def func_nd_bcast_ref(x):
    return x[0] + 0 * sum(x[1:])


def func_nd_bcast_oop(x):
    return x[0]


def func_nd_bcast_ip(x, out):
    out[:] = x[0]


def func_nd_bcast_dual(x, out=None):
    if out is None:
        return x[0]
    else:
        out[:] = x[0]


func_nd_ref = func_nd_oop
func_nd_params = [(func_nd_ref, f)
                  for f in [func_nd_oop, func_nd_ip, func_nd_dual]]
func_nd_params.extend([(func_nd_bcast_ref, func_nd_bcast_oop),
                       (func_nd_bcast_ref, func_nd_bcast_ip)])

func_nd = simple_fixture('func_nd', func_nd_params,
                         fmt=' {name} = {value[1].__name__} ')


def func_nd_other(x):
    return sum(x) + 1


def func_param_nd_oop(x, c):
    return sum(x) + c


def func_param_nd_ip(x, out, c):
    out[:] = sum(x) + c


def func_param_switched_nd_ip(x, c, out):
    out[:] = sum(x) + c


def func_param_bcast_nd_ref(x, c):
    return x[0] + c + 0 * sum(x[1:])


def func_param_bcast_nd_oop(x, c):
    return x[0] + c


def func_param_bcast_nd_ip(x, out, c):
    out[:] = x[0] + c


func_param_nd_ref = func_param_nd_oop
func_param_nd_params = [(func_param_nd_ref, f)
                        for f in [func_param_nd_oop, func_param_nd_ip,
                                  func_param_switched_nd_ip]]
func_param_nd_params.extend(
    [(func_param_bcast_nd_ref, func_param_bcast_nd_oop),
     (func_param_bcast_nd_ref, func_param_bcast_nd_ip)])
func_param_nd = simple_fixture('func_with_param', func_param_nd_params,
                               fmt=' {name} = {value[1].__name__} ')


def func_1d_ref(x):
    return x[0] * 2


def func_1d_oop(x):
    return x * 2


def func_1d_ip(x, out):
    out[:] = x * 2


func_1d_params = [(func_1d_ref, func_1d_oop), (func_1d_ref, func_1d_ip)]
func_1d_params.append((lambda x: -x[0], np.negative))
func_1d = simple_fixture('func_1d', func_1d_params,
                         fmt=' {name} = {value[1].__name__} ')


def func_complex_nd_oop(x):
    return sum(x) + 1j


# --- pytest fixtures (vector-valued test functions) --- #


def func_vec_nd_ref(x):
    return np.array([sum(x) + 1, sum(x) - 1])


def func_vec_nd_oop(x):
    return (sum(x) + 1, sum(x) - 1)


func_nd_oop_seq = FuncList([lambda x: sum(x) + 1, lambda x: sum(x) - 1])
func_nd_oop_seq.__name__ = 'func_nd_oop_seq'


def func_vec_nd_ip(x, out):
    out[0] = sum(x) + 1
    out[1] = sum(x) - 1


def comp0_nd(x, out):
    out[:] = sum(x) + 1


def comp1_nd(x, out):
    out[:] = sum(x) - 1


def func_vec_nd_dual(x, out=None):
    if out is None:
        return (sum(x) + 1, sum(x) - 1)
    else:
        out[0] = sum(x) + 1
        out[1] = sum(x) - 1


func_nd_ip_seq = FuncList([comp0_nd, comp1_nd])
func_nd_ip_seq.__name__ = 'func_nd_ip_seq'

func_vec_nd_params = [(func_vec_nd_ref, f)
                      for f in [func_vec_nd_oop, func_nd_oop_seq,
                                func_vec_nd_ip, func_nd_ip_seq]]
func_vec_nd = simple_fixture('func_vec_nd', func_vec_nd_params,
                             fmt=' {name} = {value[1].__name__} ')


def func_vec_nd_other(x):
    return np.array([sum(x) + 2, sum(x) + 3])


def func_vec_1d_ref(x):
    return np.array([x[0] * 2, x[0] + 1])


def func_vec_1d_oop(x):
    return (x * 2, x + 1)


func_1d_oop_seq = FuncList([lambda x: x * 2, lambda x: x + 1])
func_1d_oop_seq.__name__ = 'func_1d_oop_seq'


def func_vec_1d_ip(x, out):
    out[0] = x * 2
    out[1] = x + 1


def comp0_1d(x, out):
    out[:] = x * 2


def comp1_1d(x, out):
    out[:] = x + 1


func_1d_ip_seq = FuncList([comp0_1d, comp1_1d])
func_1d_ip_seq.__name__ = 'func_1d_ip_seq'

func_vec_1d_params = [(func_vec_1d_ref, f)
                      for f in [func_vec_1d_oop, func_1d_oop_seq,
                                func_vec_1d_ip, func_1d_ip_seq]]
func_vec_1d = simple_fixture('func_vec_1d', func_vec_1d_params,
                             fmt=' {name} = {value[1].__name__} ')


def func_vec_complex_nd_oop(x):
    return (sum(x) + 1j, sum(x) - 1j)


# --- pytest fixtures (tensor-valued test functions) --- #


def func_tens_ref(x):
    # Reference function where all shapes in the list are correct
    # without broadcasting
    shp = np.broadcast(*x).shape
    return np.array([[x[0] - x[1], np.zeros(shp), x[1] + 0 * x[0]],
                     [np.ones(shp), x[0] + 0 * x[1], sum(x)]])


def func_tens_oop(x):
    # Output shape 2x3, input 2-dimensional. Broadcasting supported.
    return [[x[0] - x[1], 0, x[1]],
            [1, x[0], sum(x)]]


def func_tens_ip(x, out):
    # In-place version
    out[0, 0] = x[0] - x[1]
    out[0, 1] = 0
    out[0, 2] = x[1]
    out[1, 0] = 1
    out[1, 1] = x[0]
    out[1, 2] = sum(x)


# Array of functions. May contain constants. Should yield the same as func.
func_tens_oop_seq = FuncList([[lambda x: x[0] - x[1], 0, lambda x: x[1]],
                              [1, lambda x: x[0], lambda x: sum(x)]])
func_tens_oop_seq.__name__ = 'func_tens_oop_seq'


# In-place component functions, cannot use lambdas
def comp00(x, out):
    out[:] = x[0] - x[1]


def comp01(x, out):
    out[:] = 0


def comp02(x, out):
    out[:] = x[1]


def comp10(x, out):
    out[:] = 1


def comp11(x, out):
    out[:] = x[0]


def comp12(x, out):
    out[:] = sum(x)


func_tens_ip_seq = FuncList([[comp00, comp01, comp02],
                             [comp10, comp11, comp12]])
func_tens_ip_seq.__name__ = 'func_tens_ip_seq'


def func_tens_dual(x, out=None):
    if out is None:
        return [[x[0] - x[1], 0, x[1]],
                [1, x[0], sum(x)]]
    else:
        out[0, 0] = x[0] - x[1]
        out[0, 1] = 0
        out[0, 2] = x[1]
        out[1, 0] = 1
        out[1, 1] = x[0]
        out[1, 2] = sum(x)


func_tens_params = [(func_tens_ref, f)
                    for f in [func_tens_oop, func_tens_oop_seq,
                              func_tens_ip, func_tens_ip_seq]]
func_tens = simple_fixture('func_tens', func_tens_params,
                           fmt=' {name} = {value[1].__name__} ')


def func_tens_other(x):
    return np.array([[x[0] + x[1], sum(x), sum(x)],
                     [sum(x), 2 * x[0] - x[1], sum(x)]])


def func_tens_complex_oop(x):
    return [[x[0], 0, 1j * x[0]],
            [1j, x, sum(x) + 1j]]


# --- FunctionSpaceElement tests --- #


def test_point_collocation_scalar(domain_ndim, out_dtype, func_nd):
    """Check evaluation of scalar-valued function elements."""
    domain = odl.IntervalProd([0] * domain_ndim, [1] * domain_ndim)
    points = _points(domain, 3)
    mesh_shape = tuple(range(2, 2 + domain_ndim))
    mesh = _meshgrid(domain, mesh_shape)
    point = [0.5] * domain_ndim

    func_ref, func = func_nd

    true_values_points = func_ref(points)
    true_values_mesh = func_ref(mesh)
    true_value_point = func_ref(point)

    collocator = partial(point_collocation, func)

    # Out of place
    result_points = collocator(points)
    result_mesh = collocator(mesh)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)
    assert result_points.flags.writeable
    assert result_mesh.flags.writeable

    # In place
    out_points = np.empty(3, dtype=out_dtype)
    out_mesh = np.empty(mesh_shape, dtype=out_dtype)
    collocator(points, out=out_points)
    collocator(mesh, out=out_mesh)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Single point evaluation
    result_point = collocator(point)
    assert all_almost_equal(result_point, true_value_point)


def test_fspace_scal_elem_with_param_eval(func_param_nd):
    """Check evaluation of scalar-valued function elements with parameters."""
    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace_scal = FunctionSpace(intv)
    points = _points(fspace_scal.domain, 3)
    mesh_shape = (2, 3)
    mesh = _meshgrid(fspace_scal.domain, mesh_shape)

    func_ref, func = func_param_nd

    true_values_points = func_ref(points, c=2.5)
    true_values_mesh = func_ref(mesh, c=2.5)

    func_elem = fspace_scal.element(func)

    # Out of place
    result_points = func_elem(points, c=2.5)
    result_mesh = func_elem(mesh, c=2.5)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)

    # In place
    out_points = np.empty(3, dtype=fspace_scal.scalar_out_dtype)
    out_mesh = np.empty(mesh_shape, dtype=fspace_scal.scalar_out_dtype)
    func_elem(points, out=out_points, c=2.5)
    func_elem(mesh, out=out_mesh, c=2.5)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Complex output
    fspace_complex = FunctionSpace(intv, out_dtype=complex)
    true_values_points = func_ref(points, c=2j)
    true_values_mesh = func_ref(mesh, c=2j)

    func_elem = fspace_complex.element(func)

    result_points = func_elem(points, c=2j)
    result_mesh = func_elem(mesh, c=2j)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)


def test_fspace_vec_elem_eval(func_vec_nd, out_dtype):
    """Check evaluation of scalar-valued function elements."""
    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))
    points = _points(fspace_vec.domain, 3)
    mesh_shape = (2, 3)
    mesh = _meshgrid(fspace_vec.domain, mesh_shape)
    point = [0.5, 0.5]
    values_points_shape = (2, 3)
    values_mesh_shape = (2, 2, 3)

    func_ref, func = func_vec_nd

    true_values_points = func_ref(points)
    true_values_mesh = func_ref(mesh)
    true_value_point = func_ref(point)

    func_elem = fspace_vec.element(func)

    # Out of place
    result_points = func_elem(points)
    result_mesh = func_elem(mesh)
    assert all_almost_equal(result_points, true_values_points)
    assert all_almost_equal(result_mesh, true_values_mesh)
    assert result_points.dtype == fspace_vec.scalar_out_dtype
    assert result_mesh.dtype == fspace_vec.scalar_out_dtype
    assert result_points.flags.writeable
    assert result_mesh.flags.writeable

    # In place
    out_points = np.empty(values_points_shape,
                          dtype=fspace_vec.scalar_out_dtype)
    out_mesh = np.empty(values_mesh_shape,
                        dtype=fspace_vec.scalar_out_dtype)
    func_elem(points, out=out_points)
    func_elem(mesh, out=out_mesh)
    assert all_almost_equal(out_points, true_values_points)
    assert all_almost_equal(out_mesh, true_values_mesh)

    # Single point evaluation
    result_point = func_elem(point)
    assert all_almost_equal(result_point, true_value_point)
    out_point = np.empty((2,), dtype=fspace_vec.scalar_out_dtype)
    func_elem(point, out=out_point)
    assert all_almost_equal(out_point, true_value_point)


def test_fspace_tens_eval(func_tens):
    """Test tensor-valued function evaluation."""
    intv = odl.IntervalProd([0, 0], [1, 1])
    fspace_tens = FunctionSpace(intv, out_dtype=(float, (2, 3)))
    points = _points(fspace_tens.domain, 4)
    mesh_shape = (4, 5)
    mesh = _meshgrid(fspace_tens.domain, mesh_shape)
    point = [0.5, 0.5]
    values_points_shape = (2, 3, 4)
    values_mesh_shape = (2, 3, 4, 5)
    value_point_shape = (2, 3)

    func_ref, func = func_tens

    true_result_points = np.array(func_ref(points))
    true_result_mesh = np.array(func_ref(mesh))
    true_result_point = np.array(func_ref(np.array(point)[:, None])).squeeze()

    func_elem = fspace_tens.element(func)

    result_points = func_elem(points)
    result_mesh = func_elem(mesh)
    result_point = func_elem(point)
    assert all_almost_equal(result_points, true_result_points)
    assert all_almost_equal(result_mesh, true_result_mesh)
    assert all_almost_equal(result_point, true_result_point)
    assert result_points.flags.writeable
    assert result_mesh.flags.writeable
    assert result_point.flags.writeable

    out_points = np.empty(values_points_shape, dtype=float)
    out_mesh = np.empty(values_mesh_shape, dtype=float)
    out_point = np.empty(value_point_shape, dtype=float)
    func_elem(points, out=out_points)
    func_elem(mesh, out=out_mesh)
    func_elem(point, out=out_point)
    assert all_almost_equal(out_points, true_result_points)
    assert all_almost_equal(out_mesh, true_result_mesh)
    assert all_almost_equal(out_point, true_result_point)


def test_fspace_elem_eval_unusual_dtypes():
    """Check evaluation with unusual data types (int and string)."""
    str3 = odl.Strings(3)
    fspace = FunctionSpace(str3, out_dtype=int)
    strings = np.array(['aa', 'b', 'cab', 'aba'])
    out_vec = np.empty((4,), dtype=int)

    # Vectorized for arrays only
    func_elem = fspace.element(
        lambda s: np.array([str(si).count('a') for si in s]))
    true_values = [2, 0, 1, 2]

    assert func_elem('abc') == 1
    assert all_equal(func_elem(strings), true_values)
    func_elem(strings, out=out_vec)
    assert all_equal(out_vec, true_values)


def test_fspace_elem_eval_vec_1d(func_vec_1d):
    """Test evaluation in 1d since it's a corner case regarding shapes."""
    intv = odl.IntervalProd(0, 1)
    fspace_vec = FunctionSpace(intv, out_dtype=(float, (2,)))
    points = _points(fspace_vec.domain, 3)
    mesh_shape = (4,)
    mesh = _meshgrid(fspace_vec.domain, mesh_shape)
    point1 = 0.5
    point2 = [0.5]
    values_points_shape = (2, 3)
    values_mesh_shape = (2, 4)
    value_point_shape = (2,)

    func_ref, func = func_vec_1d

    true_result_points = np.array(func_ref(points))
    true_result_mesh = np.array(func_ref(mesh))
    true_result_point = np.array(func_ref(np.array([point1]))).squeeze()

    func_elem = fspace_vec.element(func)

    result_points = func_elem(points)
    result_mesh = func_elem(mesh)
    result_point1 = func_elem(point1)
    result_point2 = func_elem(point2)
    assert all_almost_equal(result_points, true_result_points)
    assert all_almost_equal(result_mesh, true_result_mesh)
    assert all_almost_equal(result_point1, true_result_point)
    assert all_almost_equal(result_point2, true_result_point)

    out_points = np.empty(values_points_shape, dtype=float)
    out_mesh = np.empty(values_mesh_shape, dtype=float)
    out_point1 = np.empty(value_point_shape, dtype=float)
    out_point2 = np.empty(value_point_shape, dtype=float)
    func_elem(points, out=out_points)
    func_elem(mesh, out=out_mesh)
    func_elem(point1, out=out_point1)
    func_elem(point2, out=out_point2)
    assert all_almost_equal(out_points, true_result_points)
    assert all_almost_equal(out_mesh, true_result_mesh)
    assert all_almost_equal(out_point1, true_result_point)
    assert all_almost_equal(out_point2, true_result_point)

if __name__ == '__main__':
    odl.util.test_file(__file__)
