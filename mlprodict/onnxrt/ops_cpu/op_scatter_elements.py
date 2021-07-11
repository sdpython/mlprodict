# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ..shape_object import ShapeObject
from ._op import OpRun


def scatter_elements(data, indices, updates, axis=0):
    """
    ::
        // for 3-dim and axis=0
        //    output[indices[i][j][k]][j][k] = updates[i][j][k]
        // for axis 1
        //    output[i][indices[i][j][k]][k] = updates[i][j][k]
        // and so on
    """
    if len(data.shape) == 1 and axis == 0:
        scattered = numpy.copy(data)
        for pos, up in zip(indices, updates):
            scattered[pos] = up
        return scattered

    if axis < 0:
        axis = data.ndim + axis

    idx_xsection_shape = indices.shape[:axis] + indices.shape[axis + 1:]

    def make_slice(arr, axis, i):
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = unpacked, packed[i]
        return unpacked

    # We use indices and axis parameters to create idx
    # idx is in a form that can be used as a NumPy advanced
    # indices for scattering of updates param. in data
    idx = [[unpack(numpy.indices(idx_xsection_shape).reshape(indices.ndim - 1, -1)),
            indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0]]
           for i in range(indices.shape[axis])]
    idx = list(numpy.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())

    # updates_idx is a NumPy advanced indices for indexing
    # of elements in the updates
    updates_idx = list(idx)
    updates_idx.pop(axis)
    updates_idx.insert(axis, numpy.repeat(numpy.arange(indices.shape[axis]),
                                          numpy.prod(idx_xsection_shape)))

    scattered = numpy.copy(data)
    scattered[tuple(idx)] = updates[tuple(updates_idx)]
    return scattered


class ScatterElements(OpRun):

    atts = {'axis': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, data, indices, updates):  # pylint: disable=W0221
        res = scatter_elements(data, indices, updates, axis=self.axis)
        return (res, )

    def _infer_shapes(self, data, indices, updates):  # pylint: disable=W0221
        return (ShapeObject(data.shape, dtype=data.dtype), )

    def _infer_types(self, data, indices, updates):  # pylint: disable=W0221
        return (data, )

    def _infer_sizes(self, *args):  # pylint: disable=W0221
        res = self.run(*args)
        return (dict(temp=0), ) + res
