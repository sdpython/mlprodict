# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def gather_numpy_2(self, dim, index):
    res = []
    for a, b in zip(self, index):
        res.append(a[b[0]])
    res = numpy.array(
        res, dtype=self.dtype).reshape(
            index.shape)
    return res


def gather_numpy(self, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:

    ::

        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values

    See `How to do scatter and gather operations in numpy?
    <https://stackoverflow.com/questions/46065873/
    how-to-do-scatter-and-gather-operations-in-numpy/46204790#46204790>`_
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError(  # pragma: no cover
            "Except for dimension {}, all dimensions of "
            "index and self should be the same size".format(dim))
    data_swaped = numpy.swapaxes(self, 0, dim)
    index_swaped = numpy.swapaxes(index, 0, dim)
    try:
        gathered = numpy.choose(index_swaped, data_swaped)
    except ValueError as e:
        if len(index_swaped.shape) == 2 and len(data_swaped.shape) == 2:
            return gather_numpy_2(self, dim, index)
        raise e  # pragma: no cover

    return numpy.swapaxes(gathered, 0, dim)


class GatherElements(OpRun):

    atts = {'axis': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=GatherElements.atts,
                       **options)

    def _run(self, data, indices):  # pylint: disable=W0221
        if indices.size == 0:
            return (numpy.empty((0, ), dtype=data.dtype), )
        y = gather_numpy(data, self.axis, indices)
        return (y, )

    def _infer_shapes(self, data, indices):  # pylint: disable=W0221
        return (indices, )

    def to_python(self, inputs):
        lines = ['data_swaped = numpy.swapaxes(%s, 0, axis)' % inputs[0],
                 'index_swaped = numpy.swapaxes(%s, 0, axis)' % inputs[1],
                 "gathered = numpy.choose(index_swaped, data_swaped, mode='wrap')",
                 'return numpy.swapaxes(gathered, 0, axis)']
        return "import numpy", "\n".join(lines)
