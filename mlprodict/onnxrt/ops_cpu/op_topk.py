# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class TopK_10(OpRun):

    atts = {'axis': -1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=TopK_10.atts,
                       **options)

    def _run(self, data, ink):  # pylint: disable=W0221
        """
        Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what :epkg:`onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        k = ink[0]
        axis = self.axis if self.axis >= 0 else (self.axis + len(data.shape))
        sorti = numpy.argsort(data, axis=axis)
        sort = numpy.sort(data, axis=axis)
        if k > 0:
            shapes = [0 for s in data.shape]
            shapes[axis] = data.shape[axis] - k
            indices = tuple(slice(b, e) for b, e in zip(shapes, data.shape))
            return (sort[indices], sorti[indices])
        else:
            return (sort, sorti)

    def _infer_shapes(self, data, ink):  # pylint: disable=W0221
        axis = self.axis if self.axis >= 0 else (self.axis + len(data))
        sh = data.copy()
        pref = str(hex(id(self))[2:])
        sh[axis] = "ntopk%s" % pref
        shi = sh.copy(dtype=numpy.int64)
        return (sh, shi)


def topk_sorted_implementation(X, k, axis, largest):
    """
    Retrieves the top-k elements.

    @param      X           data
    @param      k           k in top-k
    @param      axis        axis chosen to select the top-k elements
    @param      largest     largest (1) or smallest (0)
    @return                 top-k values, top-k indices
    """
    sorted_indices = numpy.argsort(X, axis=axis)
    sorted_values = numpy.sort(X, axis=axis)
    if largest:
        sorted_indices = numpy.flip(sorted_indices, axis=axis)
        sorted_values = numpy.flip(sorted_values, axis=axis)
    ark = numpy.arange(k)
    topk_sorted_indices = numpy.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = numpy.take(sorted_values, ark, axis=axis)
    return topk_sorted_values, topk_sorted_indices


class TopK(OpRun):

    atts = {'axis': -1, 'largest': 1, 'sorted': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=TopK.atts,
                       **options)
        if self.sorted not in (True, 1):
            raise RuntimeError(
                "TopK does not implement anything for sorted=0.")

    def _run(self, data, ink):  # pylint: disable=W0221
        """
        Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what :epkg:`onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        k = ink[0]
        axis = self.axis if self.axis >= 0 else (self.axis + len(data.shape))
        sort, sorti = topk_sorted_implementation(data, k, axis, self.largest)
        return (sort, sorti)

    def _infer_shapes(self, data, ink):  # pylint: disable=W0221
        axis = self.axis if self.axis >= 0 else (self.axis + len(data))
        sh = data.copy()
        pref = str(hex(id(self))[2:])
        sh[axis] = "ntopk%s" % pref
        shi = sh.copy(dtype=numpy.int64)
        return (sh, shi)
