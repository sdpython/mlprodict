# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun


def topk_sorted_implementation(X, k, axis, largest):
    """
    Retrieves the top-k elements.

    @param      X           data
    @param      k           k in top-k
    @param      axis        axis chosen to select the top-k elements
    @param      largest     largest (1) or smallest (0)
    @return                 top-k values, top-k indices

    See function `_kneighbors_reduce_func
    <https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/neighbors/base.py#L304>`_.
    """
    if len(X.shape) == 2 and axis == 1:
        sample_range = numpy.arange(X.shape[0])[:, None]
        if largest == 0:
            sorted_indices = numpy.argpartition(X, axis=axis, kth=k - 1)
            sorted_indices = sorted_indices[:, :k]
            # argpartition doesn't guarantee sorted order, so we sort again
            sorted_indices = sorted_indices[
                sample_range, numpy.argsort(X[sample_range, sorted_indices])]
        else:
            sorted_indices = numpy.argpartition(-X, axis=axis, kth=k - 1)
            sorted_indices = sorted_indices[:, :k]
            # argpartition doesn't guarantee sorted order, so we sort again
            sorted_indices = sorted_indices[
                sample_range, numpy.argsort(-X[sample_range, sorted_indices])]
        sorted_distances = X[sample_range, sorted_indices]
        return sorted_distances, sorted_indices

    sorted_indices = numpy.argsort(X, axis=axis)
    sorted_values = numpy.sort(X, axis=axis)
    if largest:
        sorted_indices = numpy.flip(sorted_indices, axis=axis)
        sorted_values = numpy.flip(sorted_values, axis=axis)
    ark = numpy.arange(k)
    topk_sorted_indices = numpy.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = numpy.take(sorted_values, ark, axis=axis)
    return topk_sorted_values, topk_sorted_indices


class _CommonTopK(OpRun):

    atts = {'axis': -1}

    def __init__(self, *args, **options):
        OpRun.__init__(self, *args, **options)

    def _common_run(self, data, ink, largest=1):  # pylint: disable=W0221
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
        sort, sorti = topk_sorted_implementation(data, k, axis, 1)
        return (sort, sorti.astype(numpy.int64))

    def _infer_shapes(self, data, ink):  # pylint: disable=W0221
        axis = self.axis if self.axis >= 0 else (self.axis + len(data))
        sh = data.copy()
        pref = str(hex(id(self))[2:])
        sh[axis] = "ntopk%s" % pref
        shi = sh.copy(dtype=numpy.int64)
        return (sh, shi)


class TopK_10(_CommonTopK):

    atts = {'axis': -1}

    def __init__(self, onnx_node, desc=None, **options):
        _CommonTopK.__init__(self, onnx_node, desc=desc,
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
        return _CommonTopK._common_run(self, data, ink)


class TopK_11(_CommonTopK):

    atts = {'axis': -1, 'largest': 1, 'sorted': 1}

    def __init__(self, onnx_node, desc=None, **options):
        _CommonTopK.__init__(self, onnx_node, desc=desc,
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
        return _CommonTopK._common_run(self, data, ink, self.largest)


if onnx_opset_version() >= 11:
    TopK = TopK_11
else:
    TopK = TopK_10
