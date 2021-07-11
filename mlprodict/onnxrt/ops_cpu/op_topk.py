# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun
from ._op_onnx_numpy import (  # pylint: disable=E0611,E0401
    topk_element_min_double, topk_element_max_double, topk_element_fetch_double,
    topk_element_min_float, topk_element_max_float, topk_element_fetch_float,
    topk_element_min_int64, topk_element_max_int64, topk_element_fetch_int64)


def topk_sorted_implementation(X, k, axis, largest):
    """
    Retrieves the top-k elements.

    @param      X           data
    @param      k           k in top-k
    @param      axis        axis chosen to select the top-k elements
    @param      largest     largest (1) or smallest (0)
    @return                 top-k values, top-k indices

    See function `_kneighbors_reduce_func
    <https://github.com/scikit-learn/scikit-learn/tree/master/
    sklearn/neighbors/base.py#L304>`_.
    """
    if isinstance(k, numpy.ndarray):
        if k.size != 1:
            raise RuntimeError(  # pragma: no cover
                "k must be an integer not %r." % k)
        k = k[0]
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


def topk_sorted_implementation_cpp(X, k, axis, largest, th_para=50):
    """
    Retrieves the top-k elements using a C++
    implementation when the axis is the last dimension,
    otherwise, it falls back to
    @see fn topk_sorted_implementation.

    @param      X           data
    @param      k           k in top-k
    @param      axis        axis chosen to select the top-k elements
    @param      largest     largest (1) or smallest (0)
    @param      th_para     threshold for parallelisation
    @return                 top-k values, top-k indices
    """
    if isinstance(k, numpy.ndarray):
        if k.size != 1:
            raise RuntimeError(  # pragma: no cover
                "k must be an integer not %r." % k)
    if axis != len(X.shape) - 1:
        if k == 0:
            return numpy.empty((0,), dtype=numpy.int64)
        return topk_sorted_implementation(X, k, axis, largest)
    if X.dtype == numpy.float64:
        if k == 0:
            return numpy.empty((0,), dtype=X.dtype), numpy.empty((0,), dtype=numpy.int64)
        if largest:
            topk_sorted_indices = topk_element_max_double(X, k, True, th_para)
        else:
            topk_sorted_indices = topk_element_min_double(X, k, True, th_para)
        topk_sorted_values = topk_element_fetch_double(X, topk_sorted_indices)
    elif X.dtype == numpy.float32:
        if k == 0:
            return numpy.empty((0,), dtype=X.dtype), numpy.empty((0,), dtype=numpy.int64)
        if largest:
            topk_sorted_indices = topk_element_max_float(X, k, True, th_para)
        else:
            topk_sorted_indices = topk_element_min_float(X, k, True, th_para)
        topk_sorted_values = topk_element_fetch_float(X, topk_sorted_indices)
    elif X.dtype == numpy.int64:
        if k == 0:
            return numpy.empty((0,), dtype=X.dtype), numpy.empty((0,), dtype=numpy.int64)
        if largest:
            topk_sorted_indices = topk_element_max_int64(X, k, True, th_para)
        else:
            topk_sorted_indices = topk_element_min_int64(X, k, True, th_para)
        topk_sorted_values = topk_element_fetch_int64(X, topk_sorted_indices)
    else:
        if k == 0:
            return numpy.empty((0,), dtype=numpy.int64)
        return topk_sorted_implementation(X, k, axis, largest)
    return topk_sorted_values, topk_sorted_indices


class _CommonTopK(OpRun):
    """
    Ths class hides a parameter used as a threshold above
    which the parallelisation is started: ``th_para``.
    """

    atts = {'axis': -1}

    def __init__(self, *args, **options):
        OpRun.__init__(self, *args, **options)
        self.th_para = 50

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
        sort, sorti = topk_sorted_implementation_cpp(
            data, k, axis, largest, self.th_para)
        return (sort, sorti.astype(numpy.int64))

    def _infer_shapes(self, data, ink):  # pylint: disable=W0221
        axis = self.axis if self.axis >= 0 else (self.axis + len(data))
        sh = data.copy()
        pref = str(hex(id(self))[2:])
        sh[axis] = "ntopk%s" % pref
        shi = sh.copy(dtype=numpy.int64)
        return (sh, shi)

    def _infer_types(self, x, ink):  # pylint: disable=E0202,W0221
        return (x, numpy.int64)


class TopK_1(_CommonTopK):

    atts = {'axis': -1, 'k': None}

    def __init__(self, onnx_node, desc=None, **options):
        _CommonTopK.__init__(self, onnx_node, desc=desc,
                             expected_attributes=TopK_10.atts,
                             **options)

    def _run(self, data):  # pylint: disable=W0221
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
        return _CommonTopK._common_run(self, data, [self.k])

    def _infer_shapes(self, data):  # pylint: disable=W0221
        return _CommonTopK._infer_shapes(self, data, [self.k])

    def _infer_types(self, data):  # pylint: disable=W0221
        return (data, )

    def _infer_sizes(self, *args):  # pylint: disable=W0221
        res = self.run(*args)
        x = args[0]
        return (dict(temp=x.dtype.itemsize * self.k * 2), ) + res


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

    def _infer_sizes(self, data, ink):  # pylint: disable=W0221
        res = self.run(data, ink)
        return (dict(temp=data.dtype.itemsize * ink[0] * 2), ) + res


class TopK_11(_CommonTopK):

    atts = {'axis': -1, 'largest': 1, 'sorted': 1}

    def __init__(self, onnx_node, desc=None, **options):
        _CommonTopK.__init__(self, onnx_node, desc=desc,
                             expected_attributes=TopK_11.atts,
                             **options)
        if self.sorted not in (True, 1):
            raise RuntimeError(  # pragma: no cover
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

    def _infer_sizes(self, data, ink):  # pylint: disable=W0221
        res = self.run(data, ink)
        return (dict(temp=data.dtype.itemsize * ink[0] * 2), ) + res


if onnx_opset_version() >= 11:
    TopK = TopK_11
elif onnx_opset_version() >= 10:  # pragma: no cover
    TopK = TopK_10
else:  # pragma: no cover
    TopK = TopK_1
