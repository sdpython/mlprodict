# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject
from ._op_onnx_numpy import (  # pylint: disable=E0611,E0401
    array_feature_extractor_double,
    array_feature_extractor_int64,
    array_feature_extractor_float)


def _array_feature_extrator(data, indices):
    """
    Implementation of operator *ArrayFeatureExtractor*
    with :epkg:`numpy`.
    """
    if len(indices.shape) == 2 and indices.shape[0] == 1:
        index = indices.ravel().tolist()
        add = len(index)
    elif len(indices.shape) == 1:
        index = indices.tolist()
        add = len(index)
    else:
        add = 1
        for s in indices.shape:
            add *= s
        index = indices.ravel().tolist()
    if len(data.shape) == 1:
        new_shape = (1, add)
    else:
        new_shape = list(data.shape[:-1]) + [add]
    tem = data[..., index]
    res = tem.reshape(new_shape)
    return res


def sizeof_dtype(dty):
    if dty == numpy.float64:
        return 8
    if dty == numpy.float32:
        return 4
    if dty == numpy.int64:
        return 8
    raise ValueError(
        "Unable to get bytes size for type {}.".format(numpy.dtype))


class ArrayFeatureExtractor(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, data, indices):  # pylint: disable=W0221
        """
        Runtime for operator *ArrayFeatureExtractor*.

        .. warning::
            ONNX specifications may be imprecise in some cases.
            When the input data is a vector (one dimension),
            the output has still two like a matrix with one row.
            The implementation follows what :epkg:`onnxruntime` does in
            `array_feature_extractor.cc
            <https://github.com/microsoft/onnxruntime/blob/master/
            onnxruntime/core/providers/cpu/ml/array_feature_extractor.cc#L84>`_.
        """
        if data.dtype == numpy.float64:
            res = array_feature_extractor_double(data, indices)
        elif data.dtype == numpy.float32:
            res = array_feature_extractor_float(data, indices)
        elif data.dtype == numpy.int64:
            res = array_feature_extractor_int64(data, indices)
        else:
            # for strings, still not C++
            res = _array_feature_extrator(data, indices)
        return (res, )

    def _infer_shapes(self, data, indices):  # pylint: disable=W0221
        """
        Infer the shapes for the output.
        """
        add = indices.product()

        if len(data) == 1:
            dim = ShapeObject((1, add), dtype=data.dtype)
        else:
            dim = data.copy()
            dim.append(add)
        return (dim, )

    def _infer_types(self, data, indices):  # pylint: disable=W0221
        """
        Returns the type of the output.
        """
        return (data, )
