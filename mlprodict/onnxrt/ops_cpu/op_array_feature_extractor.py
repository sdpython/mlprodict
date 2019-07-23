# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun


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
            <https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/array_feature_extractor.cc#L84>`_.
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
        return (res, )
