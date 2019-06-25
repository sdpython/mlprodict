# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun


class ArrayFeatureExtractor(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        if desc is None:
            raise ValueError("desc should not be None.")
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, data, indices):  # pylint: disable=W0221
        if len(indices.shape) == 2 and indices.shape[0] == 1:
            index = indices.ravel().tolist()
        elif len(indices.shape) == 1:
            index = indices.tolist()
        else:
            raise RuntimeError("Unable to extract indices {} from data shape {}".format(
                indices, data.shape))
        if len(data.shape) == 1:
            # https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/ml/array_feature_extractor.cc#L84
            # ONNX specifications does not say anything specific about it.
            new_shape = (1, len(index))
        else:
            new_shape = list(data.shape[:-1]) + [len(index)]
        res = data[..., index].reshape(new_shape)
        return (res, )
