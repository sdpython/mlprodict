# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""

import numpy
from numpy.random import RandomState
from onnx.defs import onnx_opset_version
from ._op import OpRun


def _dropout(X, drop_probability=0.5, seed=0,
             training_mode=False, return_mask=False):
    if drop_probability == 0 or not training_mode:
        if return_mask:
            return X, numpy.ones(X.shape, dtype=bool)
        return (X, )

    rnd = RandomState(seed)
    mask = rnd.uniform(0, 1.0, X.shape) >= drop_probability
    scale = (1. / (1. - drop_probability))
    return (
        (mask * X * scale, mask.astype(bool))
        if return_mask else (mask * X * scale, ))


class DropoutBase(OpRun):

    def __init__(self, onnx_node, desc=None, expected_attributes=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=expected_attributes,
                       **options)
        self.nb_outputs = len(onnx_node.output)

    def _private_run(self, X, seed=None, ratio=0.5, training_mode=False):  # pylint: disable=W0221
        return _dropout(X, ratio, seed=seed, return_mask=self.nb_outputs == 2,
                        training_mode=training_mode)

    def _infer_shapes(self, *inputs):  # pylint: disable=W0221
        X = inputs[0]
        if self.nb_outputs == 1:
            return (X.copy(), )
        if self.nb_outputs == 2:
            return (X.copy(), X.copy())
        raise RuntimeError(  # pragma: no cover
            "Unexpected numbers of output {} > 2.".format(self.nb_outputs))


class Dropout_7(DropoutBase):

    atts = {'ratio': 0.5}

    def __init__(self, onnx_node, desc=None, **options):
        DropoutBase.__init__(self, onnx_node, desc=desc,
                             expected_attributes=Dropout_7.atts,
                             **options)

    def _run(self, X):  # pylint: disable=W0221
        return self._private_run(X, self.ratio)


class Dropout_12(DropoutBase):

    atts = {'seed': 0}

    def __init__(self, onnx_node, desc=None, **options):
        DropoutBase.__init__(self, onnx_node, desc=desc,
                             expected_attributes=Dropout_12.atts,
                             **options)

    def _run(self, *inputs):  # pylint: disable=W0221
        X = inputs[0]
        ratio = 0.5 if len(inputs) <= 1 else inputs[1]
        training_mode = False if len(inputs) <= 2 else inputs[2]
        return self._private_run(X, seed=self.seed, ratio=ratio,
                                 training_mode=training_mode)


if onnx_opset_version() >= 12:
    Dropout = Dropout_12
else:
    Dropout = Dropout_7  # pragma: no cover
