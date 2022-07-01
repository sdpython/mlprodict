# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun


def reshape_reference_implementation(data, shape):
    new_shape = numpy.copy(shape)
    zeros_index = numpy.where(shape == 0)
    if len(data.shape) == 1 and data.shape[0] == 0:
        reshaped = numpy.reshape(data, shape)
    else:
        try:
            new_shape[zeros_index] = numpy.array(data.shape)[zeros_index]
        except IndexError as e:  # pragma: no cover
            raise RuntimeError(
                "Unable to reshape from shape %r to shape %r (or %r)."
                "" % (data.shape, shape, new_shape)) from e
        reshaped = numpy.reshape(data, new_shape)
    return reshaped


class CommonReshape(OpRun):

    def __init__(self, onnx_node, desc=None, expected_attributes=None, **options):
        OpRun.__init__(
            self, onnx_node, desc=desc,
            expected_attributes=expected_attributes, **options)

    def _run(self, data, shape, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (reshape_reference_implementation(data, shape), )


class Reshape_5(CommonReshape):

    def __init__(self, onnx_node, desc=None, expected_attributes=None, **options):
        CommonReshape.__init__(self, onnx_node, desc=desc, **options)


class Reshape_13(Reshape_5):
    pass


class Reshape_14(CommonReshape):

    atts = {'allowzero': 0}

    def __init__(self, onnx_node, desc=None, **options):
        CommonReshape.__init__(
            self, onnx_node, desc=desc,
            expected_attributes=Reshape_14.atts, **options)


if onnx_opset_version() >= 14:
    Reshape = Reshape_14
else:
    Reshape = Reshape_5  # pragma: no cover
