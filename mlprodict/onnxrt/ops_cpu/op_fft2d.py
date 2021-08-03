# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from numpy.fft import fft2
from ..shape_object import ShapeObject
from ._op import OpRun
from ._new_ops import OperatorSchema


class FFT2D(OpRun):

    atts = {'axes': [-2, -1]}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=FFT2D.atts,
                       **options)
        if self.axes is not None:
            self.axes = tuple(self.axes)
            if len(self.axes) != 2:
                raise ValueError(  # pragma: no cover
                    "axes must a set of 1 integers not %r." % self.axes)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "FFT2D":
            return FFT2DSchema()
        raise RuntimeError(  # pragma: no cover
            "Unable to find a schema for operator '{}'.".format(op_name))

    def _run(self, a, fft_length=None):  # pylint: disable=W0221
        if fft_length is None:
            y = fft2(a, axes=self.axes)
        else:
            y = fft2(a, tuple(fft_length), axes=self.axes)
        if a.dtype in (numpy.float32, numpy.complex64):
            return (y.astype(numpy.complex64), )
        if a.dtype in (numpy.float64, numpy.complex128):
            return (y.astype(numpy.complex128), )
        raise TypeError(  # pragma: no cover
            "Unexpected input type: %r." % a.dtype)

    def _infer_shapes(self, a, b=None):  # pylint: disable=W0221,W0237
        if a.dtype in (numpy.float32, numpy.complex64):
            return (ShapeObject(a.shape, dtype=numpy.complex64), )
        if a.dtype in (numpy.float64, numpy.complex128):
            return (ShapeObject(a.shape, dtype=numpy.complex128), )
        raise TypeError(  # pragma: no cover
            "Unexpected input type: %r." % a.dtype)

    def _infer_types(self, a, b=None):  # pylint: disable=W0221,W0237
        if a.dtype in (numpy.float32, numpy.complex64):
            return (numpy.complex64, )
        if a.dtype in (numpy.float64, numpy.complex128):
            return (numpy.complex128, )
        raise TypeError(  # pragma: no cover
            "Unexpected input type: %r." % a.dtype)

    def to_python(self, inputs):
        if self.axes is not None:
            axes = tuple(self.axes)
        else:
            axes = None
        if len(inputs) == 1:
            return ('from numpy.fft import fft2',
                    "return fft2({}, axes={})".format(
                        inputs[0], axes))
        return ('from numpy.fft import fft2',
                "return fft2({}, tuple({}), axes={})".format(
                    inputs[0], inputs[1], axes))


class FFT2DSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl FFT.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'FFT2D')
        self.attributes = FFT2D.atts
