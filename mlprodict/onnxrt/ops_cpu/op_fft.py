# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from numpy.fft import fft
from ._op import OpRun
from ._new_ops import OperatorSchema


class FFT(OpRun):

    atts = {'axis': -1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=FFT.atts,
                       **options)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "FFT":
            return FFTSchema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")

    def _run(self, a, fft_length=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if fft_length is not None:
            fft_length = fft_length[0]
            y = fft(a, fft_length, axis=self.axis)
        else:
            y = fft(a, axis=self.axis)
        if a.dtype in (numpy.float32, numpy.complex64):
            return (y.astype(numpy.complex64), )
        if a.dtype in (numpy.float64, numpy.complex128):
            return (y.astype(numpy.complex128), )
        raise TypeError(  # pragma: no cover
            f"Unexpected input type: {a.dtype!r}.")

    def to_python(self, inputs):
        if len(inputs) == 1:
            return ('from numpy.fft import fft',
                    f"return fft({inputs[0]}, axis={self.axis})")
        return ('from numpy.fft import fft',
                f"return fft({inputs[0]}, {inputs[1]}[0], axis={self.axis})")


class FFTSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl FFT.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'FFT')
        self.attributes = FFT.atts
