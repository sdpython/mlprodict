# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from collections import OrderedDict
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunUnaryNum


class Clip_6(OpRunUnaryNum):

    atts = {'min': -3.4028234663852886e+38,
            'max': 3.4028234663852886e+38}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Clip_6.atts,
                               **options)

    def _run(self, data):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return self._run_inplace(data)
        res = numpy.clip(data, self.min, self.max)
        return (res, ) if res.dtype == data.dtype else (res.astype(data.dtype), )

    def _run_inplace(self, data):
        return (numpy.clip(data, self.min, self.max, out=data), )

    def to_python(self, inputs):
        return ("import numpy",
                "return numpy.clip(%s, min_, max_)" % inputs[0])


class Clip_11(OpRunUnaryNum):

    version_higher_than = 11
    mandatory_inputs = ['X']
    optional_inputs = OrderedDict([
        ('min', -3.4028234663852886e+38),
        ('max', 3.4028234663852886e+38)
    ])

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               **options)

    def run(self, x, *minmax):  # pylint: disable=E0202,W0221
        """
        Calls method ``_run``.
        """
        try:
            res = self._run(x, *minmax)
        except TypeError as e:
            raise TypeError("Issues with types {} (binary operator {}).".format(
                ", ".join(str(type(_)) for _ in [x]),
                self.__class__.__name__)) from e
        return res

    def _run(self, data, *minmax):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return self._run_inplace(data, *minmax)
        le = len(minmax)
        amin = minmax[0] if le > 0 else None  # -3.4028234663852886e+38
        amax = minmax[1] if le > 1 else None  # 3.4028234663852886e+38
        res = numpy.clip(data, amin, amax)
        return (res, ) if res.dtype == data.dtype else (res.astype(data.dtype), )

    def _run_inplace(self, data, *minmax):  # pylint: disable=W0221
        le = len(minmax)
        amin = minmax[0] if le > 0 else None  # -3.4028234663852886e+38
        amax = minmax[1] if le > 1 else None  # 3.4028234663852886e+38
        res = numpy.clip(data, amin, amax, out=data)
        return (res, )

    def infer_shapes(self, x, *minmax):  # pylint: disable=E0202,W0221
        try:
            return self._infer_shapes(x)
        except TypeError as e:
            raise TypeError("Issues with types {} (operator {}).".format(
                x.dtype, self.__class__.__name__)) from e

    def infer_types(self, x, *minmax):  # pylint: disable=E0202,W0221
        try:
            return self._infer_types(x)
        except TypeError as e:
            raise TypeError("Issues with types {} (operator {}).".format(
                x.dtype, self.__class__.__name__)) from e

    def to_python(self, inputs):
        return ("import numpy",
                "return numpy.clip(%s, min_, max_)" % inputs[0])


if onnx_opset_version() >= 11:
    Clip = Clip_11
else:
    Clip = Clip_6
