# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Normalizer(OpRunUnaryNum):

    atts = {'norm': 'MAX'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Normalizer.atts,
                               **options)
        if self.norm == b'MAX':  # pylint: disable=E1101
            self._norm = Normalizer.norm_max
        elif self.norm == b'L1':  # pylint: disable=E1101
            self._norm = Normalizer.norm_l1
        elif self.norm == b'L2':  # pylint: disable=E1101
            self._norm = Normalizer.norm_l2
        else:
            raise ValueError(  # pragma: no cover
                "Unexpected value for norm='{}'.".format(self.norm))  # pylint: disable=E1101

    @staticmethod
    def norm_max(x, inplace):
        "max normalization"
        if inplace:
            return Normalizer._norm_max_inplace(x)
        return x / numpy.abs(x).max(axis=1).reshape((x.shape[0], -1))

    @staticmethod
    def _norm_max_inplace(x):
        numpy.divide(x, numpy.abs(x).max(axis=1).reshape((x.shape[0], -1)),
                     out=x)
        return x

    @staticmethod
    def norm_l1(x, inplace):
        "L1 normalization"
        if inplace:
            return Normalizer._norm_L1_inplace(x)
        return x / numpy.abs(x).sum(axis=1).reshape((x.shape[0], -1))

    @staticmethod
    def _norm_L1_inplace(x):
        numpy.divide(x, numpy.abs(x).sum(axis=1).reshape((x.shape[0], -1)),
                     out=x)
        return x

    @staticmethod
    def norm_l2(x, inplace):
        "L2 normalization"
        xn = numpy.square(x).sum(axis=1)
        numpy.sqrt(xn, out=xn)
        norm = xn.reshape((x.shape[0], -1))
        if inplace:
            numpy.divide(x, norm, out=x)
            return x
        return x / norm

    def _run(self, x):  # pylint: disable=W0221
        return (self._norm(x, inplace=self.inplaces.get(0, False)), )
