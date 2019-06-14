# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
import numpy
from ._op import OpRun


class Normalizer(OpRun):
    """
    Implements a normalization.
    """
    atts = {'norm': 'MAX'}

    def __init__(self, onnx_node, desc=None, **options):
        if desc is None:
            raise ValueError("desc should not be None.")
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Normalizer.atts,
                       **options)
        if self.norm == b'MAX':  # pylint: disable=E1101
            self._norm = Normalizer.norm_max
        elif self.norm == b'L1':  # pylint: disable=E1101
            self._norm = Normalizer.norm_l1
        elif self.norm == b'L2':  # pylint: disable=E1101
            self._norm = Normalizer.norm_l2
        else:
            raise ValueError(
                "Unexpected value for norm='{}'.".format(self.norm))  # pylint: disable=E1101

    @staticmethod
    def norm_max(x):
        "max normalization"
        mx = numpy.max(x)
        return x / mx

    @staticmethod
    def norm_l1(x):
        "L1 normalization"
        su = numpy.sum(numpy.abs(x))
        return x / su

    @staticmethod
    def norm_l2(x):
        "L2 normalization"
        su = numpy.sum(numpy.square(x)) ** 0.5
        return x / su

    def _run(self, x):  # pylint: disable=W0221
        return (self._norm(x), )
