# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Normalizer(OpRun):

    atts = {'norm': 'MAX'}

    def __init__(self, onnx_node, desc=None, **options):
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
        return x / numpy.abs(x).max(axis=1).reshape((x.shape[0], -1))

    @staticmethod
    def norm_l1(x):
        "L1 normalization"
        return x / numpy.abs(x).sum(axis=1).reshape((x.shape[0], -1))

    @staticmethod
    def norm_l2(x):
        "L2 normalization"
        return x / numpy.square(x).sum(axis=1).reshape((x.shape[0], -1))

    def _run(self, x):  # pylint: disable=W0221
        return (self._norm(x), )
