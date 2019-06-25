# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Cast(OpRun):

    atts = {'to': None}

    def __init__(self, onnx_node, desc=None, **options):
        if desc is None:
            raise ValueError("desc should not be None.")
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Cast.atts,
                       **options)
        if self.to == 1:  # pylint: disable=E1101
            self._cast = lambda x: x.astype(numpy.float32)
        elif self.to == 6:  # pylint: disable=E1101
            self._cast = lambda x: x.astype(numpy.int32)
        elif self.to == 7:  # pylint: disable=E1101
            self._cast = lambda x: x.astype(numpy.int64)
        else:
            raise ValueError("Unexpected value for to='{}'.".format(
                self.to))  # pylint: disable=E1101

    def _run(self, x):  # pylint: disable=W0221
        return (self._cast(x), )
