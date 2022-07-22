# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunBinaryNumpy


class BitShift(OpRunBinaryNumpy):

    atts = {'direction': b''}

    def __init__(self, onnx_node, desc=None, **options):
        "constructor"
        OpRunBinaryNumpy.__init__(self, numpy.add, onnx_node,
                                  expected_attributes=BitShift.atts,
                                  desc=desc, **options)
        if self.direction not in (b'LEFT', b'RIGHT'):
            raise ValueError(  # pragma: no cover
                f"Unexpected value for direction ({self.direction!r}).")
        if self.direction == b'LEFT':
            self.numpy_fct = numpy.left_shift
        else:
            self.numpy_fct = numpy.right_shift
