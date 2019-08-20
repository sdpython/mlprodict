# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunBinaryNumpy


class Max(OpRunBinaryNumpy):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNumpy.__init__(self, numpy.maximum, onnx_node,
                                  desc=desc, **options)
