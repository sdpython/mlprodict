# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Range(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Range.atts,
                       **options)

    def _run(self, starts, ends, steps, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (numpy.arange(starts, ends, steps).astype(starts.dtype), )
