# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Clip(OpRun):

    atts = {'min': -3.4028234663852886e+38,
            'max': 3.4028234663852886e+38}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Clip.atts,
                       **options)

    def _run(self, data):  # pylint: disable=W0221
        return (numpy.clip(data, self.min, self.max), )
