# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun


class Scaler(OpRun):

    atts = {'offset': None, 'scale': None}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Scaler.atts,
                       **options)

    def _run(self, x):  # pylint: disable=W0221
        return ((x - self.offset) * self.scale, )
