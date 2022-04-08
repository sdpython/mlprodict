# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun


class DepthToSpace(OpRun):

    atts = {'blocksize': 0, 'mode': 'DCR'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=DepthToSpace.atts,
                       **options)

    def _run(self, data):  # pylint: disable=W0221
        raise NotImplementedError()


class SpaceToDepth(OpRun):

    atts = {'blocksize': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=SpaceToDepth.atts,
                       **options)

    def _run(self, data):  # pylint: disable=W0221
        raise NotImplementedError()
