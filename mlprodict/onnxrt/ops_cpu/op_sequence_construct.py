# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.

.. versionadded:: 0.7
"""
from ._op import OpRun


class SequenceConstruct(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       atts=SequenceConstruct.atts, **options)

    def _run(self, *data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (data, )
