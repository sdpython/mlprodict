# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.

.. versionadded:: 0.9
"""
from ._op import OpRun


class SequenceEmpty(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       atts=SequenceEmpty.atts, **options)

    def _run(self, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return ([], )
