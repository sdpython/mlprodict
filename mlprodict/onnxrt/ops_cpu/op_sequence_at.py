# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.

.. versionadded:: 0.8
"""
from ._op import OpRun


class SequenceAt(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       atts=SequenceAt.atts, **options)

    def _run(self, seq, index, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (seq[index], )
