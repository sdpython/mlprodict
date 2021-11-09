# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.

.. versionadded:: 0.8
"""
from ._op import OpRun
from ..shape_object import ShapeObject


class SequenceAt(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       atts=SequenceAt.atts, **options)

    def _run(self, seq, index):  # pylint: disable=W0221
        return (seq[index], )

    def _infer_shapes(self, seq, index):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=seq.subtype.dtype), )

    def _infer_types(self, *data):  # pylint: disable=W0221
        return (None, )

    def _infer_sizes(self, *args):  # pylint: disable=W0221
        res = self.run(*args)
        return (dict(temp=0), ) + res
