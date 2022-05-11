# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.

.. versionadded:: 0.9
"""
from ._op import OpRun
from ..shape_object import ShapeObject


class SequenceEmpty(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       atts=SequenceEmpty.atts, **options)

    def _run(self, verbose=0, fLOG=None):  # pylint: disable=W0221
        return ([], )

    def _infer_shapes(self):  # pylint: disable=W0221
        return (ShapeObject(None, dtype="sequence", subtype=None), )

    def _infer_types(self):  # pylint: disable=W0221
        return ([], )

    def _infer_sizes(self, *args):  # pylint: disable=W0221
        res = self.run(*args)
        return (dict(temp=0), ) + res
