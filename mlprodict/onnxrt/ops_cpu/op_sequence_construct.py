# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.

.. versionadded:: 0.7
"""
from ._op import OpRun
from ..shape_object import ShapeObject


class SequenceConstruct(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       atts=SequenceConstruct.atts, **options)

    def _run(self, *data):  # pylint: disable=W0221
        return (data, )

    def _infer_shapes(self, *data):  # pylint: disable=W0221
        return (ShapeObject(None, dtype="sequence"), )

    def _infer_types(self, *data):  # pylint: disable=W0221
        return (list, )

    def _infer_sizes(self, *args):  # pylint: disable=W0221
        res = self.run(*args)
        return (dict(temp=0), ) + res
