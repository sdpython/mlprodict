# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class ConcatFromSequence(OpRun):

    atts = {'axis': 0, 'new_axis': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ConcatFromSequence.atts,
                       **options)

    def _run(self, seq):  # pylint: disable=W0221
        if seq is None:
            raise RuntimeError(  # pragma: no cover
                "A sequence cannot be null.")
        if self.new_axis == 1:
            seq2 = [s[..., numpy.newaxis] for s in seq]
            res = numpy.concatenate(seq2, axis=-1)
        else:
            res = numpy.concatenate(seq, axis=self.axis)
        return (res, )

    def _infer_shapes(self, seq):  # pylint: disable=W0221
        return (ShapeObject(None, seq.dtype), )

    def _infer_types(self, seq):  # pylint: disable=W0221
        return (seq, )

    def _infer_sizes(self, seq):  # pylint: disable=W0221
        res = self.run(seq)
        if self.new_axis == 1:
            return (dict(temp=sum(o.size for o in seq)), ) + res
        return (dict(temp=0), ) + res
