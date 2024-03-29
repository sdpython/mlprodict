# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def _concat_from_sequence(seq, axis, new_axis=0):
    if new_axis == 1:
        seq2 = [s[..., numpy.newaxis] for s in seq]
        res = numpy.concatenate(seq2, axis=-1)
    else:
        res = numpy.concatenate(seq, axis=axis)
    return res


class ConcatFromSequence(OpRun):

    atts = {'axis': 0, 'new_axis': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ConcatFromSequence.atts,
                       **options)

    def _run(self, seq, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if seq is None:
            raise RuntimeError(  # pragma: no cover
                "A sequence cannot be null.")
        res = _concat_from_sequence(seq, self.axis, new_axis=self.new_axis)
        return (res, )
