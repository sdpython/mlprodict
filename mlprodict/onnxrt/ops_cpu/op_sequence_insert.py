# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.

.. versionadded:: 0.7
"""
from ._op import OpRun


class SequenceInsert(OpRun):

    atts = {'axis': 0, 'new_axis': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       atts=SequenceInsert.atts, **options)

    def _run(self, S, T, ind=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        S = S.copy()
        if ind is not None:
            S.insert(ind[0], T)
        else:
            S.append(T)
        return (S, )
