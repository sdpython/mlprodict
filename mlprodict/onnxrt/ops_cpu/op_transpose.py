# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Transpose(OpRun):

    atts = {'perm': []}

    def __init__(self, onnx_node, desc=None, **options):
        if desc is None:
            raise ValueError("desc should not be None.")
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Transpose.atts,
                       **options)

    def _run(self, data):  # pylint: disable=W0221
        return (numpy.transpose(data, axes=self.perm), )
