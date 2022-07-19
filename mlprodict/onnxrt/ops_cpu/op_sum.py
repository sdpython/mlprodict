# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun


class Sum(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, *args, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (sum(args), )

    def to_python(self, inputs):
        return None, f"return sum([{', '.join(inputs)}])"
