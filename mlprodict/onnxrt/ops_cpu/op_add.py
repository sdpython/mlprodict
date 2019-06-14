# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
from ._op import OpRun


class Add(OpRun):
    """
    Implements an addition.
    """

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=None, **options)

    def _run(self, a, b):  # pylint: disable=W0221
        return (a + b, )
