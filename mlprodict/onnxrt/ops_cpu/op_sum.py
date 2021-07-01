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

    def _run(self, *args):  # pylint: disable=W0221
        return (sum(args), )

    def _infer_shapes(self, *args):  # pylint: disable=W0221
        return (args[0], )

    def _infer_types(self, *args):  # pylint: disable=W0221
        return (args[0], )

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res

    def to_python(self, inputs):
        return None, "return sum([%s])" % ", ".join(inputs)
