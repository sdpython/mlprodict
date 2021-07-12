# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun


class Mean(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, *args):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return self._run_inplace(*args)
        res = args[0].copy()
        for m in args[1:]:
            res += m
        return (res / len(args), )

    def _run_inplace(self, *args):
        res = args[0]
        for m in args[1:]:
            res += m
        return (res / len(args), )

    def _infer_shapes(self, *args):  # pylint: disable=W0221
        return (args[0], )

    def _infer_types(self, *args):  # pylint: disable=W0221
        return (args[0], )

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res
