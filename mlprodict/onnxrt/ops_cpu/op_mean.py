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

    def _run(self, *args, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and args[0].flags['WRITEABLE']:
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
