# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRunUnary


class Scaler(OpRunUnary):

    atts = {'offset': None, 'scale': None}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=Scaler.atts,
                            **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return self._run_no_checks_(x, verbose=verbose, fLOG=fLOG)

    def _run_no_checks_(self, x, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and x.flags['WRITEABLE']:
            return self._run_inplace(x)
        dx = x - self.offset
        return (dx * self.scale, )

    def _run_inplace(self, x):
        x -= self.offset
        x *= self.scale
        return (x, )
