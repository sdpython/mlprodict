# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun, DefaultNone


class Compress(OpRun):

    atts = {'axis': DefaultNone}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Compress.atts,
                       **options)

    def _run(self, x, condition, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and x.flags['WRITEABLE']:
            return (numpy.compress(condition, x, axis=self.axis, out=x), )
        return (numpy.compress(condition, x, axis=self.axis), )

    def to_python(self, inputs):
        if self.axis is None:
            return "import numpy\nreturn numpy.compress(%s, %s)" % tuple(inputs)
        return "import numpy\nreturn numpy.compress(%s, %s, axis=%d)" % (
            tuple(inputs) + (self.axis, ))
