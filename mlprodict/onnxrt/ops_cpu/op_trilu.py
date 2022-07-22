# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Trilu(OpRun):

    atts = {'upper': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Trilu.atts,
                       **options)
        if self.upper not in (0, 1):
            raise ValueError(f"upper must be 0 or 1 not {self.upper!r}.")

    def _run(self, *inputs, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        x = inputs[0]
        k = 0 if len(inputs) == 1 else int(inputs[1])
        if self.upper:
            return (numpy.triu(x, k), )
        return (numpy.tril(x, k), )

    def to_python(self, inputs):
        name = "triu" if self.upper else "tril"
        return (
            "import numpy",
            "return numpy.%s(%s, int(%s))" % (
                name, inputs[0], 0 if len(inputs) == 1 else inputs[1]))
