# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Einsum(OpRun):

    atts = {'equation': ''}
    python_inputs = ['*inputs']

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Einsum.atts,
                       **options)
        if not isinstance(self.equation, (str, bytes)):
            raise TypeError(  # pragma: no cover
                f"equation must be string but is {type(self.equation)!r}.")
        self.equation = self.equation.strip()
        if len(self.equation) == 0:
            raise TypeError("equation is empty.")  # pragma: no cover

    def _run(self, *args, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        try:
            return (numpy.einsum(self.equation, *args, optimize=True), )
        except TypeError:
            return (numpy.einsum(self.equation, *args), )

    def to_python(self, inputs):
        return ("import numpy",
                "return numpy.einsum(equation, *inputs)")
