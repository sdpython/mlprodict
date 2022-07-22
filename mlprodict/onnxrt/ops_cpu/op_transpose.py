# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Transpose(OpRunUnaryNum):

    atts = {'perm': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Transpose.atts,
                               **options)
        self.perm_ = None if len(self.perm) == 0 else self.perm

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.perm_ is None:
            return (numpy.transpose(data), )
        if len(self.perm_) != len(data.shape):
            raise RuntimeError(  # pragma: no cover
                f"Inconsistent permutation {self.perm_!r} with shape {data.shape!r}.")
        return (numpy.transpose(data, axes=self.perm_), )

    def to_python(self, inputs):
        """
        Returns a python code equivalent to this operator.

        @param      inputs      inputs name
        @return                 imports, python code, both as strings
        """
        lines = [
            "if perm is None:",
            f"    return numpy.transpose({inputs[0]})",
            f"return numpy.transpose({inputs[0]}, axes=perm)"
        ]
        return "import numpy", "\n".join(lines)
