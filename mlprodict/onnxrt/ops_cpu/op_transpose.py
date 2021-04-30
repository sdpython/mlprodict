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

    def _run(self, data):  # pylint: disable=W0221
        if self.perm_ is None:
            return (numpy.transpose(data), )
        if len(self.perm_) != len(data.shape):
            raise RuntimeError(
                "Inconsistent permutation %r with shape %r." % (
                    self.perm_, data.shape))
        return (numpy.transpose(data, axes=self.perm_), )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        return (x.transpose(perm=self.perm), )

    def to_python(self, inputs):
        """
        Returns a python code equivalent to this operator.

        @param      inputs      inputs name
        @return                 imports, python code, both as strings
        """
        lines = [
            "if perm is None:",
            "    return numpy.transpose(%s)" % inputs[0],
            "return numpy.transpose(%s, axes=perm)" % inputs[0]
        ]
        return "import numpy", "\n".join(lines)
