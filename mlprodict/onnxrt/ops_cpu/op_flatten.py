# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnary


class Flatten(OpRunUnary):

    atts = {'axis': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=Flatten.atts,
                            **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        i = self.axis
        shape = x.shape
        new_shape = ((1, -1) if i == 0 else
                     (numpy.prod(shape[:i]).astype(int), -1))
        return (x.reshape(new_shape), )

    def to_python(self, inputs):
        lines = ['new_shape = ((1, -1) if axis == 0 else',
                 f'    (numpy.prod({inputs[0]}.shape[:axis]).astype(int), -1))',
                 f'return {inputs[0]}.reshape(new_shape)']
        return 'import numpy', '\n'.join(lines)
