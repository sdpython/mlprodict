# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Concat(OpRun):

    atts = {'axis': 0}
    python_inputs = ['*inputs']

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Concat.atts,
                       **options)

    def _preprocess(self, a):
        if len(a.shape) == 0:
            raise RuntimeError(  # pragma: no cover
                f"Concat: one input has an empty shape: {a!r}.")
        if self.axis >= len(a.shape):
            new_shape = a.shape + (1, ) * (self.axis + 1 - len(a.shape))
            return a.reshape(new_shape)
        return a

    def _run(self, *args, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        targs = tuple(self._preprocess(a) for a in args)
        return (numpy.concatenate(targs, self.axis), )

    def to_python(self, inputs):
        return "import numpy", "return numpy.concatenate(inputs, axis=axis)"
