# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ._op_helper import proto2dtype, dtype_name


class EyeLike(OpRun):

    atts = {'k': 0, 'dtype': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=EyeLike.atts,
                       **options)
        self.dtype_ = proto2dtype(self.dtype)

    def _run(self, data, *args, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        shape = data.shape
        if len(shape) == 1:
            sh = (shape[0], shape[0])
        elif len(shape) == 2:
            sh = shape
        else:
            raise RuntimeError(  # pragma: no cover
                f"EyeLike only accept 1D or 2D tensors not {shape!r}.")
        return (numpy.eye(*sh, k=self.k, dtype=self.dtype_), )

    def to_python(self, inputs):
        return (
            "import numpy",
            "return numpy.eye(*(%s.shape), k=%d, dtype=numpy.%s)" % (
                inputs[0], self.k, dtype_name(self.dtype_)))
