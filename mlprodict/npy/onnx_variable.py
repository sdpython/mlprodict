"""
@file
@brief Intermediate class between :epkg:`numpy` and :epkg:`onnx`.
"""
import numpy
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAdd, OnnxAnd,
    OnnxCast,
    OnnxDiv,
    OnnxEqual,
    OnnxGreater,
    OnnxLess,
    OnnxMatMul, OnnxMul,
    OnnxOr,
    OnnxReshape,
    OnnxSlice, OnnxSub,
    OnnxTranspose
)


class OnnxVar:
    """
    Variables used into :epkg:`onnx` computation.

    :param inputs: variable name or object
    :param onnx_op: :epkg:`ONNX` operator
    """

    def __init__(self, *inputs, op=None, **kwargs):
        self.inputs = inputs
        self.onnx_op = op
        self.alg_ = None
        self.onnx_op_kwargs = kwargs

    def to_algebra(self, op_version=None):
        """
        Converts the variable into an operator.
        """
        if self.alg_ is None:
            if self.onnx_op is None:
                if len(self.inputs) != 1:
                    raise RuntimeError("Unexpected numer of inputs, 1 expected, "
                                       "got {} instead.".format(self.inputs))
                self.alg_ = self.inputs[0]
            else:
                new_inputs = []
                for inp in self.inputs:
                    if isinstance(inp, (
                            int, float, str, numpy.ndarray, numpy.int32,
                            numpy.int64, numpy.float32, numpy.float64,
                            numpy.bool_, numpy.str, numpy.int8, numpy.uint8,
                            numpy.int16, numpy.uint16, numpy.uint32, numpy.uint64)):
                        new_inputs.append(inp)
                    else:
                        new_inputs.append(
                            inp.to_algebra(op_version=op_version))
                self.alg_ = self.onnx_op(*new_inputs, op_version=op_version,
                                         **self.onnx_op_kwargs)
        return self.alg_

    @property
    def T(self):
        "Transpose."
        return OnnxVar(self, op=OnnxTranspose)

    def astype(self, dtype):
        "Cast"
        from ..onnxrt.onnx2py_helper import guess_proto_dtype
        return OnnxVar(self, op=OnnxCast, to=guess_proto_dtype(dtype))

    def reshape(self, shape):
        "Reshape"
        if isinstance(shape, (tuple, list)):
            shape = numpy.array(shape, dtype=numpy.int64)
        return OnnxVar(self, shape, op=OnnxReshape)

    def __add__(self, y):
        "Addition."
        return OnnxVar(self, y, op=OnnxAdd)

    def __sub__(self, y):
        "Subtraction."
        return OnnxVar(self, y, op=OnnxSub)

    def __mul__(self, y):
        "Multiplication."
        return OnnxVar(self, y, op=OnnxMul)

    def __matmul__(self, y):
        "Matrix multiplication."
        return OnnxVar(self, y, op=OnnxMatMul)

    def __truediv__(self, y):
        "Division, no difference between `/` and `//`."
        return OnnxVar(self, y, op=OnnxDiv)

    def __floordiv__(self, y):
        "Division, no difference between `/` and `//`."
        return OnnxVar(self, y, op=OnnxDiv)

    def __eq__(self, y):
        "Equality."
        return OnnxVar(self, y, op=OnnxEqual)

    def __gt__(self, y):
        "Greater."
        return OnnxVar(self, y, op=OnnxGreater)

    def __lt__(self, y):
        "Less."
        return OnnxVar(self, y, op=OnnxLess)

    def __and__(self, y):
        "And."
        return OnnxVar(self, y, op=OnnxAnd)

    def __or__(self, y):
        "And."
        return OnnxVar(self, y, op=OnnxOr)

    def __getitem__(self, index):
        """
        Deals with multiple scenarios.
        """
        if not isinstance(index, tuple):
            index = (index, )
        starts = []
        ends = []
        axes = []
        steps = []
        for i, ind in enumerate(index):
            if isinstance(ind, int):
                starts.append(ind)
                ends.append(ind + 1)
                axes.append(i)
                steps.append(1)
                continue
            if isinstance(ind, slice):
                if ind.start is None and ind.stop is None and ind.step is None:
                    continue
                start = 0 if ind.start is None else ind.start
                end = -1 if ind.stop is None else ind.stop
                step = 1 if ind.step is None else ind.step
                starts.append(start)
                ends.append(end)
                axes.append(i)
                steps.append(step)
                continue
            raise NotImplementedError(  # pragma: no cover
                "Not implemented for type %r." % type(ind))
        if max(steps) == min(steps) == 1:
            steps = None
        else:
            steps = numpy.array(steps, dtype=numpy.int64)
        starts = numpy.array(starts, dtype=numpy.int64)
        ends = numpy.array(ends, dtype=numpy.int64)
        axes = numpy.array(axes, dtype=numpy.int64)
        if steps is None:
            return OnnxVar(self, starts, ends, axes, op=OnnxSlice)
        return OnnxVar(self, starts, ends, axes, steps, op=OnnxSlice)
