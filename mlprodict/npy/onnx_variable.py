"""
@file
@brief Intermediate class between :epkg:`numpy` and :epkg:`onnx`.
"""
import numpy


class OnnxVar:
    """
    Variables used into :epkg:`onnx` computation.

    :param inputs: variable name or object
    :param onnx_op: :epkg:`ONNX` operator
    """

    def __init__(self, *inputs, op=None):
        self.inputs = inputs
        self.onnx_op = op
        self.alg_ = None

    def to_algebra(self, op_version=None):
        """
        Converts the variable into an operator.
        """
        if self.alg_ is None:
            if self.onnx_op is None:
                if len(self.inputs) != 1:
                    print(self.inputs)
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
                self.alg_ = self.onnx_op(*new_inputs, op_version=op_version)
        return self.alg_

    def __add__(self, y):
        "Addition."
        from skl2onnx.algebra.onnx_ops import OnnxAdd  # pylint: disable=E0611
        return OnnxVar(self, y, op=OnnxAdd)

    def __sub__(self, y):
        "Subtraction."
        from skl2onnx.algebra.onnx_ops import OnnxSub  # pylint: disable=E0611
        return OnnxVar(self, y, op=OnnxSub)

    def __mul__(self, y):
        "Multiplication."
        from skl2onnx.algebra.onnx_ops import OnnxMul  # pylint: disable=E0611
        return OnnxVar(self, y, op=OnnxMul)

    def __matmul__(self, y):
        "Matrix multiplication."
        from skl2onnx.algebra.onnx_ops import OnnxMatMul  # pylint: disable=E0611
        return OnnxVar(self, y, op=OnnxMatMul)

    def __truediv__(self, y):
        "Division, no difference between `/` and `//`."
        from skl2onnx.algebra.onnx_ops import OnnxDiv  # pylint: disable=E0611
        return OnnxVar(self, y, op=OnnxDiv)

    def __floordiv__(self, y):
        "Division, no difference between `/` and `//`."
        from skl2onnx.algebra.onnx_ops import OnnxDiv  # pylint: disable=E0611
        return OnnxVar(self, y, op=OnnxDiv)
