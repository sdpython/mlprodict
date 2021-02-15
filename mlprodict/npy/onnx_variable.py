"""
@file
@brief Intermediate class between :epkg:`numpy` and :epkg:`onnx`.
"""


class OnnxVar:
    """
    Variables used into :epkg:`onnx` computation.

    :param inputs: variable name or object
    :param onnx_op: :epkg:`ONNX` operator
    """

    def __init__(self, *inputs, op=None):
        self.inputs = inputs
        self.onnx_op = op

    def to_algebra(self, op_version=None):
        """
        Converter the variable into an operator.
        """
        if self.onnx_op is None:
            if len(self.inputs) != 1:
                print(self.inputs)
                raise RuntimeError("Unexpected numer of inputs, 1 expected, "
                                   "got {} instead.".format(self.inputs))
            return self.inputs[0]
        new_inputs = []
        for inp in self.inputs:
            new_inputs.append(inp.to_algebra(op_version=op_version))
        return self.onnx_op(*new_inputs, op_version=op_version)
