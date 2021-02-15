"""
@file
@brief Wraps :epkg:`numpy` functions into :epkg:`onnx`.
"""
from .onnx_numpy_compiler import OnnxNumpyCompiler


def onnxnumpy(fct, op_version=None, runtime=None):
    """
    Wrappers to declare a function implements with
    :epkg:`ONNX` operators using :epkg:`numpy` operators.

    :param fct: function to convert
    :param op_version: :epkg:`ONNX` opset version
    :param runtime: see @see fct
    """
    compiled = OnnxNumpyCompiler(fct, op_version=op_version,
                                 runtime=runtime)

    def wrapper(*args, compiled=compiled):
        return compiled(*args)

    wrapper.__doc__ = fct.__doc__
    return wrapper
