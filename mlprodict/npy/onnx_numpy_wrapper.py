"""
@file
@brief Wraps :epkg:`numpy` functions into :epkg:`onnx`.
"""
from .onnx_numpy_compiler import OnnxNumpyCompiler


class wrapper_onnxnumpy:
    """
    Intermediate wrapper to store a pointer
    on the compiler (type: @see cl OnnxNumpyCompiler).

    :param compiled: instance of @see cl OnnxNumpyCompiler
    """

    def __init__(self, compiled):
        self.compiled = compiled

    def __call__(self, *args):
        """
        Calls the compiled function with arguments `args`.
        """
        return self.compiled(*args)


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

    newclass = type(
        "onnxnumpy_%s" % fct.__name__,
        (wrapper_onnxnumpy,), {'__doc__': fct.__doc__})

    return newclass(compiled)
