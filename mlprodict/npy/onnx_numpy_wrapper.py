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


def onnxnumpy(op_version=None, runtime=None):
    """
    Decorator to declare a function implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: see @see fct

    Equivalent to `onnxnumpy(arg)(foo)`.
    The decorator must be called with `onnxnumpy()`.
    """
    def decorator_fct(fct):
        compiled = OnnxNumpyCompiler(fct, op_version=op_version,
                                     runtime=runtime)
        newclass = type(
            "onnxnumpy_%s" % fct.__name__,
            (wrapper_onnxnumpy,), {'__doc__': fct.__doc__})

        return newclass(compiled)
    return decorator_fct


def onnxnumpy_default(fct):
    """
    Decorator with options to declare a function implemented
    using :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param fct: function to wrap
    :param runtime: see @see fct
    """
    return onnxnumpy()(fct)
