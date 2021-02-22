"""
@file
@brief Wraps :epkg:`numpy` functions into :epkg:`onnx`.

.. versionadded:: 0.6
"""
from .onnx_numpy_compiler import OnnxNumpyCompiler


class wrapper_onnxnumpy:
    """
    Intermediate wrapper to store a pointer
    on the compiler (type: @see cl OnnxNumpyCompiler).

    :param compiled: instance of @see cl OnnxNumpyCompiler

    .. versionadded:: 0.6
    """

    def __init__(self, compiled):
        self.compiled = compiled

    def __call__(self, *args):
        """
        Calls the compiled function with arguments `args`.
        """
        return self.compiled(*args)


def onnxnumpy(op_version=None, runtime=None, signature=None):
    """
    Decorator to declare a function implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: see @see fct
    :param signature: it should be used when the function
        is not annoatated.

    Equivalent to `onnxnumpy(arg)(foo)`.

    .. versionadded:: 0.6
    """
    def decorator_fct(fct):
        compiled = OnnxNumpyCompiler(
            fct, op_version=op_version, runtime=runtime,
            signature=signature)
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

    .. versionadded:: 0.6
    """
    return onnxnumpy()(fct)
