"""
@file
@brief Wraps :epkg:`numpy` functions into :epkg:`onnx`.

.. versionadded:: 0.6
"""
from .onnx_numpy_annotation import get_args_kwargs
from .onnx_numpy_compiler import OnnxNumpyCompiler


class _created_classes:
    """
    Class to store all dynamic classes created by wrappers.
    """

    def __init__(self):
        self.stored = {}

    def append(self, name, cl):
        """
        Adds a class into `globals()` to enable pickling on dynamic
        classes.
        """
        if name in self.stored:
            raise RuntimeError(
                "Class %r already exists in\n%r\n---\n%r" % (
                    name, "\n".join(sorted(self.stored)), cl))
        self.stored[name] = cl
        globals()[name] = cl


_created_classes_inst = _created_classes()


class wrapper_onnxnumpy:
    """
    Intermediate wrapper to store a pointer
    on the compiler (type: @see cl OnnxNumpyCompiler).

    :param compiled: instance of @see cl OnnxNumpyCompiler

    .. versionadded:: 0.6
    """

    def __init__(self, compiled):
        self.compiled = compiled

    def __call__(self, *args, **kwargs):
        """
        Calls the compiled function with arguments `args`.
        """
        return self.compiled(*args, **kwargs)

    def __getstate__(self):
        """
        Serializes everything but the function which generates
        the ONNX graph, not needed anymore.
        """
        return dict(compiled=self.compiled)

    def __setstate__(self, state):
        """
        Serializes everything but the function which generates
        the ONNX graph, not needed anymore.
        """
        self.compiled = state['compiled']


def onnxnumpy(op_version=None, runtime=None, signature=None):
    """
    Decorator to declare a function implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: see @see cl OnnxInference
    :param signature: it should be used when the function
        is not annoatated.

    Equivalent to `onnxnumpy(arg)(foo)`.

    .. versionadded:: 0.6
    """
    def decorator_fct(fct):
        compiled = OnnxNumpyCompiler(
            fct, op_version=op_version, runtime=runtime,
            signature=signature)
        name = "onnxnumpy_%s_%s_%s" % (fct.__name__, str(op_version), runtime)
        newclass = type(
            name, (wrapper_onnxnumpy,), {'__doc__': fct.__doc__})
        _created_classes_inst.append(name, newclass)
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


class wrapper_onnxnumpy_np:
    """
    Intermediate wrapper to store a pointer
    on the compiler (type: @see cl OnnxNumpyCompiler)
    supporting multiple signatures.

    .. versionadded:: 0.6
    """

    def __init__(self, **kwargs):
        self.fct = kwargs['fct']
        self.signature = kwargs['signature']
        self.args, self.kwargs = get_args_kwargs(
            self.fct,
            0 if self.signature is None else self.signature.n_optional)
        self.data = kwargs
        self.signed_compiled = {}

    def __getstate__(self):
        """
        Serializes everything but the function which generates
        the ONNX graph, not needed anymore.
        """
        data_copy = {k: v for k, v in self.data.items() if k != 'fct'}
        return dict(signature=self.signature, args=self.args,
                    kwargs=self.kwargs, data=data_copy,
                    signed_compiled=self.signed_compiled)

    def __setstate__(self, state):
        """
        Restores serialized data.
        """
        for k, v in state.items():
            setattr(self, k, v)

    def __getitem__(self, dtype):
        """
        Returns the instance of @see cl wrapper_onnxnumpy
        mapped to *dtype*.

        :param dtype: numpy dtype corresponding to the input dtype
            of the function
        :return: instance of @see cl wrapper_onnxnumpy
        """
        if isinstance(dtype, dict):
            if len(self.args) == 0:
                raise RuntimeError(  # pragma: no cover
                    "Signature does not have any arguments, use directly dtypes.")
            others = tuple(dtype.get(k, self.kwargs[k]) for k in self.kwargs)
            inkey = (dtype['dtype_onnx']
                     if isinstance(dtype['dtype_onnx'], tuple)
                     else (dtype['dtype_onnx'], ))
            key = inkey + others
            self._populate(key)
        elif dtype not in self.signed_compiled:
            self._populate(dtype)
            key = dtype
        else:
            key = dtype
        return self.signed_compiled[key]

    def __call__(self, *args, **kwargs):
        """
        Calls the compiled function assuming the type of the first
        tensor in *args* defines the templated version of the function
        to convert into *ONNX*.
        """
        key = tuple(a if a is None else a.dtype.type for a in args)
        if len(self.kwargs) == 0:
            return self[key](*args)
        others = tuple(kwargs.get(k, self.kwargs[k]) for k in self.kwargs)
        return self[key + others](*args)

    def _populate(self, version):
        """
        Creates the appropriate runtime for function *fct*
        """
        compiled = OnnxNumpyCompiler(
            fct=self.data["fct"], op_version=self.data["op_version"],
            runtime=self.data["runtime"], signature=self.data["signature"],
            version=version)
        newclass = type(
            "onnxnumpy_np_%s_%s_%s_%s" % (
                self.data["fct"].__name__, str(self.data["op_version"]),
                self.data["runtime"], str(version).split('.')[-1]),
            (wrapper_onnxnumpy,), {'__doc__': self.data["fct"].__doc__})

        self.signed_compiled[version] = newclass(compiled)


def onnxnumpy_np(op_version=None, runtime=None, signature=None):
    """
    Decorator to declare a function implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: see @see cl OnnxInference
    :param signature: it should be used when the function
        is not annoatated.

    Equivalent to `onnxnumpy(arg)(foo)`.

    .. versionadded:: 0.6
    """
    def decorator_fct(fct):
        name = "onnxnumpy_nb_%s_%s_%s" % (
            fct.__name__, str(op_version), runtime)
        newclass = type(
            name, (wrapper_onnxnumpy_np,), {
                '__doc__': fct.__doc__,
                '__getstate__': wrapper_onnxnumpy_np.__getstate__,
                '__setstate__': wrapper_onnxnumpy_np.__setstate__})
        _created_classes_inst.append(name, newclass)
        return newclass(
            fct=fct, op_version=op_version, runtime=runtime,
            signature=signature)

    return decorator_fct
