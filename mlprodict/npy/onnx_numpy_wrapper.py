"""
@file
@brief Wraps :epkg:`numpy` functions into :epkg:`onnx`.

.. versionadded:: 0.6
"""
import warnings
from .onnx_version import FctVersion
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
            warnings.warn(  # pragma: no cover
                "Class %r overwritten in\n%r\n---\n%r" % (
                    name, ", ".join(sorted(self.stored)), cl),
                RuntimeWarning)
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
        from .onnx_variable import OnnxVar
        try:
            return self.compiled(*args, **kwargs)
        except (TypeError, RuntimeError, ValueError) as e:
            if any(map(lambda a: isinstance(a, OnnxVar), args)):
                return self.__class__.__fct__(  # pylint: disable=E1101
                    *args, **kwargs)
            raise RuntimeError(
                "Unable to call the compiled version, args is %r. "
                "kwargs=%r." % ([type(a) for a in args], kwargs)) from e

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

    def to_onnx(self, **kwargs):
        """
        Returns the ONNX graph for the wrapped function.
        It takes additional arguments to distinguish between multiple graphs.
        This happens when a function needs to support multiple type.

        :return: ONNX graph
        """
        return self.compiled.to_onnx(**kwargs)


def onnxnumpy(op_version=None, runtime=None, signature=None):
    """
    Decorator to declare a function implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: `'onnxruntime'` or one implemented by
        @see cl OnnxInference
    :param signature: it should be used when the function
        is not annoatated.

    Equivalent to `onnxnumpy(arg)(foo)`.

    .. versionadded:: 0.6
    """
    def decorator_fct(fct):
        compiled = OnnxNumpyCompiler(
            fct, op_version=op_version, runtime=runtime,
            signature=signature)
        name = f"onnxnumpy_{fct.__name__}_{str(op_version)}_{runtime}"
        newclass = type(
            name, (wrapper_onnxnumpy,),
            {'__doc__': fct.__doc__, '__name__': name, '__fct__': fct})
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
        self.fctsig = kwargs.get('fctsig', None)
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
        if not isinstance(dtype, FctVersion):
            raise TypeError(  # pragma: no cover
                f"dtype must be of type 'FctVersion' not {type(dtype)}: {dtype}.")
        if dtype not in self.signed_compiled:
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
        from .onnx_variable import OnnxVar
        if len(self.kwargs) == 0:
            others = None
        else:
            others = tuple(kwargs.get(k, self.kwargs[k]) for k in self.kwargs)
        try:
            key = FctVersion(  # pragma: no cover
                tuple(a if (a is None or hasattr(a, 'fit'))
                      else a.dtype.type for a in args),
                others)
            return self[key](*args)
        except AttributeError as e:
            if any(map(lambda a: isinstance(a, OnnxVar), args)):
                return self.__class__.__fct__(  # pylint: disable=E1101
                    *args, **kwargs)
            raise RuntimeError(
                "Unable to call the compiled version, args is %r. "
                "kwargs=%r." % ([type(a) for a in args], kwargs)) from e

    def _populate(self, version):
        """
        Creates the appropriate runtime for function *fct*
        """
        compiled = OnnxNumpyCompiler(
            fct=self.data["fct"], op_version=self.data["op_version"],
            runtime=self.data["runtime"], signature=self.data["signature"],
            version=version, fctsig=self.data.get('fctsig', None))
        name = "onnxnumpy_np_%s_%s_%s_%s" % (
            self.data["fct"].__name__, str(self.data["op_version"]),
            self.data["runtime"], version.as_string())
        newclass = type(
            name, (wrapper_onnxnumpy,),
            {'__doc__': self.data["fct"].__doc__, '__name__': name})

        self.signed_compiled[version] = newclass(compiled)

    def _validate_onnx_data(self, X):
        return X

    def to_onnx(self, **kwargs):
        """
        Returns the ONNX graph for the wrapped function.
        It takes additional arguments to distinguish between multiple graphs.
        This happens when a function needs to support multiple type.

        :return: ONNX graph
        """
        if len(self.signed_compiled) == 0:
            raise RuntimeError(  # pragma: no cover
                "No ONNX graph was compiled.")
        if len(kwargs) == 0 and len(self.signed_compiled) == 1:
            # We take the only one.
            key = list(self.signed_compiled)[0]
            cpl = self.signed_compiled[key]
            return cpl.to_onnx()
        if len(kwargs) == 0:
            raise ValueError(
                "There are multiple compiled ONNX graphs associated "
                "with keys %r (add key=...)." % list(self.signed_compiled))
        if list(kwargs) != ['key']:
            raise ValueError(
                f"kwargs should contain one parameter key=... but it is {kwargs!r}.")
        key = kwargs['key']
        if key in self.signed_compiled:
            return self.signed_compiled[key].compiled.onnx_
        found = []
        for k, v in self.signed_compiled.items():
            if k.args == key:
                found.append((k, v))
            elif isinstance(key, tuple) and k.args == key:
                found.append((k, v))
            elif k.args == (key, ) * len(k.args):
                found.append((k, v))
        if len(found) == 1:
            return found[0][1].compiled.onnx_
        raise ValueError(
            "Unable to find signature with key=%r among %r found=%r." % (
                key, list(self.signed_compiled), found))


def onnxnumpy_np(op_version=None, runtime=None, signature=None):
    """
    Decorator to declare a function implemented using
    :epkg:`numpy` syntax but executed with :epkg:`ONNX`
    operators.

    :param op_version: :epkg:`ONNX` opset version
    :param runtime: `'onnxruntime'` or one implemented by @see cl OnnxInference
    :param signature: it should be used when the function
        is not annoatated.

    Equivalent to `onnxnumpy(arg)(foo)`.

    .. versionadded:: 0.6
    """
    def decorator_fct(fct):
        name = f"onnxnumpy_nb_{fct.__name__}_{str(op_version)}_{runtime}"
        newclass = type(
            name, (wrapper_onnxnumpy_np,), {
                '__doc__': fct.__doc__,
                '__name__': name,
                '__getstate__': wrapper_onnxnumpy_np.__getstate__,
                '__setstate__': wrapper_onnxnumpy_np.__setstate__,
                '__fct__': fct})
        _created_classes_inst.append(name, newclass)
        return newclass(
            fct=fct, op_version=op_version, runtime=runtime,
            signature=signature)

    return decorator_fct
