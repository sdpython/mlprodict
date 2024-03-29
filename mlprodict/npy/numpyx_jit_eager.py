"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .numpyx_var import Input, Var
from .numpyx_tensors import (
    BackendNumpyTensor, EagerNumpyTensor, BackendEagerTensor)
from .numpyx_types import EagerNotAllowedError, TensorType


class JitEager:
    """
    Converts a function into an executable function
    based on a backend. The new function is converted
    to onnx on the first call.

    :param f: function to convert
    :param tensor_class: wrapper around a class defining the backend,
        if None, it defaults to :class:`onnx.reference.ReferenceEvalutor`
    :param target_opsets: dictionary `{opset: version}`
    :param output_types: shape and type inference cannot be run before
        the onnx graph is created and type is needed to do such,
        if not specified, the class assumes there is only one output
        of the same type as the input
    :param ir_version: defines the IR version to use
    """

    def __init__(self, f: Callable, tensor_class: type,
                 target_opsets: Optional[Dict[str, int]] = None,
                 output_types: Optional[Dict[Any, TensorType]] = None,
                 ir_version: Optional[int] = None):
        self.f = f
        self.tensor_class = tensor_class
        self.versions = {}
        self.onxs = {}
        self.target_opsets = tensor_class.get_opsets(target_opsets)
        self.output_types = output_types
        self.ir_version = tensor_class.get_ir_version(ir_version)

    @property
    def n_versions(self):
        """
        Returns the number of jitted functions.
        There is one per type and number of dimensions.
        """
        return len(self.onxs)

    @property
    def available_versions(self):
        """
        Returns the key used to distinguish between every jitted version.
        """
        return list(sorted(self.onxs))

    def get_onnx(self, key: Optional[int] = None):
        """
        Returns the jitted function associated to one key.
        If key is None, the assumes there is only one available jitted function
        and it returns it.
        """
        if key is None:
            if len(self.onxs) != 1:
                raise ValueError(
                    f"There is more than one jitted function. "
                    f"The key must be specified among "
                    f"{self.available_versions!r}.")
            return self.onxs[self.available_versions[0]]
        if key not in self.onxs:
            raise ValueError(
                f"Not jitted function indexed with "
                f"key={key!r} in {self.available_versions!r}.")
        return self.onxs[key]

    @staticmethod
    def make_key(*values, **kwargs):
        """
        Builds a key based on the input types and parameters.
        Every set of inputs or parameters producing the same
        key (or signature) must use the same compiled ONNX.
        """
        if len(kwargs) == 0:
            key = tuple(v.key for v in values)
        else:
            res = [v.key for v in values]
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (int, float, str)):
                    res.append(k)
                    res.append(v)
                else:
                    raise TypeError(
                        f"Type {type(v)} is not yet supported, "
                        f"v={v} and parameter {k!r}.")
            key = tuple(res)
        return key

    def to_jit(self, *values, **kwargs):
        """
        Converts the function into ONNX based on the provided inputs
        and parameters. It then wraps it by calling
        `self.tensor_class.create_function`.
        The onnx graph built by the function defines the input
        types and the expected number of dimensions.
        """
        constraints = {f"x{i}": v.tensor_type_dims
                       for i, v in enumerate(values)}
        if self.output_types is not None:
            constraints.update(self.output_types)
        inputs = [Input(f"x{i}") for i in range(len(values))]
        var = self.f(*inputs, **kwargs)
        onx = var.to_onnx(constraints=constraints,
                          target_opsets=self.target_opsets,
                          ir_version=self.ir_version)
        names = [f"x{i}" for i in range(len(values))]
        exe = self.tensor_class.create_function(names, onx)
        return onx, exe

    def cast_to_tensor_class(self, inputs: List[Any]) -> List[BackendEagerTensor]:
        """
        Wraps input into `self.tensor_class`.

        :param inputs: python inputs (including numpy)
        :return: wrapped inputs
        """
        values = []
        for i, a in enumerate(inputs):
            try:
                values.append(self.tensor_class(a))
            except TypeError as e:
                raise TypeError(
                    f"Unable to convert input {i}, with type {type(a)}.") from e
        return values

    def cast_from_tensor_class(self, results: List[BackendEagerTensor]
                               ) -> Union[Any, Tuple[Any]]:
        """
        Wraps input from `self.tensor_class` to python types.

        :param results: python inputs (including numpy)
        :return: wrapped inputs
        """
        if isinstance(results, (tuple, list)):
            if len(results) == 1:
                return results[0].value
            return tuple(r.value for r in results)
        return results.value

    def jit_call(self, *values, **kwargs):
        """
        The method builds a key which identifies the signature
        (input types + parameters value).
        It then checks if the function was already converted into ONNX
        from a previous. If not, it converts it and caches the results
        indexed by the previous key. Finally, it executes the onnx graph
        and returns the result or the results in a tuple if there are several.
        """
        key = self.make_key(*values, **kwargs)
        if key in self.versions:
            fct = self.versions[key]
        else:
            onx, fct = self.to_jit(*values, **kwargs)
            self.versions[key] = fct
            self.onxs[key] = onx
        res = fct.run(*values)
        return res


class JitOnnx(JitEager):
    """
    Converts a function into an executable function
    based on a backend. The new function is converted
    to onnx on the first call.

    :param f: function to convert
    :param tensor_class: wrapper around a class defining the backend,
        if None, it defaults to :class:`onnx.reference.ReferenceEvalutor`
    :param target_opsets: dictionary `{opset: version}`
    :param output_types: shape and type inference cannot be run before
        the onnx graph is created and type is needed to do such,
        if not specified, the class assumes there is only one output
        of the same type as the input
    :param ir_version: defines the IR version to use
    """

    def __init__(self, f: Callable, tensor_class: type = None,
                 target_opsets: Optional[Dict[str, int]] = None,
                 output_types: Optional[Dict[Any, TensorType]] = None,
                 ir_version: Optional[int] = None):
        if tensor_class is None:
            tensor_class = BackendNumpyTensor
        JitEager.__init__(self, f, tensor_class, target_opsets=target_opsets,
                          output_types=output_types)

    def __call__(self, *args, **kwargs):
        """
        The method builds a key which identifies the signature
        (input types + parameters value).
        It then checks if the function was already converted into ONNX
        from a previous. If not, it converts it and caches the results
        indexed by the previous key. Finally, it executes the onnx graph
        and returns the result or the results in a tuple if there are several.
        The method first wraps the inputs with `self.tensor_class`
        and converts them into python types just after.
        """
        values = self.cast_to_tensor_class(args)
        res = self.jit_call(*values, **kwargs)
        return self.cast_from_tensor_class(res)


class EagerOnnx(JitEager):
    """
    Converts a function into an executable function
    based on a backend. The new function is converted
    to onnx on the first call.

    :param f: function to convert
    :param tensor_class: wrapper around a class defining the backend,
        if None, it defaults to :class:`onnx.reference.ReferenceEvalutor`
    :param target_opsets: dictionary `{opset: version}`
    :param output_types: shape and type inference cannot be run before
        the onnx graph is created and type is needed to do such,
        if not specified, the class assumes there is only one output
        of the same type as the input
    :param ir_version: defines the IR version to use
    """

    def __init__(self, f: Callable, tensor_class: type = None,
                 target_opsets: Optional[Dict[str, int]] = None,
                 output_types: Optional[Dict[Any, TensorType]] = None,
                 ir_version: Optional[int] = None):
        if tensor_class is None:
            tensor_class = EagerNumpyTensor
        JitEager.__init__(self, f, tensor_class, target_opsets=target_opsets,
                          output_types=output_types)
        self.has_eager_parameter = "eager" in set(
            p for p in signature(f).parameters)
        self._eager_cache = False

    def __call__(self, *args, **kwargs):
        """
        The method builds a key which identifies the signature
        (input types + parameters value).
        It then checks if the function was already converted into ONNX
        from a previous. If not, it converts it and caches the results
        indexed by the previous key. Finally, it executes the onnx graph
        and returns the result or the results in a tuple if there are several.
        """
        values = self.cast_to_tensor_class(args)

        if self._eager_cache:
            # The function was already converted into onnx
            # reuse it or create a new one for different types.
            res = self.jit_call(*values, **kwargs)
        else:
            # tries to call the version
            jit_call = False
            try:
                res = self.f(*values)
            except EagerNotAllowedError:
                jit_call = True
            except (AttributeError, TypeError) as e:
                inp1 = ", ".join(map(str, map(type, args)))
                inp2 = ", ".join(map(str, map(type, values)))
                raise TypeError(
                    f"Unexpected types, input types is {inp1} "
                    f"and {inp2}.") from e

            if (jit_call or isinstance(res, Var) or
                    any(map(lambda x: isinstance(x, Var), res))):
                # The function returns instance of type Var.
                # It does not support eager mode and needs
                # to be converted into onnx.
                res = self.jit_call(*values, **kwargs)
                self._eager_cache = True
        return self.cast_from_tensor_class(res)


def jit_onnx(*args, **kwargs):
    """
    Returns an instance of :class:`JitOnnx`.
    """
    return JitOnnx(*args, **kwargs)


def eager_onnx(*args, **kwargs):
    """
    Returns an instance of :class:`EagerOnnx`.
    """
    return EagerOnnx(*args, **kwargs)
